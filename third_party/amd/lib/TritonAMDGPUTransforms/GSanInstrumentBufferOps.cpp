#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANINSTRUMENTBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace amdgpu = mlir::triton::amdgpu;

namespace {

// The bound of a buffer op with a pointer base: the compiler-built
// descriptor's num_records (see BufferEmitter::createResourceDescriptor).
constexpr int64_t kDefaultNumRecords = std::numeric_limits<int32_t>::max() - 1;

// The global base pointer and bound (null: the default) of a buffer op.
struct BufferBase {
  Value ptr;
  Value numRecords;
};

// A descriptor is traced only to an amdg.make_buffer_desc that defines it
// directly. Block arguments (function arguments, loop-carried values) and
// values chosen at run time (select) have no single known base and bound.
FailureOr<BufferBase> getBufferBase(Operation *op, Value base) {
  if (isa<tt::PointerType>(base.getType()))
    return BufferBase{base, Value()};
  if (auto makeDesc = base.getDefiningOp<amdgpu::MakeBufferDescOp>())
    return BufferBase{makeDesc.getBase(), makeDesc.getNumRecords()};
  op->emitError("GSan cannot find the amdg.make_buffer_desc of the buffer "
                "descriptor; it must be built in the same function (not "
                "passed as an argument, loop-carried, or selected), so that "
                "GSan knows its base and bound");
  return failure();
}

// AxisInfo hints for the reconstructed address and access mask of a buffer
// load or store. GSan merges contiguous checked elements into one access only
// if the analysis proves the address contiguous and the mask constant over
// them; it must, because 16-byte shadow pools require whole 16-byte accesses.
// The reconstruction goes through byte offsets and an i8 pointer, which the
// analysis cannot see through, so its facts are derived here from the buffer
// op's operands, before any rewriting.
struct AccessHints {
  SmallVector<int32_t> contiguity, divisibility, constancy, maskConstancy;
};

std::optional<AccessHints> computeHints(tt::ModuleAxisInfoAnalysis &axisInfo,
                                        Value basePtr, Value offsets,
                                        Value soffset, Value mask,
                                        int64_t elemBytes) {
  auto offsetsTy = cast<RankedTensorType>(offsets.getType());
  const tt::AxisInfo *baseInfo = axisInfo.getAxisInfo(basePtr);
  const tt::AxisInfo *offInfo = axisInfo.getAxisInfo(offsets);
  const tt::AxisInfo *soffInfo =
      soffset ? axisInfo.getAxisInfo(soffset) : nullptr;
  const tt::AxisInfo *maskInfo = mask ? axisInfo.getAxisInfo(mask) : nullptr;
  if (!baseInfo || !offInfo || (soffset && !soffInfo) || (mask && !maskInfo))
    return std::nullopt;
  bool tensorSoffset = soffset && isa<RankedTensorType>(soffset.getType());
  // An i32 offset that is a multiple of d gives a byte offset
  // (elemBytes * offset mod 2^32) that is a multiple of elemBytes * d, as
  // long as that divides 2^32.
  auto byteDivisibility = [&](int64_t div) {
    return elemBytes * std::min<int64_t>(div, int64_t(1) << 28);
  };
  AccessHints hints;
  for (int d = 0, rank = offsetsTy.getRank(); d < rank; ++d) {
    int64_t contig = offInfo->getContiguity(d);
    int64_t constancy = offInfo->getConstancy(d);
    int64_t div = std::gcd(std::min<int64_t>(baseInfo->getDivisibility(0),
                                             int64_t(1) << 31),
                           byteDivisibility(offInfo->getDivisibility(d)));
    if (tensorSoffset) {
      // The same rules as AxisInfo uses for an addition.
      contig = std::max(std::gcd(contig, soffInfo->getConstancy(d)),
                        std::gcd(constancy, soffInfo->getContiguity(d)));
      constancy = std::gcd(constancy, soffInfo->getConstancy(d));
    }
    if (soffset)
      div = std::gcd(div, byteDivisibility(
                              soffInfo->getDivisibility(tensorSoffset ? d : 0)));
    // Along a contiguous run the accessed bytes increase, so once an element
    // is out of bounds all later ones are. Claiming the in-bounds mask
    // constant over the run makes a merged check count the run if its first
    // element is in bounds: this never drops an access, but counts the
    // out-of-bounds tail of a run that the bound cuts.
    int64_t maskConstancy = std::max(contig, constancy);
    if (maskInfo)
      maskConstancy = std::min(maskConstancy, maskInfo->getConstancy(d));
    hints.contiguity.push_back(contig);
    hints.divisibility.push_back(div);
    hints.constancy.push_back(constancy);
    hints.maskConstancy.push_back(maskConstancy);
  }
  return hints;
}

DenseIntElementsAttr getHintAttr(MLIRContext *ctx, ArrayRef<int32_t> values) {
  auto ty = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                  IntegerType::get(ctx, 32));
  return DenseIntElementsAttr::get(ty, values);
}

// The tensor of global addresses a buffer op accesses, and the mask of the
// elements it actually accesses.
struct BufferAccess {
  Value ptr;
  Value mask;
};

// Mirrors the buffer op lowering (BufferEmitter) and the hardware range
// check (see amdg.make_buffer_desc):
// - byte offsets: voff = elemBytes * offsets and soff = elemBytes * soffset,
//   each wrapping in i32 and read as unsigned;
// - in bounds: zext(soff) + zext(voff) + size <= zext(num_records), without
//   wrapping, where size is min(elemBytes, 4) because the hardware checks
//   wider accesses per 4-byte piece: an element whose first 4 bytes are in
//   bounds is (at least partly) accessed and counted as fully accessed;
// - address: base + zext(soff) + zext(voff), in bytes.
// A masked-off element gets an out-of-bounds offset in the lowering, and an
// out-of-bounds one is dropped by the hardware, so neither accesses memory.
// The stride, cache policy, contiguity, and (non-atomic) scope of a buffer op
// affect neither which bytes are accessed nor synchronization.
BufferAccess buildBufferAccess(OpBuilder &b, Location loc, BufferBase base,
                               Value offsets, Value soffset, Value userMask,
                               Type elemTy) {
  auto offsetsTy = cast<RankedTensorType>(offsets.getType());
  auto i64TensorTy = offsetsTy.clone(b.getI64Type());
  int64_t elemBytes = std::max(8u, elemTy.getIntOrFloatBitWidth()) / 8;

  Value elemBytesTensor = arith::ConstantOp::create(
      b, loc, DenseElementsAttr::get(offsetsTy, b.getI32IntegerAttr(elemBytes)));
  Value voff = arith::MulIOp::create(b, loc, offsets, elemBytesTensor);
  Value bytes = arith::ExtUIOp::create(b, loc, i64TensorTy, voff);
  if (soffset) {
    Value soffTensor;
    if (isa<RankedTensorType>(soffset.getType())) {
      Value soff = arith::MulIOp::create(b, loc, soffset, elemBytesTensor);
      soffTensor = arith::ExtUIOp::create(b, loc, i64TensorTy, soff);
    } else {
      Value soff = arith::MulIOp::create(
          b, loc, soffset,
          arith::ConstantIntOp::create(b, loc, elemBytes, /*width=*/32));
      Value soff64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), soff);
      soffTensor = tt::SplatOp::create(b, loc, i64TensorTy, soff64);
    }
    bytes = arith::AddIOp::create(b, loc, soffTensor, bytes);
  }

  auto i64Splat = [&](int64_t value) -> Value {
    return arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(i64TensorTy, b.getI64IntegerAttr(value)));
  };
  Value end = arith::AddIOp::create(b, loc, bytes,
                                    i64Splat(std::min<int64_t>(elemBytes, 4)));
  Value bound;
  if (base.numRecords) {
    Value numRecords64 =
        arith::ExtUIOp::create(b, loc, b.getI64Type(), base.numRecords);
    bound = tt::SplatOp::create(b, loc, i64TensorTy, numRecords64);
  } else {
    bound = i64Splat(kDefaultNumRecords);
  }
  Value mask =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, end, bound);
  if (userMask)
    mask = arith::AndIOp::create(b, loc, userMask, mask);

  auto basePtrTy = cast<tt::PointerType>(base.ptr.getType());
  auto bytePtrTy =
      tt::PointerType::get(b.getI8Type(), basePtrTy.getAddressSpace());
  Value bytePtr = tt::BitcastOp::create(b, loc, bytePtrTy, base.ptr);
  Value bytePtrs =
      tt::SplatOp::create(b, loc, offsetsTy.clone(bytePtrTy), bytePtr);
  bytePtrs = tt::AddPtrOp::create(b, loc, bytePtrs.getType(), bytePtrs, bytes);
  Value ptrs = tt::BitcastOp::create(b, loc, offsetsTy.clone(basePtrTy),
                                     bytePtrs);
  return {ptrs, mask};
}

Type getElementType(Value tensor) {
  return cast<RankedTensorType>(tensor.getType()).getElementType();
}

struct TritonAMDGPUGSanInstrumentBufferOpsPass
    : impl::TritonAMDGPUGSanInstrumentBufferOpsBase<
          TritonAMDGPUGSanInstrumentBufferOpsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<Operation *> bufferOps;
    mod.walk([&](Operation *op) {
      if (isa<amdgpu::BufferLoadOp, amdgpu::BufferStoreOp,
              amdgpu::BufferAtomicRMWOp, amdgpu::BufferAtomicCASOp,
              amdgpu::BufferAtomicLoadOp, amdgpu::BufferAtomicStoreOp>(op))
        bufferOps.push_back(op);
    });
    if (bufferOps.empty())
      return;

    // Hints for the checks of loads and stores, computed before rewriting.
    tt::ModuleAxisInfoAnalysis axisInfo(mod);
    DenseMap<Operation *, AccessHints> hints;
    bool failed = false;
    for (Operation *op : bufferOps) {
      Value base, offsets, soffset, mask, data;
      if (auto load = dyn_cast<amdgpu::BufferLoadOp>(op)) {
        std::tie(base, offsets, soffset, mask, data) =
            std::tuple{load.getPtr(), load.getOffsets(), load.getSoffset(),
                       load.getMask(), load.getResult()};
      } else if (auto store = dyn_cast<amdgpu::BufferStoreOp>(op)) {
        std::tie(base, offsets, soffset, mask, data) =
            std::tuple{store.getPtr(), store.getOffsets(), store.getSoffset(),
                       store.getMask(), store.getValue()};
      } else {
        continue;
      }
      auto bufferBase = getBufferBase(op, base);
      if (::mlir::failed(bufferBase)) {
        failed = true;
        continue;
      }
      Type elemTy = getElementType(data);
      if (!elemTy.isIntOrFloat())
        continue;
      int64_t elemBytes = std::max(8u, elemTy.getIntOrFloatBitWidth()) / 8;
      if (auto h = computeHints(axisInfo, bufferBase->ptr, offsets, soffset,
                                mask, elemBytes))
        hints[op] = std::move(*h);
    }
    if (failed)
      return signalPassFailure();

    for (Operation *op : bufferOps) {
      if (::mlir::failed(instrument(op, hints.lookup(op))))
        failed = true;
    }
    if (failed)
      signalPassFailure();
  }

  // Builds the access of `op`, or fails with an error.
  FailureOr<BufferAccess> getAccess(OpBuilder &b, Operation *op, Value base,
                                    Value offsets, Value soffset, Value mask,
                                    Value data) {
    auto bufferBase = getBufferBase(op, base);
    if (::mlir::failed(bufferBase))
      return failure();
    Type elemTy = getElementType(data);
    if (!elemTy.isIntOrFloat()) {
      op->emitError("GSan cannot check a buffer op on elements of type ")
          << elemTy;
      return failure();
    }
    return buildBufferAccess(b, op->getLoc(), *bufferBase, offsets, soffset,
                             mask, elemTy);
  }

  void setHints(BufferAccess access, const AccessHints &hints) {
    if (hints.contiguity.empty())
      return;
    MLIRContext *ctx = access.ptr.getContext();
    Operation *ptrOp = access.ptr.getDefiningOp();
    ptrOp->setDiscardableAttr("tt.contiguity",
                              getHintAttr(ctx, hints.contiguity));
    ptrOp->setDiscardableAttr("tt.divisibility",
                              getHintAttr(ctx, hints.divisibility));
    ptrOp->setDiscardableAttr("tt.constancy",
                              getHintAttr(ctx, hints.constancy));
    access.mask.getDefiningOp()->setDiscardableAttr(
        "tt.constancy", getHintAttr(ctx, hints.maskConstancy));
  }

  LogicalResult instrument(Operation *op, const AccessHints &hints) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    if (auto load = dyn_cast<amdgpu::BufferLoadOp>(op)) {
      auto access = getAccess(b, op, load.getPtr(), load.getOffsets(),
                              load.getSoffset(), load.getMask(),
                              load.getResult());
      if (::mlir::failed(access))
        return failure();
      setHints(*access, hints);
      tti::ExperimentalGSanTensorAccessOp::create(b, loc, access->ptr,
                                                  access->mask,
                                                  /*isStore=*/false);
      return success();
    }
    if (auto store = dyn_cast<amdgpu::BufferStoreOp>(op)) {
      auto access = getAccess(b, op, store.getPtr(), store.getOffsets(),
                              store.getSoffset(), store.getMask(),
                              store.getValue());
      if (::mlir::failed(access))
        return failure();
      setHints(*access, hints);
      tti::ExperimentalGSanTensorAccessOp::create(b, loc, access->ptr,
                                                  access->mask,
                                                  /*isStore=*/true);
      return success();
    }
    // GSan atomics perform the user atomic themselves, so the buffer atomic
    // is replaced (it becomes a global atomic). Inherent attributes such as
    // the operand segment sizes belong to the buffer op and are not copied.
    // A dropped (masked-off or out-of-bounds) element of a buffer atomic
    // reads zero on the hardware, while a GSan atomic leaves it undefined, so
    // results are selected to keep that behavior.
    auto replaceWithResult = [&](Operation *newOp, Value mask) {
      newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      if (op->getNumResults() == 0) {
        op->erase();
        return;
      }
      Value result = newOp->getResult(0);
      Value zero = arith::ConstantOp::create(
          b, loc, cast<TypedAttr>(b.getZeroAttr(result.getType())));
      result = arith::SelectOp::create(b, loc, mask, result, zero);
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    };
    if (auto rmw = dyn_cast<amdgpu::BufferAtomicRMWOp>(op)) {
      auto access = getAccess(b, op, rmw.getPtr(), rmw.getOffsets(),
                              /*soffset=*/Value(), rmw.getMask(),
                              rmw.getValue());
      if (::mlir::failed(access))
        return failure();
      replaceWithResult(tti::ExperimentalGSanAtomicRMWOp::create(
                            b, loc, rmw.getType(), rmw.getAtomicRmwOp(),
                            access->ptr, rmw.getValue(), access->mask,
                            rmw.getSem(), rmw.getScope()),
                        access->mask);
      return success();
    }
    if (auto cas = dyn_cast<amdgpu::BufferAtomicCASOp>(op)) {
      // Buffer CAS has no user mask, but its out-of-bounds lanes still must
      // not access memory.
      auto access =
          getAccess(b, op, cas.getPtr(), cas.getOffsets(), /*soffset=*/Value(),
                    /*mask=*/Value(), cas.getVal());
      if (::mlir::failed(access))
        return failure();
      replaceWithResult(tti::ExperimentalGSanAtomicCASOp::create(
                            b, loc, cas.getType(), access->ptr, cas.getCmp(),
                            cas.getVal(), access->mask, cas.getSem(),
                            cas.getScope()),
                        access->mask);
      return success();
    }
    if (auto atomicLoad = dyn_cast<amdgpu::BufferAtomicLoadOp>(op)) {
      auto access = getAccess(b, op, atomicLoad.getPtr(),
                              atomicLoad.getOffsets(), atomicLoad.getSoffset(),
                              atomicLoad.getMask(), atomicLoad.getResult());
      if (::mlir::failed(access))
        return failure();
      // Documented: masked-off and out-of-bounds elements read zero.
      replaceWithResult(tti::ExperimentalGSanAtomicLoadOp::create(
                            b, loc, atomicLoad.getType(), access->ptr,
                            access->mask, atomicLoad.getSem(),
                            atomicLoad.getScope()),
                        access->mask);
      return success();
    }
    auto atomicStore = cast<amdgpu::BufferAtomicStoreOp>(op);
    auto access = getAccess(b, op, atomicStore.getPtr(),
                            atomicStore.getOffsets(), atomicStore.getSoffset(),
                            atomicStore.getMask(), atomicStore.getValue());
    if (::mlir::failed(access))
      return failure();
    replaceWithResult(tti::ExperimentalGSanAtomicStoreOp::create(
                          b, loc, access->ptr, atomicStore.getValue(),
                          access->mask, atomicStore.getSem(),
                          atomicStore.getScope()),
                      access->mask);
    return success();
  }
};

} // namespace

} // namespace mlir
