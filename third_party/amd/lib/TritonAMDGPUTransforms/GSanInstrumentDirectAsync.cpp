#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANINSTRUMENTDIRECTASYNC
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace ttg = mlir::triton::gpu;
namespace amdgpu = mlir::triton::amdgpu;

namespace {

// Recover an element's position within the issued dword.  Blocked layouts
// enumerate contiguous per-lane registers along their fastest dimension;
// require the offset and mask facts used by the lowering to permit at least
// a dword there.  This also supports unaligned scalar offsets: grouping is
// relative to the vector start, not the absolute global address.
FailureOr<Value> getSubwordPosition(amdgpu::BufferLoadToLocalOp op,
                                    OpBuilder &b, int64_t elemBytes,
                                    tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto ty = cast<RankedTensorType>(op.getOffsets().getType());
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
  int64_t pack = 4 / elemBytes;
  if (!blocked)
    return op.emitError("GSan sub-dword buffer async copies require a blocked "
                        "layout with provably contiguous dword groups");
  unsigned dim = blocked.getOrder()[0];
  const auto *offsetInfo = axisInfo.getAxisInfo(op.getOffsets());
  const auto *maskInfo =
      op.getMask() ? axisInfo.getAxisInfo(op.getMask()) : nullptr;
  if (blocked.getSizePerThread()[dim] < pack || ty.getShape()[dim] < pack ||
      (!offsetInfo || std::max<int64_t>(offsetInfo->getContiguity(dim),
                                        op.getContiguity()) < pack) ||
      (op.getMask() && (!maskInfo || maskInfo->getConstancy(dim) < pack)))
    return op.emitError(
        "GSan sub-dword buffer async copies require provably "
        "contiguous, uniformly masked dword groups in each lane");

  // Slice away every other dimension, then restore singleton dimensions to
  // broadcast the logical coordinate over the original blocked tensor.
  auto encoding = cast<ttg::DistributedEncodingTrait>(blocked);
  SmallVector<unsigned> sliced;
  for (int d = ty.getRank() - 1; d >= 0; --d) {
    if (d == dim)
      continue;
    encoding = cast<ttg::DistributedEncodingTrait>(
        ttg::SliceEncodingAttr::get(b.getContext(), d, encoding));
    sliced.push_back(d);
  }
  auto rangeTy =
      RankedTensorType::get({ty.getShape()[dim]}, b.getI32Type(), encoding);
  Value position =
      tt::MakeRangeOp::create(b, op.getLoc(), rangeTy, 0, ty.getShape()[dim]);
  for (unsigned d : llvm::reverse(sliced))
    position = tt::ExpandDimsOp::create(b, op.getLoc(), position, d);
  position = tt::BroadcastOp::create(b, op.getLoc(), ty, position);
  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantOp::create(
        b, op.getLoc(), DenseElementsAttr::get(ty, b.getI32IntegerAttr(value)));
  };
  position = arith::RemUIOp::create(b, op.getLoc(), position, constant(pack));
  return arith::MulIOp::create(b, op.getLoc(), position, constant(elemBytes))
      .getResult();
}

LogicalResult instrumentBufferCopy(amdgpu::BufferLoadToLocalOp op,
                                   tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value base = op.getPtr(), bound;
  if (isa<amdgpu::BufferDescType>(base.getType())) {
    auto makeDesc = base.getDefiningOp<amdgpu::MakeBufferDescOp>();
    if (!makeDesc)
      return op.emitError(
          "GSan direct async copies require a buffer descriptor "
          "defined by amdg.make_buffer_desc; opaque, selected, "
          "and loop-carried descriptors are unsupported");
    base = makeDesc.getBase();
    bound = makeDesc.getNumRecords();
  }
  auto baseTy = cast<tt::PointerType>(base.getType());
  Type elemTy = baseTy.getPointeeType();
  if (!elemTy.isIntOrFloat() && !isa<tt::PackedFloat4E2M1Type>(elemTy))
    return op.emitError("GSan cannot check direct buffer copies of ") << elemTy;
  int64_t elemBytes = std::max(8, getIntOrFloatOrPtrBitWidth(elemTy)) / 8;
  if (elemBytes > 4 && elemBytes % 4 != 0)
    return op.emitError("GSan direct buffer copies require whole dwords for "
                        "elements wider than four bytes");

  OpBuilder b(op);
  Location loc = op.getLoc();
  auto offsetTy = cast<RankedTensorType>(op.getOffsets().getType());
  auto i64Ty = offsetTy.clone(b.getI64Type());
  auto constant = [&](RankedTensorType ty, int64_t value) -> Value {
    return arith::ConstantOp::create(
        b, loc,
        DenseElementsAttr::get(ty,
                               b.getIntegerAttr(ty.getElementType(), value)));
  };

  // BufferEmitter multiplies voffset and soffset in i32 independently.
  // The hardware treats those wrapped fields as unsigned and adds them in
  // the address domain, without wrapping their sum at 32 bits.
  Value subwordPosition;
  if (elemBytes < 4) {
    auto position = getSubwordPosition(op, b, elemBytes, axisInfo);
    if (failed(position))
      return failure();
    subwordPosition = *position;
  }
  Value voff = arith::MulIOp::create(b, loc, op.getOffsets(),
                                     constant(offsetTy, elemBytes));
  Value bytes = arith::ExtUIOp::create(b, loc, i64Ty, voff);
  Value scalarBytes;
  if (Value soffset = op.getSoffset()) {
    Value soff = arith::MulIOp::create(
        b, loc, soffset, arith::ConstantIntOp::create(b, loc, elemBytes, 32));
    Value soff64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), soff);
    Value splat = tt::SplatOp::create(b, loc, i64Ty, soff64);
    scalarBytes = splat;
    bytes = arith::AddIOp::create(b, loc, bytes, splat);
  }
  Value bounds;
  if (bound) {
    Value bound64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), bound);
    bounds = tt::SplatOp::create(b, loc, i64Ty, bound64);
  } else {
    // Pointer-based buffer ops use this compiler-built num_records value.
    bounds = constant(i64Ty, std::numeric_limits<int32_t>::max() - 1);
  }
  auto bytePtrTy =
      tt::PointerType::get(b.getI8Type(), baseTy.getAddressSpace());
  Value byteBase = tt::BitcastOp::create(b, loc, bytePtrTy, base);
  Value basePtrs =
      tt::SplatOp::create(b, loc, offsetTy.clone(bytePtrTy), byteBase);

  // The hardware bounds-checks wide accesses per dword.  Separate checks
  // retain exact footprints when num_records cuts an element, including in
  // byte-granular shadow pools.  Never claim mask constancy across a dynamic
  // bound merely to merge these checks: that would count dropped tail bytes.
  Value subwordEnd;
  if (subwordPosition) {
    // Subtract the register's position before unsigned extension, so a
    // vector whose voffset crosses 2^32 still uses its first hardware field.
    Value groupOffset = arith::SubIOp::create(b, loc, voff, subwordPosition);
    Value groupStart = arith::ExtUIOp::create(b, loc, i64Ty, groupOffset);
    if (scalarBytes)
      groupStart = arith::AddIOp::create(b, loc, groupStart, scalarBytes);
    subwordEnd = arith::AddIOp::create(b, loc, groupStart, constant(i64Ty, 4));
  }
  int64_t pieceBytes = std::min<int64_t>(elemBytes, 4);
  Type pieceTy = elemBytes <= 4 ? elemTy : b.getI32Type();
  auto piecePtrTy = tt::PointerType::get(pieceTy, baseTy.getAddressSpace());
  for (int64_t piece = 0; piece < elemBytes; piece += pieceBytes) {
    Value start = bytes;
    if (piece)
      start = arith::AddIOp::create(b, loc, start, constant(i64Ty, piece));
    Value addr =
        tt::AddPtrOp::create(b, loc, basePtrs.getType(), basePtrs, start);
    Value end = subwordEnd;
    if (!end)
      end = arith::AddIOp::create(b, loc, start, constant(i64Ty, pieceBytes));
    Value mask =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, end, bounds);
    if (op.getMask())
      mask = arith::AndIOp::create(b, loc, op.getMask(), mask);
    Value ptr = tt::BitcastOp::create(b, loc, offsetTy.clone(piecePtrTy), addr);
    tti::ExperimentalGSanTensorAccessOp::create(b, loc, ptr, mask,
                                                /*isStore=*/false);
  }
  return success();
}

struct TritonAMDGPUGSanInstrumentDirectAsyncPass
    : impl::TritonAMDGPUGSanInstrumentDirectAsyncBase<
          TritonAMDGPUGSanInstrumentDirectAsyncPass> {
  void runOnOperation() override {
    SmallVector<Operation *> copies;
    getOperation().walk([&](Operation *op) {
      if (isa<amdgpu::BufferLoadToLocalOp, amdgpu::AsyncCopyLocalToGlobalOp>(
              op))
        copies.push_back(op);
    });
    if (copies.empty())
      return;
    bool failed = false;
    tt::ModuleAxisInfoAnalysis axisInfo(getOperation());
    for (Operation *op : copies) {
      if (auto copy = dyn_cast<amdgpu::BufferLoadToLocalOp>(op)) {
        if (::mlir::failed(instrumentBufferCopy(copy, axisInfo)))
          failed = true;
      } else if (auto copy = dyn_cast<amdgpu::AsyncCopyLocalToGlobalOp>(op)) {
        OpBuilder b(copy);
        tti::ExperimentalGSanTensorAccessOp::create(
            b, copy.getLoc(), copy.getDst(), copy.getMask(), /*isStore=*/true);
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
