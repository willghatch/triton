#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANINSTRUMENTDIRECTASYNC
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace amdgpu = mlir::triton::amdgpu;

namespace {

LogicalResult instrumentBufferCopy(amdgpu::BufferLoadToLocalOp op) {
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
  Value voff = arith::MulIOp::create(b, loc, op.getOffsets(),
                                     constant(offsetTy, elemBytes));
  Value bytes = arith::ExtUIOp::create(b, loc, i64Ty, voff);
  if (Value soffset = op.getSoffset()) {
    Value soff = arith::MulIOp::create(
        b, loc, soffset, arith::ConstantIntOp::create(b, loc, elemBytes, 32));
    Value soff64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), soff);
    Value splat = tt::SplatOp::create(b, loc, i64Ty, soff64);
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
  int64_t pieceBytes = std::min<int64_t>(elemBytes, 4);
  Type pieceTy = elemBytes <= 4 ? elemTy : b.getI32Type();
  auto piecePtrTy = tt::PointerType::get(pieceTy, baseTy.getAddressSpace());
  for (int64_t piece = 0; piece < elemBytes; piece += pieceBytes) {
    Value start = bytes;
    if (piece)
      start = arith::AddIOp::create(b, loc, start, constant(i64Ty, piece));
    Value addr =
        tt::AddPtrOp::create(b, loc, basePtrs.getType(), basePtrs, start);
    Value end =
        arith::AddIOp::create(b, loc, start, constant(i64Ty, pieceBytes));
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
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (auto copy = dyn_cast<amdgpu::BufferLoadToLocalOp>(op)) {
        if (::mlir::failed(instrumentBufferCopy(copy)))
          failed = true;
      } else if (auto copy = dyn_cast<amdgpu::AsyncCopyLocalToGlobalOp>(op)) {
        OpBuilder b(copy);
        tti::ExperimentalGSanTensorAccessOp::create(
            b, copy.getLoc(), copy.getDst(), copy.getMask(), /*isStore=*/true);
      }
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir
