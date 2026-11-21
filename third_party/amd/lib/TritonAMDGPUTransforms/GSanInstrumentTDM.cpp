#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANINSTRUMENTTDM
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace amdgpu = triton::amdgpu;

namespace {

struct TritonAMDGPUGSanInstrumentTDMPass
    : impl::TritonAMDGPUGSanInstrumentTDMBase<
          TritonAMDGPUGSanInstrumentTDMPass> {
  void runOnOperation() override {
    SmallVector<Operation *> copies;
    getOperation().walk([&](Operation *op) {
      if (isa<amdgpu::AsyncTDMCopyGlobalToLocalOp,
              amdgpu::AsyncTDMCopyLocalToGlobalOp,
              amdgpu::AsyncTDMFusedCopyGlobalToLocalOp,
              amdgpu::AsyncTDMGatherOp, amdgpu::AsyncTDMScatterOp>(op))
        copies.push_back(op);
    });
    for (Operation *op : copies) {
      OpBuilder builder(op);
      Value zero = arith::ConstantIntOp::create(builder, op->getLoc(), 0, 32);
      auto rectangular = [&](Value desc, bool isStore) {
        auto descTy = cast<triton::TensorDescType>(desc.getType());
        SmallVector<Value> coords(descTy.getShape().size(), zero);
        // Positioned descriptors retain dynamic address, bounds and predicate.
        triton::instrument::ExperimentalGSanTensorDescAccessOp::create(
            builder, op->getLoc(), desc, coords, Value{}, isStore);
      };
      auto indexed = [&](Value desc, Value indices, bool isStore) {
        auto ty = cast<RankedTensorType>(indices.getType());
        if (ty.getElementType().isInteger(16))
          indices = arith::ExtUIOp::create(
              builder, op->getLoc(), ty.clone(builder.getI32Type()), indices);
        triton::instrument::ExperimentalGSanIndexedTensorDescAccessOp::create(
            builder, op->getLoc(), desc, indices, zero, Value{}, isStore);
      };
      if (auto fused = dyn_cast<amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(op)) {
        // Each member is a full logical transfer, regardless of its issue
        // warps.
        for (Value desc : fused.getDescs())
          rectangular(desc, false);
      } else if (auto gather = dyn_cast<amdgpu::AsyncTDMGatherOp>(op)) {
        indexed(gather.getDesc(), gather.getSrcRowIndices(), false);
      } else if (auto scatter = dyn_cast<amdgpu::AsyncTDMScatterOp>(op)) {
        indexed(scatter.getDesc(), scatter.getDstRowIndices(), true);
      } else {
        rectangular(op->getOperand(0),
                    isa<amdgpu::AsyncTDMCopyLocalToGlobalOp>(op));
      }
    }
  }
};

} // namespace
} // namespace mlir
