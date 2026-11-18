#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
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
    SmallVector<std::pair<Operation *, bool>> copies;
    getOperation().walk([&](Operation *op) {
      if (isa<amdgpu::AsyncTDMCopyGlobalToLocalOp>(op))
        copies.emplace_back(op, false);
      else if (isa<amdgpu::AsyncTDMCopyLocalToGlobalOp>(op))
        copies.emplace_back(op, true);
    });
    for (auto [op, isStore] : copies) {
      Value desc = op->getOperand(0);
      auto descTy = cast<triton::TensorDescType>(desc.getType());
      OpBuilder builder(op);
      Value zero = arith::ConstantIntOp::create(builder, op->getLoc(), 0, 32);
      SmallVector<Value> coords(descTy.getShape().size(), zero);
      // Pure TDM copies consume a descriptor already positioned by updates.
      // Its address, bounds, and predicate are recovered by the runtime.
      triton::instrument::ExperimentalGSanTensorDescAccessOp::create(
          builder, op->getLoc(), desc, coords, Value{}, isStore);
    }
  }
};

} // namespace
} // namespace mlir
