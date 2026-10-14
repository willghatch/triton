#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANREJECTUNSUPPORTED
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace amdgpu = mlir::triton::amdgpu;

namespace {

constexpr char kUncheckedOpsAttr[] = "tti.gsan_unchecked_ops";

// Global-memory operations that GSan instruments, plus operations whose
// global-memory effects are not user data accesses (debug output, GSan's own
// runtime calls).
bool isHandledByGSan(Operation *op) {
  if (isa<tt::LoadOp, tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp,
          tt::AtomicLoadOp, tt::AtomicStoreOp, tt::AtomicPollOp, tt::PrintOp,
          tt::AssertOp>(op))
    return true;
  return isa<tt::instrument::TritonInstrumentDialect>(op->getDialect());
}

// Returns the kinds of memory an op reads or writes: global memory (data GSan
// must check) and/or an unspecified resource (e.g. non-pure inline assembly).
struct MemoryAccess {
  bool global = false;
  bool unknown = false;
};

MemoryAccess getMemoryAccess(Operation *op) {
  MemoryAccess access;
  // These access global memory but declare no memory effects.
  if (isa<amdgpu::MaskedLoadOp, amdgpu::MaskedStoreOp>(op)) {
    access.global = true;
    return access;
  }
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return access;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  for (const MemoryEffects::EffectInstance &e : effects) {
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(e.getEffect()))
      continue;
    if (e.getResource() == tt::GlobalMemory::get())
      access.global = true;
    else if (e.getResource() == SideEffects::DefaultResource::get())
      access.unknown = true;
  }
  return access;
}

// Synchronization operations that GSan does not model on AMD; they touch
// shared memory only, so the global-memory check misses them.
// cond_barrier phase-shifts warps (block pingpong, warp pipelining), which
// the GSan instrumentation has not been validated with.
bool isUnsupportedSync(Operation *op) {
  return isa<amdgpu::InitBarrierOp, amdgpu::WaitBarrierOp,
             amdgpu::ArriveBarrierOp, amdgpu::AsyncCopyMbarrierArriveOp,
             amdgpu::ClusterBarrierArriveOp, amdgpu::ClusterBarrierWaitOp,
             amdgpu::CondBarrierOp>(op);
}

bool isWarpPipelined(Operation *op) {
  return op->hasAttr("triton.warp_pipeline.stage") ||
         op->hasAttr("triton.warp_pipeline.pipelined_for");
}

struct TritonAMDGPUGSanRejectUnsupportedPass
    : impl::TritonAMDGPUGSanRejectUnsupportedBase<
          TritonAMDGPUGSanRejectUnsupportedPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool failed = false;
    int64_t numUnchecked = 0;
    if (auto numCTAs = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
        numCTAs && numCTAs.getInt() > 1) {
      mod.emitError("GSan on AMD does not support num_ctas > 1 (got ")
          << numCTAs.getInt() << ")";
      failed = true;
    }
    // The device runtime's lane waterfall and slot protocol assume wave64.
    if (auto tpw =
            mod->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
        tpw && tpw.getInt() != 64) {
      mod.emitError("GSan on AMD requires 64 threads per warp (got ")
          << tpw.getInt() << ")";
      failed = true;
    }
    mod.walk([&](Operation *op) {
      if (isa<ttg::WarpSpecializeOp>(op)) {
        op->emitError("GSan on AMD does not support warp specialization");
        failed = true;
        return;
      }
      if (isWarpPipelined(op)) {
        op->emitError("GSan on AMD does not support warp pipelining");
        failed = true;
        return;
      }
      if (isUnsupportedSync(op)) {
        op->emitError("GSan on AMD does not support synchronization "
                      "operation '")
            << op->getName() << "'";
        failed = true;
        return;
      }
      if (isHandledByGSan(op))
        return;
      MemoryAccess access = getMemoryAccess(op);
      if (access.global) {
        op->emitError("GSan on AMD cannot check global-memory operation '")
            << op->getName()
            << "'; it would escape race detection (GSan instruments only "
               "tt.load, tt.store, and tt.atomic_* on AMD)";
        failed = true;
      } else if (access.unknown) {
        // Not an error: e.g. sleep loops use non-pure inline assembly.
        op->emitWarning("GSan cannot see memory accesses of '")
            << op->getName() << "'; any global-memory access it makes is "
            << "not race-checked";
        ++numUnchecked;
      }
    });
    if (failed)
      return signalPassFailure();
    // MLIR warnings are hidden by default in Triton, so record the count for
    // the compiler driver to report.
    if (numUnchecked)
      mod->setAttr(kUncheckedOpsAttr,
                   IntegerAttr::get(IntegerType::get(mod.getContext(), 32),
                                    numUnchecked));
  }
};

} // namespace

} // namespace mlir
