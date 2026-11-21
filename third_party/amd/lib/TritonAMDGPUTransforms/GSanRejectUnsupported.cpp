#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUGSANREJECTUNSUPPORTED
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace amdgpu = mlir::triton::amdgpu;

namespace {

constexpr char kUncheckedOpsAttr[] = "tti.gsan_unchecked_ops";

// Global-memory operations that GSan instruments (AMD buffer ops through
// tritonamdgpu-gsan-instrument-buffer-ops), plus operations whose
// global-memory effects are not user data accesses (debug output, GSan's own
// runtime calls). amdg.make_buffer_desc is pure, so it is never rejected.
bool isHandledByGSan(Operation *op) {
  if (isa<tt::LoadOp, tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp,
          tt::AtomicLoadOp, tt::AtomicStoreOp, tt::AtomicPollOp, tt::PrintOp,
          tt::AssertOp, ttg::AsyncCopyGlobalToLocalOp,
          amdgpu::BufferLoadToLocalOp, amdgpu::AsyncCopyLocalToGlobalOp>(op))
    return true;
  if (isa<amdgpu::BufferLoadOp, amdgpu::BufferStoreOp,
          amdgpu::BufferAtomicRMWOp, amdgpu::BufferAtomicCASOp,
          amdgpu::BufferAtomicLoadOp, amdgpu::BufferAtomicStoreOp>(op))
    return true;
  if (isa<amdgpu::AsyncTDMCopyGlobalToLocalOp,
          amdgpu::AsyncTDMCopyLocalToGlobalOp,
          amdgpu::AsyncTDMFusedCopyGlobalToLocalOp, amdgpu::AsyncTDMGatherOp,
          amdgpu::AsyncTDMScatterOp>(op))
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
bool isUnsupportedSync(Operation *op, bool localBarriersSupported) {
  if (isa<amdgpu::InitBarrierOp, amdgpu::WaitBarrierOp, amdgpu::ArriveBarrierOp,
          amdgpu::AsyncCopyMbarrierArriveOp>(op))
    return !localBarriersSupported;
  return isa<amdgpu::ClusterBarrierArriveOp, amdgpu::ClusterBarrierWaitOp,
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
    // Like NVIDIA single-CTA GSan, ordinary AMD logical blocks share one
    // participant clock. Local named/completion barriers retain their hardware
    // semantics without importing clocks from any other block. Global async
    // accesses remain checked at issue time. Partial regions and clusters need
    // separate participation/clock contracts and are rejected below.
    auto target = mod->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    bool localBarriersSupported = target &&
                                  target.getValue() == "hip:gfx1250" &&
                                  ttg::lookupNumCTAs(mod) == 1;
    bool failed = false;
    int64_t numUnchecked = 0;
    if (auto numCTAs = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
        numCTAs && numCTAs.getInt() > 1) {
      mod.emitError("GSan on AMD does not support num_ctas > 1 (got ")
          << numCTAs.getInt() << ")";
      failed = true;
    }
    // Both runtime profiles support the lane waterfall and slot protocol.
    if (auto tpw = mod->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
        tpw && tpw.getInt() != 32 && tpw.getInt() != 64) {
      mod.emitError("GSan on AMD requires 32 or 64 threads per warp (got ")
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
      if (isUnsupportedSync(op, localBarriersSupported)) {
        op->emitError("GSan on AMD does not support synchronization "
                      "operation '")
            << op->getName() << "'";
        failed = true;
        return;
      }
      if (auto copy = dyn_cast<amdgpu::AsyncTDMCopyGlobalToLocalOp>(op);
          copy && copy.getBarrier() && !localBarriersSupported) {
        op->emitError("GSan on AMD does not support TDM completion barriers");
        failed = true;
        return;
      }
      if (auto copy = dyn_cast<amdgpu::AsyncTDMCopyLocalToGlobalOp>(op);
          copy && copy.getBarrier() && !localBarriersSupported) {
        op->emitError("GSan on AMD does not support TDM completion barriers");
        failed = true;
        return;
      }
      if (auto gather = dyn_cast<amdgpu::AsyncTDMGatherOp>(op);
          gather && gather.getBarrier() && !localBarriersSupported) {
        op->emitError("GSan on AMD does not support TDM completion barriers");
        failed = true;
        return;
      }
      if (auto scatter = dyn_cast<amdgpu::AsyncTDMScatterOp>(op);
          scatter && scatter.getBarrier() && !localBarriersSupported) {
        op->emitError("GSan on AMD does not support TDM completion barriers");
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
               "supported loads, stores, atomics, direct async copies, "
               "and TDM copies on AMD)";
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
