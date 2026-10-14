#include "GSan.h"
#include "Hash.cuh"
#include "TMA.cuh"

// The algorithm below is shared between targets. Target-specific primitives
// live in the adapter sections guarded by GSAN_AMDGPU:
// - NVIDIA: built as CUDA device code (one identity per SM).
// - AMD: built as freestanding amdgcn C++ (identity = slot table entry, see
//   SlotEntry in GSan.h). Lanes of a wave cannot spin on each other's locks,
//   so load/store checks lock wave-cooperatively (tensorAccessCooperative)
//   and other per-lane locking paths run lane by lane (forEachLaneSerially,
//   and compiler-emitted atomics).
#if defined(__AMDGCN__)
#define GSAN_AMDGPU 1
#else
#define GSAN_AMDGPU 0
#endif

#if GSAN_AMDGPU
extern "C" {
GSAN_DEVICE __UINT64_TYPE__ __ockl_fprintf_stderr_begin();
GSAN_DEVICE __UINT64_TYPE__
__ockl_printf_append_string_n(__UINT64_TYPE__ msgDesc, const char *data,
                              __UINT64_TYPE__ length, __UINT32_TYPE__ isLast);
GSAN_DEVICE __UINT64_TYPE__ __ockl_printf_append_args(
    __UINT64_TYPE__ msgDesc, __UINT32_TYPE__ numArgs, __UINT64_TYPE__ value0,
    __UINT64_TYPE__ value1, __UINT64_TYPE__ value2, __UINT64_TYPE__ value3,
    __UINT64_TYPE__ value4, __UINT64_TYPE__ value5, __UINT64_TYPE__ value6,
    __UINT32_TYPE__ isLast);
// No clang builtins exist for these (s_memrealtime needs -mcpu); the generic
// bitcode calls the intrinsics directly.
GSAN_DEVICE __UINT64_TYPE__ gsanDispatchId() __asm("llvm.amdgcn.dispatch.id");
GSAN_DEVICE __UINT64_TYPE__ gsanRealtime() __asm("llvm.amdgcn.s.memrealtime");
}

static GSAN_DEVICE __UINT64_TYPE__ gsanStrlenWithNul(const char *s) {
  __UINT64_TYPE__ n = 0;
  while (s[n] != '\0')
    ++n;
  return n + 1;
}

static GSAN_DEVICE inline __UINT32_TYPE__ gsanLaneId() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Kept out of line so the hostcall code is not duplicated at every check.
__attribute__((noinline, cold)) static GSAN_DEVICE void
gsanPrintAssert(const char *assertion, const char *file, unsigned line) {
  const char *fmt = "GSan device assertion failed: '%s' at %s:%u\n";
  auto desc = __ockl_fprintf_stderr_begin();
  desc = __ockl_printf_append_string_n(desc, fmt, gsanStrlenWithNul(fmt), 0);
  desc = __ockl_printf_append_string_n(desc, assertion,
                                       gsanStrlenWithNul(assertion), 0);
  desc = __ockl_printf_append_string_n(desc, file, gsanStrlenWithNul(file), 0);
  __ockl_printf_append_args(desc, 1, line, 0, 0, 0, 0, 0, 0, 1);
}

// Print the failure once per wave (lowest active lane) via hostcall, then
// trap. The trap aborts the process (HSA exception), so a failure is fatal.
// "s_trap 2" is used instead of __builtin_trap, which is noreturn and lets
// StructurizeCFG move the trap past the waterfall convergence points.
static GSAN_DEVICE void __assert_fail(const char *assertion, const char *file,
                                      unsigned line, const char *function) {
  (void)function;
  __UINT64_TYPE__ exec = __builtin_amdgcn_read_exec();
  if (gsanLaneId() == static_cast<__UINT32_TYPE__>(__builtin_ctzll(exec)))
    gsanPrintAssert(assertion, file, line);
  __asm__ volatile("s_trap 2" ::: "memory");
}
#else
extern "C" GSAN_DEVICE void __assertfail(const char *assertion,
                                         const char *file, unsigned line,
                                         const char *function,
                                         __SIZE_TYPE__ charSize);

static GSAN_DEVICE inline void __assert_fail(const char *assertion,
                                             const char *file, unsigned line,
                                             const char *function) {
  __assertfail(assertion, file, line, function, sizeof(char));
}
#endif

namespace gsan {

struct Location {
  const char *file;
  unsigned line;
};

GSAN_DEVICE const char *getSourceFile(Location loc) {
  return loc.file == nullptr ? "<unknown>" : loc.file;
}

} // namespace gsan

#define assert_msg(loc, cond, msg)                                             \
  do {                                                                         \
    if (!(cond)) {                                                             \
      __assert_fail((msg), gsan::getSourceFile(loc), (loc).line, "");          \
    }                                                                          \
  } while (false)

namespace gsan {
namespace {
static constexpr uint32_t kWriterFlag = 1u << 31;
static constexpr epoch_t kMaxEpoch = static_cast<epoch_t>(~0u);
static constexpr uint16_t kMaxUint16 = static_cast<uint16_t>(~0u);

enum class AtomicSem : uint8_t {
  Relaxed = 1,
  Acquire = 2,
  Release = 3,
  AcquireRelease = 4,
};

GSAN_DEVICE void rwLockAcquireRead(uint32_t &lock) {
  uint32_t old = __scoped_atomic_fetch_add(&lock, 1, __ATOMIC_ACQUIRE,
                                           __MEMORY_SCOPE_WRKGRP);
  if ((old & kWriterFlag) == 0)
    return;

  do {
    old =
        __scoped_atomic_load_n(&lock, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
  } while ((old & kWriterFlag) != 0);
}

GSAN_DEVICE void rwLockAcquireWrite(uint32_t &lock) {
  uint32_t actual = 0;
  while (!__scoped_atomic_compare_exchange_n(&lock, &actual, kWriterFlag, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_WRKGRP)) {
    actual = 0;
  }
}

GSAN_DEVICE void rwLockReleaseRead(uint32_t &lock) {
  __scoped_atomic_fetch_sub(&lock, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}

GSAN_DEVICE void rwLockReleaseWrite(uint32_t &lock) {
  // Note we don't set 0 as there may be readers who've already
  // incremented optimistically
  __scoped_atomic_fetch_and(&lock, ~kWriterFlag, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_WRKGRP);
}

GSAN_DEVICE void clusterLockAcquire(uint32_t &lock) {
  uint32_t actual = 0;
  while (!__scoped_atomic_compare_exchange_n(&lock, &actual, 1, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_DEVICE)) {
    actual = 0;
  }
}

GSAN_DEVICE void clusterLockRelease(uint32_t &lock) {
  __scoped_atomic_store_n(&lock, 0, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
}

GSAN_DEVICE inline uintptr_t roundUp(uintptr_t ptr, uintptr_t align) {
  return ptr % align == 0 ? ptr : ptr + align - (ptr % align);
}

////////////////////////////////////////////
// Target adapter
////////////////////////////////////////////

#if GSAN_AMDGPU

static constexpr uint32_t kWaveSize = 64;

// Shadow-cell locks use agent scope when the runtime manages one GPU: on
// gfx950, system-scope acquire/release adds L2 writeback/invalidate to every
// cell access. With several GPUs, peers may lock the same cells, so system
// scope is required. Multi-GPU use is unverified on AMD: system scope is
// necessary, but peer CAS coherence on VMM shadow memory is untested (the
// host allocator warns once).
GSAN_DEVICE bool useSystemScopeShadowLocks(ThreadState *state) {
  return getGlobalState(state)->numDevices > 1;
}

GSAN_DEVICE uint32_t getLaneId() { return gsanLaneId(); }

// Whole-workgroup barrier; AMD has no named or partial barriers. The fences
// are agent scope so that the slot hand-off at kernel exit also publishes
// thread-state writes of all waves (the next owner may run on another XCD).
GSAN_DEVICE void syncThreads(uint32_t barrierId, uint32_t numThreads) {
  (void)barrierId;
  (void)numThreads;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

GSAN_DEVICE void atomicMaxRelaxedGpu(uint32_t *ptr, uint32_t value) {
  __scoped_atomic_fetch_max(ptr, value, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}

// Run f() for every active lane, one lane at a time (a "waterfall"). Used for
// entry points whose code takes blocking per-lane shadow/state locks (the
// descriptor checks; not reachable on AMD in M1). If lanes of one wave spun
// on a lock held by another lane of the same wave, the wave would deadlock (no
// independent forward progress of lanes on AMD). Only the leader lane runs,
// so it may spin. The leader is derived from a ballot, so it is wave-uniform.
template <typename F> GSAN_DEVICE void forEachLaneSerially(F &&f) {
  uint32_t laneId = getLaneId();
  bool pending = true;
  while (true) {
    uint64_t mask = __builtin_amdgcn_ballot_w64(pending);
    if (mask == 0)
      break;
    if (laneId == static_cast<uint32_t>(__builtin_ctzll(mask))) {
      f();
      pending = false;
    }
  }
}

// Workgroup-uniform hash of a key that is unique among live blocks (AQL
// packet address, dispatch ID, workgroup ID). It only spreads blocks over the
// slot table; ownership comes from the CAS in acquireSlot.
GSAN_DEVICE uint32_t getSlotHint(uint32_t numSlots) {
  uint32_t h = 0x9747b28cu;
  h = hash_combine32(h, __builtin_amdgcn_workgroup_id_x());
  h = hash_combine32(h, __builtin_amdgcn_workgroup_id_y());
  h = hash_combine32(h, __builtin_amdgcn_workgroup_id_z());
  h = hash_combine32(
      h, static_cast<uint32_t>(
             reinterpret_cast<uint64_t>(__builtin_amdgcn_dispatch_ptr()) >> 6));
  h = hash_combine32(h, static_cast<uint32_t>(gsanDispatchId()));
  return hash_finalize32(h) % numSlots;
}

// The block's slot index, written by __triton_gsan_init before its final
// barrier and read by every later runtime call of the block. LDS is per
// workgroup, so all waves agree, and it moves with the block under CWSR.
// This is the only static LDS the runtime uses (4 bytes; the static segment
// is padded to 16 bytes when the kernel also uses Triton's dynamic LDS).
[[clang::loader_uninitialized]] static
    __attribute__((address_space(3))) uint32_t gsanBlockSlot;

static constexpr uint64_t kSlotAcquireTimeoutTicks =
    20ull * 100 * 1000 * 1000; // 20 s of the 100 MHz s_memrealtime clock

// Called by one thread of the block at kernel entry. Waits while all slots
// are taken: slots are released by blocks that finish, which is enough as
// long as no kernel needs more co-resident blocks than there are slots. A
// kernel that does (e.g. blocks spinning on each other) fails with a
// diagnostic instead of hanging.
GSAN_DEVICE void acquireSlot(GlobalState *globals, Location loc) {
  auto *table = getSlotTable(globals);
  uint32_t numSlots = globals->numSms;
  uint32_t idx = getSlotHint(numSlots);
  uint64_t start = gsanRealtime();
  for (uint32_t probes = 1;; ++probes) {
    uint32_t expected = kSlotFree;
    if (__scoped_atomic_load_n(&table[idx].state, __ATOMIC_RELAXED,
                               __MEMORY_SCOPE_DEVICE) == kSlotFree &&
        __scoped_atomic_compare_exchange_n(
            &table[idx].state, &expected, kSlotTaken, false, __ATOMIC_ACQUIRE,
            __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE))
      break;
    idx = idx + 1 == numSlots ? 0 : idx + 1;
    if (probes % 64 == 0) {
      __builtin_amdgcn_s_sleep(8);
      assert_msg(loc, gsanRealtime() - start < kSlotAcquireTimeoutTicks,
                 "GSan slot table exhausted: no free slot for 20 s; more "
                 "co-resident blocks than TRITON_GSAN_NUM_SLOTS (kernels that "
                 "wait on other blocks are not supported)");
    }
  }
  // Diagnostic only (not used for lookup): record the owner.
  auto &entry = table[idx];
  entry.workgroupId[0] = __builtin_amdgcn_workgroup_id_x();
  entry.workgroupId[1] = __builtin_amdgcn_workgroup_id_y();
  entry.workgroupId[2] = __builtin_amdgcn_workgroup_id_z();
  entry.dispatchPtr =
      reinterpret_cast<uint64_t>(__builtin_amdgcn_dispatch_ptr());
  entry.dispatchId = gsanDispatchId();
  gsanBlockSlot = idx;
}

// Called by one thread of the block after the final barrier at kernel exit.
// The release store hands the slot's thread state to the next owner.
GSAN_DEVICE void releaseSlot(GlobalState *globals) {
  __scoped_atomic_store_n(&getSlotTable(globals)[gsanBlockSlot].state,
                          kSlotFree, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
}

// Identity index of the calling block within its device. NVIDIA: the SM id.
// AMD: the slot acquired at kernel entry.
GSAN_DEVICE uint32_t getSmId(GlobalState *) { return gsanBlockSlot; }

#else // NVIDIA

static constexpr uint32_t kWaveSize = 32;

GSAN_DEVICE bool useSystemScopeShadowLocks(ThreadState *) { return true; }

GSAN_DEVICE uint32_t getLaneId() { return __nvvm_read_ptx_sreg_laneid(); }

// Identity index of the calling block within its device: the SM id.
GSAN_DEVICE uint32_t getSmId(GlobalState *) {
  return __nvvm_read_ptx_sreg_smid();
}

GSAN_DEVICE void syncThreads(uint32_t barrierId, uint32_t numThreads) {
  asm volatile("barrier.sync %0, %1;"
               :
               : "r"(barrierId), "r"(numThreads)
               : "memory");
}

GSAN_DEVICE void atomicMaxRelaxedGpu(uint32_t *ptr, uint32_t value) {
  asm volatile("red.relaxed.gpu.max.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(value)
               : "memory");
}

// Lanes of a warp make independent forward progress, so per-lane locking is
// safe as is.
template <typename F> GSAN_DEVICE void forEachLaneSerially(F &&f) { f(); }

#endif

GSAN_DEVICE thread_id_t getDeviceThreadId(GlobalState *globals, uint32_t smid) {
  auto globalsBase = static_cast<uintptr_t>(globals->globalsBase);
  auto deviceBase = reinterpret_cast<uintptr_t>(globals);
  auto deviceIdx = (deviceBase - globalsBase) / kPerDeviceStateStride;
  return static_cast<thread_id_t>(deviceIdx * globals->numSms + smid);
}

GSAN_DEVICE ThreadState *getThreadStateById(GlobalState *globals,
                                            thread_id_t tid) {
  uint32_t deviceIdx = tid / globals->numSms;
  uint32_t smid = tid % globals->numSms;
  uintptr_t stateBase = static_cast<uintptr_t>(globals->globalsBase) +
                        deviceIdx * kPerDeviceStateStride;
  stateBase = getThreadStateBaseAddress(stateBase);
  auto stateStride = getThreadStateStrideBytes(globals);
  return reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);
}

GSAN_DEVICE ThreadState *getThreadState(GlobalState *globals) {
  uint32_t smid = getSmId(globals);
  uintptr_t stateBase =
      getThreadStateBaseAddress(reinterpret_cast<uintptr_t>(globals));
  auto stateStride = getThreadStateStrideBytes(globals);
  return reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);
}

GSAN_DEVICE epoch_t *getClockBufferBase(ThreadState *state) {
  auto *globals = getGlobalState(state);
  return state->vectorClock + globals->numThreads;
}

GSAN_DEVICE epoch_t *getClockBufferSlot(ThreadState *state, epoch_t token,
                                        Location loc) {
  assert_msg(loc, token != 0, "Invalid GSan clock token");
  assert_msg(loc, token <= state->clockBufferHead, "Future GSan clock token");
  auto *globals = getGlobalState(state);
  assert_msg(loc, state->clockBufferHead - token < globals->clockBufferSize,
             "GSan clock buffer token overwritten");
  uint32_t slot = token % globals->clockBufferSize;
  return getClockBufferBase(state) + slot * globals->numThreads;
}

GSAN_DEVICE epoch_t publishClockBuffer(ThreadState *state, Location loc) {
  auto *globals = getGlobalState(state);
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  epoch_t *slot =
      getClockBufferBase(state) +
      ((nextHead - 1) % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = state->vectorClock[i];
  state->clockBufferHead = nextHead;
  state->clockBufferDirty = 0;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE AtomicSem decodeAtomicSem(uint32_t sem) {
  switch (sem) {
  case 1:
    return AtomicSem::Relaxed;
  case 2:
    return AtomicSem::Acquire;
  case 3:
    return AtomicSem::Release;
  case 4:
    return AtomicSem::AcquireRelease;
  default:
    __builtin_trap();
    return AtomicSem::Relaxed;
  }
}

GSAN_DEVICE AtomicScope decodeAtomicScope(uint32_t scope) {
  switch (scope) {
  case 1:
    return AtomicScope::GPU;
  case 2:
    return AtomicScope::CTA;
  case 3:
    return AtomicScope::System;
  default:
    __builtin_trap();
    return AtomicScope::NonAtomic;
  }
}

GSAN_DEVICE bool hasAcquire(AtomicSem sem) {
  return sem == AtomicSem::Acquire || sem == AtomicSem::AcquireRelease;
}

GSAN_DEVICE bool hasRelease(AtomicSem sem) {
  return sem == AtomicSem::Release || sem == AtomicSem::AcquireRelease;
}

GSAN_DEVICE bool scopeCoversPair(AtomicScope scope, thread_id_t lhs,
                                 thread_id_t rhs, GlobalState *globals) {
  switch (scope) {
  case AtomicScope::CTA:
    return lhs == rhs;
  case AtomicScope::GPU:
    return lhs / globals->numSms == rhs / globals->numSms;
  case AtomicScope::System:
    return true;
  case AtomicScope::NonAtomic:
    return false;
  }
  return false;
}

GSAN_DEVICE bool areAtomicScopesCompatible(AtomicScope lhs, thread_id_t lhsTid,
                                           AtomicScope rhs, thread_id_t rhsTid,
                                           GlobalState *globals) {
  if (!isAtomicScope(lhs) || !isAtomicScope(rhs))
    return false;
  return scopeCoversPair(lhs, lhsTid, rhsTid, globals) &&
         scopeCoversPair(rhs, lhsTid, rhsTid, globals);
}

GSAN_DEVICE bool canAccumulateReleaseRmw(ThreadState *state, AtomicScope scope,
                                         const ScalarClock &previousWrite) {
  // A propagated snapshot currently carries only one scope. Only combine
  // concurrent releases when that scope remains unchanged.
  if (!previousWrite.isRelease || scope != previousWrite.scope)
    return false;
  return areAtomicScopesCompatible(scope, state->threadId, previousWrite.scope,
                                   previousWrite.threadId,
                                   getGlobalState(state));
}

GSAN_DEVICE void acquireLaunchClocks(ThreadState *state,
                                     const LaunchState &launch,
                                     uint32_t threadIdx, uint32_t numThreads,
                                     uint32_t barrierId) {
  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockAcquireWrite(state->lock);
  syncThreads(barrierId, numThreads);

  auto *globals = getGlobalState(state);
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    uint32_t epoch = 0;
    for (uint64_t j = 0; j < launch.numWaitClocks; ++j)
      epoch = epoch < launch.waitClocks[j][i] ? launch.waitClocks[j][i] : epoch;
    if (state->vectorClock[i] < epoch)
      state->vectorClock[i] = static_cast<epoch_t>(epoch);
  }
  if (threadIdx == 0) {
    state->clockBufferDirty = 1;
    state->gdcWaitCalled = 1;
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseWrite(state->lock);
}

GSAN_DEVICE void publishLaunchClock(ThreadState *state,
                                    uint32_t *completionClock,
                                    uint32_t threadIdx, uint32_t numThreads,
                                    uint32_t barrierId, Location loc) {
  syncThreads(barrierId, numThreads);
  if (threadIdx == 0) {
    rwLockAcquireRead(state->lock);
    assert_msg(loc, state->gdcWaitCalled,
               "kernel launched with programmatic dependent launch did not "
               "call gdc_wait");
  }
  syncThreads(barrierId, numThreads);

  auto *globals = getGlobalState(state);
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    uint32_t current = state->vectorClock[i];
    atomicMaxRelaxedGpu(completionClock + i, current);
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseRead(state->lock);
}

GSAN_DEVICE void initThread(GlobalState *globals, const LaunchState &launch,
                            uint32_t threadIdx, uint32_t numThreads,
                            uint32_t barrierId, Location loc) {
#if GSAN_AMDGPU
  // The runtime is built as generic amdgcn bitcode but assumes wave64.
  assert_msg(loc, __builtin_amdgcn_wavefrontsize() == kWaveSize,
             "GSan on AMD requires wave64");
  // Take exclusive ownership of an identity before touching its state.
  if (threadIdx == 0)
    acquireSlot(globals, loc);
  syncThreads(barrierId, numThreads);
#endif
  auto *state = getThreadState(globals);
  if (threadIdx == 0) {
    rwLockAcquireWrite(state->lock);
    auto st_globals = __scoped_atomic_load_n(&state->globals, __ATOMIC_ACQUIRE,
                                             __MEMORY_SCOPE_DEVICE);
    if (st_globals == nullptr) {
      // Lazily initialize per-SM thread state
      state->reserveBase = globals->reserveBase;
      state->numReads = 0;
      state->clockBufferDirty = 0;
      state->gdcWaitCalled = 0;
      state->clockBufferHead = 0;
      state->threadId = getDeviceThreadId(globals, getSmId(globals));
      __scoped_atomic_store_n(&state->globals, globals, __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_DEVICE);
    }
    state->gdcWaitCalled = launch.numWaitClocks == 0;
  }
  syncThreads(barrierId, numThreads);

  // Persistent SM state must not implicitly acquire a concurrent graph node.
  // Only the caller-supplied entry frontier is known to have completed.
  auto tid = state->threadId;
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    if (i == tid)
      continue;
    uint32_t epoch = 0;
    for (uint64_t j = 0; j < launch.numEntryClocks; ++j)
      epoch =
          epoch < launch.entryClocks[j][i] ? launch.entryClocks[j][i] : epoch;
    state->vectorClock[i] = static_cast<epoch_t>(epoch);
  }

  if (threadIdx == 0) {
    // Advance the SM's local epoch; nonlocal components came from the explicit
    // entry frontier above.
    auto *clock = state->vectorClock;
    assert_msg(loc, clock[tid] != kMaxEpoch, "Vector clock overflowed");
    clock[tid] += 1;
    state->clockBufferDirty = 1;
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseWrite(state->lock);
}

struct Range {
  uintptr_t start;
  uintptr_t end;
};

GSAN_DEVICE Range roundRange(Range x, int granularity) {
  uintptr_t mask = granularity - 1;
  // Round start down to shadow granularity
  x.start &= ~mask;
  // Round end up to shadow granularity
  x.end = (x.end + mask) & ~mask;
  return x;
}

GSAN_DEVICE void checkCoarseAccess(uintptr_t address, int nBytes,
                                   int granularity, Location loc) {
  assert_msg(loc, granularity != 16 || (address % 16 == 0 && nBytes % 16 == 0),
             "GSan 16-byte pools require complete aligned 16-byte accesses");
}

GSAN_DEVICE void checkAtomicGranularity(int granularity, int bytesPerElem,
                                        Location loc) {
  // Distinct atomic objects must not share release/acquire metadata, even when
  // several elements are accessed by one vectorized instruction.
  assert_msg(loc, granularity <= bytesPerElem,
             "GSan shadow granularity exceeds atomic access size");
}

template <int Scope> GSAN_DEVICE bool tryAcquireShadowScoped(ShadowCell *cell) {
  return (__scoped_atomic_fetch_or(&cell->readCountAndLock,
                                   ShadowCell::kLockBit, __ATOMIC_ACQUIRE,
                                   Scope) &
          ShadowCell::kLockBit) == 0;
}

// One attempt to lock the cell with a 32-bit CAS on the (numReads, lock) word.
GSAN_DEVICE bool tryAcquireShadow(ShadowCell *cell, bool systemScope) {
  return systemScope ? tryAcquireShadowScoped<__MEMORY_SCOPE_SYSTEM>(cell)
                     : tryAcquireShadowScoped<__MEMORY_SCOPE_DEVICE>(cell);
}

// Blocking lock. On AMD, callers must not spin here while another lane of the
// same wave holds a shadow lock (no independent lane progress); see
// forEachLaneSerially and tensorAccessCooperative.
GSAN_DEVICE ShadowCell *acquireShadow(uintptr_t shadowAddr, bool systemScope) {
  auto cell = reinterpret_cast<ShadowCell *>(shadowAddr);
  while (!tryAcquireShadow(cell, systemScope)) {
  }
  return cell;
}

GSAN_DEVICE void releaseShadow(ShadowCell *cell, bool systemScope) {
  if (systemScope)
    __scoped_atomic_fetch_and(&cell->readCountAndLock, ~ShadowCell::kLockBit,
                              __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  else
    __scoped_atomic_fetch_and(&cell->readCountAndLock, ~ShadowCell::kLockBit,
                              __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
}

GSAN_DEVICE epoch_t appendClockBufferSnapshot(ThreadState *state,
                                              const epoch_t *snapshot,
                                              Location loc) {
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t curHead = state->clockBufferHead;
  uint32_t nextHead = curHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  epoch_t *slot = getClockBufferBase(state) +
                  (nextHead % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = snapshot[i];
  state->clockBufferHead = nextHead;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE epoch_t publishCurrentVectorClock(ThreadState *state,
                                              Location loc) {
  if (state->clockBufferDirty) {
    auto token = appendClockBufferSnapshot(state, state->vectorClock, loc);
    state->clockBufferDirty = 0;
    return token;
  }
  return state->clockBufferHead;
}

GSAN_DEVICE const epoch_t *getSnapshotForWrite(ThreadState *state,
                                               const ScalarClock &write,
                                               Location loc) {
  if (!write.isRelease)
    return nullptr;
  auto *writerState = getThreadStateById(getGlobalState(state), write.threadId);
  return getClockBufferSlot(writerState, write.epoch, loc);
}

GSAN_DEVICE epoch_t publishCurrentVectorClockWithPriorRelease(
    ThreadState *state, const ScalarClock &previousWrite, Location loc) {
  const auto *previousSnapshot = getSnapshotForWrite(state, previousWrite, loc);
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  auto *slot = getClockBufferBase(state) +
               (nextHead % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i) {
    auto current = state->vectorClock[i];
    auto previous = previousSnapshot[i];
    slot[i] = current > previous ? current : previous;
  }
  state->clockBufferHead = nextHead;
  // The joined snapshot extends the release sequence without acquiring the
  // prior writer into this thread's vector clock.
  state->clockBufferDirty = 1;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE epoch_t propagateClockBufferSnapshot(ThreadState *state,
                                                 const ScalarClock &write,
                                                 Location loc) {
  auto *snapshot = getSnapshotForWrite(state, write, loc);
  assert_msg(loc, snapshot != nullptr, "Invalid GSan propagated clock token");
  auto token = appendClockBufferSnapshot(state, snapshot, loc);
  state->clockBufferDirty = 1;
  return token;
}

GSAN_DEVICE void incrementThreadEpoch(ThreadState *state, Location loc) {
  auto tid = state->threadId;
  auto *clock = state->vectorClock;
  assert_msg(loc, clock[tid] != kMaxEpoch, "Vector clock overflowed");
  clock[tid] += 1;
  state->clockBufferDirty = 1;
}

// A publisher that has not advanced its local epoch can only transfer its
// completed epochs. All other entries retain their usual acquire semantics.
GSAN_DEVICE bool mergeVectorClock(ThreadState *state, const epoch_t *snapshot,
                                  thread_id_t incompleteThread = kMaxThreads) {
  bool changed = false;
  auto *globals = getGlobalState(state);
  for (int i = 0; i < globals->numThreads; ++i) {
    auto epoch = snapshot[i];
    if (i == incompleteThread && epoch != 0)
      --epoch;
    if (state->vectorClock[i] < epoch) {
      state->vectorClock[i] = epoch;
      changed = true;
    }
  }
  if (changed)
    state->clockBufferDirty = 1;
  return changed;
}

GSAN_DEVICE bool dominatesSnapshot(ThreadState *state,
                                   const epoch_t *snapshot) {
  auto *globals = getGlobalState(state);
  for (int i = 0; i < globals->numThreads; ++i) {
    if (state->vectorClock[i] < snapshot[i])
      return false;
  }
  return true;
}

GSAN_DEVICE bool clockHappensBefore(ThreadState *state,
                                    const ScalarClock &clock, Location loc) {
  if (clock.epoch == 0)
    return true;
  if (const epoch_t *snapshot = getSnapshotForWrite(state, clock, loc))
    return dominatesSnapshot(state, snapshot);
  return state->vectorClock[clock.threadId] >= clock.epoch;
}

GSAN_DEVICE void assertOrderedOrCompatible(ThreadState *state,
                                           AtomicScope currentScope,
                                           const ScalarClock &prior,
                                           Location loc, const char *message) {
  if (prior.epoch == 0)
    return;
  if (isAtomicScope(currentScope) &&
      areAtomicScopesCompatible(currentScope, state->threadId, prior.scope,
                                prior.threadId, getGlobalState(state))) {
    return;
  }
  assert_msg(loc, clockHappensBefore(state, prior, loc), message);
}

GSAN_DEVICE void maybeMergeAcquire(ThreadState *state, AtomicScope currentScope,
                                   const ScalarClock &prior, Location loc) {
  if (!prior.isRelease)
    return;
  if (!areAtomicScopesCompatible(currentScope, state->threadId, prior.scope,
                                 prior.threadId, getGlobalState(state))) {
    return;
  }
  auto *snapshot = getSnapshotForWrite(state, prior, loc);
  mergeVectorClock(state, snapshot);
}

GSAN_DEVICE epoch_t publishClusterClock(ThreadState *state, Location loc) {
  rwLockAcquireWrite(state->lock);
  // Match a release atomic: publish the pre-barrier clock, then advance the
  // local epoch so later accesses are distinct from the published snapshot.
  epoch_t token = publishCurrentVectorClock(state, loc);
  incrementThreadEpoch(state, loc);
  rwLockReleaseWrite(state->lock);
  return token;
}

GSAN_DEVICE epoch_t publishMBarrierClock(ThreadState *state, Location loc) {
  rwLockAcquireWrite(state->lock);
  // Mbarriers are frequent in shared/TMEM pipelines. Do not consume an epoch
  // for each arrival: publish an immutable snapshot and let the waiter exclude
  // this thread's still-open epoch. Imported clocks remain fully transferable.
  // Reusing the ordinary snapshot cache also avoids filling the clock buffer
  // when repeated arrivals have no new ordering to publish.
  epoch_t token = publishCurrentVectorClock(state, loc);
  rwLockReleaseWrite(state->lock);
  return token;
}

GSAN_DEVICE void acquireClusterClocks(const ClusterBarrierState *barrier,
                                      ThreadState *state, int numCTAs,
                                      Location loc) {
  auto *globals = getGlobalState(state);
  rwLockAcquireWrite(state->lock);
  for (int ctaRank = 0; ctaRank < numCTAs; ++ctaRank) {
    thread_id_t threadId = barrier->threadIds[ctaRank];
    if (threadId == state->threadId)
      continue;
    auto *participant = getThreadStateById(globals, threadId);
    const epoch_t *snapshot =
        getClockBufferSlot(participant, barrier->tokens[ctaRank], loc);
    mergeVectorClock(state, snapshot);
  }
  rwLockReleaseWrite(state->lock);
}

GSAN_DEVICE void initClusterBarrier(void *scratch) {
  auto *barrier = reinterpret_cast<ClusterBarrierState *>(scratch);
  barrier->arrivalCount = 0;
  barrier->publicationCount = 0;
  barrier->generation = 0;
  barrier->registrationGeneration = 0;
  __scoped_atomic_store_n(&barrier->lock, 0, __ATOMIC_RELEASE,
                          __MEMORY_SCOPE_DEVICE);
}

GSAN_DEVICE void synchronizeClusterBarrier(GlobalState *globals, void *scratch,
                                           int numCTAs, int ctaRank,
                                           Location loc) {
  assert_msg(loc, numCTAs > 0 && numCTAs <= kMaxClusterCTAs,
             "Invalid GSan cluster size");
  assert_msg(loc, ctaRank >= 0 && ctaRank < numCTAs,
             "Invalid GSan cluster CTA rank");

  auto *state = getThreadState(globals);
  auto *barrier = reinterpret_cast<ClusterBarrierState *>(scratch);
  clusterLockAcquire(barrier->lock);
  uint32_t generation = barrier->generation;
  uint32_t arrival = barrier->arrivalCount;
  assert_msg(loc, arrival < static_cast<uint32_t>(numCTAs),
             "Too many GSan cluster barrier arrivals");
  barrier->threadIds[ctaRank] = state->threadId;
  barrier->arrivalCount = arrival + 1;
  // Publish the complete rank-to-thread mapping separately from the arrival
  // count. The final publisher can safely reset the count without a slow CTA
  // missing registration completion.
  if (arrival + 1 == static_cast<uint32_t>(numCTAs))
    __scoped_atomic_store_n(&barrier->registrationGeneration, generation + 1,
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
  clusterLockRelease(barrier->lock);

  while (__scoped_atomic_load_n(&barrier->registrationGeneration,
                                __ATOMIC_ACQUIRE,
                                __MEMORY_SCOPE_DEVICE) == generation) {
  }

  epoch_t token = publishClusterClock(state, loc);
  clusterLockAcquire(barrier->lock);
  barrier->tokens[ctaRank] = token;
  uint32_t publication = barrier->publicationCount + 1;
  if (publication == static_cast<uint32_t>(numCTAs)) {
    barrier->arrivalCount = 0;
    barrier->publicationCount = 0;
    __scoped_atomic_store_n(&barrier->generation, generation + 1,
                            __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
  } else {
    barrier->publicationCount = publication;
  }
  clusterLockRelease(barrier->lock);

  while (__scoped_atomic_load_n(&barrier->generation, __ATOMIC_ACQUIRE,
                                __MEMORY_SCOPE_DEVICE) == generation) {
  }

  // Each participant locks only its own state and acquires immutable release
  // snapshots from its peers. No participant state locks are nested.
  acquireClusterClocks(barrier, state, numCTAs, loc);
}

GSAN_DEVICE uint32_t getMBarrierKey(uint32_t offset, uint32_t ownerRank,
                                    Location loc) {
  assert_msg(loc, offset < (1u << 24), "Invalid GSan mbarrier offset");
  assert_msg(loc, ownerRank < kMaxClusterCTAs,
             "Invalid GSan mbarrier owner CTA rank");
  return offset | (ownerRank << 24);
}

GSAN_DEVICE void initMBarrierTable(void *scratch, uint32_t capacity) {
  auto *table = reinterpret_cast<MBarrierTable *>(scratch);
  table->capacity = capacity;
  for (uint32_t i = 0; i < capacity; ++i) {
    table->states[i].key = kEmptyMBarrierKey;
    table->states[i].lock = 0;
  }
  __scoped_atomic_store_n(&table->lock, 0, __ATOMIC_RELEASE,
                          __MEMORY_SCOPE_DEVICE);
}

GSAN_DEVICE MBarrierState *getMBarrierState(MBarrierTable *table, uint32_t key,
                                            bool create, Location loc) {
  MBarrierState *empty = nullptr;
  clusterLockAcquire(table->lock);
  for (uint32_t i = 0; i < table->capacity; ++i) {
    auto *state = &table->states[i];
    if (state->key == key) {
      clusterLockRelease(table->lock);
      return state;
    }
    if (state->key == kEmptyMBarrierKey && empty == nullptr)
      empty = state;
  }
  if (create && empty != nullptr) {
    empty->lock = 0;
    empty->key = key;
  }
  clusterLockRelease(table->lock);
  assert_msg(loc, create && empty != nullptr,
             create ? "GSan mbarrier table is full"
                    : "GSan mbarrier used before initialization");
  return empty;
}

GSAN_DEVICE void initMBarrier(void *scratch, uint32_t offset,
                              uint32_t ownerRank, uint32_t expectedCount,
                              Location loc) {
  assert_msg(loc, expectedCount > 0, "Invalid GSan mbarrier arrival count");
  auto *table = reinterpret_cast<MBarrierTable *>(scratch);
  auto *barrier = getMBarrierState(
      table, getMBarrierKey(offset, ownerRank, loc), /*create=*/true, loc);
  clusterLockAcquire(barrier->lock);
  barrier->expectedCount = expectedCount;
  barrier->pendingCount = expectedCount;
  barrier->generation = 0;
  for (int bank = 0; bank < 2; ++bank) {
    barrier->phases[bank].generation = 0;
    barrier->phases[bank].complete = bank;
    for (int source = 0; source < kMaxClusterCTAs; ++source)
      barrier->phases[bank].clocks[source] = {};
  }
  clusterLockRelease(barrier->lock);
}

GSAN_DEVICE void recordMBarrierArrival(GlobalState *globals, void *scratch,
                                       uint32_t offset, uint32_t recipientMask,
                                       uint32_t count, uint32_t sourceRank,
                                       bool publishClock, Location loc) {
  assert_msg(loc, sourceRank < kMaxClusterCTAs,
             "Invalid GSan mbarrier source CTA rank");
  assert_msg(loc, recipientMask != 0,
             "GSan mbarrier arrival has no recipients");
  MBarrierPublishedClock published = {};
  if (publishClock) {
    auto *threadState = getThreadState(globals);
    published = {threadState->threadId, publishMBarrierClock(threadState, loc)};
  }
  auto *table = reinterpret_cast<MBarrierTable *>(scratch);
  for (uint32_t ownerRank = 0; ownerRank < kMaxClusterCTAs; ++ownerRank) {
    if ((recipientMask & (1u << ownerRank)) == 0)
      continue;
    auto *barrier = getMBarrierState(
        table, getMBarrierKey(offset, ownerRank, loc), /*create=*/false, loc);
    clusterLockAcquire(barrier->lock);
    uint32_t generation = barrier->generation;
    auto *phase = &barrier->phases[generation & 1];
    if (phase->generation != generation + 1) {
      phase->generation = generation + 1;
      phase->complete = 0;
      for (int source = 0; source < kMaxClusterCTAs; ++source)
        phase->clocks[source] = {};
    }
    if (publishClock)
      phase->clocks[sourceRank] = published;
    assert_msg(loc, count > 0 && count <= barrier->pendingCount,
               "Invalid GSan mbarrier arrival count");
    barrier->pendingCount -= count;
    if (barrier->pendingCount == 0) {
      phase->complete = 1;
      barrier->pendingCount = barrier->expectedCount;
      barrier->generation = generation + 1;
    }
    clusterLockRelease(barrier->lock);
  }
}

GSAN_DEVICE void acquireMBarrierPhase(GlobalState *globals, void *scratch,
                                      uint32_t offset, uint32_t ownerRank,
                                      uint32_t waitPhase, Location loc) {
  assert_msg(loc, waitPhase < 2, "Invalid GSan mbarrier wait phase");
  auto *table = reinterpret_cast<MBarrierTable *>(scratch);
  auto *barrier = getMBarrierState(
      table, getMBarrierKey(offset, ownerRank, loc), /*create=*/false, loc);

  MBarrierPublishedClock clocks[kMaxClusterCTAs] = {};
  clusterLockAcquire(barrier->lock);
  const auto &phase = barrier->phases[waitPhase];
  assert_msg(loc, ((phase.generation - 1) & 1) == waitPhase,
             "Invalid GSan mbarrier phase state");
  assert_msg(loc, phase.complete,
             "GSan mbarrier wait observed an incomplete phase");
  for (int source = 0; source < kMaxClusterCTAs; ++source)
    clocks[source] = phase.clocks[source];
  clusterLockRelease(barrier->lock);

  auto *threadState = getThreadState(globals);
  rwLockAcquireWrite(threadState->lock);
  for (int source = 0; source < kMaxClusterCTAs; ++source) {
    const auto &published = clocks[source];
    if (published.token == 0 || published.threadId == threadState->threadId)
      continue;
    auto *publisher = getThreadStateById(globals, published.threadId);
    const epoch_t *snapshot =
        getClockBufferSlot(publisher, published.token, loc);
    // The publisher did not advance its epoch. Acquiring that epoch would
    // incorrectly order its post-arrival accesses as well as earlier ones.
    // Only its completed epochs, and ordering imported from other threads,
    // are covered by this deliberately weaker mbarrier model.
    mergeVectorClock(threadState, snapshot, published.threadId);
  }
  rwLockReleaseWrite(threadState->lock);
}

GSAN_DEVICE ScalarClock makeScalarClock(ThreadState *state, AtomicScope scope) {
  auto tid = state->threadId;
  return ScalarClock{state->vectorClock[tid], tid, scope, false};
}

GSAN_DEVICE ScalarClock makePublishedClock(ThreadState *state,
                                           AtomicScope scope, epoch_t token) {
  return ScalarClock{token, state->threadId, scope, true};
}

GSAN_DEVICE void recordRead(ThreadState *state, ShadowCell *cell,
                            AtomicScope scope) {
  auto numReads =
      __scoped_atomic_load_n(&cell->readCountAndLock, __ATOMIC_RELAXED,
                             __MEMORY_SCOPE_SYSTEM) &
      ShadowCell::kReadCountMask;
  // Only the lock owner changes the count. Use an atomic store because waiters
  // concurrently OR the lock bit, which stays set throughout this update.
  if (numReads < kMaxUint16)
    __scoped_atomic_store_n(&cell->readCountAndLock,
                            ShadowCell::kLockBit | (numReads + 1),
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);

  auto scalarClock = makeScalarClock(state, scope);
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto readClock = cell->readClocks[iRead];
    if (readClock.threadId == state->threadId || readClock.epoch == 0) {
      cell->readClocks[iRead] = scalarClock;
      return;
    }
  }

  auto threadNumReads = __scoped_atomic_fetch_add(
      &state->numReads, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  auto seed = getGlobalState(state)->rngSeed;
  uint32_t rand = hash2x32(threadNumReads, state->threadId, seed);
  rand = rand % numReads;
  if (rand < ShadowCell::kReadClockSize) {
    cell->readClocks[rand] = scalarClock;
  }
}

GSAN_DEVICE void doWrite(ThreadState *state, ShadowCell *cell, Location loc) {
  // Check WAR
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    assertOrderedOrCompatible(state, AtomicScope::NonAtomic,
                              cell->readClocks[iRead], loc,
                              "Write after read race detected");
  }
  // Check WAW
  assertOrderedOrCompatible(state, AtomicScope::NonAtomic, cell->writeClock,
                            loc, "Write after write race detected");
  // Update write
  cell->writeClock = makeScalarClock(state, AtomicScope::NonAtomic);
}

GSAN_DEVICE void doWrite(ThreadState *state, WriteOnceShadowCell *cell,
                         Location loc) {
  auto clock = makeScalarClock(state, AtomicScope::NonAtomic);
  ScalarClock expected{};
  bool firstWrite = __scoped_atomic_compare_exchange(
      &cell->writeClock, &expected, &clock, false, __ATOMIC_RELAXED,
      __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  assert_msg(loc, firstWrite, "Write-once memory written more than once");
}

GSAN_DEVICE void doRead(ThreadState *state, WriteOnceShadowCell *cell,
                        Location loc) {
  ScalarClock clock;
  __scoped_atomic_load(&cell->writeClock, &clock, __ATOMIC_RELAXED,
                       __MEMORY_SCOPE_SYSTEM);
  assert_msg(loc, clock.epoch != 0,
             "Read of write-once memory before its first write");
  assertOrderedOrCompatible(state, AtomicScope::NonAtomic, clock, loc,
                            "Read after write race detected");
}

GSAN_DEVICE void writeRange(ThreadState *state, uintptr_t write_addr,
                            int nBytes, Location loc) {
  auto reserveBase = state->reserveBase;
  auto category = categorize(write_addr, reserveBase);
  bool writeOnce = category == AddressCategory::WriteOnce;
  int granularity = getShadowGranularity(write_addr);
  auto range = roundRange(Range{write_addr, write_addr + nBytes}, granularity);
  assert_msg(loc,
             range.start == range.end ||
                 categorize(range.end - 1, reserveBase) == category,
             "Access crosses a GSan memory category boundary");
  if (category == AddressCategory::Unmanaged)
    return;
  checkCoarseAccess(write_addr, nBytes, granularity, loc);

  bool systemScope = useSystemScopeShadowLocks(state);
  for (uintptr_t addr = range.start; addr < range.end; addr += granularity) {
    auto shadowAddr = getShadowAddress(addr);
    if (writeOnce) {
      doWrite(state, reinterpret_cast<WriteOnceShadowCell *>(shadowAddr), loc);
      continue;
    }
    auto cell = acquireShadow(shadowAddr, systemScope);
    doWrite(state, cell, loc);
    releaseShadow(cell, systemScope);
  }
}

// Handles tl.store(ptrs, values, mask)
GSAN_DEVICE void tensorStore(ThreadState *state, const char *stackPtr,
                             int nElems, int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  bool acquired = false;
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask) {
      if (!acquired) {
        rwLockAcquireRead(state->lock);
        acquired = true;
      }
      writeRange(state, ptr, bytesPerElem, loc);
    }
  }
  if (acquired)
    rwLockReleaseRead(state->lock);
}

#if GSAN_AMDGPU
GSAN_DEVICE void doRead(ThreadState *state, ShadowCell *cell, Location loc);

// Wave-cooperative tl.load/tl.store check (AMD). Each lane walks its own
// shadow cells; in every round each lane makes one try-lock attempt and, on
// success, checks, updates, and releases the cell in the same round. The loop
// runs until no lane of the wave has work left (wave-uniform exit), so a lane
// never waits for a lock held by a lane of its own wave across rounds, and
// lanes on distinct cells proceed in parallel. (Lane-serial waterfalling
// would also be deadlock-free but costs one full lock round trip per lane.)
//
// Hot cells (e.g. every lane loading one scalar): in each round, lanes whose
// current cell is the first active lane's cell do not compete; they skip
// that cell once the leader has processed it. Repeating the same check and
// update of one cell by the same identity within one access has no effect
// (apart from inflating numReads). Rounds where no lane got a lock back off.
template <bool IsStore>
GSAN_DEVICE void tensorAccessCooperative(ThreadState *state,
                                         const char *stackPtr, int nElems,
                                         int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  uintptr_t reserveBase = state->reserveBase;
  int elem = 0;
  uintptr_t addr = 0;
  uintptr_t end = 0;
  // Shadow granularity of the current element's pool: the step between cells.
  int step = kShadowMemGranularityBytes;
  // Advance to the next masked element with lockable GSan-managed cells,
  // applying the same per-element checks as readRange/writeRange. Write-once
  // cells are updated lock-free right here, so they never need a lock round.
  auto nextElement = [&]() {
    while (elem < nElems) {
      int i = elem++;
      if (!maskPtr[i])
        continue;
      auto category = categorize(ptrsPtr[i], reserveBase);
      int granularity = getShadowGranularity(ptrsPtr[i]);
      auto range =
          roundRange(Range{ptrsPtr[i], ptrsPtr[i] + bytesPerElem}, granularity);
      assert_msg(loc,
                 range.start == range.end ||
                     categorize(range.end - 1, reserveBase) == category,
                 "Access crosses a GSan memory category boundary");
      if (category == AddressCategory::Unmanaged)
        continue;
      checkCoarseAccess(ptrsPtr[i], bytesPerElem, granularity, loc);
      if (category == AddressCategory::WriteOnce) {
        for (uintptr_t a = range.start; a < range.end; a += granularity) {
          auto *cell =
              reinterpret_cast<WriteOnceShadowCell *>(getShadowAddress(a));
          if (IsStore)
            doWrite(state, cell, loc);
          else
            doRead(state, cell, loc);
        }
        continue;
      }
      addr = range.start;
      end = range.end;
      step = granularity;
      if (addr < end)
        return true;
    }
    return false;
  };
  // Take the state read lock before nextElement() records any write-once
  // access. Lanes hold no shadow locks while waiting for it.
  bool acquired = false;
  for (int i = 0; i < nElems && !acquired; ++i)
    acquired = maskPtr[i] != 0;
  if (acquired)
    rwLockAcquireRead(state->lock);
  bool active = nextElement();
  bool systemScope = useSystemScopeShadowLocks(state);
  uint32_t laneId = getLaneId();
  while (true) {
    uint64_t activeMask = __builtin_amdgcn_ballot_w64(active);
    if (activeMask == 0)
      break;
    // Wave-uniform leader and its current address.
    int leader = __builtin_ctzll(activeMask);
    uintptr_t leaderAddr =
        static_cast<uintptr_t>(static_cast<uint32_t>(__builtin_amdgcn_readlane(
            static_cast<int>(static_cast<uint32_t>(addr)), leader))) |
        (static_cast<uintptr_t>(static_cast<uint32_t>(__builtin_amdgcn_readlane(
             static_cast<int>(static_cast<uint32_t>(addr >> 32)), leader)))
         << 32);
    bool follower =
        active && laneId != static_cast<uint32_t>(leader) && addr == leaderAddr;
    bool locked = false;
    if (active && !follower) {
      auto *cell = reinterpret_cast<ShadowCell *>(getShadowAddress(addr));
      if (tryAcquireShadow(cell, systemScope)) {
        if (IsStore)
          doWrite(state, cell, loc);
        else
          doRead(state, cell, loc);
        releaseShadow(cell, systemScope);
        locked = true;
      }
    }
    uint64_t lockedMask = __builtin_amdgcn_ballot_w64(locked);
    bool leaderLocked = (lockedMask >> leader) & 1;
    if (locked || (follower && leaderLocked)) {
      addr += step;
      if (addr >= end)
        active = nextElement();
    }
    if (lockedMask == 0)
      __builtin_amdgcn_s_sleep(1);
  }
  if (acquired)
    rwLockReleaseRead(state->lock);
}
#endif

GSAN_DEVICE void doRead(ThreadState *state, ShadowCell *cell, Location loc) {
  assertOrderedOrCompatible(state, AtomicScope::NonAtomic, cell->writeClock,
                            loc, "Read after write race detected");
  recordRead(state, cell, AtomicScope::NonAtomic);
}

GSAN_DEVICE void readRange(ThreadState *state, uintptr_t read_addr, int nBytes,
                           Location loc) {
  auto reserveBase = state->reserveBase;
  auto category = categorize(read_addr, reserveBase);
  bool writeOnce = category == AddressCategory::WriteOnce;
  int granularity = getShadowGranularity(read_addr);
  auto range = roundRange(Range{read_addr, read_addr + nBytes}, granularity);
  assert_msg(loc,
             range.start == range.end ||
                 categorize(range.end - 1, reserveBase) == category,
             "Access crosses a GSan memory category boundary");
  if (category == AddressCategory::Unmanaged)
    return;
  checkCoarseAccess(read_addr, nBytes, granularity, loc);

  bool systemScope = useSystemScopeShadowLocks(state);
  for (uintptr_t addr = range.start; addr < range.end; addr += granularity) {
    auto shadowAddr = getShadowAddress(addr);
    if (writeOnce) {
      doRead(state, reinterpret_cast<WriteOnceShadowCell *>(shadowAddr), loc);
      continue;
    }
    auto cell = acquireShadow(shadowAddr, systemScope);
    doRead(state, cell, loc);
    releaseShadow(cell, systemScope);
  }
}

// Handles tl.load(ptrs, mask)
GSAN_DEVICE void tensorLoad(ThreadState *state, const char *stackPtr,
                            int nElems, int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  bool acquired = false;
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask) {
      if (!acquired) {
        rwLockAcquireRead(state->lock);
        acquired = true;
      }
      readRange(state, ptr, bytesPerElem, loc);
    }
  }
  if (acquired)
    rwLockReleaseRead(state->lock);
}

template <typename ElementHandler>
GSAN_DEVICE void tensorAccessRow(uintptr_t rowPtr, int rowBytes,
                                 int elementGranularity,
                                 ElementHandler &handleElement) {
  uintptr_t reserveBase = handleElement.reserveBase;
  if (!isGsanManaged(rowPtr, reserveBase))
    return;
  int granularity = getShadowGranularity(rowPtr);
  if (elementGranularity == 0) {
    checkCoarseAccess(rowPtr, rowBytes, granularity, handleElement.loc);
    elementGranularity = granularity;
  } else {
    assert_msg(handleElement.loc, !isWriteOnceAddress(rowPtr),
               "Atomic operations on write-once memory are not supported");
    checkAtomicGranularity(granularity, elementGranularity, handleElement.loc);
    elementGranularity = roundUp(elementGranularity, granularity);
  }
  auto range = roundRange(Range{rowPtr, rowPtr + rowBytes}, granularity);
  uint32_t laneId = getLaneId();
  for (uintptr_t addr = range.start + laneId * elementGranularity;
       addr < range.end; addr += kWaveSize * elementGranularity)
    handleElement(addr);
}

template <typename ElementHandler>
GSAN_DEVICE void tensorAccessDesc(const void *descriptor, const int *coords,
                                  int rank, const int16_t *blockShape,
                                  int bytesPerElem, int elementGranularity,
                                  int warpId, int numWarps,
                                  ElementHandler &handleElement) {
  TensorDescInfo desc = decodeTensorDesc(descriptor, rank, bytesPerElem);
  int lastDim = rank - 1;
  int lastExtent = blockShape[lastDim];
  int64_t lastShape = desc.shape(lastDim);
  int64_t lastStart = coords[lastDim] > 0 ? coords[lastDim] : 0;
  int64_t lastEnd = static_cast<int64_t>(coords[lastDim]) + lastExtent;
  lastEnd = lastEnd < lastShape ? lastEnd : lastShape;
  if (lastEnd <= lastStart)
    return;

  int rowBytes = static_cast<int>((lastEnd - lastStart) * bytesPerElem);
  uintptr_t rowBase = desc.base + lastStart * desc.byteStride(lastDim);
  int validShape[4];
  int numRows = 1;
  for (int dim = 0; dim < lastDim; ++dim) {
    int extent = blockShape[dim];
    int64_t shape = desc.shape(dim);
    int64_t start = coords[dim] > 0 ? coords[dim] : 0;
    int64_t end = static_cast<int64_t>(coords[dim]) + extent;
    end = end < shape ? end : shape;
    if (end <= start)
      return;

    int validExtent = static_cast<int>(end - start);
    validShape[dim] = validExtent;
    numRows *= validExtent;

    uintptr_t stride = desc.byteStride(dim);
    rowBase += start * stride;
  }

  uintptr_t innerStride = lastDim == 0 ? 0 : desc.byteStride(lastDim - 1);
  uintptr_t rowPtr = rowBase + warpId * innerStride;
  int indices[4] = {warpId, 0, 0, 0};
  for (int row = warpId; row < numRows; row += numWarps) {
    for (int i = 0; i + 1 < lastDim; ++i) {
      int dim = lastDim - 1 - i;
      int extent = validShape[dim];
      uintptr_t stride = desc.byteStride(dim);
      uintptr_t nextStride = desc.byteStride(dim - 1);
      while (indices[i] >= extent) {
        indices[i] -= extent;
        rowPtr -= extent * stride;
        ++indices[i + 1];
        rowPtr += nextStride;
      }
    }

    tensorAccessRow(rowPtr, rowBytes, elementGranularity, handleElement);

    indices[0] += numWarps;
    rowPtr += numWarps * innerStride;
  }
}

template <typename ElementHandler>
GSAN_DEVICE void tensorAccessIndexedDesc(const void *descriptor,
                                         const int *indices, int numIndices,
                                         int yOffset, int numCols,
                                         int bytesPerElem, int pred,
                                         ElementHandler &handleElement) {
  if (!pred)
    return;

  TensorDescInfo desc = decodeTensorDesc(descriptor, 2, bytesPerElem);
  int64_t rowShape = desc.shape(0);
  int64_t colShape = desc.shape(1);
  int64_t firstCol = yOffset > 0 ? yOffset : 0;
  int64_t endCol = static_cast<int64_t>(yOffset) + numCols;
  endCol = endCol < colShape ? endCol : colShape;
  if (endCol <= firstCol)
    return;

  int rowBytes = static_cast<int>((endCol - firstCol) * bytesPerElem);
  uintptr_t rowStride = desc.byteStride(0);
  uintptr_t colOffset = firstCol * desc.byteStride(1);
  for (int i = 0; i < numIndices; ++i) {
    int index = indices[i];
    if (index < 0 || static_cast<int64_t>(index) >= rowShape)
      continue;

    uintptr_t rowPtr =
        desc.base + static_cast<uintptr_t>(index) * rowStride + colOffset;
    tensorAccessRow(rowPtr, rowBytes, /*elementGranularity=*/0, handleElement);
  }
}

struct ReadShadowCellHandler {
  template <typename Cell>
  GSAN_DEVICE void operator()(ThreadState *state, Cell *cell,
                              Location loc) const {
    doRead(state, cell, loc);
  }
};

struct WriteShadowCellHandler {
  template <typename Cell>
  GSAN_DEVICE void operator()(ThreadState *state, Cell *cell,
                              Location loc) const {
    doWrite(state, cell, loc);
  }
};

template <typename ShadowCellHandler> struct NonAtomicElementHandler {
  ThreadState *state;
  uintptr_t reserveBase;
  Location loc;
  ShadowCellHandler handleShadowCell;
  bool acquired = false;

  GSAN_DEVICE void operator()(uintptr_t address) {
    if (!acquired) {
      rwLockAcquireRead(state->lock);
      acquired = true;
    }
    if (isWriteOnceAddress(address)) {
      auto *cell =
          reinterpret_cast<WriteOnceShadowCell *>(getShadowAddress(address));
      handleShadowCell(state, cell, loc);
      return;
    }
    bool systemScope = useSystemScopeShadowLocks(state);
    auto cell = acquireShadow(getShadowAddress(address), systemScope);
    handleShadowCell(state, cell, loc);
    releaseShadow(cell, systemScope);
  }
};

template <typename ShadowCellHandler>
GSAN_DEVICE void
tensorNonAtomicDesc(ThreadState *state, const void *descriptor,
                    const int *coords, int rank, const int16_t *blockShape,
                    int bytesPerElem, int warpId, int numWarps, Location loc,
                    ShadowCellHandler handleShadowCell) {
  NonAtomicElementHandler<ShadowCellHandler> handleElement{
      state, state->reserveBase, loc, handleShadowCell};
  tensorAccessDesc(descriptor, coords, rank, blockShape, bytesPerElem,
                   /*elementGranularity=*/0, warpId, numWarps, handleElement);
  if (handleElement.acquired)
    rwLockReleaseRead(state->lock);
}

template <typename ShadowCellHandler>
GSAN_DEVICE void
tensorNonAtomicIndexedDesc(ThreadState *state, const void *descriptor,
                           const int *indices, int numIndices, int yOffset,
                           int numCols, int bytesPerElem, int pred,
                           Location loc, ShadowCellHandler handleShadowCell) {
  NonAtomicElementHandler<ShadowCellHandler> handleElement{
      state, state->reserveBase, loc, handleShadowCell};
  tensorAccessIndexedDesc(descriptor, indices, numIndices, yOffset, numCols,
                          bytesPerElem, pred, handleElement);
  if (handleElement.acquired)
    rwLockReleaseRead(state->lock);
}

GSAN_DEVICE void initAtomicEventState(AtomicEventState *event) {
  event->threadState = nullptr;
  event->numCells = 0;
  for (auto &cell : event->cells)
    cell = nullptr;
}

GSAN_DEVICE void checkAtomicAddress(uintptr_t reserveBase, uintptr_t address,
                                    Location loc) {
  assert_msg(loc,
             categorize(address, reserveBase) != AddressCategory::WriteOnce,
             "Atomic operations on write-once memory are not supported");
}

GSAN_DEVICE void acquireAtomicShadowRange(ThreadState *state,
                                          AtomicEventState *event,
                                          uintptr_t address, int nBytes,
                                          int bytesPerElem, Location loc) {
  checkAtomicAddress(state->reserveBase, address, loc);
  if (!isGsanManaged(address, state->reserveBase))
    return;
  int granularity = getShadowGranularity(address);
  checkAtomicGranularity(granularity, bytesPerElem, loc);
  auto range = roundRange(Range{address, address + nBytes}, granularity);
  auto reserveBase = state->reserveBase;
  uint8_t numCells = 0;
  for (uintptr_t addr = range.start; addr < range.end; addr += granularity) {
    if (isGsanManaged(addr, reserveBase))
      ++numCells;
  }
  assert_msg(loc, numCells <= kMaxAtomicShadowCells,
             "Atomic access spans too many GSan shadow cells");
  if (numCells == 0)
    return;

  // FIXME: Deadlock risk. If two concurrent accesses have different types, they
  // may partially acquire the shadow cells and block other threads from making
  // progress.
  rwLockAcquireWrite(state->lock);
  event->threadState = state;
  event->numCells = 0;
  for (uintptr_t addr = range.start; addr < range.end; addr += granularity) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    event->cells[event->numCells++] =
        acquireShadow(getShadowAddress(addr), useSystemScopeShadowLocks(state));
  }
}

GSAN_DEVICE void releaseAtomicShadowRange(AtomicEventState *event) {
  if (event->threadState == nullptr)
    return;
  for (uint8_t i = 0; i < event->numCells; ++i)
    releaseShadow(event->cells[i],
                  useSystemScopeShadowLocks(event->threadState));
  rwLockReleaseWrite(event->threadState->lock);
  initAtomicEventState(event);
}

GSAN_DEVICE void beginAtomicAccess(GlobalState *globals,
                                   AtomicEventState *event, bool pred,
                                   uintptr_t address, int nBytes,
                                   int bytesPerElem, bool doesRead,
                                   uint32_t semRaw, uint32_t scopeRaw,
                                   Location loc) {
  initAtomicEventState(event);
  if (!pred)
    return;

  auto *state = getThreadState(globals);
  acquireAtomicShadowRange(state, event, address, nBytes, bytesPerElem, loc);
  if (event->threadState == nullptr)
    return;

  auto sem = decodeAtomicSem(semRaw);
  auto scope = decodeAtomicScope(scopeRaw);
  if (doesRead) {
    for (uint8_t i = 0; i < event->numCells; ++i) {
      auto *cell = event->cells[i];
      auto write = cell->writeClock;
      assertOrderedOrCompatible(state, scope, write, loc,
                                "Read after write race detected");
      recordRead(state, cell, scope);
    }
    if (hasAcquire(sem)) {
      for (uint8_t i = 0; i < event->numCells; ++i) {
        auto write = event->cells[i]->writeClock;
        maybeMergeAcquire(state, scope, write, loc);
      }
    }
  }
}

GSAN_DEVICE void endAtomicAccess(AtomicEventState *event, bool pred,
                                 bool didWrite, bool isRmw, uint32_t semRaw,
                                 uint32_t scopeRaw, Location loc) {
  if (!pred || event->threadState == nullptr)
    return;

  auto *state = event->threadState;
  auto sem = decodeAtomicSem(semRaw);
  auto scope = decodeAtomicScope(scopeRaw);

  if (didWrite) {
    for (uint8_t i = 0; i < event->numCells; ++i) {
      auto *cell = event->cells[i];
      for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
        assertOrderedOrCompatible(state, scope, cell->readClocks[iRead], loc,
                                  "Write after read race detected");
      }
      assertOrderedOrCompatible(state, scope, cell->writeClock, loc,
                                "Write after write race detected");
    }

    auto previousWrite = event->cells[0]->writeClock;
    ScalarClock newWriteClock;
    if (hasRelease(sem)) {
      epoch_t token;
      if (isRmw && canAccumulateReleaseRmw(state, scope, previousWrite))
        token = publishCurrentVectorClockWithPriorRelease(state, previousWrite,
                                                          loc);
      else if (isRmw && previousWrite.isRelease) {
        const auto *previousSnapshot =
            getSnapshotForWrite(state, previousWrite, loc);
        assert_msg(loc, dominatesSnapshot(state, previousSnapshot),
                   "GSan detected atomic release accumulation with mixed "
                   "scopes, which is not supported.");
        token = publishCurrentVectorClock(state, loc);
      } else {
        token = publishCurrentVectorClock(state, loc);
      }
      newWriteClock = makePublishedClock(state, scope, token);
    } else {
      if (isRmw && previousWrite.isRelease) {
        auto token = propagateClockBufferSnapshot(state, previousWrite, loc);
        newWriteClock = makePublishedClock(state, scope, token);
      } else {
        newWriteClock = makeScalarClock(state, scope);
      }
    }

    for (uint8_t i = 0; i < event->numCells; ++i)
      event->cells[i]->writeClock = newWriteClock;

    if (hasRelease(sem))
      incrementThreadEpoch(state, loc);
  }

  releaseAtomicShadowRange(event);
}

struct AtomicElementHandler {
  GlobalState *globals;
  uintptr_t reserveBase;
  int bytesPerElem;
  int sem;
  int scope;
  Location loc;

  GSAN_DEVICE void operator()(uintptr_t address) const {
    AtomicEventState event;
    beginAtomicAccess(globals, &event, /*pred=*/true, address, bytesPerElem,
                      bytesPerElem, /*doesRead=*/true, sem, scope, loc);
    endAtomicAccess(&event, /*pred=*/true, /*didWrite=*/true,
                    /*isRmw=*/true, sem, scope, loc);
  }
};

GSAN_DEVICE void tensorAtomicDesc(GlobalState *globals, const void *descriptor,
                                  const int *coords, int rank,
                                  const int16_t *blockShape, int bytesPerElem,
                                  int sem, int scope, int warpId, int numWarps,
                                  Location loc) {
  AtomicElementHandler handleElement{
      globals, globals->reserveBase, bytesPerElem, sem, scope, loc};
  tensorAccessDesc(descriptor, coords, rank, blockShape, bytesPerElem,
                   bytesPerElem, warpId, numWarps, handleElement);
}

} // namespace
} // namespace gsan

extern "C" GSAN_DEVICE void
__triton_gsan_load_tensor(void *globalState, const char *stackPtr, int numElems,
                          int bytesPerElem, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
#if GSAN_AMDGPU
  gsan::tensorAccessCooperative<false>(threadState, stackPtr, numElems,
                                       bytesPerElem, loc);
#else
  gsan::tensorLoad(threadState, stackPtr, numElems, bytesPerElem, loc);
#endif
}

extern "C" GSAN_DEVICE void __triton_gsan_load_tensor_desc(
    void *globalState, const void *descriptor, const int *coords, int rank,
    const gsan::int16_t *blockShape, int bytesPerElem, int pred, int warpId,
    int numWarps, const char *file, unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::forEachLaneSerially([&] {
    gsan::tensorNonAtomicDesc(threadState, descriptor, coords, rank, blockShape,
                              bytesPerElem, warpId, numWarps, loc,
                              gsan::ReadShadowCellHandler{});
  });
}

extern "C" GSAN_DEVICE void __triton_gsan_load_indexed_tensor_desc(
    void *globalState, const void *descriptor, const int *indices,
    int numIndices, int yOffset, int numCols, int bytesPerElem, int pred,
    const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::forEachLaneSerially([&] {
    gsan::tensorNonAtomicIndexedDesc(threadState, descriptor, indices,
                                     numIndices, yOffset, numCols, bytesPerElem,
                                     pred, loc, gsan::ReadShadowCellHandler{});
  });
}

extern "C" GSAN_DEVICE void
__triton_gsan_init(void *globalState, const gsan::LaunchState *launchTable,
                   gsan::uint64_t launchIndex, unsigned threadIdx,
                   unsigned numThreads, unsigned barrierId, const char *file,
                   unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::initThread(reinterpret_cast<gsan::GlobalState *>(globalState),
                   launchTable[launchIndex], threadIdx, numThreads, barrierId,
                   loc);
}

extern "C" GSAN_DEVICE void __triton_gsan_kernel_exit(
    void *globalState, const gsan::LaunchState *launchTable,
    gsan::uint64_t launchIndex, unsigned threadIdx, unsigned numThreads,
    unsigned barrierId, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  auto *state = gsan::getThreadState(globals);
  gsan::publishLaunchClock(state, launchTable[launchIndex].completionClock,
                           threadIdx, numThreads, barrierId, loc);
#if GSAN_AMDGPU
  // publishStreamClock ends with a barrier after the last state use of all
  // waves, and thread 0 released its read lock before this.
  if (threadIdx == 0)
    gsan::releaseSlot(globals);
#endif
}

extern "C" GSAN_DEVICE void __triton_gsan_grid_dependency_wait(
    void *globalState, const gsan::LaunchState *launchTable,
    gsan::uint64_t launchIndex, unsigned threadIdx, unsigned numThreads,
    unsigned barrierId) {
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  auto *state = gsan::getThreadState(globals);
  gsan::acquireLaunchClocks(state, launchTable[launchIndex], threadIdx,
                            numThreads, barrierId);
}

extern "C" GSAN_DEVICE void __triton_gsan_cluster_barrier_init(void *scratch,
                                                               int pred) {
  if (pred)
    gsan::initClusterBarrier(scratch);
}

extern "C" GSAN_DEVICE void
__triton_gsan_cluster_barrier_sync(void *globalState, void *scratch, int pred,
                                   int numCTAs, int ctaRank, const char *file,
                                   unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  gsan::synchronizeClusterBarrier(
      reinterpret_cast<gsan::GlobalState *>(globalState), scratch, numCTAs,
      ctaRank, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_mbarrier_table_init(void *scratch, int pred, unsigned capacity) {
  if (pred)
    gsan::initMBarrierTable(scratch, capacity);
}

extern "C" GSAN_DEVICE void
__triton_gsan_mbarrier_init(void *scratch, unsigned offset, int pred,
                            unsigned ownerRank, unsigned expectedCount,
                            const char *file, unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  gsan::initMBarrier(scratch, offset, ownerRank, expectedCount, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_mbarrier_arrive(void *globalState, void *scratch, unsigned offset,
                              int pred, unsigned recipientMask, unsigned count,
                              unsigned sourceRank, int publishClock,
                              const char *file, unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  gsan::recordMBarrierArrival(
      reinterpret_cast<gsan::GlobalState *>(globalState), scratch, offset,
      recipientMask, count, sourceRank, publishClock != 0, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_mbarrier_wait(void *globalState, void *scratch, unsigned offset,
                            int pred, unsigned ownerRank, unsigned phase,
                            const char *file, unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  gsan::acquireMBarrierPhase(reinterpret_cast<gsan::GlobalState *>(globalState),
                             scratch, offset, ownerRank, phase, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_store_tensor(void *globalState, const char *stackPtr,
                           int numElems, int bytesPerElem, const char *file,
                           unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
#if GSAN_AMDGPU
  gsan::tensorAccessCooperative<true>(threadState, stackPtr, numElems,
                                      bytesPerElem, loc);
#else
  gsan::tensorStore(threadState, stackPtr, numElems, bytesPerElem, loc);
#endif
}

extern "C" GSAN_DEVICE void __triton_gsan_store_tensor_desc(
    void *globalState, const void *descriptor, const int *coords, int rank,
    const gsan::int16_t *blockShape, int bytesPerElem, int pred, int warpId,
    int numWarps, const char *file, unsigned line) {
  if (!pred)
    return;
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::forEachLaneSerially([&] {
    gsan::tensorNonAtomicDesc(threadState, descriptor, coords, rank, blockShape,
                              bytesPerElem, warpId, numWarps, loc,
                              gsan::WriteShadowCellHandler{});
  });
}

extern "C" GSAN_DEVICE void __triton_gsan_store_indexed_tensor_desc(
    void *globalState, const void *descriptor, const int *indices,
    int numIndices, int yOffset, int numCols, int bytesPerElem, int pred,
    const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::forEachLaneSerially([&] {
    gsan::tensorNonAtomicIndexedDesc(threadState, descriptor, indices,
                                     numIndices, yOffset, numCols, bytesPerElem,
                                     pred, loc, gsan::WriteShadowCellHandler{});
  });
}

extern "C" GSAN_DEVICE void __triton_gsan_atomic_tensor_desc(
    void *globalState, const void *descriptor, const int *coords, int rank,
    const gsan::int16_t *blockShape, int bytesPerElem, int sem, int scope,
    int warpId, int numWarps, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  gsan::forEachLaneSerially([&] {
    gsan::tensorAtomicDesc(globals, descriptor, coords, rank, blockShape,
                           bytesPerElem, sem, scope, warpId, numWarps, loc);
  });
}

extern "C" GSAN_DEVICE void
__triton_gsan_atomic_begin_scalar(void *globalState, void *eventState, int pred,
                                  gsan::uintptr_t address, int nBytes,
                                  int bytesPerElem, int doesRead, int sem,
                                  int scope, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::beginAtomicAccess(
      reinterpret_cast<gsan::GlobalState *>(globalState),
      reinterpret_cast<gsan::AtomicEventState *>(eventState), pred != 0,
      address, nBytes, bytesPerElem, doesRead != 0, sem, scope, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_atomic_end_scalar(void *eventState, int pred, int didWrite,
                                int isRmw, int sem, int scope, const char *file,
                                unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::endAtomicAccess(reinterpret_cast<gsan::AtomicEventState *>(eventState),
                        pred != 0, didWrite != 0, isRmw != 0, sem, scope, loc);
}
