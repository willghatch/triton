// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritonamdgpu-gsan-instrument-direct-async -tritonamdgpu-gsan-instrument-tdm -tritoninstrument-global-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @instrumented
  // CHECK-SAME: !tt.ptr<i8> {tti.gsan_global_state}
  // CHECK-SAME: !tt.ptr<i8> {tti.gsan_launch_table}
  // CHECK-SAME: i64 {tti.gsan_launch_index}
  tt.func @instrumented(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>,
                        %mask: tensor<128xi1, #blocked>,
                        %other: tensor<128xf32, #blocked>,
                        %vals: tensor<128xf32, #blocked>) {
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, false, %{{.*}}
    // CHECK-NEXT: %[[LD:.*]] = tt.load
    %0 = tt.load %ptrs, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, true, %{{.*}}
    // CHECK-NEXT: tt.store
    tt.store %ptrs, %vals, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @instrumented_atomic_poll
  tt.func @instrumented_atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32) {
    // CHECK: %[[MATCHED:.*]] = tt.atomic_poll acquire, sys, %[[PTR:.*]], %{{.*}} : !tt.ptr<i32>, i32 -> i1
    // CHECK-NEXT: tti.experimental_gsan_atomic_poll acquire, sys, %[[PTR]], %[[MATCHED]] : !tt.ptr<i32>
    // CHECK-NEXT: ttg.barrier local
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_copy
  tt.func @instrumented_async_copy(%ptrs: tensor<128x!tt.ptr<f16>, #blocked>,
                                   %mask: tensor<128xi1, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensor_access %[[PTRS:.*]], false, %[[MASK:.*]] :
    // CHECK-NEXT: ttg.async_copy_global_to_local %[[PTRS]], {{.*}} mask %[[MASK]]
    %tok = ttg.async_copy_global_to_local %ptrs, %buf mask %mask : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_copy
  tt.func @instrumented_async_tma_copy(%desc: !tt.tensordesc<8x8x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_access %arg0[%{{.*}}, %{{.*}}, %{{.*}}], false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<8x8x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_access %arg0[%{{.*}}, %{{.*}}, %{{.*}}], true
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32, %c0_i32] %buf : !tt.tensordesc<8x8x32xf32, #shared>, !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @instrumented_async_tma_reduce
  tt.func @instrumented_async_tma_reduce(%desc: !tt.tensordesc<8x8x32xi32, #shared>) {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x32xi32, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_atomic_tensordesc_access %arg0[%{{.*}}, %{{.*}}, %{{.*}}], relaxed, gpu
    // CHECK-NEXT: ttng.async_tma_reduce add
    ttng.async_tma_reduce add, %desc[%c0_i32, %c0_i32, %c0_i32] %buf : !tt.tensordesc<8x8x32xi32, #shared>, !ttg.memdesc<64x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_pair = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_release_acquire
  tt.func @mbarrier_release_acquire(%phase: i32, %pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 192 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_table_init %[[SCRATCH]], 1 : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 1
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER]], %{{.*}}, 1
    // CHECK-SAME: multicast = false, multicastMasks = array<i32>, sourceBroadcastMask = 0 : i32
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttng.arrive_barrier %[[BARRIER]], 1, %{{.*}}
    ttng.arrive_barrier %barrier, 1, %pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %{{.*}}
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %{{.*}}
    // CHECK-NEXT: ttg.barrier local
    ttng.wait_barrier %barrier, %phase, %pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_pair = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_warp_specialized
  tt.func @mbarrier_warp_specialized(%pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
    // CHECK: ttg.warp_specialize(%[[BARRIER:.*]], %[[PRED:.*]], %[[SCRATCH]])
    ttg.warp_specialize(%barrier, %pred)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%[[PARTITION_BARRIER:.*]]: !ttg.memdesc<1xi64, #{{.*}}, #smem, mutable>, %[[PARTITION_PRED:.*]]: i1, %[[PARTITION_SCRATCH:.*]]: !tt.ptr<i8>)
    partition0(%partition_barrier: !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>, %partition_pred: i1) num_warps(4) {
      // CHECK: tti.experimental_gsan_mbarrier_arrive %[[PARTITION_SCRATCH]], %[[PARTITION_BARRIER]], %[[PARTITION_PRED]], 1
      ttng.arrive_barrier %partition_barrier, 1, %partition_pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
      %phase = arith.constant 0 : i32
      // CHECK: tti.experimental_gsan_mbarrier_wait %[[PARTITION_SCRATCH]], %[[PARTITION_BARRIER]], %{{.*}}, %[[PARTITION_PRED]]
      ttng.wait_barrier %partition_barrier, %phase, %partition_pred : !ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #mbarrier_pair, #smem, mutable>, i1) -> ()
    tt.return
  }
}

// -----

#mbarrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_dynamic_index
  tt.func @mbarrier_dynamic_index() {
    // Four barriers per CTA, including those reached on later loop iterations.
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1424 : i32,
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_table_init %[[SCRATCH]], 8 : <i8>
    %barriers = ttg.local_alloc : () -> !ttg.memdesc<4x2xi64, #mbarrier, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %barrier = ttg.memdesc_index %barriers[%i] : !ttg.memdesc<4x2xi64, #mbarrier, #smem, mutable> -> !ttg.memdesc<2xi64, #mbarrier, #smem, mutable>
      ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #mbarrier, #smem, mutable>
      ttng.inval_barrier %barrier : !ttg.memdesc<2xi64, #mbarrier, #smem, mutable>
      // Reinitialization must not count the same allocation again.
      ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #mbarrier, #smem, mutable>
      ttng.inval_barrier %barrier : !ttg.memdesc<2xi64, #mbarrier, #smem, mutable>
    }
    tt.return
  }
}

// -----

#mbarrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @mbarrier_loop_carried_allocations
  tt.func @mbarrier_loop_carried_allocations() {
    // Both loop-carried allocations can reach the same init operation.
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 720 : i32,
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_table_init %[[SCRATCH]], 4 : <i8>
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %r:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%current = %a, %next = %b) -> (!ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>, !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>) : i32 {
      scf.for %j = %c0 to %c2 step %c1 : i32 {
        %barrier = ttg.memdesc_index %current[%j] : !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable> -> !ttg.memdesc<1xi64, #mbarrier, #smem, mutable>
        ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier, #smem, mutable>
        ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #mbarrier, #smem, mutable>
      }
      scf.yield %next, %current : !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>, !ttg.memdesc<2x1xi64, #mbarrier, #smem, mutable>
    }
    tt.return
  }
}

// -----

#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_tcgen5_commit
  tt.func @mbarrier_tcgen5_commit(%pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %[[PRED:.*]], 1
    // CHECK-SAME: publishClock = false
    // CHECK-NEXT: ttng.tc_gen5_commit %[[BARRIER]], %[[PRED]]
    ttng.tc_gen5_commit %barrier, %pred : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared_commit = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_tcgen5_commit_multicast
  tt.func @mbarrier_tcgen5_commit_multicast(%desc: !ttg.memdesc<128x128xf16, #shared_commit, #smem>, %wait_pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 2
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 2 : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %{{.*}}, 1
    // CHECK-SAME: multicast = true, multicastMasks = array<i32: 1>, publishClock = false, sourceBroadcastMask = 0 : i32
    // CHECK-NEXT: ttng.tc_gen5_commit %[[BARRIER]] descs %{{.*}}
    ttng.tc_gen5_commit %barrier descs %desc : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared_commit, #smem>
    %phase = arith.constant 0 : i32
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %[[WAIT_PRED:.*]] :
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %[[WAIT_PRED]]
    ttng.wait_barrier %barrier, %phase, %wait_pred : !ttg.memdesc<2xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

#mbarrier_partial = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [1]]}>
#tma_partial = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0], [0, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_multicast_partial_wait
  tt.func @mbarrier_multicast_partial_wait(%desc: !tt.tensordesc<256x128xf16, #tma_partial>, %wait_pred: i1) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %true = arith.constant true
    %zero = arith.constant 0 : i32
    %signal = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: ttng.init_barrier %[[BARRIER:.*]], 1
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_init %[[SCRATCH]], %[[BARRIER]], 2
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER]], %{{.*}}, 1
    // CHECK-SAME: multicast = false, multicastMasks = array<i32>, sourceBroadcastMask = 0 : i32
    ttng.barrier_expect %barrier, 32768, %true : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}}, %[[BARRIER]], %{{.*}} {multicast}
    ttng.async_tma_copy_global_to_local %desc[%zero, %zero] %signal, %barrier, %true {multicast} : !tt.tensordesc<256x128xf16, #tma_partial>, !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable> -> !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    %phase = arith.constant 0 : i32
    // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE:.*]], %[[WAIT_PRED:.*]] deps %{{.*}} :
    // CHECK-NEXT: tti.experimental_gsan_mbarrier_wait %[[SCRATCH]], %[[BARRIER]], %[[PHASE]], %[[WAIT_PRED]]
    ttng.wait_barrier %barrier, %phase, %wait_pred deps %signal : !ttg.memdesc<2xi64, #mbarrier_partial, #smem, mutable>, !ttg.memdesc<256x128xf16, #tma_partial, #smem, mutable>
    tt.return
  }
}

// -----

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#mbarrier_local = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>

// Gluon represents two-CTA execution on the MMA op itself and does not add the
// module-level ttng.two-ctas attribute.
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @mbarrier_two_cta_mma_without_module_attr
  tt.func @mbarrier_two_cta_mma_without_module_attr() {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>
    %d = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    // CHECK: tti.experimental_gsan_mbarrier_arrive %[[SCRATCH]], %[[BARRIER:.*]], %{{.*}}, 1
    // CHECK-SAME: multicast = true, multicastMasks = array<i32: 1>, publishClock = false, sourceBroadcastMask = 1 : i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[BARRIER]][%{{.*}}] {is_async, two_ctas}
    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %barrier[%true] {is_async, two_ctas} :
      !ttg.memdesc<256x128xf16, #shared_a, #smem, mutable>,
      !ttg.memdesc<128x128xf16, #shared_b, #smem, mutable>,
      !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #mbarrier_local, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @cluster_barriers
  tt.func @cluster_barriers() {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 128 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK-NOT: tti.experimental_gsan_cluster_barrier_sync
    ttng.cluster_barrier
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @cluster_barrier_equivalents
  tt.func @cluster_barrier_equivalents(%ptr: !tt.ptr<i32>) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier {relaxed = true}
    // CHECK: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    %one = arith.constant 1 : i32
    %release = tt.atomic_rmw add, release, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %acquire = tt.atomic_rmw add, acquire, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %acq_rel = tt.atomic_rmw add, acq_rel, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK-NEXT: %{{.*}} = tti.experimental_gsan_atomic_rmw
    %relaxed = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32

    // CHECK: %{{.*}} = ttg.convert_layout
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %value = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %converted = ttg.convert_layout %value : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
    tt.return
  }
}

// -----

#blockedLocalBroadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
#blockedCTABroadcast = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @atomic_result_barrier_scope
  tt.func @atomic_result_barrier_scope(%ptr: !tt.ptr<i32>) {
    // CHECK: %[[SCRATCH:.*]] = ttg.global_scratch_alloc
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_init %[[SCRATCH]] : <i8>
    %local_ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>
    %local_ones = arith.constant dense<1> : tensor<32xi32, #blockedLocalBroadcast>

    // A CTA-local scratch barrier cannot supply the acquire cluster barrier,
    // even if shared-memory allocation has already attached an offset.
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %local_acquire = tt.atomic_rmw add, acquire, gpu, %local_ptrs, %local_ones {allocation.offset = 0 : i32} : (tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>, tensor<32xi32, #blockedLocalBroadcast>) -> tensor<32xi32, #blockedLocalBroadcast>
    tt.store %local_ptrs, %local_acquire : tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>

    // A relaxed atomic with only a CTA-local result broadcast has no cluster
    // synchronization effect.
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NOT: tti.experimental_gsan_cluster_barrier_sync
    %local_relaxed = tt.atomic_rmw add, relaxed, gpu, %local_ptrs, %local_ones {allocation.offset = 0 : i32} : (tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>, tensor<32xi32, #blockedLocalBroadcast>) -> tensor<32xi32, #blockedLocalBroadcast>
    tt.store %local_ptrs, %local_relaxed : tensor<32x!tt.ptr<i32>, #blockedLocalBroadcast>

    %cluster_ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>
    %cluster_ones = arith.constant dense<1> : tensor<128xi32, #blockedCTABroadcast>
    // CHECK: %{{.*}} = tti.experimental_gsan_atomic_rmw
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NEXT: tti.experimental_gsan_cluster_barrier_sync %[[SCRATCH]] : <i8>
    // CHECK-NEXT: ttng.cluster_barrier
    %cluster_relaxed = tt.atomic_rmw add, relaxed, gpu, %cluster_ptrs, %cluster_ones {allocation.offset = 0 : i32} : (tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>, tensor<128xi32, #blockedCTABroadcast>) -> tensor<128xi32, #blockedCTABroadcast>
    tt.store %cluster_ptrs, %cluster_relaxed : tensor<128x!tt.ptr<i32>, #blockedCTABroadcast>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_rank_reducing_tma_copy
  tt.func @instrumented_rank_reducing_tma_copy(%desc: !tt.tensordesc<1x1x1x32x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_access %arg0[%{{.*}}], false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<1x1x1x32x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_access %arg0[%{{.*}}], true
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] %buf : !tt.tensordesc<1x1x1x32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @instrumented_call_in_warp_specialize
  // CHECK-SAME: %[[STATE:[^:, )]+]]: !tt.ptr<i8>
  // CHECK-SAME: %[[LAUNCH_TABLE:[^:, )]+]]: !tt.ptr<i8>
  // CHECK-SAME: %[[LAUNCH_INDEX:[^:, )]+]]: i64
  tt.func public @instrumented_call_in_warp_specialize(%value: i32) {
    // CHECK: ttg.warp_specialize(%{{.*}}, %[[STATE]], %[[LAUNCH_TABLE]], %[[LAUNCH_INDEX]])
    ttg.warp_specialize(%value)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%[[VALUE:[^:, )]+]]: i32, %[[PARTITION_STATE:[^:, )]+]]: !tt.ptr<i8>,
    // CHECK-SAME: %[[PARTITION_TABLE:[^:, )]+]]: !tt.ptr<i8>, %[[PARTITION_INDEX:[^:, )]+]]: i64) num_warps(4)
    partition0(%partition_value: i32) num_warps(4) {
      // CHECK: tt.call @identity(%[[VALUE]], %[[PARTITION_STATE]], %[[PARTITION_TABLE]], %[[PARTITION_INDEX]])
      // CHECK-SAME: : (i32, !tt.ptr<i8>, !tt.ptr<i8>, i64) -> i32
      %result = tt.call @identity(%partition_value) : (i32) -> i32
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }

  tt.func private @identity(%value: i32) -> i32 {
    tt.return %value : i32
  }
}

// -----

#blocked_rows_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_rows = #ttg.slice<{dim = 0, parent = #blocked_rows_parent}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_gather_scatter
  tt.func @instrumented_async_tma_gather_scatter(%desc: !tt.tensordesc<1x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32, #blocked_rows>
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_indexed_tensordesc_access %arg0[%{{.*}}, %{{.*}}], false, %{{.*}} : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #{{.*}}>
    // CHECK-NEXT: ttng.async_tma_gather
    ttng.async_tma_gather %desc[%x_offsets, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #blocked_rows>, i32, !ttg.memdesc<1xi64, #bar, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    // CHECK: tti.experimental_gsan_indexed_tensordesc_access %arg0[%{{.*}}, %{{.*}}], true : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #{{.*}}>
    // CHECK-NEXT: ttng.async_tma_scatter
    ttng.async_tma_scatter %desc[%x_offsets, %c0_i32] %buf : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #blocked_rows>, i32, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @instrumented_async_tma_copy_device_desc
  tt.func @instrumented_async_tma_copy_device_desc(%raw_desc: !tt.ptr<i8>,
                                                   %base: !tt.ptr<f32>,
                                                   %shape0: i32, %shape1: i32,
                                                   %stride0: i64) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    ttng.tensormap_create %raw_desc, %base, [%c32_i32, %c32_i32], [%shape1, %shape0], [%stride0], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<f32>, i32, i32, i32, i32, i64, i32, i32) -> ()
    // CHECK: %[[DESC:.*]] = ttng.reinterpret_tensor_descriptor %arg0
    %desc = ttng.reinterpret_tensor_descriptor %raw_desc : !tt.ptr<i8> to !tt.tensordesc<32x32xf32, #shared>
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: tti.experimental_gsan_tensordesc_access %[[DESC]][%{{.*}}, %{{.*}}], false, %{{.*}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// AMD direct copies retain their async tokens and check only global memory at
// issue time.  The buffer address uses independently wrapped unsigned i32
// byte offsets and descriptor bounds, so masks and OOB zero-fill do not add
// false global accesses.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: @amd_direct_store
  // CHECK-SAME: %[[DST:[a-zA-Z0-9_]+]]: tensor<64x!tt.ptr<f32>, #blocked>
  // CHECK-SAME: %[[MASK:[a-zA-Z0-9_]+]]: tensor<64xi1, #blocked>
  tt.func @amd_direct_store(%src: !ttg.memdesc<64xf32, #shared, #smem, mutable>, %dst: tensor<64x!tt.ptr<f32>, #blocked>, %mask: tensor<64xi1, #blocked>) {
    // CHECK: tti.experimental_gsan_tensor_access %[[DST]], true, %[[MASK]]
    // CHECK: %[[TOK:[a-zA-Z0-9_]+]] = amdg.async_copy_local_to_global {{.*}}, %[[DST]] mask %[[MASK]]
    %token = amdg.async_copy_local_to_global %src, %dst mask %mask : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64x!tt.ptr<f32>, #blocked>
    // CHECK: ttg.async_wait %[[TOK]]
    ttg.async_wait %token {num = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @amd_buffer_async
  // CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<f32>
  // CHECK-SAME: %[[BOUND:[a-zA-Z0-9_]+]]: i32
  // CHECK-SAME: %[[OFF:[a-zA-Z0-9_]+]]: tensor<64xi32, #blocked>
  // CHECK-SAME: %[[SOFF:[a-zA-Z0-9_]+]]: i32
  // CHECK-SAME: %[[USERMASK:[a-zA-Z0-9_]+]]: tensor<64xi1, #blocked>
  tt.func @amd_buffer_async(%base: !tt.ptr<f32>, %bound: i32, %offsets: tensor<64xi32, #blocked>, %soffset: i32, %mask: tensor<64xi1, #blocked>, %other: tensor<64xf32, #blocked>, %dst: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %desc = amdg.make_buffer_desc %base, %bound : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK-DAG: %[[FOUR:[a-zA-Z0-9_]+]] = arith.constant dense<4> : tensor<64xi32
    // CHECK-DAG: %[[VOFF:[a-zA-Z0-9_]+]] = arith.muli %[[OFF]], %[[FOUR]]
    // CHECK-DAG: %[[V64:[a-zA-Z0-9_]+]] = arith.extui %[[VOFF]]
    // CHECK-DAG: %[[SFOUR:[a-zA-Z0-9_]+]] = arith.constant 4 : i32
    // CHECK-DAG: %[[SWRAP:[a-zA-Z0-9_]+]] = arith.muli %[[SOFF]], %[[SFOUR]]
    // CHECK-DAG: %[[S64:[a-zA-Z0-9_]+]] = arith.extui %[[SWRAP]]
    // CHECK-DAG: %[[SSPLAT:[a-zA-Z0-9_]+]] = tt.splat %[[S64]]
    // CHECK: %[[BYTES:[a-zA-Z0-9_]+]] = arith.addi %[[V64]], %[[SSPLAT]]
    // CHECK-DAG: %[[B64:[a-zA-Z0-9_]+]] = arith.extui %[[BOUND]]
    // CHECK-DAG: %[[BSPLAT:[a-zA-Z0-9_]+]] = tt.splat %[[B64]]
    // CHECK-DAG: %[[BYTEBASE:[a-zA-Z0-9_]+]] = tt.bitcast %[[BASE]] : !tt.ptr<f32> -> !tt.ptr<i8>
    // CHECK: %[[PTRS:[a-zA-Z0-9_]+]] = tt.splat %[[BYTEBASE]]
    // CHECK: %[[ADDRS:[a-zA-Z0-9_]+]] = tt.addptr %[[PTRS]], %[[BYTES]]
    // CHECK-DAG: %[[END:[a-zA-Z0-9_]+]] = arith.addi %[[BYTES]], {{.*}}
    // CHECK: %[[INBOUNDS:[a-zA-Z0-9_]+]] = arith.cmpi ule, %[[END]], %[[BSPLAT]]
    // CHECK: %[[ACTIVE:[a-zA-Z0-9_]+]] = arith.andi %[[USERMASK]], %[[INBOUNDS]]
    // CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tt.bitcast %[[ADDRS]]
    // CHECK: tti.experimental_gsan_tensor_access %[[TYPED]], false, %[[ACTIVE]]
    // CHECK: %[[COPY:[a-zA-Z0-9_]+]] = amdg.buffer_load_to_local {{.*}}[%[[OFF]]] mask = %[[USERMASK]] other = {{.*}} soffset = %[[SOFF]]
    %token = amdg.buffer_load_to_local %desc[%offsets] mask = %mask other = %other soffset = %soffset into %dst : !amdg.buffer_desc<f32>[tensor<64xi32, #blocked>] tensor<64xf32, #blocked> -> <64xf32, #shared, #smem, mutable>
    // CHECK: ttg.async_wait %[[COPY]]
    ttg.async_wait %token {num = 0 : i32}
    tt.return
  }
}

// -----

// A dynamic bound can cut an f64 element between its two dwords.  Only its
// in-bounds dwords should enter shadow memory; counting the whole element
// would spuriously report races past the bound in fine-grained pools.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  // CHECK-LABEL: @amd_buffer_async_partial_dword
  tt.func @amd_buffer_async_partial_dword(%base: !tt.ptr<f64>, %bound: i32, %offsets: tensor<128xi32, #blocked>, %dst: !ttg.memdesc<128xf64, #shared, #smem, mutable>) {
    %desc = amdg.make_buffer_desc %base, %bound : !tt.ptr<f64> -> !amdg.buffer_desc<f64>
    // CHECK: %[[FIRSTADDR:[a-zA-Z0-9_]+]] = tt.addptr
    // CHECK: %[[FIRSTEND:[a-zA-Z0-9_]+]] = arith.addi
    // CHECK: %[[FIRSTMASK:[a-zA-Z0-9_]+]] = arith.cmpi ule, %[[FIRSTEND]], %[[BOUND:[a-zA-Z0-9_]+]]
    // CHECK: %[[FIRSTPTR:[a-zA-Z0-9_]+]] = tt.bitcast %[[FIRSTADDR]] : {{.*}} -> tensor<128x!tt.ptr<i32>
    // CHECK: tti.experimental_gsan_tensor_access %[[FIRSTPTR]], false, %[[FIRSTMASK]]
    // CHECK: %[[SECONDADDR:[a-zA-Z0-9_]+]] = tt.addptr
    // CHECK: %[[SECONDEND:[a-zA-Z0-9_]+]] = arith.addi
    // CHECK: %[[SECONDMASK:[a-zA-Z0-9_]+]] = arith.cmpi ule, %[[SECONDEND]], %[[BOUND]]
    // CHECK: %[[SECONDPTR:[a-zA-Z0-9_]+]] = tt.bitcast %[[SECONDADDR]] : {{.*}} -> tensor<128x!tt.ptr<i32>
    // CHECK: tti.experimental_gsan_tensor_access %[[SECONDPTR]], false, %[[SECONDMASK]]
    // CHECK-NOT: tti.experimental_gsan_tensor_access
    // CHECK: amdg.buffer_load_to_local
    %token = amdg.buffer_load_to_local %desc[%offsets] into %dst : !amdg.buffer_desc<f64>[tensor<128xi32, #blocked>] -> <128xf64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Positioned and loop-carried AMD descriptors must be checked as the current
// SSA value, preserving their bounds/predicate updates and both directions.
#shared = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @instrumented_tdm_updates(
  // CHECK-SAME: [[DESC:%[a-zA-Z0-9_]+]]: !tt.tensordesc
  // CHECK-SAME: [[OFFSET:%[a-zA-Z0-9_]+]]: i32
  // CHECK-SAME: [[BOUND:%[a-zA-Z0-9_]+]]: i32
  // CHECK-SAME: [[PRED:%[a-zA-Z0-9_]+]]: i32
  tt.func @instrumented_tdm_updates(%desc: !tt.tensordesc<64x64xf16, #shared>, %offset: i32, %bound: i32, %pred: i32, %iters: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: [[POSITIONED:%[a-zA-Z0-9_]+]] = amdg.update_tensor_descriptor [[DESC]] add_offsets = {{\[}}[[OFFSET]], %{{.*}}{{\]}} set_bounds = {{\[}}[[BOUND]], [[BOUND]]{{\]}} pred = [[PRED]]
    %positioned = amdg.update_tensor_descriptor %desc add_offsets=[%offset, %zero] set_bounds=[%bound, %bound] pred=%pred : !tt.tensordesc<64x64xf16, #shared>
    // CHECK: [[ZERO:%[a-zA-Z0-9_]+]] = arith.constant 0 : i32
    // CHECK: tti.experimental_gsan_tensordesc_access [[POSITIONED]]{{\[}}[[ZERO]], [[ZERO]]{{\]}}, false
    // CHECK-NEXT: {{.*}}amdg.async_tdm_copy_global_to_local [[POSITIONED]]
    %token = amdg.async_tdm_copy_global_to_local %positioned into %buf {warp_used_hint = 5 : i32} : !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %loaded = amdg.async_tdm_wait %token {num = 0 : i32}
    ttg.barrier local
    // CHECK: scf.for {{.*}}iter_args([[CARRIED:%[a-zA-Z0-9_]+]] = [[POSITIONED]])
    %result = scf.for %i = %zero to %iters step %one iter_args(%carried = %positioned) -> (!tt.tensordesc<64x64xf16, #shared>) : i32 {
      // CHECK: tti.experimental_gsan_tensordesc_access [[CARRIED]]{{\[}}%{{.*}}, %{{.*}}{{\]}}, true
      // CHECK-NEXT: {{.*}}amdg.async_tdm_copy_local_to_global [[CARRIED]]
      %stored = amdg.async_tdm_copy_local_to_global %carried from %buf : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x64xf16, #shared>
      %completed = amdg.async_tdm_wait %stored {num = 0 : i32}
      %next = amdg.update_tensor_descriptor %carried add_offsets=[%one, %zero] {clamp_bounds} : !tt.tensordesc<64x64xf16, #shared>
      scf.yield %next : !tt.tensordesc<64x64xf16, #shared>
    }
    tt.return
  }
}
