// RUN: triton-opt %s -split-input-file -tritonamdgpu-gsan-reject-unsupported -tritonamdgpu-gsan-instrument-direct-async -verify-diagnostics | FileCheck %s

// Operations GSan instruments pass through unchanged.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @supported
  tt.func @supported(%ptr: !tt.ptr<f32>, %ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %iptr: !tt.ptr<i32>) {
    %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %v : tensor<256x!tt.ptr<f32>, #blocked>
    %c1 = arith.constant 1.0 : f32
    %true = arith.constant true
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %c1, %true : (!tt.ptr<f32>, f32, i1) -> f32
    %c0_i32 = arith.constant 0 : i32
    %1 = tt.atomic_cas acquire, gpu, %iptr, %c0_i32, %c0_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.print ": " {hex = false, isSigned = array<i32: 0>} : %1 : i32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load(%arg0: !tt.ptr<f32>, %offset: tensor<256xi32, #blocked>) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.buffer_load'}}
    %ret = amdg.buffer_load %arg0[%offset] : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_store(%arg0: !tt.ptr<f32>, %offset: tensor<256xi32, #blocked>, %value: tensor<256xf32, #blocked>) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.buffer_store'}}
    amdg.buffer_store %value, %arg0[%offset] : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @async_copy(%ptrs: tensor<32x64x!tt.ptr<f16>, #blocked>, %dst: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    %0 = ttg.async_copy_global_to_local %ptrs, %dst : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @descriptor_load(%desc: !tt.tensordesc<32x64xf16>, %i: i32) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'tt.descriptor_load'}}
    %0 = tt.descriptor_load %desc[%i, %i] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 8 : i32} {
  tt.func @warp_specialize() {
    // expected-error@+1 {{GSan on AMD does not support warp specialization}}
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mbarrier(%bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) {
    // expected-error@+1 {{GSan on AMD does not support synchronization operation 'amdg.init_barrier'}}
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

// expected-error@+1 {{GSan on AMD does not support num_ctas > 1 (got 2)}}
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @multi_cta() {
    tt.return
  }
}

// -----

// Atomic load/store/poll and assertions are instrumented (or harmless) too.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @supported_atomic_load_store_poll
  tt.func @supported_atomic_load_store_poll(%ptr: !tt.ptr<i32>, %cmp: i1) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %v = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
    tt.atomic_store release, gpu, %ptr, %v, %false : !tt.ptr<i32>
    %timeout = arith.constant 100 : i64
    %matched = tt.atomic_poll acquire, gpu, %ptr, %c0 timeout %timeout : !tt.ptr<i32>, i32 -> i1
    tt.assert %cmp, "cond must be true " : i1
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local(%ptr: !tt.ptr<f32>, %offsets: tensor<32x64xi32, #blocked>, %buffer: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %buffer : !tt.ptr<f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_atomic_rmw(%ptr: !tt.ptr<f32>, %offsets: tensor<256xi32, #blocked>, %value: tensor<256xf32, #blocked>) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.buffer_atomic_rmw'}}
    %0 = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %value, %ptr[%offsets] : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @async_copy_local_to_global(%buffer: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, %dst: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %token = amdg.async_copy_local_to_global %buffer, %dst : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @tdm_copy(%desc: !tt.tensordesc<128x128xf16, #shared>, %dst: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.async_tdm_copy_global_to_local'}}
    %token = amdg.async_tdm_copy_global_to_local %desc into %dst : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mbarrier_wait_arrive(%bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %phase: i32) {
    // expected-error@+1 {{GSan on AMD does not support synchronization operation 'amdg.wait_barrier'}}
    amdg.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-error@+1 {{GSan on AMD does not support synchronization operation 'amdg.arrive_barrier'}}
    %prior = amdg.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cond_barrier(%pred: i1) {
    // expected-error@+1 {{GSan on AMD does not support synchronization operation 'amdg.cond_barrier'}}
    amdg.cond_barrier %pred
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_pipeline_stage(%ptr: !tt.ptr<f32>) {
    %v = arith.constant 0.0 : f32
    // expected-error@+1 {{GSan on AMD does not support warp pipelining}}
    scf.execute_region {
      tt.store %ptr, %v : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
// The count of unchecked ops is recorded for the compiler driver to report.
// CHECK: module attributes {{.*}}tti.gsan_unchecked_ops = 1 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @impure_inline_asm
  tt.func @impure_inline_asm(%x: tensor<256xi32, #blocked>) {
    // expected-warning@+1 {{GSan cannot see memory accesses of 'tt.elementwise_inline_asm'}}
    %0 = tt.elementwise_inline_asm "s_sleep 1" {constraints = "=v,v", packed_element = 1 : i32, pure = false} %x : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

// expected-error@+1 {{GSan on AMD requires 64 threads per warp (got 32)}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wave32() {
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @masked_load_store(%ptr: !llvm.ptr<1>, %mask: i1, %other: f32) {
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.masked_load'}}
    %v = amdg.masked_load %ptr, %mask, %other : (!llvm.ptr<1>, i1, f32) -> f32
    // expected-error@+1 {{GSan on AMD cannot check global-memory operation 'amdg.masked_store'}}
    amdg.masked_store %ptr, %v, %mask : !llvm.ptr<1>, f32, i1
    tt.return
  }
}


// -----

// A copy through an opaque descriptor must fail explicitly rather than
// inventing a base address or silently escaping issue-time race checking.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @opaque_async_buffer(%desc: !amdg.buffer_desc<f32>, %offsets: tensor<64xi32, #blocked>, %dst: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    // expected-error@+1 {{GSan direct async copies require a buffer descriptor defined by amdg.make_buffer_desc}}
    %token = amdg.buffer_load_to_local %desc[%offsets] into %dst : !amdg.buffer_desc<f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Linear layouts can encode valid subword copies, but the direct GSan pass
// currently recovers issued dword groups only for blocked layouts.
#linear = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @linear_subword_buffer_async(%base: !tt.ptr<f16>, %dst: !ttg.memdesc<128xf16, #shared, #smem, mutable>) {
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #linear>
    // expected-error@+1 {{GSan sub-dword buffer async copies require a blocked layout with provably contiguous dword groups}}
    %token = amdg.buffer_load_to_local %base[%offsets] into %dst : !tt.ptr<f16>[tensor<128xi32, #linear>] -> <128xf16, #shared, #smem, mutable>
    tt.return
  }
}
