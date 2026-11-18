// RUN: triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --convert-scf-to-cf --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --convert-scf-to-cf --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=NO-NV

// No NVIDIA-only constructs anywhere in the output.
// NO-NV-NOT: nvg.
// NO-NV-NOT: llvm.inline_asm

// GSan instrumentation lowered for AMD (gfx950, wave64).
// The shared GSan lowering was NVIDIA-only; on AMD it must:
// - not use NVIDIA-only ops (named barrier IDs, PTX inline-asm atomics),
// - put runtime argument buffers in private memory (addrspace 5), since
//   AMDGPU rejects allocas in other address spaces,
// - lane-serialize ("waterfall") each atomic_begin -> user atomic ->
//   atomic_end sequence, because the runtime holds per-lane spin locks across
//   the user atomic and lanes of a wave cannot make independent progress.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Init and exit must not use the NVIDIA named-barrier ID op, and the
  // tensor-check argument buffer is a private-memory alloca in the entry block.
  // CHECK-LABEL: llvm.func @load_store
  // CHECK: [[BUF:%[a-zA-Z0-9_.]+]] = llvm.alloca %{{.*}} x !llvm.struct<(array<1 x i64>, array<1 x i8>)> : (i32) -> !llvm.ptr<5>
  // CHECK: llvm.call @__triton_gsan_init(
  // CHECK: llvm.call @__triton_gsan_load_tensor(
  // CHECK: [[GENERIC:%[a-zA-Z0-9_.]+]] = llvm.addrspacecast [[BUF]] : !llvm.ptr<5> to !llvm.ptr
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, [[GENERIC]],
  // CHECK: llvm.call @__triton_gsan_kernel_exit(
  tt.func public @load_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>,
                      %other: tensor<256xf32, #blocked>, %vals: tensor<256xf32, #blocked>) {
    %loaded = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // The user RMW is a native LLVM atomic between begin and end. All three run
  // only in the lane chosen from a ballot of pending lanes; that lane then
  // clears its pending bit on the loop back edge and carries the old value.
  // Only lanes whose mask is set start pending, and the loop exit value is
  // the RMW result.
  // CHECK-LABEL: llvm.func @atomic_rmw
  // CHECK: [[MASK:%[a-zA-Z0-9_.]+]] = llvm.extractvalue %arg2[0]
  // CHECK: llvm.br ^[[HEADER:[a-zA-Z0-9_.]+]]([[MASK]],
  // CHECK: ^[[HEADER]](%[[PENDING:[a-zA-Z0-9_.]+]]: i1
  // CHECK: [[BALLOT:%[a-zA-Z0-9_.]+]] = rocdl.ballot %[[PENDING]] : i64
  // CHECK: llvm.cond_br %{{.*}}, ^{{[a-zA-Z0-9_.]+}}, ^[[EXIT:[a-zA-Z0-9_.]+]](
  // CHECK: "llvm.intr.cttz"([[BALLOT]])
  // CHECK: llvm.cond_br %{{.*}}, ^[[CRIT:[a-zA-Z0-9_.]+]], ^[[HEADER]](
  // CHECK: ^[[CRIT]]:
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: [[OLD:%[a-zA-Z0-9_.]+]] = llvm.atomicrmw add %{{.*}}, %{{.*}} syncscope("agent") release
  // CHECK: ^{{[a-zA-Z0-9_.]+}}([[MERGED:%[a-zA-Z0-9_.]+]]: i32):
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  // CHECK: [[DONE:%[a-zA-Z0-9_.]+]] = llvm.mlir.constant(false) : i1
  // CHECK: llvm.br ^[[HEADER]]([[DONE]], [[MERGED]] : i1, i32)
  // CHECK: ^[[EXIT]]([[RESULT:%[a-zA-Z0-9_.]+]]: i32):
  // CHECK: llvm.insertvalue [[RESULT]]
  tt.func public @atomic_rmw(%ptrs: tensor<256x!tt.ptr<i32>, #blocked>, %vals: tensor<256xi32, #blocked>, %mask: tensor<256xi1, #blocked>) {
    %0 = tt.atomic_rmw add, release, gpu, %ptrs, %vals, %mask : (tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Float CAS is performed on the integer bits (LLVM cmpxchg needs an integer
  // type). A failed CAS still reads, so acquire is kept on failure.
  // CHECK-LABEL: llvm.func @atomic_cas
  // CHECK: rocdl.ballot
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: [[PAIR:%[a-zA-Z0-9_.]+]] = llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} syncscope("workgroup") acq_rel acquire
  // CHECK: llvm.extractvalue [[PAIR]][0]
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  tt.func public @atomic_cas(%ptr: !tt.ptr<f32>, %cmp: f32, %val: f32) {
    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<f32>, f32, f32) -> f32
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A masked GSan CAS (from an AMD buffer CAS, whose out-of-bounds lanes the
  // hardware suppresses) runs the check and the CAS only in lanes whose mask
  // is set: only they start pending.
  // CHECK-LABEL: llvm.func @masked_atomic_cas
  // CHECK: [[MASK:%[a-zA-Z0-9_.]+]] = llvm.extractvalue %arg3[0]
  // CHECK: llvm.br ^[[HEADER:[a-zA-Z0-9_.]+]]([[MASK]],
  // CHECK: ^[[HEADER]](%[[PENDING:[a-zA-Z0-9_.]+]]: i1
  // CHECK: rocdl.ballot %[[PENDING]] : i64
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: llvm.cmpxchg
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  tt.func public @masked_atomic_cas(%ptrs: tensor<256x!tt.ptr<i32>, #blocked>, %cmp: tensor<256xi32, #blocked>, %vals: tensor<256xi32, #blocked>, %mask: tensor<256xi1, #blocked>) {
    %0 = tti.experimental_gsan_atomic_cas acq_rel, gpu, %ptrs, %cmp, %vals, %mask : (tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>, tensor<256xi32, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Atomic load, store, and poll checks also hold runtime locks between begin
  // and end, so each is lane-serialized with the user access inside.
  // CHECK-LABEL: llvm.func @atomic_load_store_poll
  // CHECK: rocdl.ballot
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: llvm.load %{{.*}} atomic syncscope("agent") monotonic
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  // CHECK: rocdl.ballot
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: llvm.store %{{.*}}, %{{.*}} atomic syncscope("agent") monotonic
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  // CHECK: rocdl.ballot
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar(
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar(
  tt.func public @atomic_load_store_poll(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %flag: !tt.ptr<i32>, %mask: i1) {
    %loaded = tt.atomic_load acquire, gpu, %ptr, %mask : (!tt.ptr<i32>, i1) -> i32
    tt.atomic_store release, gpu, %out, %loaded, %mask : !tt.ptr<i32>
    %c1 = arith.constant 1 : i32
    %matched = tt.atomic_poll acquire, gpu, %flag, %c1 : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A check inside a loop still allocates its buffer in the entry block;
  // a dynamic alloca would re-grow the private stack every iteration.
  // CHECK-LABEL: llvm.func @load_in_loop
  // CHECK-NOT: ^bb
  // CHECK: llvm.alloca %{{.*}} x !llvm.struct<(array<1 x i64>, array<1 x i8>)> : (i32) -> !llvm.ptr<5>
  // CHECK: ^bb
  // CHECK: llvm.call @__triton_gsan_load_tensor(
  tt.func public @load_in_loop(%ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %loaded = tt.load %ptrs : tensor<256x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}

// -----

// Dynamic native AMD descriptors, including host arguments, must be passed to
// rectangular checks by value through private memory. Treating them as NVIDIA
// descriptor pointers would lose the actual address, strides, bounds, and pred.
#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @tdm_descriptor_checks(
  // CHECK-SAME: [[DESC:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  // CHECK: [[DESC_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> : (i32) -> !llvm.ptr<5>
  // CHECK: [[GENERIC:%[a-zA-Z0-9_]+]] = llvm.addrspacecast [[DESC_BUFFER]] : !llvm.ptr<5> to !llvm.ptr
  // CHECK: llvm.store [[DESC]], [[GENERIC]]
  // CHECK: llvm.call @__triton_gsan_load_tensor_desc({{%[^,]+}}, [[GENERIC]],
  tt.func public @tdm_descriptor_checks(%desc: !tt.tensordesc<128x128xf16, #shared>) {
    %zero = arith.constant 0 : i32
    tti.experimental_gsan_tensordesc_access %desc[%zero, %zero], false : !tt.tensordesc<128x128xf16, #shared>
    tt.return
  }
  // CHECK-LABEL: llvm.func @tdm_descriptor_store(
  // CHECK-SAME: [[STORE_DESC:%[a-zA-Z0-9_]+]]: !llvm.struct
  // CHECK: [[STORE_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> : (i32) -> !llvm.ptr<5>
  // CHECK: [[STORE_GENERIC:%[a-zA-Z0-9_]+]] = llvm.addrspacecast [[STORE_BUFFER]] : !llvm.ptr<5> to !llvm.ptr
  // CHECK: llvm.store [[STORE_DESC]], [[STORE_GENERIC]]
  // CHECK: llvm.call @__triton_gsan_store_tensor_desc({{%[^,]+}}, [[STORE_GENERIC]],
  tt.func public @tdm_descriptor_store(%desc: !tt.tensordesc<128x128xf16, #shared>) {
    %zero = arith.constant 0 : i32
    tti.experimental_gsan_tensordesc_access %desc[%zero, %zero], true : !tt.tensordesc<128x128xf16, #shared>
    tt.return
  }
}

// -----

// A legal 32768-element TDM tile exceeds signed int16_t but fits its hardware
// unsigned 16-bit shape field. The runtime must preserve this extent.
#shared = #ttg.padded_shared<[128:+4] {order = [0], shape = [32768]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @tdm_unsigned_block_extent(
  // CHECK: [[EXTENT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(-32768 : i16)
  // CHECK: llvm.store [[EXTENT]],
  // CHECK: llvm.call @__triton_gsan_load_tensor_desc(
  tt.func public @tdm_unsigned_block_extent(%desc: !tt.tensordesc<32768xi8, #shared>) {
    %zero = arith.constant 0 : i32
    tti.experimental_gsan_tensordesc_access %desc[%zero], false : !tt.tensordesc<32768xi8, #shared>
    tt.return
  }
}
