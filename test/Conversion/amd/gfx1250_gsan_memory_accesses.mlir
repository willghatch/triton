// RUN: triton-opt %s -tritoninstrument-global-sanitizer --convert-scf-to-cf --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 -cse | FileCheck %s

#bytes = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1250"} {
  // Four consecutive bytes with independent masks must remain four one-byte
  // accesses. Rounding their masks up to a shadow word touches inactive bytes.
  // CHECK-LABEL: llvm.func @independently_masked_bytes
  // CHECK-DAG: %[[FOUR:[a-zA-Z0-9_.]+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[ONE:[a-zA-Z0-9_.]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, %{{.*}}, %[[FOUR]], %[[ONE]],
  // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %[[FOUR]], %[[ONE]],
  tt.func public @independently_masked_bytes(%base: !tt.ptr<i8> {tt.divisibility = 16 : i32},
                                             %mask: tensor<128xi1, #bytes>) {
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #bytes>
    %bases = tt.splat %base : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>, #bytes>
    %ptrs = tt.addptr %bases, %offsets : tensor<128x!tt.ptr<i8>, #bytes>, tensor<128xi32, #bytes>
    %values = arith.constant dense<1> : tensor<128xi8, #bytes>
    tt.store %ptrs, %values, %mask : tensor<128x!tt.ptr<i8>, #bytes>
    %loaded = tt.load %ptrs, %mask : tensor<128x!tt.ptr<i8>, #bytes>
    tt.return
  }

  // Register replication represents a single logical access.
  // CHECK-LABEL: llvm.func @gsan_canonical_register_accesses
  // CHECK-DAG: %[[CANONICAL_COUNT:[a-zA-Z0-9_.]+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[CANONICAL_WIDTH:[a-zA-Z0-9_.]+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__triton_gsan_load_tensor({{%[^,]+}}, {{%[^,]+}}, %[[CANONICAL_COUNT]], %[[CANONICAL_WIDTH]],
  // CHECK: llvm.call @__triton_gsan_store_tensor({{%[^,]+}}, {{%[^,]+}}, %[[CANONICAL_COUNT]], %[[CANONICAL_WIDTH]],
  tt.func public @gsan_canonical_register_accesses(
      %ptrs: tensor<1x!tt.ptr<i32>, #bytes>,
      %vals: tensor<1xi32, #bytes>,
      %mask: tensor<1xi1, #bytes>) {
    %loaded = tt.load %ptrs, %mask : tensor<1x!tt.ptr<i32>, #bytes>
    tt.store %ptrs, %vals, %mask : tensor<1x!tt.ptr<i32>, #bytes>
    tt.return
  }

  // Distinct register addresses each need their own shadow access.
  // CHECK-LABEL: llvm.func @gsan_distinct_register_accesses
  // CHECK-DAG: %[[DISTINCT_COUNT:[a-zA-Z0-9_.]+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__triton_gsan_load_tensor({{%[^,]+}}, {{%[^,]+}}, %[[DISTINCT_COUNT]], %[[DISTINCT_COUNT]],
  // CHECK: llvm.call @__triton_gsan_store_tensor({{%[^,]+}}, {{%[^,]+}}, %[[DISTINCT_COUNT]], %[[DISTINCT_COUNT]],
  tt.func public @gsan_distinct_register_accesses(
      %ptrs: tensor<128x!tt.ptr<i32>, #bytes>,
      %vals: tensor<128xi32, #bytes>,
      %mask: tensor<128xi1, #bytes>) {
    %loaded = tt.load %ptrs, %mask : tensor<128x!tt.ptr<i32>, #bytes>
    tt.store %ptrs, %vals, %mask : tensor<128x!tt.ptr<i32>, #bytes>
    tt.return
  }
}
