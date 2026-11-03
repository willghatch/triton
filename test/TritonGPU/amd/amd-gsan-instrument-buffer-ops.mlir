// RUN: triton-opt %s -split-input-file -tritonamdgpu-gsan-instrument-buffer-ops -verify-diagnostics | FileCheck %s

// GSan checks AMD buffer ops on the tensor of global addresses they access.
// Element i of a buffer op accesses the bytes at
//   base + zext(elemBytes * soffset) + zext(elemBytes * offsets[i])
// (each product wraps in i32), and only if the user mask is set and the
// access is within the descriptor bound:
//   zext(soffset bytes) + zext(offset bytes) + size <= zext(num_records),
// where a pointer base has the default bound 2^31 - 2.
// Out-of-bounds lanes are suppressed by the hardware, so they must not be
// checked as accesses, and a check on the scalar base alone would miss most of
// the footprint.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A load from a pointer base with a scalar soffset and a mask: one read
  // check, before the load, on base + soffset + offsets in bytes, under the
  // user mask and the default bound.
  // CHECK-LABEL: @load_ptr_base
  // CHECK-SAME: (%[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<f32>, %[[OFF:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>, %[[SOFF:[a-zA-Z0-9_]+]]: i32, %[[MASK:[a-zA-Z0-9_]+]]: tensor<256xi1, #blocked>)
  // CHECK-DAG: %[[FOUR:.+]] = arith.constant dense<4> : tensor<256xi32, #blocked>
  // CHECK-DAG: %[[VOFF:.+]] = arith.muli %[[OFF]], %[[FOUR]] : tensor<256xi32, #blocked>
  // CHECK-DAG: %[[VOFF64:.+]] = arith.extui %[[VOFF]] : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
  // CHECK-DAG: %[[SOFFB:.+]] = arith.muli %[[SOFF]], %{{.+}} : i32
  // CHECK-DAG: %[[SOFF64:.+]] = arith.extui %[[SOFFB]] : i32 to i64
  // CHECK-DAG: %[[SOFFT:.+]] = tt.splat %[[SOFF64]] : i64 -> tensor<256xi64, #blocked>
  // CHECK-DAG: %[[BYTES:.+]] = arith.addi %[[SOFFT]], %[[VOFF64]] : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant dense<4> : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[END:.+]] = arith.addi %[[BYTES]], %[[SIZE]] : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[BOUND:.+]] = arith.constant dense<2147483646> : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[INB:.+]] = arith.cmpi ule, %[[END]], %[[BOUND]] : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[EFF:.+]] = arith.andi %[[MASK]], %[[INB]] : tensor<256xi1, #blocked>
  // CHECK-DAG: %[[BASE8:.+]] = tt.bitcast %[[BASE]] : !tt.ptr<f32> -> !tt.ptr<i8>
  // CHECK-DAG: %[[SPLAT:.+]] = tt.splat %[[BASE8]] : !tt.ptr<i8> -> tensor<256x!tt.ptr<i8>, #blocked>
  // CHECK-DAG: %[[ADDR8:.+]] = tt.addptr %[[SPLAT]], %[[BYTES]] : tensor<256x!tt.ptr<i8>, #blocked>, tensor<256xi64, #blocked>
  // CHECK-DAG: %[[ADDR:.+]] = tt.bitcast %[[ADDR8]] : tensor<256x!tt.ptr<i8>, #blocked> -> tensor<256x!tt.ptr<f32>, #blocked>
  // CHECK: tti.experimental_gsan_tensor_access %[[ADDR]], false, %[[EFF]]
  // CHECK-NEXT: amdg.buffer_load %[[BASE]][%[[OFF]]], %[[MASK]] soffset = %[[SOFF]]
  // CHECK-NOT: tti.experimental_gsan
  tt.func @load_ptr_base(%base: !tt.ptr<f32>, %offsets: tensor<256xi32, #blocked>, %soffset: i32, %mask: tensor<256xi1, #blocked>) -> tensor<256xf32, #blocked> {
    %0 = amdg.buffer_load %base[%offsets], %mask soffset = %soffset : i32 : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return %0 : tensor<256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A store through a descriptor with a tensor soffset and no mask: one write
  // check on the descriptor base, bounded by the descriptor's num_records
  // read as unsigned.
  // CHECK-LABEL: @store_desc_base
  // CHECK-SAME: (%[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<f16>, %[[NUM:[a-zA-Z0-9_]+]]: i32, %[[OFF:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>, %[[SOFF:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>
  // CHECK-DAG: %[[TWO:.+]] = arith.constant dense<2> : tensor<256xi32, #blocked>
  // CHECK-DAG: %[[VOFF:.+]] = arith.muli %[[OFF]], %[[TWO]]
  // CHECK-DAG: %[[VOFF64:.+]] = arith.extui %[[VOFF]]
  // CHECK-DAG: %[[SOFFB:.+]] = arith.muli %[[SOFF]], %[[TWO]]
  // CHECK-DAG: %[[SOFF64:.+]] = arith.extui %[[SOFFB]]
  // CHECK-DAG: %[[BYTES:.+]] = arith.addi %[[SOFF64]], %[[VOFF64]]
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant dense<2> : tensor<256xi64, #blocked>
  // CHECK-DAG: %[[END:.+]] = arith.addi %[[BYTES]], %[[SIZE]]
  // CHECK-DAG: %[[NUM64:.+]] = arith.extui %[[NUM]] : i32 to i64
  // CHECK-DAG: %[[BOUND:.+]] = tt.splat %[[NUM64]] : i64 -> tensor<256xi64, #blocked>
  // CHECK-DAG: %[[INB:.+]] = arith.cmpi ule, %[[END]], %[[BOUND]]
  // CHECK-DAG: %[[BASE8:.+]] = tt.bitcast %[[BASE]] : !tt.ptr<f16> -> !tt.ptr<i8>
  // CHECK-DAG: %[[SPLAT:.+]] = tt.splat %[[BASE8]]
  // CHECK-DAG: %[[ADDR8:.+]] = tt.addptr %[[SPLAT]], %[[BYTES]]
  // CHECK-DAG: %[[ADDR:.+]] = tt.bitcast %[[ADDR8]] : tensor<256x!tt.ptr<i8>, #blocked> -> tensor<256x!tt.ptr<f16>, #blocked>
  // CHECK: tti.experimental_gsan_tensor_access %[[ADDR]], true, %[[INB]]
  // CHECK-NEXT: amdg.buffer_store
  tt.func @store_desc_base(%base: !tt.ptr<f16>, %num_records: i32, %offsets: tensor<256xi32, #blocked>, %soffset: tensor<256xi32, #blocked>, %value: tensor<256xf16, #blocked>) {
    %desc = amdg.make_buffer_desc %base, %num_records : !tt.ptr<f16> -> !amdg.buffer_desc<f16>
    amdg.buffer_store %value, %desc[%offsets] soffset = %soffset : tensor<256xi32, #blocked> : !amdg.buffer_desc<f16> -> tensor<256xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // The hardware checks accesses wider than 4 bytes per 4-byte piece, so an
  // 8-byte element whose first piece is in bounds is (partly) accessed. GSan
  // counts it as accessed when its first 4 bytes are in bounds.
  // CHECK-LABEL: @load_f64_partial_bound
  // CHECK-DAG: %[[EIGHT:.+]] = arith.constant dense<8> : tensor<64xi32, #blocked>
  // CHECK-DAG: arith.muli %{{.+}}, %[[EIGHT]]
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant dense<4> : tensor<64xi64, #blocked>
  // CHECK-DAG: %[[END:.+]] = arith.addi %{{.+}}, %[[SIZE]] : tensor<64xi64, #blocked>
  // CHECK-DAG: %[[INB:.+]] = arith.cmpi ule, %[[END]],
  // CHECK: tti.experimental_gsan_tensor_access %{{.+}}, false, %[[INB]] : tensor<64x!tt.ptr<f64>, #blocked>
  tt.func @load_f64_partial_bound(%base: !tt.ptr<f64>, %num_records: i32, %offsets: tensor<64xi32, #blocked>) -> tensor<64xf64, #blocked> {
    %desc = amdg.make_buffer_desc %base, %num_records : !tt.ptr<f64> -> !amdg.buffer_desc<f64>
    %0 = amdg.buffer_load %desc[%offsets] : !amdg.buffer_desc<f64> -> tensor<64xf64, #blocked>
    tt.return %0 : tensor<64xf64, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Buffer atomics become GSan atomics, which perform the user atomic
  // themselves, on the reconstructed addresses. Kind, semantics, scope,
  // value, and result uses are preserved; the mask also drops out-of-bounds
  // lanes. Attributes such as the operand segment sizes are not copied.
  // CHECK-LABEL: @atomic_rmw
  // CHECK-SAME: (%[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<f32>, %[[OFF:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>, %[[VAL:[a-zA-Z0-9_]+]]: tensor<256xf32, #blocked>, %[[MASK:[a-zA-Z0-9_]+]]: tensor<256xi1, #blocked>)
  // CHECK: %[[INB:.+]] = arith.cmpi ule
  // CHECK: %[[EFF:.+]] = arith.andi %[[MASK]], %[[INB]]
  // CHECK: %[[ADDR:.+]] = tt.bitcast %{{.+}} : tensor<256x!tt.ptr<i8>, #blocked> -> tensor<256x!tt.ptr<f32>, #blocked>
  // CHECK: %[[OLD:.+]] = tti.experimental_gsan_atomic_rmw fadd, acq_rel, gpu, %[[ADDR]], %[[VAL]], %[[EFF]] {foo = 1 : i32} :
  // CHECK-NOT: amdg.buffer_atomic_rmw
  // CHECK: tt.return %[[OLD]]
  tt.func @atomic_rmw(%base: !tt.ptr<f32>, %offsets: tensor<256xi32, #blocked>, %value: tensor<256xf32, #blocked>, %mask: tensor<256xi1, #blocked>) -> tensor<256xf32, #blocked> {
    %0 = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %value, %base[%offsets], %mask {foo = 1 : i32} : !tt.ptr<f32> -> tensor<256xf32, #blocked>
    tt.return %0 : tensor<256xf32, #blocked>
  }

  // Buffer CAS has no user mask, but out-of-bounds lanes still must not
  // access memory, so the GSan CAS gets the in-bounds mask.
  // CHECK-LABEL: @atomic_cas
  // CHECK-SAME: (%[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>, %[[OFF:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>, %[[CMP:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>, %[[VAL:[a-zA-Z0-9_]+]]: tensor<256xi32, #blocked>)
  // CHECK: %[[INB:.+]] = arith.cmpi ule
  // CHECK: %[[ADDR:.+]] = tt.bitcast %{{.+}} : tensor<256x!tt.ptr<i8>, #blocked> -> tensor<256x!tt.ptr<i32>, #blocked>
  // CHECK: %[[OLD:.+]] = tti.experimental_gsan_atomic_cas release, cta, %[[ADDR]], %[[CMP]], %[[VAL]], %[[INB]] :
  // CHECK-NOT: amdg.buffer_atomic_cas
  // CHECK: tt.return %[[OLD]]
  tt.func @atomic_cas(%base: !tt.ptr<i32>, %offsets: tensor<256xi32, #blocked>, %cmp: tensor<256xi32, #blocked>, %value: tensor<256xi32, #blocked>) -> tensor<256xi32, #blocked> {
    %0 = amdg.buffer_atomic_cas release, cta, %cmp, %value, %base[%offsets] : !tt.ptr<i32> -> tensor<256xi32, #blocked>
    tt.return %0 : tensor<256xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Buffer atomic loads and stores become GSan atomic loads and stores.
  // Masked-off and out-of-bounds elements of a buffer atomic load read zero.
  // CHECK-LABEL: @atomic_load_store
  // CHECK-SAME: (%[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>, %[[NUM:[a-zA-Z0-9_]+]]: i32, %[[OFF:[a-zA-Z0-9_]+]]: tensor<64xi32, #blocked>, %[[SOFF:[a-zA-Z0-9_]+]]: i32, %[[MASK:[a-zA-Z0-9_]+]]: tensor<64xi1, #blocked>)
  // CHECK: %[[LEFF:.+]] = arith.andi %[[MASK]], %{{.+}}
  // CHECK: %[[LADDR:.+]] = tt.bitcast %{{.+}} : tensor<64x!tt.ptr<i8>, #blocked> -> tensor<64x!tt.ptr<i32>, #blocked>
  // CHECK: %[[LOADED:.+]] = tti.experimental_gsan_atomic_load acquire, gpu, %[[LADDR]], %[[LEFF]] :
  // CHECK: %[[ZERO:.+]] = arith.constant dense<0> : tensor<64xi32, #blocked>
  // CHECK: %[[RESULT:.+]] = arith.select %[[LEFF]], %[[LOADED]], %[[ZERO]]
  // CHECK: %[[SINB:.+]] = arith.cmpi ule
  // CHECK: %[[SADDR:.+]] = tt.bitcast %{{.+}} : tensor<64x!tt.ptr<i8>, #blocked> -> tensor<64x!tt.ptr<i32>, #blocked>
  // CHECK: tti.experimental_gsan_atomic_store release, gpu, %[[SADDR]], %[[RESULT]], %[[SINB]] :
  // CHECK-NOT: amdg.buffer_atomic
  tt.func @atomic_load_store(%base: !tt.ptr<i32>, %num_records: i32, %offsets: tensor<64xi32, #blocked>, %soffset: i32, %mask: tensor<64xi1, #blocked>) {
    %desc = amdg.make_buffer_desc %base, %num_records : !tt.ptr<i32> -> !amdg.buffer_desc<i32>
    %0 = amdg.buffer_atomic_load acquire, gpu, %desc[%offsets], %mask soffset = %soffset : i32 : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    amdg.buffer_atomic_store release, gpu, %0, %desc[%offsets] soffset = %soffset : i32 : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A descriptor whose amdg.make_buffer_desc is not visible (here a function
  // argument) has an unknown base and bound, so it is rejected.
  tt.func @desc_argument(%desc: !amdg.buffer_desc<i32>, %offsets: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked> {
    // expected-error@+1 {{GSan cannot find the amdg.make_buffer_desc of the buffer descriptor}}
    %0 = amdg.buffer_load %desc[%offsets] : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
    tt.return %0 : tensor<64xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // A loop-carried descriptor is rejected too.
  tt.func @desc_loop_carried(%base: !tt.ptr<i32>, %num_records: i32, %offsets: tensor<64xi32, #blocked>, %value: tensor<64xi32, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %desc = amdg.make_buffer_desc %base, %num_records : !tt.ptr<i32> -> !amdg.buffer_desc<i32>
    %0 = scf.for %i = %c0 to %n step %c1 iter_args(%d = %desc) -> (!amdg.buffer_desc<i32>) : i32 {
      // expected-error@+1 {{GSan cannot find the amdg.make_buffer_desc of the buffer descriptor}}
      amdg.buffer_store %value, %d[%offsets] : !amdg.buffer_desc<i32> -> tensor<64xi32, #blocked>
      scf.yield %d : !amdg.buffer_desc<i32>
    }
    tt.return
  }
}
