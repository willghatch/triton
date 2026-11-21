from __future__ import annotations

import functools
import inspect
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)

from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna4, is_hip_cdna5, is_hopper_or_newer, run_in_process
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._testing_utils import atomic_poll, shadow_cell_from_address
from triton.tools.tensor_descriptor import TensorDescriptor

THREADS_PER_WARP = gl.constexpr(64 if is_hip_cdna4() else 32)
amd_global_load_to_shared = (gl.amd.cdna5.async_copy.global_to_shared
                             if is_hip_cdna5() else gl.amd.cdna4.async_copy.global_load_to_shared)
# Cross-block atomic_load spin loops are validated on gfx1250. Keep gfx950
# excluded until its previously observed GSan polling hang is revalidated.
_GFX950_ATOMIC_LOAD_SPIN_HANG = "GSan atomic_load spin loops previously hung on gfx950; revalidation required"

pytestmark = pytest.mark.skipif(not (is_cuda() or is_hip_cdna4() or is_hip_cdna5()),
                                reason="requires CUDA or AMD gfx950/gfx1250 backend")

# On AMD, a GSan race report traps on the device, which aborts the whole
# process and by default writes a large, slow GPU core dump into the cwd.
# The runtime reads this knob when HSA initializes, which happens while the
# subprocess imports this module, so it must already be in the environment the
# subprocess (forkserver) inherits. This changes the environment of the whole
# pytest session, which only affects HSA exception handling.
if is_hip():
    os.environ.setdefault("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1")
HIP_DEFAULT_PROCESS_TIMEOUT_S = "300"

RELEASE_ACQUIRE_SYNC_CASES = (
    pytest.param("release", "acquire", id="release-acquire"),
    pytest.param("acq_rel", "acq_rel", id="acq-rel-acq-rel"),
)

# Missing release and missing acquire are distinct failures; exercise each once
# per operation, with both scopes represented rather than crossed with each case.
CROSS_SM_SEMANTIC_MISMATCH_CASES = (
    pytest.param("relaxed", "acquire", "gpu", id="producer-relaxed-consumer-acquire-scope-gpu"),
    pytest.param("release", "relaxed", "sys", id="producer-release-consumer-relaxed-scope-sys"),
)

TRANSITIVE_RELAY_MISMATCH_CASES = (
    pytest.param("release", "relaxed", "gpu", id="relay-relaxed-scope-gpu"),
    pytest.param("acq_rel", "release", "sys", id="relay-release-scope-sys"),
)


@gluon.jit
def _raw_kernel(ptr, scratch_ptr, counter_ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    pid = gl.program_id(0)
    if pid == 0:
        gl.store(ptr + offsets, 1)
        gl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        while gl.atomic_add(counter_ptr, 0, sem="relaxed") != 1:
            pass
        value = gl.load(ptr + offsets)
        gl.store(scratch_ptr + offsets, value)


@gluon.jit
def _war_kernel(ptr, scratch_ptr, counter_ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    pid = gl.program_id(0)
    if pid == 0:
        value = gl.load(ptr + offsets)
        gl.store(scratch_ptr + offsets, value)
        gl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        while gl.atomic_add(counter_ptr, 0, sem="relaxed") != 1:
            pass
        gl.store(ptr + offsets, 1)


@gluon.jit
def _waw_kernel(ptr, scratch_ptr, counter_ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    pid = gl.program_id(0)
    if pid == 0:
        gl.store(ptr + offsets, 1)
        gl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        while gl.atomic_add(counter_ptr, 0, sem="relaxed") != 1:
            pass
        gl.store(ptr + offsets, 2)


@triton.jit
def _store_after_all_blocks_arrive_kernel(ptr, value, arrived_ptr, BLOCK: tl.constexpr):
    # Keep every block resident until all have started, so the blocks of a
    # launch hold distinct GSan identities; relaxed atomics add no ordering.
    tl.atomic_add(arrived_ptr, 1, sem="relaxed")
    atomic_poll(arrived_ptr, tl.num_programs(0))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, value)


@triton.jit
def _release_loop_kernel(flag_ptr, num_iters):
    for _ in range(num_iters):
        tl.atomic_add(flag_ptr, 1, sem="release")


@triton.jit
def _relaxed_cas_handoff_kernel(payload_ptr, flag_ptr, arrived_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        # Wait until the consumer is resident, so it cannot take over this
        # block's GSan identity (slot) after it exits and miss the race.
        atomic_poll(arrived_ptr, 1)
        tl.store(payload_ptr, 42)
        tl.atomic_cas(flag_ptr, 0, 1, sem="release")
    else:
        tl.atomic_add(arrived_ptr, 1, sem="relaxed")
        # A relaxed CAS observes the flag but does not acquire the payload.
        seen = tl.atomic_cas(flag_ptr, 1, 1, sem="relaxed")
        while seen != 1:
            seen = tl.atomic_cas(flag_ptr, 1, 1, sem="relaxed")
        tl.store(out_ptr, tl.load(payload_ptr))


@triton.jit
def _pdl_producer_kernel(payload_ptr):
    pid = tl.program_id(0)
    tl.store(payload_ptr + pid, 1000 + pid)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_consumer_without_wait_kernel(payload_ptr, scratch_ptr):
    value = tl.load(payload_ptr + 1)
    tl.store(scratch_ptr, value)


@triton.jit
def _pdl_conditional_wait_kernel(should_wait_ptr, scratch_ptr):
    if tl.load(should_wait_ptr) != 0:
        tl.extra.cuda.gdc_wait()
    tl.store(scratch_ptr, 1)


@triton.jit
def _pdl_transitively_acquired_producer_kernel(payload_ptr, flag_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    else:
        atomic_poll(flag_ptr, 1, sem="acquire", scope="gpu")
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_transitively_acquired_consumer_without_wait_kernel(payload_ptr, scratch_ptr):
    value = tl.load(payload_ptr)
    tl.store(scratch_ptr + tl.program_id(0), value)


@triton.jit
def _cross_sm_atomic_sync_kernel(payload_ptr, flag_ptr, counter_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                 consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag_ptr, 0, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _atomic_poll_cross_sm_sync_kernel(payload_ptr, flag_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                      consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
    elif pid == 1:
        tl.atomic_poll(flag_ptr, 1, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _atomic_load_store_cross_sm_sync_kernel(payload_ptr, flag_ptr, counter_ptr, scratch_ptr, producer_sem: tl.constexpr,
                                            consumer_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_store(flag_ptr, 1, sem=producer_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = tl.atomic_load(flag_ptr, sem=consumer_sem, scope=scope)
        while ready != 1:
            ready = tl.atomic_load(flag_ptr, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _transitive_atomic_sync_kernel(payload_ptr, flag0_ptr, flag1_ptr, counter_ptr, scratch_ptr,
                                   release_sem: tl.constexpr, relay_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag0_ptr, 1, sem=release_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 1:
        atomic_poll(counter_ptr, 1)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag0_ptr, 0, sem=relay_sem, scope=scope)
        tl.atomic_xchg(flag1_ptr, 1, sem=release_sem, scope=scope)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    elif pid == 2:
        atomic_poll(counter_ptr, 2)
        ready = 0
        while ready != 1:
            ready = tl.atomic_add(flag1_ptr, 0, sem="acquire", scope=scope)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _tma_raw_kernel(ptr, scratch_ptr, counter_ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
        values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
        desc.store([row_idx, col_idx], values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        value = tl.load(ptr + row_idx * stride_0 + col_idx)
        tl.store(scratch_ptr, value)


@triton.jit
def _host_tma_war_kernel(target_ptr, target_desc, scratch_desc, counter_ptr, row_idx, col_idx, stride_0):
    pid = tl.program_id(0)
    if pid == 0:
        block = target_desc.load([row_idx, col_idx])
        scratch_desc.store([row_idx, col_idx], block)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)


@triton.jit
def _host_tma_gather_war_kernel(target_ptr, target_desc, x_offsets_ptr, scratch_ptr, counter_ptr, row_idx, y_offset,
                                stride_0, scratch_stride_0, scratch_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        values = target_desc.gather(x_offsets, y_offset)
        indices_x = tl.arange(0, BLOCK_X)[:, None] * scratch_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * scratch_stride_1
        tl.store(scratch_ptr + indices_x + indices_y, values)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)


@triton.jit
def _host_tma_scatter_war_kernel(target_ptr, target_desc, x_offsets_ptr, src_ptr, src_stride_0, src_stride_1,
                                 scratch_ptr, counter_ptr, row_idx, y_offset, stride_0, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = target_desc.block_shape[1]
    pid = tl.program_id(0)
    if pid == 0:
        value = tl.load(target_ptr + row_idx * stride_0 + y_offset)
        tl.store(scratch_ptr, value)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
        indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
        indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
        values = tl.load(src_ptr + indices_x + indices_y)
        target_desc.scatter(values, x_offsets, y_offset)


@triton.jit
def _host_tma_atomic_flag_publish_kernel(payload_ptr, flag_ptr, flag_desc, counter_ptr, scratch_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        BLOCK_X: tl.constexpr = flag_desc.block_shape[0]
        BLOCK_Y: tl.constexpr = flag_desc.block_shape[1]
        values = tl.full((BLOCK_X, BLOCK_Y), 1, dtype=tl.int32)
        # TMA atomics on the released flag are relaxed.gpu and must not acquire
        # the producer's prior payload store.
        flag_desc.atomic_add([0, 0], values)
        result = tl.load(payload_ptr)
        tl.store(scratch_ptr, result)


@triton.jit
def _mixed_scope_release_rmw_kernel(counter_ptr, ready_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        tl.atomic_xchg(ready_ptr, 1, sem="relaxed", scope="gpu")
    elif pid == 1:
        atomic_poll(ready_ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="release", scope="sys")


@gluon.jit
def _gluon_cluster_barrier_kernel(payload_ptr, out_ptr, RELAXED: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.barrier(cluster=True)
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    if RELAXED:
        hopper.cluster.barrier(relaxed=True)
    else:
        gl.barrier(cluster=True)
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)


@gluon.jit
def _gluon_mbarrier_current_epoch_kernel(payload_ptr, out_ptr, AFTER_ARRIVAL: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    offsets = gl.arange(0, 256, data_layout)

    payload_ptrs = payload_ptr + offsets * 0
    if not AFTER_ARRIVAL:
        gl.store(payload_ptrs, 1, mask=offsets == 128)
    hopper.mbarrier.arrive(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=0)
    if AFTER_ARRIVAL:
        gl.store(payload_ptrs, 1, mask=offsets == 128)
    hopper.cluster.barrier(relaxed=True)
    value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
    gl.store(out_ptr + offsets * 0, value, mask=offsets == 0)
    hopper.mbarrier.invalidate(barrier)


@gluon.jit
def _gluon_tcgen05_commit_no_release_kernel(payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    signal_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 128], gl.float16, cga_layout=((0, 0), ))
    signal_desc = gl.allocate_shared_memory(gl.float16, [128, 128], signal_layout)

    barrier = hopper.mbarrier.allocate_mbarrier()
    hopper.mbarrier.init(barrier, count=2)
    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    tcgen05_commit(barrier, descs=[signal_desc])
    hopper.mbarrier.wait(barrier, phase=0)
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)
    hopper.mbarrier.invalidate(barrier)


@gluon.jit
def _gluon_mbarrier_multicast_partial_wait_kernel(input_desc, payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1], [2]])
    signal = gl.allocate_shared_memory(gl.float16, [256, 128], input_desc.layout)
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 256], gl.float16, cga_layout=((0, 1), (0, 2)))
    b = gl.allocate_shared_memory(gl.float16, [128, 256], b_layout)
    acc_layout: gl.constexpr = TensorMemoryLayout([128, 128], col_stride=1, cga_layout=((1, 0), (0, 1)), two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [256, 256], acc_layout)

    # Four CTAs lower this to a two-element mbarrier with CGA layout [[0], [1]].
    # Each pair of CTAs arrives on a different barrier, and only its leader CTA
    # executes the lowered wait because the first CTA-layout basis is zero.
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    offsets = gl.arange(0, 512, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.expect(barrier, input_desc.nbytes_per_cta)
    hopper.tma.async_load(input_desc, [0, 0], barrier, signal, multicast=True)
    hopper.mbarrier.wait(barrier, phase=0, deps=[signal])
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)
    tcgen05_mma(signal, b, acc, use_acc=False)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(barrier)


def _cuda_byte_allocator(size: int, _align: int, _stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def run_with_gsan(fn):

    @functools.wraps(fn)
    def wrapped(*args, **kwargs) -> None:
        triton.knobs.compilation.instrumentation_mode = "gsan"
        pool = create_mem_pool(shadow_granularity=kwargs.pop("shadow_granularity", 4))
        with torch.cuda.use_mem_pool(pool):
            fn(*args, **kwargs)

    return wrapped


def _run_race_case(kernel, dtype, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    width = 16 // torch.empty((), dtype=dtype).element_size() if shadow_granularity == 16 else 1
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    auxiliary_pool = create_mem_pool(shadow_granularity=min(shadow_granularity, 4))
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros(width, dtype=dtype, device="cuda")
    # The relaxed scheduling flag does not establish a happens-before edge.
    # Keep it in a fine pool because coarse pools cannot contain atomic flags.
    with torch.cuda.use_mem_pool(auxiliary_pool):
        scratch = torch.zeros(width, dtype=torch.int32, device="cuda")
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel[(2, )](target, scratch, counter, WIDTH=width, num_warps=1)
    torch.cuda.synchronize()


def _run_raw_case(dtype=torch.int32, shadow_granularity=4) -> None:
    _run_race_case(_raw_kernel, dtype, shadow_granularity)


def _run_war_case(dtype=torch.int32, shadow_granularity=4) -> None:
    _run_race_case(_war_kernel, dtype, shadow_granularity)


def _run_waw_case(dtype=torch.int32, shadow_granularity=4) -> None:
    _run_race_case(_waw_kernel, dtype, shadow_granularity)


@run_with_gsan
def _run_pdl_without_wait_case() -> None:
    # Keep the producer CTAs in separate cells even in an 8-byte pool so the
    # failure comes from the missing consumer wait, not producer false sharing.
    payload = torch.zeros(2, dtype=torch.int64, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    _pdl_producer_kernel[(2, )](payload, num_warps=1)
    _pdl_consumer_without_wait_kernel[(1, )](payload, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@run_with_gsan
def _run_pdl_missing_wait_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    should_wait = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    _pdl_producer_kernel[(1, )](payload, num_warps=1)
    _pdl_conditional_wait_kernel[(1, )](should_wait, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@triton.jit
def _graph_unordered_write_kernel(payload, REVERSE: tl.constexpr):
    pid = tl.program_id(0)
    if REVERSE:
        pid = tl.num_programs(0) - 1 - pid
    tl.store(payload + pid, 1)


@triton.jit
def _graph_pdl_consumer_before_wait_kernel(payload_ptr, scratch_ptr, BLOCK: tl.constexpr):
    # Read every producer CTA's output so the race does not depend on a single
    # writer and this consumer being scheduled on different SMs.
    offsets = tl.arange(0, BLOCK)
    values = tl.load(payload_ptr + offsets)
    # Wait after the read so only the access ordering is invalid.
    tl.extra.cuda.gdc_wait()
    tl.store(scratch_ptr + offsets, values)


@run_with_gsan
def _run_explicit_graph_failure_case(kind: str) -> None:
    from triton.experimental.gsan import graph
    from triton.experimental.gsan._testing_utils import ExplicitGraph

    n = 512
    payload = torch.zeros(n, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(n, dtype=torch.int32, device="cuda")
    should_wait = torch.zeros(1, dtype=torch.int32, device="cuda")
    pdl = kind != "unordered"
    nodes = [graph.GraphNode(), graph.GraphNode((), (0, )) if pdl else graph.GraphNode()]
    plan = graph.GraphPlan(torch.cuda.current_device(), nodes)
    with ExplicitGraph(plan) as executable:
        if pdl:
            producer = _pdl_producer_kernel.warmup(payload, grid=(n, ), num_warps=1)
            if kind == "missing_wait":
                consumer = _pdl_conditional_wait_kernel.warmup(should_wait, scratch, grid=(1, ), num_warps=1,
                                                               launch_pdl=True)
                arguments = (should_wait, scratch)
            else:
                consumer = _graph_pdl_consumer_before_wait_kernel.warmup(payload, scratch, BLOCK=n, grid=(1, ),
                                                                         num_warps=1, launch_pdl=True)
                arguments = (payload, scratch)
            executable.add_kernel(producer, (payload, ), n)
            executable.add_kernel(consumer, arguments, 1)
        else:
            for reverse in (False, True):
                kernel = _graph_unordered_write_kernel.warmup(payload, REVERSE=reverse, grid=(n, ), num_warps=1)
                executable.add_kernel(kernel, (payload, ), n)
        executable.instantiate()
        executable.launch()
        torch.cuda.synchronize()


@run_with_gsan
def _run_pdl_persistent_state_without_wait_case() -> None:
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(num_sms, dtype=torch.int32, device="cuda")
    _pdl_transitively_acquired_producer_kernel[(num_sms, )](payload, flag, num_warps=1)
    _pdl_transitively_acquired_consumer_without_wait_kernel[(num_sms, )](payload, scratch, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()


@run_with_gsan
def _run_independent_streams_waw_case() -> None:
    num_blocks = 4
    target = torch.zeros(num_blocks, dtype=torch.int32, device="cuda")
    arrived = torch.zeros(2, dtype=torch.int32, device="cuda")
    # The side streams do not wait for the zero fills (which GSan does not see).
    torch.cuda.synchronize()
    # Nothing orders the two side streams, whichever kernel runs first.
    with torch.cuda.stream(torch.cuda.Stream()):
        _store_after_all_blocks_arrive_kernel[(1, )](target, 1, arrived[0:1], BLOCK=num_blocks, num_warps=1)
    # A block that takes over the first launch's GSan identity (slot) shares
    # its epoch, so the earlier store looks like its own and is not reported.
    # The co-resident blocks of this launch hold distinct slots, so at most
    # one of them can (this needs at least 4 free slots; with fewer, slot
    # acquisition fails after 20 s).
    with torch.cuda.stream(torch.cuda.Stream()):
        _store_after_all_blocks_arrive_kernel[(num_blocks, )](target, 2, arrived[1:2], BLOCK=1, num_warps=1)


@run_with_gsan
def _run_relaxed_cas_handoff_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    arrived = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _relaxed_cas_handoff_kernel[(2, )](payload, flag, arrived, out, num_warps=1)


@run_with_gsan
def _run_release_epoch_overflow_case() -> None:
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    # Every release advances the block's 16-bit epoch; wrapping it silently
    # would make later accesses look ordered.
    _release_loop_kernel[(1, )](flag, 70000, num_warps=1)


@run_with_gsan
def _run_mixed_scope_release_rmw_case() -> None:
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
    _mixed_scope_release_rmw_kernel[(2, )](counter, ready, num_warps=1)


@run_with_gsan
def _run_gluon_relaxed_cluster_barrier_case() -> None:
    probe_payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    probe_out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_cluster_barrier_kernel[(1, )](probe_payload, probe_out, RELAXED=False, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()
    probe_cell = shadow_cell_from_address(probe_payload.data_ptr())
    assert probe_cell.write_clock.thread_id != probe_cell.read_clocks[0].thread_id

    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_cluster_barrier_kernel[(1, )](payload, out, RELAXED=True, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_mbarrier_current_epoch_case(after_arrival: bool) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_mbarrier_current_epoch_kernel[(1, )](payload, out, after_arrival, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_tcgen05_commit_no_release_case() -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_tcgen05_commit_no_release_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)


@run_with_gsan
def _run_gluon_mbarrier_multicast_partial_wait_case() -> None:
    source = torch.zeros((256, 128), dtype=torch.float16, device="cuda")
    signal_layout = gl.NVMMASharedLayout.get_default_for([256, 128], gl.float16, cga_layout=((1, 0), (0, 0)))
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(source, [256, 128], signal_layout)
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _gluon_mbarrier_multicast_partial_wait_kernel[(1, )](input_desc, payload, out, num_warps=4, num_ctas=4)


@run_with_gsan
def _run_tma_raw_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:, :n_size]
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    triton.set_allocator(_cuda_byte_allocator)
    _tma_raw_kernel[(2, )](target, scratch, counter, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block)


@run_with_gsan
def _run_host_tma_war_case() -> None:
    block = 32
    m_size = 35
    n_size = 37
    padded_n = 40
    row_idx = 5
    col_idx = 8

    target_storage = torch.zeros((m_size, padded_n), dtype=torch.int32, device="cuda")
    scratch_storage = torch.zeros_like(target_storage)
    target = target_storage[:, :n_size]
    scratch = scratch_storage[:, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [block, block])
    scratch_desc = TensorDescriptor.from_tensor(scratch, [block, block])
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_war_kernel[(2, )](target, target_desc, scratch_desc, counter, row_idx, col_idx, target.stride(0))


@run_with_gsan
def _run_host_tma_gather_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    scratch = torch.zeros((block_x, block_y), dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_gather_war_kernel[(2, )](target, target_desc, x_offsets, scratch, counter, row_idx, y_offset,
                                       target.stride(0), scratch.stride(0), scratch.stride(1), BLOCK_X=block_x)


@run_with_gsan
def _run_host_tma_scatter_war_case() -> None:
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 13
    padded_m = 16
    padded_n = 16
    row_idx = 5
    y_offset = 8
    x_offsets_values = [5, 7, 9, 10, 1, 3, 11, 13]

    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    x_offsets = torch.tensor(x_offsets_values, dtype=torch.int32, device="cuda")
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    scratch = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _host_tma_scatter_war_kernel[(2, )](target, target_desc, x_offsets, src, src.stride(0), src.stride(1), scratch,
                                        counter, row_idx, y_offset, target.stride(0), BLOCK_X=block_x)


@run_with_gsan
def _run_host_tma_atomic_flag_publish_case() -> None:
    flag = torch.zeros((1, 16), dtype=torch.int32, device="cuda")
    flag_desc = TensorDescriptor.from_tensor(flag, [1, 16])
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _host_tma_atomic_flag_publish_kernel[(2, )](payload, flag, flag_desc, counter, scratch, num_warps=1)


@run_with_gsan
def _run_cross_sm_atomic_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_atomic_sync_kernel[(2, )](
        payload,
        flags,
        counter,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )


@run_with_gsan
def _run_atomic_poll_cross_sm_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _atomic_poll_cross_sm_sync_kernel[(2, )](
        payload,
        flag,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=4,
    )


@run_with_gsan
def _run_atomic_load_store_cross_sm_sync_case(producer_sem: str, consumer_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _atomic_load_store_cross_sm_sync_kernel[(2, )](
        payload,
        flag,
        counter,
        scratch,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )


@run_with_gsan
def _run_transitive_atomic_sync_case(release_sem: str, relay_sem: str, scope: str) -> None:
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _transitive_atomic_sync_kernel[(3, )](
        payload,
        flag0,
        flag1,
        counter,
        scratch,
        release_sem=release_sem,
        relay_sem=relay_sem,
        scope=scope,
        num_warps=1,
    )


def _expected_file_line(source_function, marker: str) -> str:
    source_lines, starting_line = inspect.getsourcelines(source_function)
    for line_offset, line in enumerate(source_lines):
        if marker in line:
            return f"{Path(inspect.getsourcefile(source_function)).name}:{starting_line + line_offset}"
    raise AssertionError(f"Could not find marker {marker!r} for function {source_function!r}")


def _run_failure_case(case: str, shadow_granularity: int | None = None, *, runner, source_function, marker: str,
                      error: str, runner_args=(), runner_kwargs=None) -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA device")

    if shadow_granularity is not None:
        runner_kwargs = {**(runner_kwargs or {}), "shadow_granularity": shadow_granularity}
    if runner_kwargs is None:
        runner_kwargs = {}

    if is_hip():
        # A hang (e.g. a lock-progress bug) must fail the test, not block it.
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRITON_TEST_PROCESS_TIMEOUT",
                      os.environ.get("TRITON_TEST_PROCESS_TIMEOUT", HIP_DEFAULT_PROCESS_TIMEOUT_S))
            result = run_in_process(runner, runner_args, runner_kwargs, allow_abort=True)
    else:
        result = run_in_process(runner, runner_args, runner_kwargs)
    print(result.driver_stderr_output)
    assert isinstance(result.exc, RuntimeError), (f"case={case} completed without the expected GSan failure\n"
                                                  f"exc={result.exc!r}\n"
                                                  f"driver stderr:\n{result.driver_stderr_output}")
    if is_hip():
        # HIP_SKIP_ABORT_ON_GPU_ERROR can surface a trap as launch failure or
        # hipErrorIllegalAddress. Require the sanitizer's exact diagnostic and
        # source location for the latter, so an unrelated memory fault cannot
        # stand in for the expected GSan report.
        aborted = getattr(result.exc, "exitcode", 0) < 0
        diagnostic = f"GSan device assertion failed: '{error}' at "
        reported_illegal_address = "hipErrorIllegalAddress" in str(result.exc) and any(
            line.startswith(diagnostic) and line.endswith(_expected_file_line(source_function, marker))
            for line in result.driver_stderr_output.splitlines())
        assert aborted or "launch failure" in str(
            result.exc) or reported_illegal_address, (f"case={case} did not fail with a device trap: {result.exc!r}")
    assert "GSanLibrary.cu" not in result.driver_stderr_output
    assert Path(inspect.getsourcefile(source_function)).name in result.driver_stderr_output
    assert _expected_file_line(source_function, marker) in result.driver_stderr_output
    assert error in result.driver_stderr_output


# Test races at every cell size with byte accesses, plus the default-pool word
# case. Multi-cell word accesses are checked by the shadow-update tests.
RACE_ACCESS_CASES = [(granularity, torch.int8) for granularity in (1, 2, 4, 8, 16)] + [(4, torch.int32)]


@pytest.mark.parametrize("shadow_granularity,dtype", RACE_ACCESS_CASES, indirect=["shadow_granularity"])
def test_read_after_write(shadow_granularity, dtype):
    _run_failure_case("raw", shadow_granularity, runner=_run_raw_case, source_function=_raw_kernel.fn,
                      marker="value = gl.load(ptr + offsets)", error="Read after write race detected",
                      runner_kwargs={"dtype": dtype})


@pytest.mark.parametrize("shadow_granularity,dtype", RACE_ACCESS_CASES, indirect=["shadow_granularity"])
def test_write_after_read(shadow_granularity, dtype):
    _run_failure_case("war", shadow_granularity, runner=_run_war_case, source_function=_war_kernel.fn,
                      marker="gl.store(ptr + offsets, 1)", error="Write after read race detected",
                      runner_kwargs={"dtype": dtype})


@pytest.mark.parametrize("shadow_granularity,dtype", RACE_ACCESS_CASES, indirect=["shadow_granularity"])
def test_write_after_write(shadow_granularity, dtype):
    _run_failure_case("waw", shadow_granularity, runner=_run_waw_case, source_function=_waw_kernel.fn,
                      marker="gl.store(ptr + offsets, 2)", error="Write after write race detected",
                      runner_kwargs={"dtype": dtype})


@triton.jit
def _coarse_pool_access_kernel(ptr, out, KIND: tl.constexpr):
    if KIND == "load":
        value = tl.load(ptr)
        tl.store(out, value)
    elif KIND == "store":
        tl.store(ptr, 1)
    else:
        tl.atomic_add(ptr, 1)


@triton.jit
def _coarse_pool_tma_tail_kernel(ptr):
    desc = tl.make_tensor_descriptor(ptr, shape=[32, 37], strides=[40, 1], block_shape=[32, 32])
    desc.store([0, 32], tl.full((32, 32), 1, tl.int32))


@gluon.jit
def _atomic_granularity_kernel(ptr, out, KIND: gl.constexpr):
    offsets = gl.arange(0, 256, layout=gl.BlockedLayout([8], [THREADS_PER_WARP], [1], [0]))
    if KIND == "load":
        value = gl.atomic_load(ptr + offsets, sem="relaxed")
        gl.store(out + offsets, value)
    elif KIND == "store":
        gl.atomic_store(ptr + offsets, 1, sem="relaxed")
    elif KIND == "rmw":
        gl.atomic_add(ptr, 1, sem="relaxed")
    elif KIND == "cas":
        gl.atomic_cas(ptr, 0, 1, sem="relaxed")
    else:
        gl.atomic_poll(ptr, 0, sem="relaxed")


@triton.jit
def _atomic_granularity_tma_kernel(desc):
    values = tl.full((1, 32), 1, desc.dtype)
    desc.atomic_add([0, 0], values)


@run_with_gsan
def _run_atomic_granularity_case(kind, dtype):
    target = torch.zeros((1, 256), dtype=dtype, device="cuda")
    if kind == "tma":
        desc = TensorDescriptor.from_tensor(target, [1, 32])
        _atomic_granularity_tma_kernel[(1, )](desc)
    else:
        output = torch.empty_like(target)
        _atomic_granularity_kernel[(1, )](target, output, kind, num_warps=1)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kind,marker", [
    ("load", "value = gl.atomic_load"),
    ("store", "gl.atomic_store"),
    ("rmw", "gl.atomic_add"),
    ("cas", "gl.atomic_cas"),
    ("poll", "gl.atomic_poll"),
])
def test_atomic_access_rejects_coarser_granularity(kind, marker):
    _run_failure_case("atomic_granularity", 8, runner=_run_atomic_granularity_case,
                      source_function=_atomic_granularity_kernel.fn, marker=marker,
                      error="GSan shadow granularity exceeds atomic access size", runner_args=(kind, torch.int32))


@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize("dtype,granularity", [(torch.int8, 2), (torch.int16, 4)])
def test_subword_atomic_access_rejects_coarser_granularity(kind, dtype, granularity):
    _run_failure_case("subword_atomic_granularity", granularity, runner=_run_atomic_granularity_case,
                      source_function=_atomic_granularity_kernel.fn, marker=f"gl.atomic_{kind}",
                      error="GSan shadow granularity exceeds atomic access size", runner_args=(kind, dtype))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="TMA requires Hopper or newer")
@pytest.mark.parametrize("dtype,granularity", [(torch.float16, 4), (torch.int32, 8)])
def test_tma_atomic_access_rejects_coarser_granularity(dtype, granularity):
    _run_failure_case("tma_atomic_granularity", granularity, runner=_run_atomic_granularity_case,
                      source_function=_atomic_granularity_tma_kernel.fn, marker="desc.atomic_add",
                      error="GSan shadow granularity exceeds atomic access size", runner_args=("tma", dtype))


@run_with_gsan
def _run_coarse_pool_access_case(kind):
    target = torch.zeros((32, 40), dtype=torch.int32, device="cuda")
    output = torch.empty_like(target)
    if kind == "tma_tail":
        triton.set_allocator(_cuda_byte_allocator)
        _coarse_pool_tma_tail_kernel[(1, )](target)
    else:
        _coarse_pool_access_kernel[(1, )](target, output, kind, num_warps=1)


@pytest.mark.parametrize("kind,marker", [
    ("load", "value = tl.load(ptr)"),
    ("store", "tl.store(ptr, 1)"),
    ("atomic", "tl.atomic_add(ptr, 1)"),
])
@pytest.mark.parametrize("shadow_granularity", [4, 16], indirect=True)
def test_pool_partial_and_atomic_access_restrictions(shadow_granularity, kind, marker):
    if shadow_granularity != 16:
        result = run_in_process(_run_coarse_pool_access_case, args=(kind, ),
                                kwargs={"shadow_granularity": shadow_granularity})
        assert result.exc is None, result.driver_stderr_output
        return
    error = ("GSan shadow granularity exceeds atomic access size"
             if kind == "atomic" else "GSan 16-byte pools require complete aligned 16-byte accesses")
    _run_failure_case("coarse_pool", shadow_granularity, runner=_run_coarse_pool_access_case,
                      source_function=_coarse_pool_access_kernel.fn, marker=marker, error=error, runner_args=(kind, ))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="TMA requires Hopper or newer")
@pytest.mark.parametrize("shadow_granularity", [4, 16], indirect=True)
def test_pool_partial_tma_tail_restrictions(shadow_granularity):
    if shadow_granularity != 16:
        result = run_in_process(_run_coarse_pool_access_case, args=("tma_tail", ),
                                kwargs={"shadow_granularity": shadow_granularity})
        assert result.exc is None, result.driver_stderr_output
        return
    _run_failure_case("coarse_pool_tma", shadow_granularity, runner=_run_coarse_pool_access_case,
                      source_function=_coarse_pool_tma_tail_kernel.fn, marker="desc.store",
                      error="GSan 16-byte pools require complete aligned 16-byte accesses", runner_args=("tma_tail", ))


def test_explicit_graph_unordered_nodes_report_race():
    _run_failure_case("graph_unordered", runner=_run_explicit_graph_failure_case, runner_args=("unordered", ),
                      source_function=_graph_unordered_write_kernel.fn, marker="tl.store(payload + pid, 1)",
                      error="Write after write race detected")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Graph PDL requires SM90 or newer")
@pytest.mark.parametrize("missing_wait", [False, True])
def test_explicit_graph_programmatic_edge_requires_wait(missing_wait):
    source = _pdl_conditional_wait_kernel if missing_wait else _graph_pdl_consumer_before_wait_kernel
    _run_failure_case(
        "graph_pdl", runner=_run_explicit_graph_failure_case,
        runner_args=("missing_wait" if missing_wait else "before_wait", ), source_function=source.fn,
        marker="def _pdl_conditional_wait_kernel" if missing_wait else "values = tl.load(payload_ptr + offsets)",
        error="did not call gdc_wait" if missing_wait else "Read after write race detected")


def test_independent_streams_write_after_write():
    _run_failure_case("independent_streams_waw", runner=_run_independent_streams_waw_case,
                      source_function=_store_after_all_blocks_arrive_kernel.fn, marker="tl.store(ptr + offsets, value)",
                      error="Write after write race detected")


def test_release_epoch_overflow_is_reported():
    _run_failure_case("release_epoch_overflow", runner=_run_release_epoch_overflow_case,
                      source_function=_release_loop_kernel.fn, marker="tl.atomic_add(flag_ptr, 1",
                      error="Vector clock overflowed")


def test_relaxed_cas_handoff_read_after_write():
    _run_failure_case("relaxed_cas_handoff", runner=_run_relaxed_cas_handoff_case,
                      source_function=_relaxed_cas_handoff_kernel.fn, marker="tl.store(out_ptr, tl.load(payload_ptr))",
                      error="Read after write race detected")


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_programmatic_dependent_launch_requires_wait(shadow_granularity):
    _run_failure_case(
        "pdl_missing_wait",
        shadow_granularity,
        runner=_run_pdl_missing_wait_case,
        source_function=_pdl_conditional_wait_kernel.fn,
        marker="def _pdl_conditional_wait_kernel",
        error="kernel launched with programmatic dependent launch did not call gdc_wait",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_programmatic_dependent_launch_without_wait_reports_race(shadow_granularity):
    _run_failure_case(
        "pdl_without_wait",
        shadow_granularity,
        runner=_run_pdl_without_wait_case,
        source_function=_pdl_consumer_without_wait_kernel.fn,
        marker="value = tl.load(payload_ptr + 1)",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_programmatic_dependent_launch_does_not_inherit_persistent_sm_dependencies(shadow_granularity):
    _run_failure_case(
        "pdl_persistent_state_without_wait",
        shadow_granularity,
        runner=_run_pdl_persistent_state_without_wait_case,
        source_function=_pdl_transitively_acquired_consumer_without_wait_kernel.fn,
        marker="value = tl.load(payload_ptr)",
        error="Read after write race detected",
    )


@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_mixed_scope_release_rmw_accumulation(shadow_granularity):
    _run_failure_case(
        "mixed_scope_release_rmw",
        shadow_granularity,
        runner=_run_mixed_scope_release_rmw_case,
        source_function=_mixed_scope_release_rmw_kernel.fn,
        marker='tl.atomic_add(counter_ptr, 1, sem="release", scope="sys")',
        error="GSan detected atomic release accumulation with mixed scopes, which is not supported.",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_relaxed_cluster_barrier_does_not_synchronize_vector_clocks(shadow_granularity):
    _run_failure_case(
        "relaxed_cluster_barrier",
        shadow_granularity,
        runner=_run_gluon_relaxed_cluster_barrier_case,
        source_function=_gluon_cluster_barrier_kernel.fn,
        marker="value = gl.load(payload_ptr",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.parametrize("after_arrival", [False, True], ids=["before-arrival", "after-arrival"])
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_mbarrier_wait_does_not_acquire_current_epoch(shadow_granularity, after_arrival):
    _run_failure_case(
        f"mbarrier_current_epoch_{after_arrival}",
        shadow_granularity,
        runner=_run_gluon_mbarrier_current_epoch_case,
        runner_args=(after_arrival, ),
        source_function=_gluon_mbarrier_current_epoch_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_tcgen05_commit_does_not_publish_vector_clock(shadow_granularity):
    _run_failure_case(
        "tcgen05_commit_no_release",
        shadow_granularity,
        runner=_run_gluon_tcgen05_commit_no_release_case,
        source_function=_gluon_tcgen05_commit_no_release_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_mbarrier_multicast_partial_wait_only_acquires_on_leader_ctas(shadow_granularity):
    _run_failure_case(
        "mbarrier_multicast_partial_wait",
        shadow_granularity,
        runner=_run_gluon_mbarrier_multicast_partial_wait_case,
        source_function=_gluon_mbarrier_multicast_partial_wait_kernel.fn,
        marker="value = gl.load(payload_ptrs",
        error="Read after write race detected",
    )


@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_tma_read_after_write(shadow_granularity):
    _run_failure_case("tma_raw", shadow_granularity, runner=_run_tma_raw_case, source_function=_tma_raw_kernel.fn,
                      marker="value = tl.load(ptr + row_idx * stride_0 + col_idx)",
                      error="Read after write race detected")


@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_host_tma_write_after_read(shadow_granularity):
    _run_failure_case("host_tma_war", shadow_granularity, runner=_run_host_tma_war_case,
                      source_function=_host_tma_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + col_idx, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_host_tma_gather_write_after_read(shadow_granularity):
    _run_failure_case("host_tma_gather_war", shadow_granularity, runner=_run_host_tma_gather_war_case,
                      source_function=_host_tma_gather_war_kernel.fn,
                      marker="tl.store(target_ptr + row_idx * stride_0 + y_offset, 1)",
                      error="Write after read race detected")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_host_tma_scatter_write_after_read(shadow_granularity):
    _run_failure_case("host_tma_scatter_war", shadow_granularity, runner=_run_host_tma_scatter_war_case,
                      source_function=_host_tma_scatter_war_kernel.fn,
                      marker="target_desc.scatter(values, x_offsets, y_offset)", error="Write after read race detected")


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_host_tma_atomic_on_release_flag_does_not_publish_data(shadow_granularity):
    _run_failure_case("host_tma_atomic_flag_publish", shadow_granularity, runner=_run_host_tma_atomic_flag_publish_case,
                      source_function=_host_tma_atomic_flag_publish_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_cross_sm_semantic_mismatch_read_after_write(shadow_granularity, producer_sem, consumer_sem, scope):
    _run_failure_case(f"cross_sm_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}", shadow_granularity,
                      runner=_run_cross_sm_atomic_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_cross_sm_atomic_sync_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_atomic_poll_semantic_mismatch_read_after_write(shadow_granularity, producer_sem, consumer_sem, scope):
    _run_failure_case(f"atomic_poll_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}", shadow_granularity,
                      runner=_run_atomic_poll_cross_sm_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_atomic_poll_cross_sm_sync_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.xfail(is_hip_cdna4(), run=False, reason=_GFX950_ATOMIC_LOAD_SPIN_HANG)
@pytest.mark.parametrize("producer_sem, consumer_sem, scope", CROSS_SM_SEMANTIC_MISMATCH_CASES)
def test_atomic_load_store_semantic_mismatch_read_after_write(producer_sem, consumer_sem, scope):
    _run_failure_case(f"atomic_load_store_semantic_mismatch_{producer_sem}_{consumer_sem}_{scope}",
                      runner=_run_atomic_load_store_cross_sm_sync_case, runner_args=(producer_sem, consumer_sem, scope),
                      source_function=_atomic_load_store_cross_sm_sync_kernel.fn,
                      marker="result = tl.load(payload_ptr)", error="Read after write race detected")


@pytest.mark.parametrize("producer_sem, consumer_sem", RELEASE_ACQUIRE_SYNC_CASES)
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_cross_sm_cta_scope_read_after_write(shadow_granularity, producer_sem, consumer_sem):
    _run_failure_case(f"cross_sm_cta_scope_{producer_sem}_{consumer_sem}", shadow_granularity,
                      runner=_run_cross_sm_atomic_sync_case, runner_args=(producer_sem, consumer_sem, "cta"),
                      source_function=_cross_sm_atomic_sync_kernel.fn,
                      marker="ready = tl.atomic_add(flag_ptr, 0, sem=consumer_sem, scope=scope)",
                      error="Read after write race detected")


@pytest.mark.parametrize("release_sem, relay_sem, scope", TRANSITIVE_RELAY_MISMATCH_CASES)
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_transitive_release_acquire_requires_middle_acquire(shadow_granularity, release_sem, relay_sem, scope):
    _run_failure_case(f"transitive_sync_{release_sem}_{relay_sem}_{scope}", shadow_granularity,
                      runner=_run_transitive_atomic_sync_case, runner_args=(release_sem, relay_sem, scope),
                      source_function=_transitive_atomic_sync_kernel.fn, marker="result = tl.load(payload_ptr)",
                      error="Read after write race detected")


@pytest.mark.parametrize("release_sem, relay_sem", RELEASE_ACQUIRE_SYNC_CASES)
@pytest.mark.gsan_fine_granularity("This race test uses scalar accesses or atomic synchronization")
def test_transitive_cta_scope_read_after_write(shadow_granularity, release_sem, relay_sem):
    _run_failure_case(f"transitive_cta_scope_{release_sem}_{relay_sem}", shadow_granularity,
                      runner=_run_transitive_atomic_sync_case, runner_args=(release_sem, relay_sem, "cta"),
                      source_function=_transitive_atomic_sync_kernel.fn,
                      marker="ready = tl.atomic_add(flag0_ptr, 0, sem=relay_sem, scope=scope)",
                      error="Read after write race detected")


# On AMD, tensor accesses through a kernel-argument base are converted to
# buffer ops (amdg.buffer_*), which GSan checks on the reconstructed addresses.
@triton.jit
def _buffer_race_kernel(ptr, scratch_ptr, counter_ptr, KIND: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    if pid == 0:
        if KIND == "war":
            first_read = tl.load(ptr + offsets)
            tl.store(scratch_ptr + offsets, first_read)
        else:
            tl.store(ptr + offsets, offsets)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        if KIND == "raw":
            second_read = tl.load(ptr + offsets)
            tl.store(scratch_ptr + offsets, second_read)
        else:
            tl.store(ptr + offsets, offsets + 1)


def _assert_buffer_ops(compiled, *ops) -> None:
    ttgir = compiled.asm["ttgir"]
    for op in ops:
        assert f"amdg.{op} " in ttgir, f"expected amdg.{op} in TTGIR:\n{ttgir}"


@run_with_gsan
def _run_buffer_race_case(kind: str) -> None:
    block = 64
    target = torch.zeros(block, dtype=torch.int32, device="cuda")
    scratch = torch.zeros_like(target)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    args = (target, scratch, counter)
    compiled = _buffer_race_kernel.warmup(*args, KIND=kind, BLOCK=block, num_warps=1, grid=(2, ))
    _assert_buffer_ops(compiled, "buffer_load" if kind != "waw" else "buffer_store", "buffer_store")
    _buffer_race_kernel[(2, )](*args, KIND=kind, BLOCK=block, num_warps=1)
    torch.cuda.synchronize()


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer ops")
@pytest.mark.parametrize("kind,marker,error", [
    ("raw", "second_read = tl.load(ptr + offsets)", "Read after write race detected"),
    ("war", "tl.store(ptr + offsets, offsets + 1)", "Write after read race detected"),
    ("waw", "tl.store(ptr + offsets, offsets + 1)", "Write after write race detected"),
])
def test_buffer_op_race(kind, marker, error):
    _run_failure_case(f"buffer_{kind}", runner=_run_buffer_race_case, runner_args=(kind, ),
                      source_function=_buffer_race_kernel.fn, marker=marker, error=error)


@gluon.jit
def _gluon_buffer_desc_race_kernel(ptr, scratch_ptr, counter_ptr):
    offsets = gl.arange(0, 64, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0]))
    pid = gl.program_id(0)
    if pid == 0:
        gl.store(ptr + 64 + offsets, 1)
        gl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        while gl.atomic_add(counter_ptr, 0, sem="relaxed") != 1:
            pass
        desc = gl.amd.cdna4.buffer.make_buffer_descriptor(ptr, 128 * 4)
        # Only the scalar offset moves these reads onto block 0's writes.
        values = gl.amd.cdna4.buffer.load(desc, offsets, soffset=64)
        gl.store(scratch_ptr + offsets, values)


@run_with_gsan
def _run_gluon_buffer_desc_race_case() -> None:
    target = torch.zeros(128, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(64, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _gluon_buffer_desc_race_kernel[(2, )](target, scratch, counter, num_warps=1)
    torch.cuda.synchronize()


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer ops")
def test_gluon_buffer_descriptor_soffset_read_after_write():
    _run_failure_case("gluon_buffer_desc_raw", runner=_run_gluon_buffer_desc_race_case,
                      source_function=_gluon_buffer_desc_race_kernel.fn,
                      marker="values = gl.amd.cdna4.buffer.load(desc, offsets, soffset=64)",
                      error="Read after write race detected")


@triton.jit
def _buffer_atomic_handoff_kernel(payload_ptr, flag_ptr, counter_ptr, scratch_ptr, PRODUCER_SEM: tl.constexpr,
                                  CONSUMER_SEM: tl.constexpr, CONSUMER_OP: tl.constexpr):
    # The flag is element 0 of a pair; element 1 stays 0, and masked-off
    # atomic results are undefined. (A one-element tensor offset folds away,
    # and only base + tensor offset becomes a buffer op.)
    pair = tl.arange(0, 2)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr + pair, 1000)
        tl.atomic_xchg(flag_ptr + pair, 1, mask=pair < 1, sem=PRODUCER_SEM)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        ready = 0
        while ready != 1:
            if CONSUMER_OP == "cas":
                ready = tl.sum(tl.atomic_cas(flag_ptr + pair, 1, 1, sem=CONSUMER_SEM))
            else:
                ready = tl.sum(tl.where(pair < 1, tl.atomic_add(flag_ptr + pair, 0, mask=pair < 1, sem=CONSUMER_SEM),
                                        0))
        result = tl.load(payload_ptr + pair)
        tl.store(scratch_ptr + pair, result)


@run_with_gsan
def _run_buffer_atomic_handoff_case(producer_sem: str, consumer_sem: str, consumer_op: str) -> None:
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    scratch = torch.full((2, ), -1, dtype=torch.int32, device="cuda")
    args = (payload, flag, counter, scratch, producer_sem, consumer_sem, consumer_op)
    compiled = _buffer_atomic_handoff_kernel.warmup(*args, num_warps=1, grid=(2, ))
    _assert_buffer_ops(compiled, "buffer_atomic_rmw", "buffer_load",
                       *(["buffer_atomic_cas"] if consumer_op == "cas" else []))
    _buffer_atomic_handoff_kernel[(2, )](*args, num_warps=1)
    torch.cuda.synchronize()


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer atomics")
@pytest.mark.parametrize("producer_sem, consumer_sem", [("relaxed", "acquire"), ("release", "relaxed")])
@pytest.mark.parametrize("consumer_op", ["rmw", "cas"])
def test_buffer_atomic_semantic_mismatch_read_after_write(producer_sem, consumer_sem, consumer_op):
    _run_failure_case(f"buffer_atomic_handoff_{producer_sem}_{consumer_sem}_{consumer_op}",
                      runner=_run_buffer_atomic_handoff_case, runner_args=(producer_sem, consumer_sem, consumer_op),
                      source_function=_buffer_atomic_handoff_kernel.fn, marker="result = tl.load(payload_ptr + pair)",
                      error="Read after write race detected")


@gluon.jit
def _buffer_i64_bound_kernel(data, out, num_records, KIND: gl.constexpr, MASK_PARTIAL: gl.constexpr,
                             CONSTANT_BOUND: gl.constexpr):
    offsets = gl.arange(0, 64, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0]))
    if CONSTANT_BOUND:
        num_records = 12
    desc = gl.amd.cdna4.buffer.make_buffer_descriptor(data, num_records)
    mask = (offsets == 0) | ((offsets == 1) & MASK_PARTIAL)
    if KIND == "store":
        values = gl.full((64, ), 0x4444444433333333, gl.int64, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [1],
                                                                                       [0]))
        gl.amd.cdna4.buffer.atomic_store(values, desc, offsets, mask=mask, sem="release")
    else:
        values = gl.amd.cdna4.buffer.atomic_load(desc, offsets, mask=mask, sem="acquire")
        gl.store(out + offsets, values)


def _run_buffer_i64_bound_case(kind, num_records, mask_partial, gsan, constant_bound=False):
    triton.knobs.compilation.instrumentation_mode = "gsan" if gsan else ""
    pool = create_mem_pool(shadow_granularity=4)
    with torch.cuda.use_mem_pool(pool):
        data = torch.full((64, ), 0x2222222211111111, dtype=torch.int64, device="cuda")
        out = torch.full_like(data, -1)
        _buffer_i64_bound_kernel[(1, )](data, out, num_records, kind, mask_partial, constant_bound, num_warps=1)
        torch.cuda.synchronize()
        data, out = data.tolist(), out.tolist()
        expected_data = [0x2222222211111111] * 64
        expected_out = [-1] * 64 if kind == "store" else [0] * 64
        bound = 12 if constant_bound else num_records
        if bound >= 8:
            if kind == "store":
                expected_data[0] = 0x4444444433333333
            else:
                expected_out[0] = 0x2222222211111111
        if mask_partial and bound >= 12 and (bound >= 16 or not is_hip_cdna5()):
            if kind == "store":
                expected_data[1] = 0x4444444433333333 if bound >= 16 else 0x2222222233333333
            else:
                expected_out[1] = 0x2222222211111111 if bound >= 16 else 0x11111111
        assert data == expected_data
        assert out == expected_out


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer atomics")
@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize("num_records,mask_partial,constant_bound", [
    (16, True, False),
    (12, False, False),
    (0, True, False),
    pytest.param(12, True, False, marks=pytest.mark.skipif(not is_hip_cdna5(),
                                                           reason="gfx950 accesses a partial dword")),
    pytest.param(12, True, True, marks=pytest.mark.skipif(not is_hip_cdna5(),
                                                          reason="gfx950 accesses a partial dword")),
])
def test_buffer_i64_atomic_full_or_dropped_accesses_preserved(kind, num_records, mask_partial, constant_bound,
                                                              monkeypatch):
    # Full accesses, masked partial elements, and wholly out-of-bounds accesses
    # remain supported. Compare actual bits and every byte of the allocation.
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", HIP_DEFAULT_PROCESS_TIMEOUT_S)
    baseline = run_in_process(_run_buffer_i64_bound_case, args=(kind, num_records, mask_partial, False, constant_bound))
    sanitized = run_in_process(_run_buffer_i64_bound_case, args=(kind, num_records, mask_partial, True, constant_bound))
    assert baseline.exc is None, baseline.exc
    assert sanitized.exc is None, sanitized.exc


@pytest.mark.skipif(not is_hip_cdna4(), reason="gfx950 accesses partial buffer atomics")
@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize("constant_bound", [False, True], ids=["dynamic-bound", "constant-bound"])
def test_buffer_i64_atomic_partial_access_rejected(kind, constant_bound, monkeypatch):
    # Off: the low dword is accessed, the high dword is zero on loads and
    # untouched on stores. On: fail clearly instead of issuing a global i64
    # access that would change those memory effects.
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", HIP_DEFAULT_PROCESS_TIMEOUT_S)
    baseline = run_in_process(_run_buffer_i64_bound_case, args=(kind, 12, True, False, constant_bound))
    assert baseline.exc is None, baseline.exc
    _run_failure_case("buffer_i64_partial", runner=_run_buffer_i64_bound_case,
                      source_function=_buffer_i64_bound_kernel.fn, marker="gl.amd.cdna4.buffer.atomic_" + kind + "(",
                      error="GSan on AMD does not support partially in-bounds buffer atomics",
                      runner_args=(kind, 12, True, True, constant_bound))


@gluon.jit
def _amd_direct_async_handoff_kernel(payload, flag, output, BUFFER: gl.constexpr, ORDERED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    index = gl.arange(0, 64, layout)
    if gl.program_id(0) == 0:
        gl.store(payload + index, 7)
        gl.atomic_add(flag, 1, sem="release" if ORDERED else "relaxed")
    else:
        while gl.atomic_add(flag, 0, sem="acquire" if ORDERED else "relaxed") == 0:
            pass
        smem = gl.allocate_shared_memory(gl.int32, [64], shared)
        if BUFFER:
            desc = gl.amd.cdna4.buffer.make_buffer_descriptor(payload, 64 * 4)
            gl.amd.cdna4.buffer.load_to_shared(smem, desc, index)
        else:
            amd_global_load_to_shared(smem, payload + index)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.store(output + index, smem.load(layout))


@run_with_gsan
def _run_amd_direct_async_handoff(buffer, ordered):
    payload = torch.zeros(64, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.zeros_like(payload)
    _amd_direct_async_handoff_kernel[(2, )](payload, flag, output, buffer, ordered, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 7))


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD gfx950/gfx1250 direct async GSan")
@pytest.mark.parametrize("buffer", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not is_hip_cdna4(), reason="Buffer-to-LDS instructions require gfx950"))
])
def test_gsan_amd_direct_async_read_after_write(buffer):
    # The relaxed flag forces the writer to run first without ordering its
    # payload.  The async issue must report a source-located RAW diagnostic.
    marker = "buffer.load_to_shared" if buffer else "amd_global_load_to_shared"
    _run_failure_case("amd direct async raw", runner=_run_amd_direct_async_handoff,
                      source_function=_amd_direct_async_handoff_kernel.fn, marker=marker,
                      error="Read after write race detected", runner_args=(buffer, False))


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD gfx950/gfx1250 direct async GSan")
@pytest.mark.parametrize("buffer", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not is_hip_cdna4(), reason="Buffer-to-LDS instructions require gfx950"))
])
def test_gsan_amd_direct_async_release_acquire_handoff(buffer, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", HIP_DEFAULT_PROCESS_TIMEOUT_S)
    result = run_in_process(_run_amd_direct_async_handoff, args=(buffer, True), allow_abort=True)
    assert result.exc is None, result.driver_stderr_output


@gluon.jit
def _amd_tdm_handoff_kernel(payload, flag, output, KIND: gl.constexpr, ORDERED: gl.constexpr,
                            COMPLETION: gl.constexpr = False):
    parent: gl.constexpr = gl.BlockedLayout([16, 1], [1, 32], [1, 4], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    row = gl.arange(0, 16, gl.SliceLayout(1, parent))
    col = gl.arange(0, 64, gl.SliceLayout(0, parent))
    ptrs = payload + row[:, None] * 64 + col[None, :]
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(payload, [16, 64], [64, 1], [16, 64], shared)
    if gl.program_id(0) == 0:
        if KIND == "raw" or KIND == "fused":
            gl.store(ptrs, 7)
        elif KIND == "war":
            gl.store(output + row[:, None] * 64 + col[None, :], gl.load(ptrs))
        else:
            gl.store(ptrs, 3)
        gl.atomic_add(flag, 1, sem="release" if ORDERED else "relaxed")
    else:
        while gl.atomic_add(flag, 0, sem="acquire" if ORDERED else "relaxed") == 0:
            pass
        smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
        bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
        if COMPLETION:
            gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
        if KIND == "raw":
            gl.amd.cdna5.tdm.async_gather(desc, row, smem, mbarrier=bar if COMPLETION else None)
            if COMPLETION:
                gl.amd.cdna5.mbarrier.wait(bar, 0)
            else:
                gl.amd.cdna5.tdm.async_wait(0)
            gl.store(output + row[:, None] * 64 + col[None, :], smem.load(parent))
        elif KIND == "fused":
            out_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(output, [16, 64], [64, 1], [16, 64], shared)
            other = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
            gl.amd.cdna5.tdm.async_load_fused([(out_desc, other, 3), (desc, smem, 12)])
            gl.amd.cdna5.tdm.async_wait(0)
            gl.store(output + row[:, None] * 64 + col[None, :], smem.load(parent))
        else:
            smem.store(gl.full([16, 64], 7, gl.int32, parent))
            gl.amd.cdna5.tdm.async_scatter(desc, row, smem, mbarrier=bar if COMPLETION else None)
            if COMPLETION:
                gl.amd.cdna5.mbarrier.wait(bar, 0)
            else:
                gl.amd.cdna5.tdm.async_wait(0)


@run_with_gsan
def _run_amd_tdm_handoff(kind, ordered, completion=False):
    payload = torch.full((16, 64), 1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.zeros_like(payload)
    _amd_tdm_handoff_kernel[(2, )](payload, flag, output, kind, ordered, completion)
    torch.cuda.synchronize()
    if kind in ("raw", "fused"):
        torch.testing.assert_close(output, torch.full_like(output, 7))
    else:
        torch.testing.assert_close(payload, torch.full_like(payload, 7))
        if kind == "war":
            torch.testing.assert_close(output, torch.full_like(output, 1))


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("kind", ["raw", "war", "waw", "fused"])
def test_gsan_gfx1250_tdm_unordered_handoff(kind, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    marker = {
        "raw": "tdm.async_gather", "war": "tdm.async_scatter", "waw": "tdm.async_scatter", "fused":
        "tdm.async_load_fused"
    }[kind]
    error = {
        "raw": "Read after write", "war": "Write after read", "waw": "Write after write", "fused": "Read after write"
    }[kind] + " race detected"
    _run_failure_case("amd advanced tdm " + kind, runner=_run_amd_tdm_handoff,
                      source_function=_amd_tdm_handoff_kernel.fn, marker=marker, error=error, runner_args=(kind, False))


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("kind", ["raw", "war", "waw", "fused"])
def test_gsan_gfx1250_tdm_ordered_handoff(kind, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_amd_tdm_handoff, args=(kind, True), allow_abort=True)
    assert result.exc is None, result.driver_stderr_output


@pytest.mark.skipif(not is_hip_cdna5(), reason="Completion barriers require AMD gfx1250")
@pytest.mark.parametrize("kind", ["raw", "war", "waw"])
@pytest.mark.parametrize("ordered", [False, True])
def test_gsan_gfx1250_local_completion_barrier_handoff(kind, ordered, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    if ordered:
        result = run_in_process(_run_amd_tdm_handoff, args=(kind, True, True), allow_abort=True)
        assert result.exc is None, result.driver_stderr_output
    else:
        marker = "tdm.async_gather" if kind == "raw" else "tdm.async_scatter"
        error = {"raw": "Read after write", "war": "Write after read", "waw": "Write after write"}[kind]
        # A local completion wait orders no producer in another logical block.
        _run_failure_case("amd local completion " + kind, runner=_run_amd_tdm_handoff,
                          source_function=_amd_tdm_handoff_kernel.fn, marker=marker, error=error + " race detected",
                          runner_args=(kind, False, True))


@gluon.jit
def _amd_ws_handoff_default(payload, flag, ack, output, ORDERED: gl.constexpr, ITERATIONS: gl.constexpr,
                            WORKER_WARPS: gl.constexpr):
    pass


@gluon.jit(noinline=True)
def _amd_ws_handoff_worker(payload, flag, ack, output, ORDERED: gl.constexpr, ITERATIONS: gl.constexpr,
                           WORKER_WARPS: gl.constexpr):
    offsets = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [WORKER_WARPS], [0]))
    for iteration in range(ITERATIONS):
        if gl.program_id(0) == 0:
            if iteration > 0:
                while gl.atomic_add(ack, 0, sem="acquire") != iteration:
                    pass
            gl.store(payload + offsets, offsets + 100 * iteration, offsets < 123)
            gl.atomic_xchg(flag, iteration + 1, sem="release" if ORDERED else "relaxed")
        else:
            while gl.atomic_add(flag, 0, sem="acquire") != iteration + 1:
                pass
            values = gl.load(payload + offsets, offsets < 123, other=0)
            gl.store(output + iteration * 128 + offsets, values, offsets < 123)
            gl.atomic_xchg(ack, iteration + 1, sem="release")


@gluon.jit
def _amd_ws_handoff_kernel(payload, flag, ack, output, ORDERED: gl.constexpr, ITERATIONS: gl.constexpr,
                           WORKER_WARPS: gl.constexpr):
    gl.warp_specialize([
        (_amd_ws_handoff_default, (payload, flag, ack, output, ORDERED, ITERATIONS, WORKER_WARPS)),
        (_amd_ws_handoff_worker, (payload, flag, ack, output, ORDERED, ITERATIONS, WORKER_WARPS)),
    ], [WORKER_WARPS], [24])


@run_with_gsan
def _run_amd_ws_handoff(ordered, iterations, worker_warps=4):
    payload = torch.zeros(128, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    ack = torch.zeros_like(flag)
    output = torch.zeros(iterations * 128, dtype=torch.int32, device="cuda")
    compiled = _amd_ws_handoff_kernel[(2, )](payload, flag, ack, output, ordered, iterations, worker_warps, num_warps=4)
    torch.cuda.synchronize()
    expected = torch.cat([
        torch.cat((torch.arange(123, dtype=torch.int32, device="cuda") + 100 * i,
                   torch.zeros(5, dtype=torch.int32, device="cuda"))) for i in range(iterations)
    ])
    torch.testing.assert_close(output, expected)
    return compiled.asm["llir"], compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna5(), reason="requires MI450 warp specialization")
@pytest.mark.parametrize("worker_warps", [1, 2, 4])
def test_gsan_amd_warp_specialize_noinline_scalar_handoff(monkeypatch, worker_warps):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_amd_ws_handoff, args=(True, 4, worker_warps), allow_abort=True)
    assert result.exc is None, result.driver_stderr_output


@pytest.mark.skipif(not is_hip_cdna5(), reason="requires MI450 warp specialization")
def test_gsan_amd_warp_specialize_noinline_missing_release(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    _run_failure_case("nondefault noinline region missing release", runner=_run_amd_ws_handoff,
                      source_function=_amd_ws_handoff_worker.fn, marker="values = gl.load",
                      error="Read after write race detected", runner_args=(False, 1))


@gluon.jit
def _amd_warp_pipeline_handoff_kernel(payload, flag, output, ORDERED: gl.constexpr):
    offsets = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [8], [0]))
    if gl.program_id(0) == 0:
        gl.store(payload + offsets, offsets)
        gl.atomic_xchg(flag, 1, sem="release" if ORDERED else "relaxed")
    else:
        while gl.atomic_add(flag, 0, sem="acquire") != 1:
            pass
        for iteration in range(4):
            with gl.amd.warp_pipeline_stage("read"):
                values = gl.load(payload + offsets)
            with gl.amd.warp_pipeline_stage("write"):
                gl.store(output + iteration * 256 + offsets, values + iteration)


@run_with_gsan
def _run_amd_warp_pipeline_handoff(ordered):
    payload = torch.zeros(256, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.zeros(4 * 256, dtype=torch.int32, device="cuda")
    compiled = _amd_warp_pipeline_handoff_kernel[(2, )](payload, flag, output, ordered, num_warps=8)
    torch.cuda.synchronize()
    expected = torch.cat([torch.arange(256, dtype=torch.int32, device="cuda") + i for i in range(4)])
    torch.testing.assert_close(output, expected)
    return compiled.asm["llir"], compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna5(), reason="requires MI450 warp pipelining")
def test_gsan_amd_warp_pipeline_repeated_handoff(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_amd_warp_pipeline_handoff, args=(True, ), allow_abort=True)
    assert result.exc is None, result.driver_stderr_output


@pytest.mark.skipif(not is_hip_cdna5(), reason="requires MI450 warp pipelining")
def test_gsan_amd_warp_pipeline_missing_release(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    _run_failure_case("warp pipeline missing release", runner=_run_amd_warp_pipeline_handoff,
                      source_function=_amd_warp_pipeline_handoff_kernel.fn, marker="values = gl.load",
                      error="Read after write race detected", runner_args=(False, ))


@gluon.jit(noinline=True)
def _amd_ws_tdm_completion_worker(src, dst):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    source = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [16, 64], [64, 1], [16, 64], shared)
    destination = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [16, 64], [64, 1], [16, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
    for iteration in range(4):
        gl.amd.cdna5.tdm.async_load(source, [0, 0], smem, mbarrier=bar)
        gl.amd.cdna5.mbarrier.wait(bar, 0)
        gl.amd.cdna5.tdm.async_store(destination, [0, 0], smem, mbarrier=bar)
        gl.amd.cdna5.mbarrier.wait(bar, 1)


@gluon.jit
def _amd_ws_tdm_completion_default(src, dst):
    pass


@gluon.jit
def _amd_ws_tdm_completion_kernel(src, dst):
    gl.warp_specialize([(_amd_ws_tdm_completion_default, (src, dst)), (_amd_ws_tdm_completion_worker, (src, dst))], [4],
                       [24])


@run_with_gsan
def _run_amd_ws_tdm_completion():
    src = torch.arange(16 * 64, dtype=torch.int32, device="cuda")
    dst = torch.full_like(src, -1)
    compiled = _amd_ws_tdm_completion_kernel[(1, )](src, dst, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src)
    return compiled.asm["llir"], compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna5(), reason="requires MI450 TDM and warp specialization")
def test_gsan_amd_warp_specialize_noinline_tdm_completion(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_amd_ws_tdm_completion, allow_abort=True)
    assert result.exc is None, result.driver_stderr_output
