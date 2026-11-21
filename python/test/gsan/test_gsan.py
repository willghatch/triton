from __future__ import annotations

import os
import re

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia import hopper
from triton.tools.tensor_descriptor import TensorDescriptor

from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna4, is_hip_cdna5, is_ampere_or_newer, is_hopper_or_newer, is_sm12x, run_in_process
from triton.experimental.gsan import create_mem_pool
from triton.experimental.gsan._allocator import get_device_rank, gsan_free, gsan_malloc
from triton.experimental.gsan._stream_sync import _runtime_state_layout
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr
from triton.backends.compiler import GPUTarget
from triton._C.libtriton.gsan_testing import AtomicScope, ScalarClock, shadow_granularity as get_shadow_granularity, shadow_cell_address
from triton.experimental.gsan._testing_utils import (atomic_poll, load_one_i32, shadow_cell_from_address, store_one_i32,
                                                     thread_state_from_smid)

# GSan runs on NVIDIA GPUs and, as a prototype, on AMD gfx950/gfx1250.
requires_gsan = pytest.mark.skipif(not (is_cuda() or is_hip_cdna4() or is_hip_cdna5()),
                                   reason="GSan requires CUDA or AMD gfx950/gfx1250")
THREADS_PER_WARP = gl.constexpr(64 if is_hip_cdna4() else 32)
amd_global_load_to_shared = (gl.amd.cdna5.async_copy.global_to_shared
                             if is_hip_cdna5() else gl.amd.cdna4.async_copy.global_load_to_shared)


@pytest.fixture()
def with_gsan(fresh_knobs, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(pool):
        yield shadow_granularity


def _first_identity_on_device() -> int:
    """Global thread id of this device's first SM/slot (local ids add to it)."""
    device = torch.cuda.current_device()
    rank = get_device_rank(device)
    return rank * _runtime_state_layout(rank, device).num_sms


def _own_epochs() -> list[int]:
    """Current epoch of every GSan thread identity (SM or slot) on this device."""
    device = torch.cuda.current_device()
    rank = get_device_rank(device)
    layout = _runtime_state_layout(rank, device)
    states = layout.thread_state_region.view(layout.num_sms, layout.thread_state_stride_bytes)
    header = layout.thread_state_header_size_bytes
    epoch_bytes = 2  # gsan::epoch_t is uint16_t
    vector_clocks = states[:, header:header + epoch_bytes * layout.num_threads].contiguous().view(torch.int16)
    local = torch.arange(layout.num_sms, device=vector_clocks.device)
    own = vector_clocks[local, local + rank * layout.num_sms].int() & 0xFFFF
    epochs = [0] * layout.num_threads
    for local_id, epoch in enumerate(own.tolist()):
        epochs[local_id + rank * layout.num_sms] = epoch
    return epochs


def _clock_buffer_snapshot_idx(token: int, state, tid: int) -> int:
    return (token % state.clock_buffer_size) * state.num_threads + tid


ATOMIC_SCOPE_CASES = (
    pytest.param("cta", AtomicScope.CTA, id="scope-cta"),
    pytest.param("gpu", AtomicScope.GPU, id="scope-gpu"),
    pytest.param("sys", AtomicScope.SYSTEM, id="scope-sys"),
)

ATOMIC_LOAD_STORE_TYPES = (
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
)

# Check every dtype at GPU scope, and scope handling on one representative
# dtype. The ordering parameters below still cover each case's legal semantics.
ATOMIC_LOAD_STORE_SCOPE_CASES = [(dtype, min(4, dtype.itemsize), "gpu", AtomicScope.GPU)
                                 for dtype in ATOMIC_LOAD_STORE_TYPES] + [
                                     (torch.int32, 4, "cta", AtomicScope.CTA),
                                     (torch.int32, 4, "sys", AtomicScope.SYSTEM),
                                 ]

ATOMIC_SEMANTIC_CASES = (
    pytest.param("relaxed", False, id="sem-relaxed"),
    pytest.param("acquire", False, id="sem-acquire"),
    pytest.param("release", True, id="sem-release"),
    pytest.param("acq_rel", True, id="sem-acq-rel"),
)

RELEASE_SEMANTIC_CASES = (
    pytest.param("release", id="sem-release"),
    pytest.param("acq_rel", id="sem-acq-rel"),
)

ACQUIRE_SEMANTIC_CASES = (
    pytest.param("acquire", id="sem-acquire"),
    pytest.param("acq_rel", id="sem-acq-rel"),
)


def _assert_atomic_rmw_shadow(real_address: int, expected_scope: AtomicScope, *, is_release: bool) -> None:
    cell = shadow_cell_from_address(real_address)
    tid = cell.write_clock.thread_id
    state = thread_state_from_smid(tid)

    if is_release:
        token = cell.write_clock.epoch
        snapshot_idx = _clock_buffer_snapshot_idx(token, state, tid)
        published_epoch = state.clock_buffer[snapshot_idx]

        assert cell.write_clock == ScalarClock(token, tid, expected_scope, is_release=True)
        assert token == state.clock_buffer_head
        assert state.clock_buffer_dirty
        assert cell.read_clocks[0] == ScalarClock(published_epoch, tid, expected_scope)
        assert state.vector_clock[tid] == published_epoch + 1
    else:
        epoch = state.vector_clock[tid]
        assert cell.write_clock == ScalarClock(epoch, tid, expected_scope)
        assert cell.read_clocks[0] == ScalarClock(epoch, tid, expected_scope)

    assert cell.num_reads == 1


def _assert_atomic_read_only_shadow(real_address: int, expected_scope: AtomicScope) -> None:
    cell = shadow_cell_from_address(real_address)
    tid = cell.read_clocks[0].thread_id
    epoch = thread_state_from_smid(tid).vector_clock[tid]

    assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
    assert cell.read_clocks[0] == ScalarClock(epoch, tid, expected_scope)
    assert cell.num_reads == 1


def _assert_atomic_store_only_shadow(real_address: int, expected_scope: AtomicScope, *, is_release: bool) -> None:
    cell = shadow_cell_from_address(real_address)
    tid = cell.write_clock.thread_id
    state = thread_state_from_smid(tid)

    assert cell.write_clock.scope == expected_scope
    assert cell.write_clock.is_release == is_release
    assert cell.num_reads == 0
    assert all(clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC) for clock in cell.read_clocks)
    if is_release:
        assert cell.write_clock.epoch == state.clock_buffer_head
        assert state.clock_buffer_dirty
    else:
        assert cell.write_clock.epoch == state.vector_clock[tid]


def _assert_cross_sm_sync(payload_ptr: torch.Tensor, flag_ptr: torch.Tensor, expected_scope: AtomicScope) -> None:
    payload_cell = shadow_cell_from_address(payload_ptr.data_ptr())
    flag_cell = shadow_cell_from_address(flag_ptr.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch
    consumer_tid = payload_cell.read_clocks[0].thread_id
    consumer_state = thread_state_from_smid(consumer_tid)

    assert flag_cell.write_clock.scope == expected_scope
    assert flag_cell.write_clock.is_release
    assert consumer_state.vector_clock[producer_tid] >= producer_epoch


def _assert_no_gsan_runtime_output(capfd) -> None:
    captured = capfd.readouterr()
    assert "GSanLibrary.cu" not in captured.out + captured.err


@gluon.jit
def _store_wide_i32(ptr):
    offsets = gl.arange(0, 4, layout=gl.BlockedLayout([4], [32], [1], [0]))
    gl.store(ptr + offsets, 1)


@gluon.jit
def _load_wide_i32(ptr, out_ptr):
    offsets = gl.arange(0, 4, layout=gl.BlockedLayout([4], [32], [1], [0]))
    gl.store(out_ptr + offsets, gl.load(ptr + offsets))


@requires_gsan
@pytest.mark.parametrize("shadow_granularity", [4] if is_hip_cdna4() else [1, 2, 4, 8, 16], indirect=True)
def test_load_store_updates_shadow(with_gsan):
    width = max(1, with_gsan // 4)
    target = torch.zeros(width, dtype=torch.int32, device="cuda")
    scratch = torch.zeros_like(target)
    store = _store_wide_i32 if with_gsan == 16 else store_one_i32
    load = _load_wide_i32 if with_gsan == 16 else load_one_i32

    store[(1, )](target, num_warps=1)
    cell0 = shadow_cell_from_address(target.data_ptr())

    tid = cell0.write_clock.thread_id
    epoch0 = thread_state_from_smid(tid).vector_clock[tid]

    assert cell0.write_clock.thread_id == tid
    assert cell0.write_clock.epoch == epoch0
    assert cell0.read_clocks[0].thread_id == 0
    assert cell0.read_clocks[0].epoch == 0
    assert cell0.num_reads == 0

    epochs_before_load = _own_epochs()
    load[(1, )](target, scratch, num_warps=1)
    cell1 = shadow_cell_from_address(target.data_ptr())
    # The reader may run on a different SM (NVIDIA) or state slot (AMD) than
    # the writer; either way its kernel entry starts a new epoch, which the
    # read records.
    reader_tid = cell1.read_clocks[0].thread_id
    epoch1 = thread_state_from_smid(reader_tid - _first_identity_on_device()).vector_clock[reader_tid]
    assert epoch1 == epochs_before_load[reader_tid] + 1
    assert cell1.write_clock == cell0.write_clock
    assert cell1.read_clocks[0] == ScalarClock(epoch1, reader_tid, AtomicScope.NON_ATOMIC)
    # Scalar accesses are instrumented once via the redundant-thread predicate.
    assert cell1.num_reads == 1
    for byte_offset in range(0, target.numel() * target.element_size(), with_gsan):
        assert shadow_cell_from_address(target.data_ptr() + byte_offset) == cell1


@gluon.jit
def _disjoint_access_kernel(ptr, out, counter, KIND: gl.constexpr, CELL_BYTES: gl.constexpr):
    pid = gl.program_id(0)
    offsets = gl.arange(0, 128, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    mask = offsets // CELL_BYTES % 2 == pid
    if pid == 1:
        while gl.atomic_add(counter, 0, sem="relaxed") == 0:
            pass
    if (KIND == "raw" and pid == 1) or (KIND == "war" and pid == 0):
        values = gl.load(ptr + offsets, mask=mask, other=0)
        gl.store(out + offsets, values, mask=mask)
    else:
        gl.store(ptr + offsets, pid + 5, mask=mask)
    if pid == 0:
        gl.atomic_add(counter, 1, sem="relaxed")


@requires_gsan
@pytest.mark.parametrize("kind", ["raw", "war", "waw"])
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_disjoint_masked_accesses(fresh_knobs, shadow_granularity, kind):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    auxiliary_pool = create_mem_pool(shadow_granularity=min(shadow_granularity, 4))
    with torch.cuda.use_mem_pool(pool):
        target = torch.arange(128, dtype=torch.uint8, device="cuda")
        out = torch.zeros_like(target)
    with torch.cuda.use_mem_pool(auxiliary_pool):
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _disjoint_access_kernel[(2, )](target, out, counter, kind, CELL_BYTES=shadow_granularity, num_warps=1)
    expected = torch.arange(128, dtype=torch.uint8, device="cuda")
    first_cta = torch.arange(128, device="cuda") // shadow_granularity % 2 == 0
    if kind != "war":
        expected[first_cta] = 5
    if kind != "raw":
        expected[~first_cta] = 6
    torch.testing.assert_close(target, expected)
    if kind != "waw":
        read_mask = ~first_cta if kind == "raw" else first_cta
        torch.testing.assert_close(out[read_mask], expected[read_mask])

    # Inspect all cells, including cells masked off in each participating block.
    for offset in range(0, target.numel(), shadow_granularity):
        reader = (kind == "raw" and offset // shadow_granularity % 2 == 1
                  or kind == "war" and offset // shadow_granularity % 2 == 0)
        cell = shadow_cell_from_address(target.data_ptr() + offset)
        assert (cell.num_reads > 0) == reader
        if reader:
            assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
        else:
            assert cell.write_clock.epoch != 0
            assert cell.write_clock.scope == AtomicScope.NON_ATOMIC
        output_cell = shadow_cell_from_address(out.data_ptr() + offset)
        if reader:
            assert output_cell.write_clock.epoch != 0
        else:
            assert output_cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)


@triton.jit
def _byte_pool_atomic_kernel(ptr):
    tl.atomic_add(ptr, 1, sem="acq_rel")


@requires_gsan
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
@pytest.mark.parametrize(
    "dtype,shadow_granularity",
    [(dtype, g) for dtype in (torch.float16, torch.int32, torch.int64) for g in (1, 2, 4, 8) if g <= dtype.itemsize],
    indirect=["shadow_granularity"])
def test_atomic_updates_every_shadow_cell(with_gsan, dtype):
    target = torch.zeros(3, dtype=dtype, device="cuda")
    _byte_pool_atomic_kernel[(1, )](target[1:], num_warps=1)
    torch.testing.assert_close(target, torch.tensor([0, 1, 0], dtype=dtype, device="cuda"))
    for offset in range(0, target.element_size(), with_gsan):
        cell = shadow_cell_from_address(target.data_ptr() + target.element_size() + offset)
        assert cell.write_clock.is_release
        assert cell.write_clock.scope == AtomicScope.GPU
        assert cell.num_reads == 1

    for index in (0, 2):
        for offset in range(0, target.element_size(), with_gsan):
            cell = shadow_cell_from_address(target.data_ptr() + index * target.element_size() + offset)
            assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
            assert cell.num_reads == 0


@gluon.jit
def _mixed_pool_store_kernel(pointers):
    offsets = gl.arange(0, 512, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    for i in gl.static_range(len(pointers)):
        gl.store(pointers[i] + offsets, i + 1)


@requires_gsan
def test_mem_pools_mix_granularities_in_one_kernel(fresh_knobs):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    combinations = [(g, w) for g in (1, 2, 4, 8, 16) for w in (False, True)]
    pools = [create_mem_pool(shadow_granularity=g, write_once=w) for g, w in combinations]
    tensors = []
    for pool in pools:
        with torch.cuda.use_mem_pool(pool):
            tensors.append(torch.zeros(512, dtype=torch.uint8, device="cuda"))
    _mixed_pool_store_kernel[(1, )](tuple(tensors), num_warps=1)
    for i, (tensor, (granularity, write_once)) in enumerate(zip(tensors, combinations)):
        torch.testing.assert_close(tensor, torch.full_like(tensor, i + 1))
        for offset in range(0, tensor.numel(), granularity):
            cell = shadow_cell_from_address(tensor.data_ptr() + offset)
            assert (cell if write_once else cell.write_clock).epoch != 0


@requires_gsan
def test_cuda_graph_capture_is_rejected(with_gsan):
    target = torch.zeros(max(1, with_gsan // 4), dtype=torch.int32, device="cuda")
    scratch = torch.zeros_like(target)
    load = _load_wide_i32 if with_gsan == 16 else load_one_i32
    load[(1, )](target, scratch, num_warps=1)
    torch.cuda.synchronize()

    cuda_utils = triton.runtime.driver.active.utils
    assert not cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.zero_()
        assert cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)
        with pytest.raises(RuntimeError, match="GSan does not support (CUDA|HIP) graph capture"):
            load[(1, )](target, scratch, num_warps=1)

    assert not cuda_utils.is_stream_capturing(torch.cuda.current_stream().cuda_stream)


@triton.jit
def _pdl_producer_kernel(payload_ptr):
    pid = tl.program_id(0)
    tl.store(payload_ptr + pid, 1000 + pid)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_consumer_kernel(payload_ptr, result_ptr):
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    value = tl.load(payload_ptr + pid)
    tl.store(result_ptr + pid, value)


@triton.jit
def _pdl_stage_kernel(payload_ptr, INDEX: tl.constexpr, WAIT_PREDECESSOR: tl.constexpr):
    if WAIT_PREDECESSOR:
        tl.extra.cuda.gdc_wait()
    tl.store(payload_ptr + INDEX, 1000 + INDEX)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_two_back_consumer_kernel(payload_ptr, result_ptr):
    value = tl.load(payload_ptr)
    tl.store(result_ptr, value)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_programmatic_dependent_launch_wait_synchronizes_vector_clocks(with_gsan, capfd):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.zeros(2, dtype=dtype, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(2, )](payload, num_warps=1)
    compiled = _pdl_consumer_kernel[(2, )](payload, result, num_warps=1, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.tensor([1000, 1001], device="cuda", dtype=dtype))
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    assert "red.relaxed.gpu.max.u32" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_normal_launch_after_programmatic_dependent_launch_acquires_all_predecessors(with_gsan, capfd):
    payload = torch.zeros(2, dtype=torch.int32, device="cuda")
    result = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _pdl_stage_kernel[(1, )](payload, INDEX=0, WAIT_PREDECESSOR=False, num_warps=1)
    _pdl_stage_kernel[(1, )](payload, INDEX=1, WAIT_PREDECESSOR=True, num_warps=1, launch_pdl=True)
    _pdl_two_back_consumer_kernel[(1, )](payload, result, num_warps=1)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.tensor([1000], dtype=torch.int32, device="cuda"))
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_ws_completion_default(out_ptr, layout: gl.constexpr):
    offsets = gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_completion_worker(out_ptr, layout: gl.constexpr):
    offsets = 128 + gl.arange(0, 128, layout=layout)
    gl.store(out_ptr + offsets, offsets)


@gluon.jit
def _gluon_ws_pdl_wait_default(payload_ptr, result_ptr, layout: gl.constexpr):
    pass


@gluon.jit
def _gluon_ws_pdl_wait_worker(payload_ptr, result_ptr, layout: gl.constexpr):
    tl.extra.cuda.gdc_wait()
    offsets = gl.arange(0, 128, layout=layout)
    values = gl.load(payload_ptr + offsets)
    gl.store(result_ptr + offsets, values)


@gluon.jit(noinline=True)
def _gluon_ws_pdl_wait_noinline_worker(payload_ptr, result_ptr, layout: gl.constexpr):
    tl.extra.cuda.gdc_wait()
    offsets = gl.arange(0, 128, layout=layout)
    values = gl.load(payload_ptr + offsets)
    gl.store(result_ptr + offsets, values)


@gluon.jit
def _gluon_ws_pdl_wait_kernel(payload_ptr, result_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_pdl_wait_default, (payload_ptr, result_ptr, layout)),
        (_gluon_ws_pdl_wait_worker, (payload_ptr, result_ptr, layout)),
    ], [4], [24])


@gluon.jit
def _gluon_ws_pdl_wait_noinline_kernel(payload_ptr, result_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_pdl_wait_default, (payload_ptr, result_ptr, layout)),
        (_gluon_ws_pdl_wait_noinline_worker, (payload_ptr, result_ptr, layout)),
    ], [4], [24])


@gluon.jit
def _gluon_ws_completion_kernel(out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [4], [0])
    gl.warp_specialize([
        (_gluon_ws_completion_default, (out_ptr, layout)),
        (_gluon_ws_completion_worker, (out_ptr, layout)),
    ], [4], [24])


@pytest.mark.skipif(not (is_cuda() or is_hip_cdna5()), reason="requires CUDA or MI450 warp specialization")
def test_gluon_warp_specialize_completes(with_gsan):
    expected = torch.arange(256, dtype=torch.int32, device="cuda")

    out = torch.full((256, ), -1, dtype=torch.int32, device="cuda")
    compiled = _gluon_ws_completion_kernel[(1, )](out, num_warps=4)
    if is_cuda():
        assert 'asm sideeffect "barrier.sync $0, $1;"' in compiled.asm["llir"]
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_programmatic_dependent_launch_wait_inside_warp_specialize(with_gsan, capfd):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.empty((128, ), dtype=dtype, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(128, )](payload, num_warps=1)
    compiled = _gluon_ws_pdl_wait_kernel[(1, )](payload, result, num_warps=4, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.arange(1000, 1128, device="cuda", dtype=dtype))
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="PDL requires SM90 or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_programmatic_dependent_launch_wait_inside_noinline_warp_specialize(with_gsan, capfd):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.empty((128, ), dtype=dtype, device="cuda")
    result = torch.full_like(payload, -1)

    _pdl_producer_kernel[(128, )](payload, num_warps=1)
    compiled = _gluon_ws_pdl_wait_noinline_kernel[(1, )](payload, result, num_warps=4, launch_pdl=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(result, torch.arange(1000, 1128, device="cuda", dtype=dtype))
    assert "tt.call" in compiled.asm["ttgir"]
    assert "griddepcontrol.wait" in compiled.asm["ptx"]
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_ws_noinline_default(out_ptr, layout: gl.constexpr):
    pass


@gluon.jit(noinline=True)
def _gluon_ws_noinline_worker(out_ptr, layout: gl.constexpr):
    pass


@gluon.jit
def _gluon_ws_noinline_kernel(out_ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    gl.warp_specialize(
        [
            (_gluon_ws_noinline_default, (out_ptr, layout)),
            (_gluon_ws_noinline_worker, (out_ptr, layout)),
        ],
        [4],
        [24],
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
def test_gluon_two_cta_warp_specialize_noinline_call(with_gsan):
    out = torch.full((256, ), -1, dtype=torch.int32, device="cuda")

    compiled = _gluon_ws_noinline_kernel[(1, )](out, num_warps=4, num_ctas=2)
    assert "tt.call" in compiled.asm["ttgir"]
    torch.cuda.synchronize()
    assert torch.all(out == -1).item()


@gluon.jit
def _gluon_cluster_barrier_sync_kernel(payload_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload0_ptrs = payload_ptr + offsets * 0
    payload1_ptrs = payload_ptr + 1 + offsets * 0
    out0_ptrs = out_ptr + offsets * 0
    out1_ptrs = out_ptr + 1 + offsets * 0
    gl.store(payload0_ptrs, 1, mask=offsets == 0)
    gl.store(payload1_ptrs, 2, mask=offsets == 128)
    gl.barrier(cluster=True)
    value0 = gl.load(payload0_ptrs, mask=offsets == 128, other=0)
    value1 = gl.load(payload1_ptrs, mask=offsets == 0, other=0)
    gl.store(out0_ptrs, value0, mask=offsets == 128)
    gl.store(out1_ptrs, value1, mask=offsets == 0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_gluon_cluster_barrier_synchronizes_vector_clocks(with_gsan):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.zeros(2, dtype=dtype, device="cuda")
    out = torch.full((2, ), -1, dtype=dtype, device="cuda")

    _gluon_cluster_barrier_sync_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.tensor([1, 2], dtype=dtype, device="cuda"))
    for offset in range(2):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_atomic_cluster_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + offsets * 0
    out_ptrs = out_ptr + offsets * 0
    gl.store(payload_ptrs, 1, mask=offsets == 0)
    gl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
    value = gl.load(payload_ptrs, mask=offsets == 128, other=0)
    gl.store(out_ptrs, value, mask=offsets == 128)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_gluon_cluster_synchronizing_atomic_synchronizes_vector_clocks(with_gsan):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _gluon_atomic_cluster_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    assert counter.item() == 1
    assert out.item() == 1
    payload_cell = shadow_cell_from_address(payload.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch
    consumer_tid = payload_cell.read_clocks[0].thread_id
    assert consumer_tid != producer_tid
    consumer_state = thread_state_from_smid(consumer_tid)
    assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_ws_cluster_barrier_partition(payload_ptr, out_ptr, partition_offset: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload0_ptrs = payload_ptr + partition_offset + offsets * 0
    payload1_ptrs = payload_ptr + partition_offset + 1 + offsets * 0
    out0_ptrs = out_ptr + partition_offset + offsets * 0
    out1_ptrs = out_ptr + partition_offset + 1 + offsets * 0
    gl.store(payload0_ptrs, partition_offset + 1, mask=offsets == 0)
    gl.store(payload1_ptrs, partition_offset + 2, mask=offsets == 128)
    gl.barrier(cluster=True)
    value0 = gl.load(payload0_ptrs, mask=offsets == 128, other=0)
    value1 = gl.load(payload1_ptrs, mask=offsets == 0, other=0)
    gl.store(out0_ptrs, value0, mask=offsets == 128)
    gl.store(out1_ptrs, value1, mask=offsets == 0)


@gluon.jit
def _gluon_ws_cluster_barrier_kernel(payload_ptr, out_ptr):
    gl.warp_specialize([
        (_gluon_ws_cluster_barrier_partition, (payload_ptr, out_ptr, 0)),
        (_gluon_ws_cluster_barrier_partition, (payload_ptr, out_ptr, 2)),
    ], [4])


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_gluon_cluster_barriers_in_warp_specialize_synchronize_vector_clocks(with_gsan):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.zeros(4, dtype=dtype, device="cuda")
    out = torch.full((4, ), -1, dtype=dtype, device="cuda")

    _gluon_ws_cluster_barrier_kernel[(1, )](payload, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1, 5, dtype=dtype, device="cuda"))
    for offset in range(4):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_mbarrier_initial_empty_phase_kernel(out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=1)
    offsets = gl.arange(0, 256, data_layout)
    gl.store(out_ptr + offsets * 0, 1, mask=offsets == 0)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_gluon_mbarrier_initial_empty_phase_is_ready(with_gsan):
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    compiled = _gluon_mbarrier_initial_empty_phase_kernel[(1, )](out, num_warps=4, num_ctas=2)
    assert "__triton_gsan_mbarrier_wait" in compiled.asm["llir"]
    torch.cuda.synchronize()

    assert out.item() == 1


@gluon.jit
def _gluon_mbarrier_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=1)
    offsets = gl.arange(0, 256, data_layout)
    for iteration in range(4):
        payload_ptrs = payload_ptr + iteration + offsets * 0
        out_ptrs = out_ptr + iteration + offsets * 0
        gl.store(payload_ptrs, iteration + 1, mask=offsets == 128)
        # Mbarriers only propagate completed epochs. A release closes the
        # writer's epoch; direct current-epoch handoffs are tested as failures.
        gl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        hopper.mbarrier.arrive(barrier, count=1)
        hopper.mbarrier.wait(barrier, phase=iteration % 2)
        value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
        gl.store(out_ptrs, value, mask=offsets == 0)
        hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_gluon_mbarrier_wait_preserves_completed_epochs(with_gsan):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.zeros(4, dtype=dtype, device="cuda")
    counter = torch.zeros(1, dtype=dtype, device="cuda")
    out = torch.full((4, ), -1, dtype=dtype, device="cuda")

    _gluon_mbarrier_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1, 5, dtype=dtype, device="cuda"))
    for offset in range(4):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_ws_mbarrier_partition(payload_ptr, counter_ptr, out_ptr, barrier, index: gl.constexpr):
    data_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])

    offsets = gl.arange(0, 256, data_layout)
    payload_ptrs = payload_ptr + index + offsets * 0
    out_ptrs = out_ptr + index + offsets * 0
    gl.store(payload_ptrs, index + 1, mask=offsets == 128)
    gl.atomic_add(counter_ptr + index, 1, sem="release", scope="gpu")
    hopper.mbarrier.arrive(barrier, count=1)
    hopper.mbarrier.wait(barrier, phase=0)
    value = gl.load(payload_ptrs, mask=offsets == 0, other=0)
    gl.store(out_ptrs, value, mask=offsets == 0)


@gluon.jit
def _gluon_ws_mbarrier_sync_kernel(payload_ptr, counter_ptr, out_ptr):
    barriers = hopper.mbarrier.allocate_mbarrier(batch=2, two_ctas=True)
    hopper.mbarrier.init(barriers.index(0), count=1)
    hopper.mbarrier.init(barriers.index(1), count=1)
    gl.warp_specialize([
        (_gluon_ws_mbarrier_partition, (payload_ptr, counter_ptr, out_ptr, barriers.index(0), 0)),
        (_gluon_ws_mbarrier_partition, (payload_ptr, counter_ptr, out_ptr, barriers.index(1), 1)),
    ], [4])
    hopper.mbarrier.invalidate(barriers.index(0))
    hopper.mbarrier.invalidate(barriers.index(1))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("This synchronization test uses sub-16-byte scalar accesses")
def test_gluon_mbarrier_sync_in_warp_specialized_partitions(with_gsan):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    payload = torch.zeros(2, dtype=dtype, device="cuda")
    counter = torch.zeros(2, dtype=dtype, device="cuda")
    out = torch.full((2, ), -1, dtype=dtype, device="cuda")

    _gluon_ws_mbarrier_sync_kernel[(1, )](payload, counter, out, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.tensor([1, 2], dtype=dtype, device="cuda"))
    for offset in range(2):
        payload_cell = shadow_cell_from_address(payload.data_ptr() + offset * payload.element_size())
        producer_tid = payload_cell.write_clock.thread_id
        producer_epoch = payload_cell.write_clock.epoch
        consumer_tid = payload_cell.read_clocks[0].thread_id
        assert consumer_tid != producer_tid
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[producer_tid] >= producer_epoch


@gluon.jit
def _gluon_mbarrier_repeated_phases_kernel(markers, flags, iterations, EXPECT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1]])
    offsets = gl.arange(0, 256, layout)
    cta = offsets // 128
    elected = offsets % 128 == 0
    barrier = hopper.mbarrier.allocate_mbarrier(two_ctas=True)
    hopper.mbarrier.init(barrier, count=1)
    # Keep a release snapshot live throughout the loop, as fused-communication
    # completion flags do in production kernels.
    gl.atomic_xchg(flags + cta, 1, mask=elected, sem="release", scope="gpu")
    gl.store(markers + cta, 1, mask=elected)
    for iteration in range(iterations):
        if EXPECT:
            hopper.mbarrier.expect(barrier, 0)
        else:
            hopper.mbarrier.arrive(barrier)
        hopper.mbarrier.wait(barrier, iteration % 2)
        hopper.cluster.barrier(relaxed=True)
    gl.store(markers + 2 + cta, 1, mask=elected)
    hopper.mbarrier.invalidate(barrier)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.parametrize("expect", [False, True], ids=["arrive", "expect"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_gluon_mbarrier_repeated_phases_reuse_epochs_and_snapshots(with_gsan, expect):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    markers = torch.zeros(4, dtype=dtype, device="cuda")
    flags = torch.zeros(2, dtype=dtype, device="cuda")
    _gluon_mbarrier_repeated_phases_kernel[(1, )](markers, flags, 65536, expect, num_warps=4, num_ctas=2)
    torch.cuda.synchronize()

    clocks = [shadow_cell_from_address(markers.data_ptr() + i * markers.element_size()).write_clock for i in range(4)]
    assert clocks[0].thread_id != clocks[1].thread_id
    for cta in range(2):
        assert clocks[cta] == clocks[cta + 2]
        state = thread_state_from_smid(clocks[cta].thread_id)
        release = shadow_cell_from_address(flags.data_ptr() + cta * flags.element_size()).write_clock
        assert release.is_release
        assert release.thread_id == state.thread_id
        # Publishing once and importing the peer's completed epoch can create
        # a few snapshots, but the phase count must not control buffer usage.
        assert 0 <= state.clock_buffer_head - release.epoch <= 4
        assert state.vector_clock[state.thread_id] == clocks[cta].epoch
    consumer = thread_state_from_smid(clocks[0].thread_id)
    assert consumer.vector_clock[clocks[1].thread_id] == clocks[1].epoch - 1


@triton.jit
def _gsan_warm_all_sms_kernel(markers, counter, num_sms: tl.constexpr):
    smid = tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], tl.int32, is_pure=False, pack=1)
    tl.store(markers + smid, 1)
    tl.atomic_add(counter, 1, sem="relaxed", scope="gpu")
    # GSan reserves all shared memory, so these resident CTAs occupy distinct
    # SMs. Keep them alive until every SM has advanced its local epoch.
    atomic_poll(counter, num_sms)


@gluon.jit
def _gluon_mbarrier_transitive_clock_kernel(markers):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1], [2]])
    pairs01 = gl.allocate_shared_memory(gl.int64, [2], hopper.mbarrier.MBarrierLayout([[0], [1]]))
    pairs02 = gl.allocate_shared_memory(gl.int64, [2], hopper.mbarrier.MBarrierLayout([[1], [0]]))
    hopper.mbarrier.init(pairs01, count=1)
    hopper.mbarrier.init(pairs02, count=1)
    offsets = gl.arange(0, 512, layout)
    gl.store(markers + offsets // 128, 1, mask=offsets % 128 == 0)
    # CTA 2 first acquires CTA 3's completed epoch, then passes it to CTA 0.
    hopper.mbarrier.arrive(pairs01)
    hopper.mbarrier.wait(pairs01, 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.arrive(pairs02)
    hopper.mbarrier.wait(pairs02, 0)
    hopper.cluster.barrier(relaxed=True)
    hopper.mbarrier.invalidate(pairs01)
    hopper.mbarrier.invalidate(pairs02)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires Hopper or newer")
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_gluon_mbarrier_preserves_transitively_imported_clocks(with_gsan):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    warm = torch.zeros(num_sms, dtype=dtype, device="cuda")
    counter = torch.zeros(1, dtype=dtype, device="cuda")
    markers = torch.zeros(4, dtype=dtype, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _gsan_warm_all_sms_kernel[(num_sms, )](warm, counter, num_sms, num_warps=1)
    stream.synchronize()
    # A host wait does not import this independent stream's GSan clock.
    _gluon_mbarrier_transitive_clock_kernel[(1, )](markers, num_warps=4, num_ctas=4)
    torch.cuda.synchronize()

    clocks = [shadow_cell_from_address(markers.data_ptr() + i * markers.element_size()).write_clock for i in range(4)]
    assert len({clock.thread_id for clock in clocks}) == 4
    consumer = thread_state_from_smid(clocks[0].thread_id)
    for clock in clocks[1:]:
        smid = clock.thread_id % num_sms
        previous = shadow_cell_from_address(warm.data_ptr() + smid * warm.element_size()).write_clock
        assert previous.thread_id == clock.thread_id
        assert previous.epoch + 1 == clock.epoch
        assert consumer.vector_clock[clock.thread_id] == previous.epoch
    # The zero CTA-layout bases predicate the waits onto the pair leaders.
    # CTA 1 acquires CTA 3 in the second phase, but neither CTA 1 nor CTA 3
    # executes a wait that could acquire CTA 2's newly completed epoch.
    nonleader = thread_state_from_smid(clocks[1].thread_id)
    assert nonleader.vector_clock[clocks[3].thread_id] == clocks[3].epoch - 1
    for cta in (1, 3):
        state = thread_state_from_smid(clocks[cta].thread_id)
        assert state.vector_clock[clocks[2].thread_id] < clocks[2].epoch - 1


@gluon.jit
def _gsan_empty_kernel(out_ptr):
    cga_layout: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    offsets = gl.arange(0, 4, layout=gl.BlockedLayout([4], [32], [4], [0], cga_layout=cga_layout))
    gl.store(out_ptr + offsets, 0)


@pytest.mark.skipif(not is_cuda(), reason="GSan requires CUDA")
@pytest.mark.parametrize("num_ctas", [1, 2])
@pytest.mark.parametrize("max_occupancy", [None, 4])
def test_gsan_limits_occupancy(with_gsan, num_ctas, max_occupancy):
    if num_ctas > 1 and not is_hopper_or_newer():
        pytest.skip("CTA clusters require Hopper or newer")
    out = torch.empty(4, dtype=torch.int32, device="cuda")
    compiled = _gsan_empty_kernel.warmup(out, grid=(1, ), num_ctas=num_ctas, max_occupancy=max_occupancy)
    assert compiled.metadata.max_occupancy == 1


@triton.jit
def _gsan_pipelined_matmul_kernel(a_ptr, b_ptr, c_ptr, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * BN + rn[None, :])
        acc += tl.dot(a, b)
    tl.atomic_add(c_ptr + rm[:, None] * BN + rn[None, :], acc)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Checks the gfx950 matmul buffer/async lowering")
def test_gsan_amd_uses_buffer_ops_and_async_copy(with_gsan):
    # Buffer conversion and pipelined async copies both receive GSan checks.
    a = torch.randn(64, 256, dtype=torch.float16, device="cuda")
    b = torch.randn(256, 64, dtype=torch.float16, device="cuda")
    c = torch.zeros(64, 64, dtype=torch.float32, device="cuda")
    compiled = _gsan_pipelined_matmul_kernel.warmup(a, b, c, 256, BM=64, BN=64, BK=64, num_stages=2, grid=(1, ))
    ttgir = compiled.asm["ttgir"]
    assert "amdg.buffer_atomic_rmw " in ttgir
    assert "amdg.buffer_load " in ttgir or "amdg.buffer_load_to_local" in ttgir


@gluon.jit
def _gluon_amd_buffer_load_to_shared_kernel(ptr):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0])
    offsets = gl.arange(0, 64, layout)
    smem = gl.allocate_shared_memory(gl.int32, [64], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem, ptr, offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Checks rejected gfx950 buffer-to-LDS lowering")
def test_gsan_amd_rejects_buffer_load_to_local(with_gsan, capfd):
    src = torch.zeros(64, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _gluon_amd_buffer_load_to_shared_kernel.warmup(src, num_warps=1, grid=(1, ))
    captured = capfd.readouterr()
    assert "GSan on AMD cannot check global-memory operation 'amdg.buffer_load_to_local'" in captured.err


@triton.jit
def _gsan_empty_tl_kernel(out_ptr):
    # Wave-size agnostic, unlike the Gluon _gsan_empty_kernel's 32-lane layout.
    tl.store(out_ptr, 0)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD GSan LDS reservation")
def test_gsan_uses_all_available_shared_memory(with_gsan):
    out = torch.empty(4, dtype=torch.int32, device="cuda")
    compiled = _gsan_empty_tl_kernel.warmup(out, grid=(1, ))

    device = triton.runtime.driver.active.get_current_device()
    max_shared = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
    assert compiled.metadata.min_shared_mem == max_shared
    # AMD launches request dynamic LDS on top of the kernel's static LDS, and
    # a total above the limit kills the process, so dynamic = limit - static.
    static_shared = 0
    if is_hip():
        static_shared = int(re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", compiled.asm["amdgcn"]).group(1))
    assert compiled.metadata.shared + static_shared == max_shared
    assert compiled.packed_metadata[2] == compiled.metadata.shared


@triton.jit
def atomic_add_kernel(ptr, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_add(ptr, 1, sem=sem, scope=scope)


@triton.jit
def atomic_load_kernel(ptr, out_ptr, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    value = tl.atomic_load(ptr, sem=sem, scope=scope)
    tl.store(out_ptr, value)


@triton.jit
def atomic_store_kernel(ptr, value, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_store(ptr, value, sem=sem, scope=scope)


@gluon.jit
def gluon_atomic_load_store_kernel(ptr, out_ptr, value):
    gl.atomic_store(ptr, value, sem="relaxed", scope="gpu")
    value = gl.atomic_load(ptr, sem="relaxed", scope="gpu")
    gl.store(out_ptr, value)


@triton.jit
def atomic_cas_kernel(ptr, out_ptr, expect, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    old = tl.atomic_cas(ptr, expect, 2, sem=sem, scope=scope)
    tl.store(out_ptr, old)


@triton.jit
def _scalar_atomic_rmw_cluster_kernel(ptr, out_ptr):
    old = tl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")
    offsets = tl.arange(0, 32)
    tl.store(out_ptr + offsets, old)


@triton.jit
def _scalar_atomic_cas_cluster_kernel(ptr, out_ptr, expected, desired):
    old = tl.atomic_cas(ptr, expected, desired, sem="relaxed", scope="gpu")
    offsets = tl.arange(0, 32)
    tl.store(out_ptr + offsets, old)


def _assert_cluster_scalar_atomic_result(kernel):
    initial = 0x12345678
    target = torch.full((1, ), initial, dtype=torch.int32, device="cuda")
    out = torch.full((32, ), -1, dtype=torch.int32, device="cuda")

    if kernel is _scalar_atomic_rmw_cluster_kernel:
        kernel[(1, )](target, out, num_warps=1, num_ctas=2)
    else:
        kernel[(1, )](target, out, initial, initial + 1, num_warps=1, num_ctas=2)
    torch.cuda.synchronize()

    assert target.item() == initial + 1
    torch.testing.assert_close(out, torch.full_like(out, initial))


@pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                    reason="scalar multi-CTA atomics require Hopper+ and are unsupported on sm12x")
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_scalar_atomic_rmw_cluster_result_broadcast(with_gsan):
    _assert_cluster_scalar_atomic_result(_scalar_atomic_rmw_cluster_kernel)


@pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                    reason="scalar multi-CTA atomics require Hopper+ and are unsupported on sm12x")
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_scalar_atomic_cas_cluster_result_broadcast(with_gsan):
    _assert_cluster_scalar_atomic_result(_scalar_atomic_cas_cluster_kernel)


@triton.jit
def atomic_poll_kernel(ptr, expect, sem: tl.constexpr, scope: tl.constexpr = "gpu"):
    tl.atomic_poll(ptr, expect, sem=sem, scope=scope)


@triton.jit
def atomic_poll_timeout_kernel(ptr, out_ptr):
    matched = tl.atomic_poll(ptr, 1, timeout_ns=0)
    tl.store(out_ptr, matched)


@gluon.jit
def _atomic_poll_tensor_kernel(ptr, out_ptr, BLOCK: gl.constexpr, STRIDE: gl.constexpr, sem: gl.constexpr,
                               scope: gl.constexpr, TIMEOUT: gl.constexpr):
    offsets = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0]))
    matched = gl.atomic_poll(ptr + offsets * STRIDE, offsets + 1, sem=sem, scope=scope, timeout_ns=TIMEOUT)
    gl.store(out_ptr + offsets, matched)


@gluon.jit
def _atomic_poll_tensor_sync_kernel(payload_ptr, flag_ptr, out_ptr, BLOCK: gl.constexpr, STRIDE: gl.constexpr,
                                    scope: gl.constexpr):
    offsets = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0]))
    pid = gl.program_id(0)
    if pid < 2:
        mask = offsets % 2 == pid
        gl.store(payload_ptr + offsets * STRIDE, offsets + 1000, mask)
        gl.atomic_xchg(flag_ptr + offsets * STRIDE, 1, mask=mask, sem="release", scope=scope)
    else:
        gl.atomic_poll(flag_ptr + offsets * STRIDE, 1, sem="acquire", scope=scope)
        result = gl.load(payload_ptr + offsets * STRIDE)
        gl.store(out_ptr + offsets, result)


@triton.jit
def _cross_sm_atomic_sync_kernel(payload_ptr, flag_ptr, out_ptr, producer_sem: tl.constexpr, consumer_sem: tl.constexpr,
                                 scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
    elif pid == 1:
        atomic_poll(flag_ptr, 1, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@triton.jit
def _cross_sm_atomic_poll_sync_kernel(payload_ptr, flag_ptr, out_ptr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope=scope)
    elif pid == 1:
        tl.atomic_poll(flag_ptr, 1, sem="acquire", scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@triton.jit
def _cross_sm_atomic_load_store_sync_kernel(payload_ptr, flag_ptr, out_ptr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_store(flag_ptr, 1, sem="release", scope=scope)
    elif pid == 1:
        ready = tl.atomic_load(flag_ptr, sem="acquire", scope=scope)
        while ready != 1:
            ready = tl.atomic_load(flag_ptr, sem="acquire", scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@triton.jit
def _masked_atomic_load_store_kernel(ptr, out_ptr, stride: tl.constexpr):
    indices = tl.arange(0, 4)
    offsets = indices * stride
    mask = indices < 2
    values = indices % 2 if ptr.dtype.element_ty == tl.int1 else indices + 10
    tl.atomic_store(ptr + offsets, values, mask=mask, sem="relaxed")
    values = tl.atomic_load(ptr + offsets, mask=mask, sem="relaxed")
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _transitive_atomic_sync_kernel(payload_ptr, flag0_ptr, flag1_ptr, out_ptr, release_sem: tl.constexpr,
                                   acquire_sem: tl.constexpr, scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag0_ptr, 1, sem=release_sem, scope=scope)
    elif pid == 1:
        atomic_poll(flag0_ptr, 1, sem=acquire_sem, scope=scope)
        tl.atomic_xchg(flag1_ptr, 1, sem=release_sem, scope=scope)
    elif pid == 2:
        atomic_poll(flag1_ptr, 1, sem=acquire_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, is_release", ATOMIC_SEMANTIC_CASES)
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_add_updates_atomic_shadow(with_gsan, sem, is_release, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_add_kernel[(1, )](target, sem=sem, scope=scope, num_warps=1)
    assert target.item() == 1

    for byte_offset in range(0, target.element_size(), with_gsan):
        _assert_atomic_rmw_shadow(target.data_ptr() + byte_offset, expected_scope, is_release=is_release)


@requires_gsan
@pytest.mark.parametrize("sem", ["relaxed", "acquire"])
@pytest.mark.parametrize("dtype,shadow_granularity,scope,expected_scope", ATOMIC_LOAD_STORE_SCOPE_CASES,
                         indirect=["shadow_granularity"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_load_only_records_read(with_gsan, dtype, sem, scope, expected_scope):
    value = 1 if dtype == torch.bool else -37 if dtype == torch.int8 else 197
    target = torch.full((1, ), value, dtype=dtype, device="cuda")
    out = torch.zeros_like(target)

    atomic_load_kernel[(1, )](target, out, sem=sem, scope=scope, num_warps=1)

    assert out.item() == value
    _assert_atomic_read_only_shadow(target.data_ptr(), expected_scope)


@requires_gsan
@pytest.mark.parametrize("op", ["load", "store"])
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, g)
                          for dtype in (torch.int8, torch.float16, torch.int32, torch.float64)
                          for g in (1, 2, 4, 8)
                          if g <= dtype.itemsize], indirect=["shadow_granularity"])
def test_atomic_load_store_vectorized_shadow(with_gsan, op, dtype):

    @gluon.jit
    def kernel(Src, Out, STORE: gl.constexpr):
        x = gl.arange(0, 256, layout=gl.BlockedLayout([8], [THREADS_PER_WARP], [1], [0]))
        mask = x // 8 % 2 == 0
        if STORE:
            values = gl.load(Src + x)
            gl.atomic_store(Out + x, values, mask=mask, sem="release")
        else:
            values = gl.atomic_load(Src + x, mask=mask, sem="acquire")
            gl.store(Out + x, values, mask=mask)

    src = (torch.arange(256, device="cuda") - 64).to(dtype)
    out = torch.full_like(src, -1)
    compiled = kernel[(1, )](src, out, op == "store", num_warps=1)

    mask = torch.arange(256, device="cuda") // 8 % 2 == 0
    assert torch.equal(out, torch.where(mask, src, -1))
    if is_cuda():
        suffix = "b64" if dtype == torch.float64 else "v2.b32"
        opcode = "st" if op == "store" else "ld"
        assert f"{opcode}.relaxed.gpu.global.{suffix}" in compiled.asm["ptx"]
    target = out if op == "store" else src
    target_ptr = target.data_ptr()
    granularity = get_shadow_granularity(target_ptr)
    for byte_offset in range(0, src.numel() * dtype.itemsize, granularity):
        address = target_ptr + byte_offset
        if byte_offset // dtype.itemsize // 8 % 2 == 0:
            if op == "store":
                cell = shadow_cell_from_address(address)
                assert cell.write_clock.scope == AtomicScope.GPU
                assert cell.write_clock.is_release
                assert cell.num_reads == 0
            else:
                _assert_atomic_read_only_shadow(address, AtomicScope.GPU)
        else:
            cell = shadow_cell_from_address(address)
            assert cell.num_reads == 0
            assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)


@requires_gsan
@pytest.mark.parametrize("sem, is_release", [("relaxed", False), ("release", True)])
@pytest.mark.parametrize("dtype,shadow_granularity,scope,expected_scope", ATOMIC_LOAD_STORE_SCOPE_CASES,
                         indirect=["shadow_granularity"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_store_only_records_write(with_gsan, dtype, sem, is_release, scope, expected_scope):
    target = torch.zeros(1, dtype=dtype, device="cuda")

    value = 1 if dtype == torch.bool else -37 if dtype == torch.int8 else 197
    atomic_store_kernel[(1, )](target, value, sem=sem, scope=scope, num_warps=1)

    assert target.item() == value
    _assert_atomic_store_only_shadow(target.data_ptr(), expected_scope, is_release=is_release)


@requires_gsan
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, min(4, dtype.itemsize)) for dtype in ATOMIC_LOAD_STORE_TYPES],
                         indirect=["shadow_granularity"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_gluon_atomic_load_store_updates_shadow(with_gsan, dtype):
    target = torch.zeros(1, dtype=dtype, device="cuda")
    out = torch.zeros_like(target)

    expected = 1 if dtype == torch.bool else 17
    gluon_atomic_load_store_kernel[(1, )](target, out, expected, num_warps=1)

    assert target.item() == expected
    assert out.item() == expected
    cell = shadow_cell_from_address(target.data_ptr())
    assert cell.write_clock.scope == AtomicScope.GPU
    assert cell.num_reads == 1


@requires_gsan
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, g) for dtype in ATOMIC_LOAD_STORE_TYPES for g in (1, 2, 4, 8) if g <= dtype.itemsize],
                         indirect=["shadow_granularity"])
def test_masked_atomic_load_store_only_updates_active_lanes(with_gsan, dtype):
    # Byte accesses start at an odd offset so the value checks also catch
    # overwrites of neighboring bytes.
    itemsize = dtype.itemsize
    target = torch.empty(4, dtype=dtype, device="cuda")
    granularity = get_shadow_granularity(target.data_ptr())
    stride = max(1, granularity // itemsize)
    offset = 1 if itemsize == 1 else 0
    target.resize_(4 * stride + offset).zero_()
    out = torch.full_like(target, True if dtype == torch.bool else 42)

    _masked_atomic_load_store_kernel[(1, )](target[offset:], out[offset:], stride, num_warps=1)

    expected_target = torch.zeros_like(target)
    expected_out = torch.full_like(out, True if dtype == torch.bool else 42)
    expected_target[offset] = expected_out[offset] = 0 if dtype == torch.bool else 10
    expected_target[offset + stride] = expected_out[offset + stride] = 1 if dtype == torch.bool else 11
    torch.testing.assert_close(target, expected_target)
    torch.testing.assert_close(out, expected_out)
    for index in range(4):
        for byte_offset in range(0, itemsize, granularity):
            cell = shadow_cell_from_address(target[offset + index * stride].data_ptr() + byte_offset)
            if index < 2:
                assert cell.num_reads == 1
                assert cell.write_clock.scope == AtomicScope.GPU
            else:
                assert cell.num_reads == 0
                assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, _", ATOMIC_SEMANTIC_CASES)
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_cas_failed_only_records_read(with_gsan, sem, _, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_cas_kernel[(1, )](target, out, expect=1, sem=sem, scope=scope, num_warps=1)

    assert target.item() == 0
    assert out.item() == 0

    for byte_offset in range(0, target.element_size(), with_gsan):
        _assert_atomic_read_only_shadow(target.data_ptr() + byte_offset, expected_scope)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem", ["relaxed", "acquire"])
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, min(4, dtype.itemsize)) for dtype in (torch.int16, torch.int32, torch.int64)],
                         indirect=["shadow_granularity"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_poll_only_records_read(with_gsan, dtype, sem, scope, expected_scope):
    target = torch.ones(1, dtype=dtype, device="cuda")

    atomic_poll_kernel[(1, )](target, 1, sem=sem, scope=scope, num_warps=4)

    for byte_offset in range(0, target.element_size(), with_gsan):
        _assert_atomic_read_only_shadow(target.data_ptr() + byte_offset, expected_scope)


@requires_gsan
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_poll_timeout_does_not_record_read(with_gsan):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.ones(1, dtype=torch.bool, device="cuda")

    atomic_poll_timeout_kernel[(1, )](target, out, num_warps=4)

    assert not out.item()
    cell = shadow_cell_from_address(target.data_ptr())
    assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
    assert cell.num_reads == 0


# Cover sub-warp and multi-warp tensors without crossing every size with every dtype.
@requires_gsan
@pytest.mark.parametrize("dtype,shadow_granularity,block_size", [(torch.int16, 2, 16), (torch.int32, 4, 256),
                                                                 (torch.int64, 4, 256)],
                         indirect=["shadow_granularity"])
@pytest.mark.parametrize("sem", ["relaxed", "acquire"])
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("timeout", [None, 0])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_poll_tensor_only_records_matched_reads(with_gsan, block_size, dtype, sem, scope, expected_scope,
                                                       timeout):
    # Separate shadow cells let us check successful and timed-out elements independently.
    stride = max(1, with_gsan // torch.empty((), dtype=dtype).element_size())
    target = torch.zeros(block_size * stride, dtype=dtype, device="cuda")
    expected = torch.arange(1, block_size + 1, dtype=dtype, device="cuda")
    target[::stride] = expected
    if timeout is not None:
        target[::2 * stride] = 0
    out = torch.empty(block_size, dtype=torch.bool, device="cuda")

    _atomic_poll_tensor_kernel[(1, )](target, out, block_size, stride, sem, scope, timeout, num_warps=4)
    torch.testing.assert_close(out, target[::stride] == expected)

    for index in range(block_size):
        for byte_offset in range(0, target.element_size(), with_gsan):
            address = target.data_ptr() + index * stride * target.element_size() + byte_offset
            if timeout is None or index % 2:
                _assert_atomic_read_only_shadow(address, expected_scope)
            else:
                cell = shadow_cell_from_address(address)
                assert cell.write_clock == ScalarClock(0, 0, AtomicScope.NON_ATOMIC)
                assert cell.num_reads == 0


@requires_gsan
@pytest.mark.parametrize("block_size", [16, 256])
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_poll_tensor_acquires_all_producers(with_gsan, capfd, block_size, scope, expected_scope):
    stride = max(1, with_gsan // 4)
    payload = torch.zeros(block_size * stride, dtype=torch.int32, device="cuda")
    flags = torch.zeros_like(payload)
    out = torch.full((block_size, ), -1, dtype=torch.int32, device="cuda")

    _atomic_poll_tensor_sync_kernel[(3, )](payload, flags, out, block_size, stride, scope, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.arange(1000, 1000 + block_size, dtype=torch.int32, device="cuda"))
    for index in range(block_size):
        _assert_cross_sm_sync(payload[index * stride:], flags[index * stride:], expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_poll_acquire_synchronizes_cross_sm(with_gsan, capfd, scope, expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _cross_sm_atomic_poll_sync_kernel[(2, )](payload, flag, out, scope=scope, num_warps=4)
    torch.cuda.synchronize()

    assert out.item() == 1000
    _assert_cross_sm_sync(payload, flag, expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, min(4, dtype.itemsize)) for dtype in ATOMIC_LOAD_STORE_TYPES],
                         indirect=["shadow_granularity"])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_load_store_synchronizes_cross_sm(with_gsan, capfd, scope, expected_scope, dtype):
    if dtype == torch.bool and is_cuda():
        # LLVM loop peeling duplicates aligned barriers and hangs the kernel.
        pytest.xfail("LLVM loop peeling breaks convergence of atomic-load result broadcasts")

    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=dtype, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")

    _cross_sm_atomic_load_store_sync_kernel[(2, )](payload, flag, out, scope=scope, num_warps=1)
    torch.cuda.synchronize()

    assert out.item() == 1000
    _assert_cross_sm_sync(payload, flag, expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES)
@pytest.mark.parametrize("sem, is_release", ATOMIC_SEMANTIC_CASES)
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_cas_success_updates_atomic_shadow(with_gsan, sem, is_release, scope, expected_scope):
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    atomic_cas_kernel[(1, )](target, out, expect=0, sem=sem, scope=scope, num_warps=1)

    assert target.item() == 2
    assert out.item() == 0

    for byte_offset in range(0, target.element_size(), with_gsan):
        _assert_atomic_rmw_shadow(target.data_ptr() + byte_offset, expected_scope, is_release=is_release)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.parametrize("producer_sem", RELEASE_SEMANTIC_CASES)
@pytest.mark.parametrize("consumer_sem", ACQUIRE_SEMANTIC_CASES)
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_release_acquire_synchronizes_cross_sm(with_gsan, capfd, producer_sem, consumer_sem, scope,
                                                      expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _cross_sm_atomic_sync_kernel[(2, )](
        payload,
        flags,
        out,
        producer_sem=producer_sem,
        consumer_sem=consumer_sem,
        scope=scope,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert out.item() == 1000

    _assert_cross_sm_sync(payload, flags, expected_scope)
    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _cas_handoff_kernel(payload_ptr, flag_ptr, arrived_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        # Wait until the consumer is resident, so the two blocks hold distinct
        # GSan identities (slots).
        atomic_poll(arrived_ptr, 1)
        tl.store(payload_ptr, 42)
        tl.atomic_cas(flag_ptr, 0, 1, sem="release")
    else:
        tl.atomic_add(arrived_ptr, 1, sem="relaxed")
        # CAS 1 -> 1 succeeds only after the producer published the flag; the
        # successful acquire CAS orders the payload read after the store.
        seen = tl.atomic_cas(flag_ptr, 1, 1, sem="acquire")
        while seen != 1:
            seen = tl.atomic_cas(flag_ptr, 1, 1, sem="acquire")
        tl.store(out_ptr, tl.load(payload_ptr))


@requires_gsan
def test_atomic_cas_handoff_synchronizes_cross_sm(with_gsan, capfd):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    arrived = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _cas_handoff_kernel[(2, )](payload, flag, arrived, out, num_warps=1)
    torch.cuda.synchronize()

    assert out.item() == 42
    _assert_no_gsan_runtime_output(capfd)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.parametrize("release_sem", RELEASE_SEMANTIC_CASES)
@pytest.mark.parametrize("acquire_sem", ACQUIRE_SEMANTIC_CASES)
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_release_acquire_transitively_synchronizes_cross_sm(with_gsan, capfd, release_sem, acquire_sem, scope,
                                                                   expected_scope):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.full((1, ), -1, dtype=torch.int32, device="cuda")
    _transitive_atomic_sync_kernel[(3, )](
        payload,
        flag0,
        flag1,
        out,
        release_sem=release_sem,
        acquire_sem=acquire_sem,
        scope=scope,
        num_warps=1,
    )
    torch.cuda.synchronize()

    assert out.item() == 1000

    payload_cell = shadow_cell_from_address(payload.data_ptr())
    flag1_cell = shadow_cell_from_address(flag1.data_ptr())
    producer_tid = payload_cell.write_clock.thread_id
    producer_epoch = payload_cell.write_clock.epoch

    relay_state = thread_state_from_smid(flag1_cell.write_clock.thread_id)
    snapshot_idx = _clock_buffer_snapshot_idx(flag1_cell.write_clock.epoch, relay_state, producer_tid)

    assert flag1_cell.write_clock.scope == expected_scope
    assert flag1_cell.write_clock.is_release
    assert relay_state.clock_buffer[snapshot_idx] >= producer_epoch

    consumer_tid = payload_cell.read_clocks[0].thread_id
    consumer_state = thread_state_from_smid(consumer_tid)

    assert consumer_state.vector_clock[producer_tid] >= producer_epoch

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _release_rmw_chain_kernel(payload_ptr, counter_ptr, out_ptr, scope: tl.constexpr, NUM_WRITERS: tl.constexpr):
    pid = tl.program_id(0)
    if pid == NUM_WRITERS:
        atomic_poll(counter_ptr, NUM_WRITERS, sem="acquire", scope=scope)
        idx = tl.arange(0, triton.next_power_of_2(NUM_WRITERS))
        value = tl.load(payload_ptr + idx, mask=idx < NUM_WRITERS)
        tl.store(out_ptr + idx, value, mask=idx < NUM_WRITERS)
    else:
        tl.store(payload_ptr + pid, 1000 + pid)
        tl.atomic_add(counter_ptr, 1, sem="release", scope=scope)


@requires_gsan
@pytest.mark.parametrize("scope, expected_scope", ATOMIC_SCOPE_CASES[1:])
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_atomic_release_rmw_chain_synchronizes_all_writers(with_gsan, capfd, scope, expected_scope):
    dtype = torch.int64 if with_gsan == 8 else torch.int32
    num_writers = 3
    payload = torch.zeros(num_writers, dtype=dtype, device="cuda")

    counter = torch.zeros(1, dtype=dtype, device="cuda")
    out = torch.full((num_writers, ), -1, dtype=dtype, device="cuda")
    _release_rmw_chain_kernel[(num_writers + 1, )](
        payload,
        counter,
        out,
        scope=scope,
        NUM_WRITERS=num_writers,
        num_warps=1,
    )
    torch.cuda.synchronize()

    expected = torch.arange(1000, 1000 + num_writers, dtype=dtype, device="cuda")
    torch.testing.assert_close(out, expected)

    writer_tids = set()
    for index in range(num_writers):
        payload_cell = shadow_cell_from_address(payload[index].data_ptr())
        writer_tid = payload_cell.write_clock.thread_id
        writer_epoch = payload_cell.write_clock.epoch
        writer_tids.add(writer_tid)
        consumer_tid = payload_cell.read_clocks[0].thread_id
        consumer_state = thread_state_from_smid(consumer_tid)
        assert consumer_state.vector_clock[writer_tid] >= writer_epoch
    assert len(writer_tids) == num_writers

    counter_cell = shadow_cell_from_address(counter.data_ptr())
    assert counter_cell.write_clock.scope == expected_scope
    assert counter_cell.write_clock.is_release

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _ordered_mixed_scope_release_rmw_kernel(payload_ptr, counter_ptr, ready_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
        tl.atomic_xchg(ready_ptr, 1, sem="relaxed", scope="gpu")
    elif pid == 1:
        atomic_poll(ready_ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")


@requires_gsan
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_ordered_mixed_scope_release_rmw_is_allowed(with_gsan, capfd):
    payload = torch.zeros(1, dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, device="cuda")
    _ordered_mixed_scope_release_rmw_kernel[(2, )](payload, counter, ready, num_warps=1)
    torch.cuda.synchronize()

    assert counter.item() == 2

    payload_cell = shadow_cell_from_address(payload.data_ptr())
    counter_cell = shadow_cell_from_address(counter.data_ptr())
    assert counter_cell.write_clock.scope == AtomicScope.SYSTEM
    assert counter_cell.write_clock.is_release

    counter_writer_state = thread_state_from_smid(counter_cell.write_clock.thread_id)
    snapshot_idx = _clock_buffer_snapshot_idx(counter_cell.write_clock.epoch, counter_writer_state,
                                              payload_cell.write_clock.thread_id)
    assert counter_writer_state.clock_buffer[snapshot_idx] >= payload_cell.write_clock.epoch

    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _write_blocks_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(ptr + offsets, 1, mask=mask)


@triton.jit
def _read_reversed_blocks_kernel(ptr, scratch_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    src_pid = tl.num_programs(0) - 1 - pid
    src_offsets = src_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_offsets < n_elements
    value = tl.load(ptr + src_offsets, mask=mask)
    tl.store(scratch_ptr + dst_offsets, value, mask=mask)


@requires_gsan
def test_implicit_stream_ordering(with_gsan):
    block_size = 512 if with_gsan == 16 else 128
    size = block_size * 1024
    target = torch.zeros(size, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(size, dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(size, block_size), )
    _write_blocks_kernel[grid](target, size, BLOCK_SIZE=block_size)
    _read_reversed_blocks_kernel[grid](target, scratch, size, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    assert scratch.sum().item() == size


@triton.jit
def _write_blocks_after_all_arrive_kernel(ptr, arrived_ptr, BLOCK_SIZE: tl.constexpr):
    # Keep every block resident until all have started, so the blocks of a
    # launch hold distinct GSan identities; relaxed atomics add no ordering.
    tl.atomic_add(arrived_ptr, 1, sem="relaxed")
    atomic_poll(arrived_ptr, tl.num_programs(0))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(ptr + offsets, 1)


@requires_gsan
def test_freed_allocation_does_not_keep_shadow_state(with_gsan):
    # Guards against shadow state of a freed range surviving into a new
    # allocation of the same range. The driver currently returns zeroed pages
    # for the new mapping, so this does not exercise gsanMalloc's own shadow
    # zeroing.
    device = torch.cuda.current_device()
    nbytes = 1 << 20
    num_elements = 128
    num_blocks = 4
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    arrived = torch.zeros(2, dtype=torch.int32, device="cuda")
    # The side streams do not wait for the zero fill (which GSan does not see).
    torch.cuda.synchronize()

    first = gsan_malloc(nbytes, device, first_stream.cuda_stream)
    with torch.cuda.stream(first_stream):
        target = uint8_cuda_tensor_from_ptr(first, nbytes, device).view(torch.int32)
        _write_blocks_after_all_arrive_kernel[(1, )](target, arrived[0:1], BLOCK_SIZE=num_elements)
    torch.cuda.synchronize()
    target = None
    gsan_free(first, device, nbytes, first_stream.cuda_stream)

    second = gsan_malloc(nbytes, device, second_stream.cuda_stream)
    try:
        # The test only means something if the allocator hands out the same range again.
        assert second == first
        # The second stream is not ordered after the first, so shadow state
        # left over from the freed allocation would be reported as a WAW race.
        # A block that takes over the first launch's GSan identity (slot)
        # shares its epoch and would not report it, but the co-resident blocks
        # of this launch hold distinct slots, so at most one can (this needs at
        # least 4 free slots; with fewer, slot acquisition fails after 20 s).
        with torch.cuda.stream(second_stream):
            target = uint8_cuda_tensor_from_ptr(second, nbytes, device).view(torch.int32)
            _write_blocks_after_all_arrive_kernel[(num_blocks, )](target, arrived[1:2],
                                                                  BLOCK_SIZE=num_elements // num_blocks)
        torch.cuda.synchronize()
        assert target[:num_elements].eq(1).all().item()
    finally:
        target = None
        gsan_free(second, device, nbytes, second_stream.cuda_stream)


@triton.jit
def _gsan_wave32_copy_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offsets, tl.load(src + offsets) + 1)


@pytest.mark.skipif(not is_hip() or triton.runtime.driver.active.get_current_target().arch != "gfx1250",
                    reason="Requires AMD gfx1250")
def test_gsan_gfx1250_multiwave_copy(with_gsan):
    src = torch.arange(4096, dtype=torch.int32, device="cuda")
    dst = torch.empty_like(src)
    compiled = _gsan_wave32_copy_kernel[(4, )](src, dst, BLOCK=1024, num_warps=4)
    torch.testing.assert_close(dst, src + 1)
    assert compiled.metadata.warp_size == 32


@pytest.mark.skipif(not is_hip(), reason="AMD target check")
def test_gsan_rejects_unsupported_amd_target():
    # Targets beyond gfx950 and gfx1250 must fail at compile time.
    src = triton.compiler.ASTSource(fn=_write_blocks_kernel, signature={"ptr": "*i32", "n_elements": "i32"},
                                    constexprs={"BLOCK_SIZE": 128})
    with pytest.raises(ValueError, match="GSan is not supported on gfx1200"):
        triton.compile(src, target=GPUTarget("hip", "gfx1200", 32), options={"instrumentation_mode": "gsan"})


@gluon.jit
def _gluon_async_copy_masked_kernel(out_ptr, in_ptr, n_elements: gl.constexpr, start_idx: gl.constexpr,
                                    BLOCK: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    block_layout: gl.constexpr = gl.BlockedLayout([4], [32], [2], [0])
    smem = gl.allocate_shared_memory(in_ptr.dtype.element_ty, [BLOCK], smem_layout)

    # Specialize the bounds so coarse pools can use complete 16-byte accesses.
    offsets = start_idx + gl.arange(0, BLOCK, block_layout)
    mask = offsets < n_elements
    async_copy.async_load(smem, in_ptr + offsets, mask=mask)
    async_copy.commit_group()
    async_copy.wait_group(0)

    values = smem.load(block_layout)
    gl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _device_tma_masked_store_kernel(ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
    values = tl.full((BLOCK, BLOCK), 1, dtype=tl.int32)
    desc.store([row_idx, col_idx], values)


@triton.jit
def _device_tma_masked_load_kernel(out_ptr, ptr, m_size, n_size, row_idx, col_idx, stride_0, BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(ptr, [m_size, n_size], [stride_0, 1], [BLOCK, BLOCK])
    values = desc.load([row_idx, col_idx])
    offsets = tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    tl.store(out_ptr + offsets, values)


@triton.jit
def _host_tma_gather_kernel(out_ptr, out_stride_0, out_stride_1, desc, x_offsets_ptr, y_offset, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
    out = desc.gather(x_offsets, y_offset)
    indices_x = tl.arange(0, BLOCK_X)[:, None] * out_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * out_stride_1
    tl.store(out_ptr + indices_x + indices_y, out)


@triton.jit
def _host_tma_scatter_kernel(desc, x_offsets_ptr, y_offset, src_ptr, src_stride_0, src_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
    src = tl.load(src_ptr + indices_x + indices_y)
    x_offsets = tl.load(x_offsets_ptr + tl.arange(0, BLOCK_X))
    desc.scatter(src, x_offsets, y_offset)


@triton.jit
def _host_tma_reduce_add_kernel(desc, src_ptr, src_stride_0, src_stride_1, BLOCK_X: tl.constexpr):
    BLOCK_Y: tl.constexpr = desc.block_shape[1]
    indices_x = tl.arange(0, BLOCK_X)[:, None] * src_stride_0
    indices_y = tl.arange(0, BLOCK_Y)[None, :] * src_stride_1
    src = tl.load(src_ptr + indices_x + indices_y)
    desc.atomic_add([0, 0], src)


def _shadow_cells_for_tensor(tensor: torch.Tensor):
    assert tensor.ndim >= 1
    if tensor.ndim > 1:
        return [_shadow_cells_for_tensor(tensor[i]) for i in range(tensor.shape[0])]

    device_idx = tensor.device.index
    row = []
    for i in range(tensor.shape[0]):
        real_ptr = tensor[i].data_ptr()
        granularity = get_shadow_granularity(real_ptr)
        row.append(
            tuple((shadow_cell_address(real_ptr + byte_offset),
                   shadow_cell_from_address(real_ptr + byte_offset, device_index=device_idx))
                  for byte_offset in range(0, tensor.element_size(), granularity)))
    return row


def _assert_shadow_mask(before, after, changed_mask: torch.Tensor, *, access_kind: str) -> None:
    assert access_kind in {"read", "write"}
    assert len(before) == changed_mask.shape[0]
    assert len(before[0]) == changed_mask.shape[1]

    # Logical elements can share a coarse shadow cell. A change to any of them
    # updates the cell seen through all of those elements, including masked ones.
    changed_cells = {
        address
        for row_idx, col_idx in changed_mask.nonzero().tolist()
        for address, _ in before[row_idx][col_idx]
    }
    for row_idx in range(changed_mask.shape[0]):
        for col_idx in range(changed_mask.shape[1]):
            for (address, before_cell), (after_address, after_cell) in zip(before[row_idx][col_idx],
                                                                           after[row_idx][col_idx]):
                assert address == after_address
                if address in changed_cells:
                    assert after_cell != before_cell
                    if access_kind == "read":
                        assert after_cell.write_clock == before_cell.write_clock
                    else:
                        assert after_cell.write_clock != before_cell.write_clock
                        assert after_cell.write_clock.epoch != 0
                else:
                    assert after_cell == before_cell


def _masked_tma_change_mask(storage: torch.Tensor, m_size: int, n_size: int, row_idx: int, col_idx: int,
                            block: int) -> torch.Tensor:
    changed_mask = torch.zeros(storage.shape, dtype=torch.bool)
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    changed_mask[first_row:last_row, first_col:last_col] = True
    return changed_mask


def _gather_scatter_change_mask(storage: torch.Tensor, x_offsets: torch.Tensor, y_offset: int, m_size: int, n_size: int,
                                block_y: int) -> torch.Tensor:
    changed_mask = torch.zeros(storage.shape, dtype=torch.bool)
    valid_cols = max(min(n_size - y_offset, block_y), 0)
    if valid_cols == 0:
        return changed_mask

    for row_idx in x_offsets.tolist():
        if 0 <= row_idx < m_size:
            changed_mask[row_idx, y_offset:y_offset + valid_cols] = True
    return changed_mask


def _gather_reference(target: torch.Tensor, x_offsets: torch.Tensor, y_offset: int, block_y: int) -> torch.Tensor:
    result = torch.zeros((x_offsets.numel(), block_y), dtype=target.dtype, device=target.device)
    valid_rows = (x_offsets >= 0) & (x_offsets < target.shape[0])
    valid_cols = max(min(target.shape[1] - y_offset, block_y), 0)
    if valid_cols == 0:
        return result

    safe_rows = torch.where(valid_rows, x_offsets, 0)
    gathered = target[safe_rows.long(), y_offset:y_offset + valid_cols]
    result[:, :valid_cols] = gathered * valid_rows[:, None]
    return result


def _scatter_reference(dst: torch.Tensor, src: torch.Tensor, x_offsets: torch.Tensor, y_offset: int) -> torch.Tensor:
    result = torch.zeros_like(dst)
    valid_cols = max(min(dst.shape[1] - y_offset, src.shape[1]), 0)
    if valid_cols == 0:
        return result

    for src_row, dst_row in enumerate(x_offsets.tolist()):
        if 0 <= dst_row < dst.shape[0]:
            result[dst_row, y_offset:y_offset + valid_cols] = src[src_row, :valid_cols]
    return result


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_gluon_async_copy_updates_shadow(with_gsan):
    block = 128
    start_idx = 8 if with_gsan == 16 else 5
    n_elements = 116 if with_gsan == 16 else 117
    padded = 160
    inp = torch.arange(padded, dtype=torch.float32, device="cuda")
    out = torch.zeros_like(inp)
    shadow0 = [_shadow_cells_for_tensor(inp)]
    changed_mask = torch.zeros((1, inp.numel()), dtype=torch.bool)
    changed_mask[0, start_idx:n_elements] = True

    _gluon_async_copy_masked_kernel[(1, )](out, inp, n_elements, start_idx, BLOCK=block, num_warps=2)

    expected = torch.zeros_like(out)
    expected[start_idx:n_elements] = inp[start_idx:n_elements]
    torch.testing.assert_close(out, expected)

    shadow1 = [_shadow_cells_for_tensor(inp)]
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")
    assert out[n_elements].item() == 0
    assert n_elements - start_idx < block


# Cell-size coverage uses one offset; extra boundary shapes run at the default
# granularity and at 16 bytes, whose complete-cell requirement is different.
TMA_MASK_CASES = [(g, 5, 8)
                  for g in (1, 2, 4, 8, 16)] + [(g, row, col) for g in (4, 16) for row, col in ((30, 8), (5, 32))]


@requires_gsan
@pytest.mark.parametrize("num_ctas", [
    1,
    pytest.param(
        2, marks=pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                                    reason="Multi-CTA TMA requires Hopper or Blackwell")),
])
@pytest.mark.parametrize("shadow_granularity,row_idx,col_idx", TMA_MASK_CASES, indirect=["shadow_granularity"])
def test_tma_masked_load_updates_shadow(with_gsan, with_allocator, row_idx, col_idx, num_ctas):
    if with_gsan == 16 and not is_hopper_or_newer():
        pytest.skip("Pre-Hopper descriptor emulation does not guarantee complete 16-byte accesses")
    block = 32
    m_size = 35
    n_size = 36 if with_gsan == 16 else 37
    padded_m = 40
    padded_n = 40
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    target_storage = torch.arange(padded_m * padded_n, dtype=torch.int32, device="cuda").reshape(padded_m, padded_n)
    target = target_storage[:m_size, :n_size]
    output = torch.empty((block, block), dtype=torch.int32, device="cuda")
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _masked_tma_change_mask(target_storage, m_size, n_size, row_idx, col_idx, block)

    _device_tma_masked_load_kernel[(1, )](output, target, m_size, n_size, row_idx, col_idx, target.stride(0),
                                          BLOCK=block, num_ctas=num_ctas)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    expected[:last_row - first_row, :last_col - first_col] = target[first_row:last_row, first_col:last_col]
    torch.testing.assert_close(output, expected)

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")


@requires_gsan
@pytest.mark.parametrize("num_ctas", [
    1,
    pytest.param(
        2, marks=pytest.mark.skipif(not is_hopper_or_newer() or is_sm12x(),
                                    reason="Multi-CTA TMA requires Hopper or Blackwell")),
])
@pytest.mark.parametrize("shadow_granularity,row_idx,col_idx", TMA_MASK_CASES, indirect=["shadow_granularity"])
def test_tma_masked_store_updates_shadow(with_gsan, with_allocator, row_idx, col_idx, num_ctas):
    if with_gsan == 16 and not is_hopper_or_newer():
        pytest.skip("Pre-Hopper descriptor emulation does not guarantee complete 16-byte accesses")
    block = 32
    m_size = 35
    n_size = 36 if with_gsan == 16 else 37
    padded_m = 40
    padded_n = 40
    first_row = max(row_idx, 0)
    last_row = min(row_idx + block, m_size)
    first_col = max(col_idx, 0)
    last_col = min(col_idx + block, n_size)
    valid_rows = max(last_row - first_row, 0)
    valid_cols = max(last_col - first_col, 0)
    target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _masked_tma_change_mask(target_storage, m_size, n_size, row_idx, col_idx, block)

    _device_tma_masked_store_kernel[(1, )](target, m_size, n_size, row_idx, col_idx, target.stride(0), BLOCK=block,
                                           num_ctas=num_ctas)
    torch.cuda.synchronize()

    expected = torch.zeros_like(target)
    expected[first_row:last_row, first_col:last_col] = 1
    torch.testing.assert_close(target, expected)

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="write")
    assert target_storage[m_size, col_idx].item() == 0
    assert valid_rows < block
    assert valid_cols < block


@pytest.mark.skipif(not is_hopper_or_newer(), reason="TMA requires Hopper or newer")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_tma_load_store_uses_pool_granularity(fresh_knobs, with_allocator, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    output = torch.empty((32, 32), dtype=torch.int32, device="cuda")
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros((32, 32), dtype=torch.int32, device="cuda")

    _device_tma_masked_store_kernel[(1, )](target, 32, 32, 0, 0, 32, BLOCK=32)
    _device_tma_masked_load_kernel[(1, )](output, target, 32, 32, 0, 0, 32, BLOCK=32)
    torch.testing.assert_close(output, torch.ones_like(output))
    for offset in (0, shadow_granularity, 128, target.numel() * target.element_size() - shadow_granularity):
        cell = shadow_cell_from_address(target.data_ptr() + offset)
        assert cell.write_clock.epoch != 0
        assert cell.num_reads != 0


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_host_tma_gather_updates_shadow(fresh_knobs, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 12 if shadow_granularity == 16 else 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    # Scalar index loads and the small output tile need a fine-grained pool.
    auxiliary_pool = create_mem_pool(shadow_granularity=min(shadow_granularity, 4))
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(auxiliary_pool):
        x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
        out = torch.empty((block_x, block_y), dtype=torch.int32, device="cuda")
    with torch.cuda.use_mem_pool(pool):
        target_storage = torch.arange(padded_m * padded_n, dtype=torch.int32, device="cuda").reshape(padded_m, padded_n)
    target = target_storage[:m_size, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _gather_scatter_change_mask(target_storage, x_offsets, y_offset, m_size, n_size, block_y)

    compiled = _host_tma_gather_kernel[(1, )](out, out.stride(0), out.stride(1), target_desc, x_offsets, y_offset,
                                              BLOCK_X=block_x)
    assert "ttng.async_tma_gather" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(out, _gather_reference(target, x_offsets, y_offset, block_y))

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="read")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_host_tma_scatter_updates_shadow(fresh_knobs, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    block_x = 8
    block_y = 8
    m_size = 11
    n_size = 12 if shadow_granularity == 16 else 13
    padded_m = 16
    padded_n = 16
    y_offset = 8
    auxiliary_pool = create_mem_pool(shadow_granularity=min(shadow_granularity, 4))
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(auxiliary_pool):
        x_offsets = torch.tensor([1, 3, 5, 7, 9, 10, 11, 13], dtype=torch.int32, device="cuda")
        src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").reshape(block_x, block_y)
    with torch.cuda.use_mem_pool(pool):
        target_storage = torch.zeros((padded_m, padded_n), dtype=torch.int32, device="cuda")
    target = target_storage[:m_size, :n_size]
    target_desc = TensorDescriptor.from_tensor(target, [1, block_y])
    shadow0 = _shadow_cells_for_tensor(target_storage)
    changed_mask = _gather_scatter_change_mask(target_storage, x_offsets, y_offset, m_size, n_size, block_y)

    compiled = _host_tma_scatter_kernel[(1, )](target_desc, x_offsets, y_offset, src, src.stride(0), src.stride(1),
                                               BLOCK_X=block_x)
    assert "ttng.async_tma_scatter" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(target, _scatter_reference(target, src, x_offsets, y_offset))

    shadow1 = _shadow_cells_for_tensor(target_storage)
    _assert_shadow_mask(shadow0, shadow1, changed_mask, access_kind="write")
    assert target_storage[m_size, y_offset].item() == 0


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires Hopper or newer")
@pytest.mark.parametrize("dtype,shadow_granularity",
                         [(dtype, min(4, dtype.itemsize))
                          for dtype in (torch.int32, torch.float16, torch.bfloat16, torch.uint64)],
                         indirect=["shadow_granularity"])
@pytest.mark.parametrize("block_x", (1, 8))
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_host_tma_reduce_updates_atomic_shadow(with_gsan, block_x, dtype):
    block_y = 16
    target = torch.zeros((block_x, block_y), dtype=dtype, device="cuda")
    src = torch.arange(1, block_x * block_y + 1, dtype=torch.int32, device="cuda").to(dtype).reshape(block_x, block_y)
    target_desc = TensorDescriptor.from_tensor(target, [block_x, block_y])

    compiled = _host_tma_reduce_add_kernel[(1, )](target_desc, src, src.stride(0), src.stride(1), BLOCK_X=block_x)
    assert "ttng.async_tma_reduce" in compiled.asm["ttgir"]
    torch.cuda.synchronize()

    torch.testing.assert_close(target, src)
    for row in range(block_x):
        for col in range(block_y):
            for byte_offset in range(0, target.element_size(), with_gsan):
                address = target[row, col].data_ptr() + byte_offset
                _assert_atomic_rmw_shadow(address, AtomicScope.GPU, is_release=False)


@triton.jit
def _graph_copy_kernel(Input, Output, ADD: tl.constexpr, WAIT: tl.constexpr = False, SIGNAL: tl.constexpr = False):
    if WAIT:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    value = tl.load(Input + (tl.num_programs(0) - 1 - pid))
    tl.store(Output + pid, value + ADD)
    if SIGNAL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _graph_join_kernel(Left, Right, Output):
    pid = tl.program_id(0)
    tl.store(Output + pid, tl.load(Left + pid) + tl.load(Right + pid))


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("change_stream", [False, True])
def test_explicit_graph_fork_join_replay(with_gsan, pdl, change_stream):
    if pdl and not is_hopper_or_newer():
        pytest.skip("Programmatic graph dependencies require NVIDIA Hopper or newer")
    from triton.experimental.gsan import graph
    from triton.experimental.gsan._testing_utils import ExplicitGraph

    #          +--full--> left --full--+
    # root ----+                      +--> join
    #          +--edge--> right --full-+
    # "edge" is PDL when pdl=True (right calls gdc_wait), full otherwise.
    # eager write --> graph replay 0 --> ... --> graph replay 8 --> eager copy
    # Replays alternate streams when change_stream=True.
    n = 512
    source = torch.zeros(n, device="cuda", dtype=torch.int32)
    root, left, right, result = [torch.empty_like(source) for _ in range(4)]
    plan = graph.GraphPlan(torch.cuda.current_device(), [
        graph.GraphNode(),
        graph.GraphNode((0, )),
        graph.GraphNode((), (0, )) if pdl else graph.GraphNode((0, )),
        graph.GraphNode((1, 2))
    ])
    kernels = [
        (_graph_copy_kernel.warmup(source, root, ADD=1, SIGNAL=pdl, grid=(n, ), num_warps=1), (source, root)),
        (_graph_copy_kernel.warmup(root, left, ADD=2, grid=(n, ), num_warps=1), (root, left)),
        (_graph_copy_kernel.warmup(root, right, ADD=3, WAIT=pdl, **({"launch_pdl": pdl} if is_cuda() else {}),
                                   grid=(n, ), num_warps=1), (root, right)),
        (_graph_join_kernel.warmup(left, right, result, grid=(n, ), num_warps=1), (left, right, result)),
    ]
    properties = triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())
    stream = torch.cuda.Stream() if change_stream else torch.cuda.current_stream()
    with ExplicitGraph(plan) as executable:
        for compiled, arguments in kernels:
            executable.add_kernel(compiled, arguments, n)
            # Graph nodes bypass the eager launcher but need the same reservation.
            assert executable.parameters[-1][0].shared == getattr(compiled.run, "shared", compiled.metadata.shared)
            if is_cuda():
                assert 2 * compiled.run.shared > properties["max_shared_mem_per_multiprocessor"]
        executable.instantiate()
        # Consume an eager producer before replay and feed an eager consumer after.
        _write_blocks_kernel[(4, )](source, n, BLOCK_SIZE=128)
        for iteration in range(9):
            selected = stream if iteration % 2 else torch.cuda.current_stream()
            if selected != torch.cuda.current_stream():
                selected.wait_stream(torch.cuda.current_stream())
            completion = executable.launch(selected)
            # The same executable's next replay is ordered even on another stream.
        torch.cuda.current_stream().wait_stream(stream)
        graph.acquire_completions(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, (completion, ))
        _graph_copy_kernel[(n, )](result, source, ADD=0, num_warps=1)
        torch.cuda.synchronize()
        assert torch.all(source == 9)


def test_graph_completion_is_immutable_across_stream_slot_reuse(with_gsan):
    from triton.experimental.gsan import graph, _stream_sync

    # write --> snapshot --> write 0 --> ... --> write 6
    #              |
    #              +--> retained clock (unchanged as stream slots are reused)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    values = torch.zeros(1024, device="cuda", dtype=torch.int32)
    _write_blocks_kernel[(8, )](values, values.numel(), BLOCK_SIZE=128)
    snapshot = graph.record_completion(device, stream)
    expected = snapshot.clock.clone()
    for _ in range(7):
        _write_blocks_kernel[(8, )](values, values.numel(), BLOCK_SIZE=128)
    torch.cuda.synchronize()
    torch.testing.assert_close(snapshot.clock, expected)
    assert not torch.equal(snapshot.clock, _stream_sync._current_stream_clock(device, stream))


def test_explicit_graph_aborted_launch_preserves_stream_order(with_gsan):
    from triton.experimental.gsan import graph, _stream_sync

    # write --> reserve empty graph --> abort (restore the write's stream clock)
    # The empty graph would contain only entry --> exit helpers; it is not run.
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    values = torch.zeros(512, device="cuda", dtype=torch.int32)
    _write_blocks_kernel[(4, )](values, values.numel(), BLOCK_SIZE=128)
    expected = _stream_sync._current_stream_clock(device, stream).clone()
    plan = graph.GraphPlan(device, ())
    prepared = plan.prepare_launch(stream)
    prepared.finish(submitted=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(_stream_sync._current_stream_clock(device, stream), expected)
    plan.close()


@pytest.mark.skipif(not is_hip(), reason="Uses HIP native invalid-executable rejection")
def test_explicit_graph_native_rejection_preserves_stream_order(with_gsan, monkeypatch):
    from triton.experimental.gsan import graph, _stream_sync
    from triton.experimental.gsan._testing_utils import ExplicitGraph

    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    values = torch.zeros(512, device="cuda", dtype=torch.int32)
    result = torch.empty_like(values)
    _write_blocks_kernel[(4, )](values, values.numel(), BLOCK_SIZE=128)
    expected = _stream_sync._current_stream_clock(device, stream).clone()
    with ExplicitGraph(graph.GraphPlan(device, ())) as executable:
        executable.instantiate()
        native_call = executable._call

        def reject_launch(name, *args):
            # Node updates succeed; the actual HIP launch rejects a null exec.
            if name == "cuGraphLaunch":
                args = (executable.C.c_void_p(), *args[1:])
            return native_call(name, *args)

        monkeypatch.setattr(executable, "_call", reject_launch)
        with pytest.raises(RuntimeError, match="hipGraphLaunch failed"):
            executable.launch()
        # Consume the expected HIP last-error status before further torch work.
        executable.cuda.hipGetLastError()
        torch.cuda.synchronize()
        torch.testing.assert_close(_stream_sync._current_stream_clock(device, stream), expected)
        _graph_copy_kernel[(values.numel(), )](values, result, ADD=0, num_warps=1)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, values.flip(0))


def test_explicit_graph_storage_survives_executable_close(with_gsan):
    import gc
    import weakref
    from triton.experimental.gsan import graph
    from triton.experimental.gsan._testing_utils import ExplicitGraph

    # Empty graph: entry --> exit --> completion snapshot
    # Host:       launch --> close --> synchronize --> collect retained storage
    device = torch.cuda.current_device()
    plan = graph.GraphPlan(device, ())
    reference = weakref.ref(plan)
    with ExplicitGraph(plan) as executable:
        executable.instantiate()
        completion = executable.launch()
    del plan, executable
    gc.collect()
    # The submission owns storage until its CUDA completion event is observed.
    assert reference() is not None
    torch.cuda.synchronize()
    graph._collect_pending()
    assert reference() is None
    assert completion.clock.numel() > 0


@triton.jit
def _shared_cell_contention_kernel(x_ptr, c_ptr, y_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    for _ in range(ITERS):
        # Every lane of every program reads the same cells (one hot scalar and
        # four shared cells) while atomics hit four other shared cells.
        acc += tl.load(x_ptr + offs * 0)
        acc += tl.load(x_ptr + 1 + offs % 4)
        tl.atomic_add(c_ptr + offs % 4, 1, sem="relaxed")
    tl.store(y_ptr + pid * BLOCK + offs, acc)


def _run_shared_cell_contention() -> None:
    triton.knobs.compilation.instrumentation_mode = "gsan"
    with torch.cuda.use_mem_pool(create_mem_pool()):
        grid, block, iters = 64, 256, 2
        x = torch.ones(8, dtype=torch.int32, device="cuda")
        c = torch.zeros(4, dtype=torch.int32, device="cuda")
        y = torch.empty(grid * block, dtype=torch.int32, device="cuda")
        _shared_cell_contention_kernel[(grid, )](x, c, y, iters, block, num_warps=4)
        torch.cuda.synchronize()
        assert (y == 2 * iters).all()
        assert c.tolist() == [grid * block * iters // 4] * 4


@requires_gsan
def test_gsan_shared_cell_contention():
    # Many lanes, warps, and programs contend for the same shadow cells, with
    # checked loads interleaved with atomics. On AMD, lanes of a wave cannot
    # spin on each other's shadow locks, so this exercises the deadlock-free
    # locking schemes; the kernel must finish with exact results and no GSan
    # report. Runs in a subprocess so a lock-progress regression times out
    # instead of hanging the session (a report aborts the process on AMD).
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TRITON_TEST_PROCESS_TIMEOUT", os.environ.get("TRITON_TEST_PROCESS_TIMEOUT", "300"))
        result = run_in_process(_run_shared_cell_contention, allow_abort=True)
    assert result.exc is None, f"{result.exc}\n{result.driver_stderr_output}"


# On AMD, tensor accesses through a kernel-argument base are converted to
# buffer ops (amdg.buffer_*), which GSan checks on the reconstructed addresses.
def _assert_buffer_ops(compiled, *ops) -> None:
    ttgir = compiled.asm["ttgir"]
    for op in ops:
        assert f"amdg.{op} " in ttgir, f"expected amdg.{op} in TTGIR:\n{ttgir}"


@triton.jit
def _buffer_write_blocks_kernel(ptr, arrived_ptr, BLOCK: tl.constexpr):
    # Keep both blocks resident at once; the relaxed atomics order nothing.
    tl.atomic_add(arrived_ptr, 1, sem="relaxed")
    atomic_poll(arrived_ptr, tl.num_programs(0))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, offsets)


@triton.jit
def _buffer_read_other_block_kernel(ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = (tl.num_programs(0) - 1 - pid) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.load(ptr + offsets) + 1)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer ops")
@pytest.mark.parametrize("shadow_granularity", [4, 16], indirect=True)
def test_buffer_ops_disjoint_or_launch_ordered_accesses_do_not_race(with_gsan, capfd):
    # Two concurrent blocks write disjoint ranges from the same base (a check
    # on the scalar base alone would see a write-write race), then a second
    # launch on the same stream reads the other block's range. In 16-byte
    # pools this also needs the per-element checks merged into whole 16-byte
    # accesses.
    block = 256
    data = torch.zeros(2 * block, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(data)
    # Atomics need a pool finer than 16 bytes.
    with torch.cuda.use_mem_pool(create_mem_pool(shadow_granularity=4)):
        arrived = torch.zeros(1, dtype=torch.int32, device="cuda")
    _assert_buffer_ops(_buffer_write_blocks_kernel.warmup(data, arrived, BLOCK=block, num_warps=1, grid=(2, )),
                       "buffer_store")
    _assert_buffer_ops(_buffer_read_other_block_kernel.warmup(data, out, BLOCK=block, num_warps=1, grid=(2, )),
                       "buffer_load", "buffer_store")
    _buffer_write_blocks_kernel[(2, )](data, arrived, BLOCK=block, num_warps=1)
    _buffer_read_other_block_kernel[(2, )](data, out, BLOCK=block, num_warps=1)
    torch.cuda.synchronize()
    expected = torch.arange(2 * block, dtype=torch.int32, device="cuda")
    assert torch.equal(data, expected)
    assert torch.equal(out, expected + 1)
    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _buffer_atomic_handoff_kernel(payload_ptr, flag_ptr, out_ptr, CONSUMER_OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    # The flag is element 0 of a pair; element 1 stays 0, and masked-off
    # atomic results are undefined. (A one-element tensor offset folds away,
    # and only base + tensor offset becomes a buffer atomic.)
    pair = tl.arange(0, 2)
    if tl.program_id(0) == 0:
        tl.store(payload_ptr + offsets, offsets + 1000)
        tl.atomic_xchg(flag_ptr + pair, 1, mask=pair < 1, sem="release")
    else:
        ready = 0
        while ready != 1:
            if CONSUMER_OP == "cas":
                ready = tl.sum(tl.atomic_cas(flag_ptr + pair, 1, 1, sem="acquire"))
            else:
                ready = tl.sum(tl.where(pair < 1, tl.atomic_add(flag_ptr + pair, 0, mask=pair < 1, sem="acquire"), 0))
        tl.store(out_ptr + offsets, tl.load(payload_ptr + offsets))


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer atomics")
@pytest.mark.parametrize("consumer_op", ["rmw", "cas"])
def test_buffer_atomic_release_acquire_handoff_does_not_race(with_gsan, capfd, consumer_op):
    block = 64
    payload = torch.zeros(block, dtype=torch.int32, device="cuda")
    flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(payload)
    args = (payload, flag, out, consumer_op)
    compiled = _buffer_atomic_handoff_kernel.warmup(*args, BLOCK=block, num_warps=1, grid=(2, ))
    _assert_buffer_ops(compiled, "buffer_load", "buffer_store", "buffer_atomic_rmw",
                       *(["buffer_atomic_cas"] if consumer_op == "cas" else []))
    _buffer_atomic_handoff_kernel[(2, )](*args, BLOCK=block, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.arange(block, dtype=torch.int32, device="cuda") + 1000)
    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _buffer_atomic_results_kernel(counter_ptr, cas_ptr, old_ptr, cas_old_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    old = tl.atomic_add(counter_ptr + offsets, offsets, mask=offsets < BLOCK - 8, sem="relaxed")
    tl.store(old_ptr + offsets, old, mask=offsets < BLOCK - 8)
    # Only even elements hold the compare value 0.
    cas_old = tl.atomic_cas(cas_ptr + offsets, 0, offsets + 100, sem="acq_rel")
    tl.store(cas_old_ptr + offsets, cas_old)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer atomics")
def test_buffer_atomic_results_are_preserved(with_gsan, capfd):
    block = 64
    idx = torch.arange(block, dtype=torch.int32, device="cuda")
    counter = torch.full((block, ), 5, dtype=torch.int32, device="cuda")
    cas = (idx % 2).clone()
    old = torch.full_like(idx, -1)
    cas_old = torch.full_like(idx, -1)
    args = (counter, cas, old, cas_old)
    _assert_buffer_ops(_buffer_atomic_results_kernel.warmup(*args, BLOCK=block, num_warps=1, grid=(1, )),
                       "buffer_atomic_rmw", "buffer_atomic_cas")
    _buffer_atomic_results_kernel[(1, )](*args, BLOCK=block, num_warps=1)
    torch.cuda.synchronize()
    active = idx < block - 8
    assert torch.equal(counter, torch.where(active, 5 + idx, 5))
    assert torch.equal(old, torch.where(active, 5, -1))
    assert torch.equal(cas_old, idx % 2)
    assert torch.equal(cas, torch.where(idx % 2 == 0, idx + 100, 1))
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_buffer_bounds_kernel(ptr, out_ptr, arrived_ptr):
    offsets = gl.arange(0, 64, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0]))
    # Keep both blocks resident at once; the relaxed atomics order nothing.
    gl.atomic_add(arrived_ptr, 1, sem="relaxed")
    while gl.atomic_add(arrived_ptr, 0, sem="relaxed") != 2:
        pass
    if gl.program_id(0) == 0:
        # Unsynchronized writes to elements 12..63.
        gl.store(ptr + 12 + offsets, 1, mask=offsets < 52)
    else:
        # Element i is base[8 + i]. Elements 0..3 are accessed. Elements 4..7
        # (base[12..15]) are in bounds but masked off, and elements 8.. lie
        # past the 16-element bound; both overlap block 0's writes.
        desc = gl.amd.cdna4.buffer.make_buffer_descriptor(ptr, 16 * 4)
        mask = offsets < 4
        values = gl.amd.cdna4.buffer.load(desc, offsets, mask=mask, soffset=8)
        gl.amd.cdna4.buffer.store(values + 1, desc, offsets, mask=mask, soffset=8)
        gl.store(out_ptr + offsets, values)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer ops")
def test_gluon_buffer_masked_and_out_of_bounds_lanes_do_not_race(with_gsan, capfd):
    data = torch.arange(64, dtype=torch.int32, device="cuda")
    out = torch.full((64, ), -1, dtype=torch.int32, device="cuda")
    arrived = torch.zeros(1, dtype=torch.int32, device="cuda")
    _gluon_buffer_bounds_kernel[(2, )](data, out, arrived, num_warps=1)
    torch.cuda.synchronize()
    idx = torch.arange(64, dtype=torch.int32, device="cuda")
    assert torch.equal(out, torch.where(idx < 4, idx + 8, 0))
    assert torch.equal(data, torch.where(idx < 8, idx, torch.where(idx < 12, idx + 1, 1)))
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_buffer_atomic_load_store_kernel(ptr, out_ptr):
    offsets = gl.arange(0, 64, layout=gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0]))
    desc = gl.amd.cdna4.buffer.make_buffer_descriptor(ptr, 32 * 4)
    # Elements 32..47 are out of bounds and dropped; 48.. are masked off.
    gl.amd.cdna4.buffer.atomic_store(offsets + 100, desc, offsets, mask=offsets < 48, sem="release")
    # Masked-off and out-of-bounds elements read zero.
    loaded = gl.amd.cdna4.buffer.atomic_load(desc, offsets, mask=offsets % 2 == 0, sem="acquire")
    gl.store(out_ptr + offsets, loaded)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD buffer atomics")
@pytest.mark.gsan_fine_granularity("16-byte pools do not support atomics")
def test_gluon_buffer_atomic_load_store_results(with_gsan, capfd):
    data = torch.full((64, ), -1, dtype=torch.int32, device="cuda")
    out = torch.full((64, ), -1, dtype=torch.int32, device="cuda")
    _gluon_buffer_atomic_load_store_kernel[(1, )](data, out, num_warps=1)
    torch.cuda.synchronize()
    idx = torch.arange(64, dtype=torch.int32, device="cuda")
    assert torch.equal(data, torch.where(idx < 32, idx + 100, -1))
    assert torch.equal(out, torch.where((idx < 32) & (idx % 2 == 0), idx + 100, 0))
    _assert_no_gsan_runtime_output(capfd)


@triton.jit
def _packed_fp4_copy_kernel(src, dst):
    offsets = tl.arange(0, 1024)
    tl.store(dst + offsets, tl.load(src + offsets))


@gluon.jit
def _gluon_packed_fp4_copy_kernel(src, dst):
    offsets = gl.arange(0, 1024, layout=gl.BlockedLayout([16], [THREADS_PER_WARP], [1], [0]))
    src_desc = gl.amd.cdna4.buffer.make_buffer_descriptor(src, 1024)
    dst_desc = gl.amd.cdna4.buffer.make_buffer_descriptor(dst, 1024)
    values = gl.amd.cdna4.buffer.load(src_desc, offsets)
    gl.amd.cdna4.buffer.store(values, dst_desc, offsets)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="AMD packed FP4 buffer ops")
@pytest.mark.parametrize("explicit_buffer", [False, True], ids=["automatic", "gluon"])
def test_buffer_packed_fp4_bytes_preserved(with_gsan, capfd, explicit_buffer):
    src = torch.arange(1024, dtype=torch.int32, device="cuda").to(torch.uint8)
    dst = torch.zeros_like(src)
    kernel = _gluon_packed_fp4_copy_kernel if explicit_buffer else _packed_fp4_copy_kernel
    args = (triton.reinterpret(src, tl.float4e2m1x2), triton.reinterpret(dst, tl.float4e2m1x2))
    triton.knobs.amd.use_buffer_ops = True
    for mode in ("", "gsan"):
        triton.knobs.compilation.instrumentation_mode = mode
        dst.zero_()
        compiled = kernel[(1, )](*args, num_warps=1)
        _assert_buffer_ops(compiled, "buffer_load", "buffer_store")
        torch.cuda.synchronize()
        assert torch.equal(src, dst)
    _assert_no_gsan_runtime_output(capfd)


@gluon.jit
def _gluon_amd_direct_async_kernel(src, dst, n_elements, soffset, BUFFER: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS_PER_WARP], [1], [0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    index = gl.arange(0, 64, layout)
    mask = index % 2 == 0
    smem = gl.allocate_shared_memory(gl.int32, [64], shared)
    if BUFFER:
        desc = gl.amd.cdna4.buffer.make_buffer_descriptor(src, n_elements * 4)
        gl.amd.cdna4.buffer.load_to_shared(smem, desc, index, mask=mask, soffset=soffset)
    else:
        amd_global_load_to_shared(smem, src + index + soffset, mask=mask & (index + soffset < n_elements), other=0)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    gl.store(dst + index, smem.load(layout))


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_cdna5()), reason="Direct async GSAN requires AMD gfx950/gfx1250")
@pytest.mark.parametrize("buffer", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not is_hip_cdna4(), reason="Buffer-to-LDS instructions require gfx950"))
])
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4], indirect=True)
def test_gsan_amd_direct_async_masks_bounds_update_shadow(with_gsan, buffer):
    # Both the alternating user mask and the dynamic descriptor tail must
    # leave untouched shadow cells unchanged, even in byte-granular pools.
    src = torch.arange(80, dtype=torch.int32, device="cuda")
    dst = torch.empty(64, dtype=torch.int32, device="cuda")
    before = [_shadow_cells_for_tensor(src)]
    _gluon_amd_direct_async_kernel[(1, )](src, dst, 43, 5, buffer, num_warps=1)
    index = torch.arange(64, device="cuda")
    active = (index % 2 == 0) & (index + 5 < 43)
    expected = torch.where(active, src[index + 5], 0)
    torch.testing.assert_close(dst, expected)
    changed = torch.zeros((1, src.numel()), dtype=torch.bool)
    changed[0, 5:69] = active.cpu()
    _assert_shadow_mask(before, [_shadow_cells_for_tensor(src)], changed, access_kind="read")


@gluon.jit
def _gluon_amd_buffer_async_wide_kernel(src, dst, bound, scalar_offset):
    layout: gl.constexpr = gl.BlockedLayout([2], [THREADS_PER_WARP], [1], [0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    index = gl.arange(0, 128, layout)
    smem = gl.allocate_shared_memory(gl.float64, [128], shared)
    desc = gl.amd.cdna4.buffer.make_buffer_descriptor(src, bound)
    # Scaling each element offset by eight wraps independently in the two
    # hardware i32 offset fields, leaving effective offsets index and three.
    gl.amd.cdna4.buffer.load_to_shared(smem, desc, index + (1 << 29), soffset=scalar_offset)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    gl.store(dst + index, smem.load(layout))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Buffer-to-LDS instructions require gfx950")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4], indirect=True)
def test_gsan_amd_buffer_async_partial_dwords_wrapped_offsets(with_gsan):
    # The 260-byte bound cuts an f64 element after its first dword.  Inspect
    # integer storage and its shadow cells to distinguish that partial access
    # from an incorrectly checked full f64 element or signed i32 address.
    src_words = torch.arange(320, dtype=torch.int32, device="cuda")
    dst_words = torch.empty(256, dtype=torch.int32, device="cuda")
    before = [_shadow_cells_for_tensor(src_words)]
    _gluon_amd_buffer_async_wide_kernel[(1, )](src_words.view(torch.float64), dst_words.view(torch.float64), 65 * 4,
                                               (1 << 29) + 3, num_warps=1)
    expected = torch.zeros_like(dst_words)
    expected[:59] = src_words[6:65]
    torch.testing.assert_close(dst_words, expected)
    changed = torch.zeros((1, src_words.numel()), dtype=torch.bool)
    changed[0, 6:65] = True
    _assert_shadow_mask(before, [_shadow_cells_for_tensor(src_words)], changed, access_kind="read")


@gluon.jit
def _gluon_amd_buffer_async_subword_kernel(src, dst, bound, soffset, PACK: gl.constexpr, RANK: gl.constexpr):
    if RANK == 1:
        layout: gl.constexpr = gl.BlockedLayout([PACK], [THREADS_PER_WARP], [1], [0])
        shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        index = gl.arange(0, 64 * PACK, layout)
        smem = gl.allocate_shared_memory(src.dtype.element_ty, [64 * PACK], shared)
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, PACK], [1, THREADS_PER_WARP], [1, 1], [1, 0])
        shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        row = gl.arange(0, 16, gl.SliceLayout(1, layout))
        col = gl.arange(0, 64 * PACK, gl.SliceLayout(0, layout))
        index = row[:, None] * (64 * PACK) + col[None, :]
        smem = gl.allocate_shared_memory(src.dtype.element_ty, [16, 64 * PACK], shared)
    desc = gl.amd.cdna4.buffer.make_buffer_descriptor(src, bound)
    gl.amd.cdna4.buffer.load_to_shared(smem, desc, index, soffset=gl.cast(soffset, gl.uint32))
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    gl.store(dst + index, smem.load(layout))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Buffer-to-LDS instructions require gfx950")
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.float16])
@pytest.mark.parametrize("rank,soffset,bound", [(1, 0, 3), (1, 0, 5), (1, 0, 7), (2, 1, 519), (2, (1 << 31) + 1, 519)])
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4], indirect=True)
def test_gsan_amd_buffer_async_subwords_obey_dword_bounds(with_gsan, dtype, rank, soffset, bound):
    # Subword elements share each issued dword.  A bound that admits an
    # element but drops its dword must leave all its shadow cells untouched.
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    pack = 4 // elem_bytes
    n = 64 * pack * (16 if rank == 2 else 1)
    src = torch.arange(n + 8, dtype=torch.int32, device="cuda").to(dtype)
    dst = torch.empty(n, dtype=dtype, device="cuda")
    before = [_shadow_cells_for_tensor(src)]
    _gluon_amd_buffer_async_subword_kernel[(1, )](src, dst, bound, soffset, pack, rank, num_warps=1)
    effective_soffset = ((soffset * elem_bytes) % (1 << 32)) // elem_bytes
    count = max(0, (bound - effective_soffset * elem_bytes) // 4) * pack
    expected = torch.zeros_like(dst)
    expected[:count] = src[effective_soffset:effective_soffset + count]
    torch.testing.assert_close(dst, expected)
    changed = torch.zeros((1, src.numel()), dtype=torch.bool)
    changed[0, effective_soffset:effective_soffset + count] = True
    _assert_shadow_mask(before, [_shadow_cells_for_tensor(src)], changed, access_kind="read")


@gluon.jit
def _gsan_gfx1250_tdm_copy_kernel(src, dst, row, enabled, COMPLETION: gl.constexpr):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    src_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [32, 128], [128, 1], [16, 64], shared)
    dst_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [32, 128], [128, 1], [16, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    if COMPLETION:
        gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
    gl.amd.cdna5.tdm.async_load(src_desc, offsets=[row, 32], dest=smem, pred=enabled,
                                mbarrier=bar if COMPLETION else None)
    if COMPLETION:
        if enabled:
            gl.amd.cdna5.mbarrier.wait(bar, 0)
    else:
        gl.amd.cdna5.tdm.async_wait(0)
    dst_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(dst_desc, add_offsets=[row, 16], pred=enabled,
                                                         clamp_bounds=True)
    gl.amd.cdna5.tdm.async_store(dst_desc, src=smem, mbarrier=bar if COMPLETION else None)
    if COMPLETION:
        if enabled:
            gl.amd.cdna5.mbarrier.wait(bar, 1)
    else:
        gl.amd.cdna5.tdm.async_wait(0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("row", [0, 24])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("completion", [False, True])
def test_gsan_gfx1250_tdm_offsets_bounds_predicate(with_gsan, row, enabled, completion):
    src = torch.arange(32 * 128, dtype=torch.int32, device="cuda").reshape(32, 128)
    dst = torch.full_like(src, -1)
    before_src = _shadow_cells_for_tensor(src)
    before_dst = _shadow_cells_for_tensor(dst)
    _gsan_gfx1250_tdm_copy_kernel[(1, )](src, dst, row, enabled, completion)
    expected = torch.full_like(src, -1)
    reads = torch.zeros((32, 128), dtype=torch.bool)
    writes = torch.zeros_like(reads)
    if enabled:
        expected[row:row + 16, 16:80] = src[row:row + 16, 32:96]
        reads[row:row + 16, 32:96] = True
        writes[row:row + 16, 16:80] = True
    torch.testing.assert_close(dst, expected)
    _assert_shadow_mask(before_src, _shadow_cells_for_tensor(src), reads, access_kind="read")
    _assert_shadow_mask(before_dst, _shadow_cells_for_tensor(dst), writes, access_kind="write")


@gluon.jit
def _gsan_gfx1250_async_store_kernel(src, dst):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    index = gl.arange(0, 64, layout)
    values = gl.load(src + index)
    smem = gl.allocate_shared_memory(gl.int32, [64], gl.SwizzledSharedLayout(1, 1, 1, [0]), values)
    gl.amd.cdna5.async_copy.shared_to_global(dst + index, smem, mask=index % 2 == 0)
    gl.amd.cdna5.async_copy.commit_group()
    gl.amd.cdna5.async_copy.wait_group(0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Async LDS-to-global requires AMD gfx1250")
def test_gsan_gfx1250_async_store_mask_updates_shadow(with_gsan):
    src = torch.arange(64, dtype=torch.int32, device="cuda")
    dst = torch.full_like(src, -1)
    before = [_shadow_cells_for_tensor(dst)]
    _gsan_gfx1250_async_store_kernel[(1, )](src, dst, num_warps=1)
    active = torch.arange(64) % 2 == 0
    expected = torch.full_like(src, -1)
    expected[::2] = src[::2]
    torch.testing.assert_close(dst, expected)
    _assert_shadow_mask(before, [_shadow_cells_for_tensor(dst)], active[None, :], access_kind="write")


@gluon.jit
def _gsan_gfx1250_tdm_indexed_kernel(src, dst, enabled, COL: gl.constexpr, INDEX16: gl.constexpr, PADDED: gl.constexpr,
                                     ROW_BASE: gl.constexpr = 0, COMPLETION: gl.constexpr = False):
    shared: gl.constexpr = (gl.PaddedSharedLayout.with_identity_for([[64, 4]], [16, 64], [1, 0])
                            if PADDED else gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    parent: gl.constexpr = gl.BlockedLayout([16, 1], [1, 32], [1, 4], [1, 0])
    rows = ROW_BASE + 2 * gl.arange(0, 16, gl.SliceLayout(1, parent))
    if INDEX16:
        rows = rows.to(gl.uint16)
    src_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [ROW_BASE + 29, 128], [144, 1], [16, 64], shared)
    dst_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [ROW_BASE + 29, 128], [144, 1], [16, 64], shared)
    src_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(src_desc, add_offsets=[0, COL], pred=enabled,
                                                         clamp_bounds=True)
    dst_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(dst_desc, add_offsets=[0, COL], pred=enabled,
                                                         clamp_bounds=True)
    if PADDED:
        smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    else:
        parent_smem = gl.allocate_shared_memory(gl.int32, [32, 64], shared)
        smem = parent_smem.slice(16, 16, dim=0)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    if COMPLETION:
        gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
    gl.amd.cdna5.tdm.async_gather(src_desc, rows, smem, mbarrier=bar if COMPLETION else None)
    if COMPLETION:
        if enabled:
            gl.amd.cdna5.mbarrier.wait(bar, 0)
    else:
        gl.amd.cdna5.tdm.async_wait(0)
    gl.amd.cdna5.tdm.async_scatter(dst_desc, rows, smem, mbarrier=bar if COMPLETION else None)
    if COMPLETION:
        if enabled:
            gl.amd.cdna5.mbarrier.wait(bar, 1)
    else:
        gl.amd.cdna5.tdm.async_wait(0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("col", [32, 112])
@pytest.mark.parametrize("index16", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("completion", [False, True])
def test_gsan_gfx1250_tdm_indexed_footprints(with_gsan, enabled, col, index16, padded, completion):
    src = torch.arange(32 * 144, dtype=torch.int32, device="cuda").reshape(32, 144)
    dst = torch.full_like(src, -1)
    before_src = _shadow_cells_for_tensor(src)
    before_dst = _shadow_cells_for_tensor(dst)
    kernel = _gsan_gfx1250_tdm_indexed_kernel[(1, )](src, dst, enabled, col, index16, padded, COMPLETION=completion)
    expected = torch.full_like(src, -1)
    touched = torch.zeros((32, 144), dtype=torch.bool)
    if enabled:
        expected[:29:2, col:min(col + 64, 128)] = src[:29:2, col:min(col + 64, 128)]
        touched[:29:2, col:min(col + 64, 128)] = True
    torch.testing.assert_close(dst, expected)
    _assert_shadow_mask(before_src, _shadow_cells_for_tensor(src), touched, access_kind="read")
    _assert_shadow_mask(before_dst, _shadow_cells_for_tensor(dst), touched, access_kind="write")
    assert "__triton_gsan_load_indexed_tensor_desc" in kernel.asm["llir"]
    assert "__triton_gsan_store_indexed_tensor_desc" in kernel.asm["llir"]


@gluon.jit
def _gsan_gfx1250_tdm_fused_kernel(src0, src1, dst0, dst1, enabled):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc0 = gl.amd.cdna5.tdm.make_tensor_descriptor(src0, [32, 128], [128, 1], [16, 64], shared)
    desc1 = gl.amd.cdna5.tdm.make_tensor_descriptor(src1, [32, 128], [128, 1], [16, 64], shared)
    desc0 = gl.amd.cdna5.tdm.update_tensor_descriptor(desc0, add_offsets=[0, 16], clamp_bounds=True)
    desc1 = gl.amd.cdna5.tdm.update_tensor_descriptor(desc1, add_offsets=[24, 64], pred=enabled, clamp_bounds=True)
    smem0 = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    smem1 = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    gl.amd.cdna5.tdm.async_load_fused([(desc0, smem0, 3), (desc1, smem1, 12)])
    gl.amd.cdna5.tdm.async_wait(0)
    out0 = gl.amd.cdna5.tdm.make_tensor_descriptor(dst0, [16, 64], [64, 1], [16, 64], shared)
    out1 = gl.amd.cdna5.tdm.make_tensor_descriptor(dst1, [16, 64], [64, 1], [16, 64], shared)
    out1 = gl.amd.cdna5.tdm.update_tensor_descriptor(out1, pred=enabled)
    gl.amd.cdna5.tdm.async_store(out0, src=smem0)
    gl.amd.cdna5.tdm.async_store(out1, src=smem1)
    gl.amd.cdna5.tdm.async_wait(0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("enabled", [False, True])
def test_gsan_gfx1250_tdm_fused_member_footprints(with_gsan, enabled):
    src0 = torch.arange(32 * 128, dtype=torch.int32, device="cuda").reshape(32, 128)
    src1 = src0 + 10000
    dst0 = torch.full((16, 64), -1, dtype=torch.int32, device="cuda")
    dst1 = torch.full_like(dst0, -1)
    before0, before1 = _shadow_cells_for_tensor(src0), _shadow_cells_for_tensor(src1)
    _gsan_gfx1250_tdm_fused_kernel[(1, )](src0, src1, dst0, dst1, enabled)
    torch.testing.assert_close(dst0, src0[:16, 16:80])
    expected1 = torch.zeros_like(dst1) if enabled else torch.full_like(dst1, -1)
    if enabled:
        expected1[:8] = src1[24:, 64:]
    torch.testing.assert_close(dst1, expected1)
    reads0 = torch.zeros((32, 128), dtype=torch.bool)
    reads1 = torch.zeros_like(reads0)
    reads0[:16, 16:80] = True
    if enabled:
        reads1[24:, 64:] = True
    _assert_shadow_mask(before0, _shadow_cells_for_tensor(src0), reads0, access_kind="read")
    _assert_shadow_mask(before1, _shadow_cells_for_tensor(src1), reads1, access_kind="read")


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
def test_gsan_gfx1250_tdm_unsigned16_row_indices(with_gsan):
    # A signed extension would drop every high-bit index while hardware reads
    # rows >= 32768.  Inspect only that tail's shadow cells.
    src = torch.arange((32768 + 32) * 144, dtype=torch.int32, device="cuda").reshape(-1, 144)
    dst = torch.full_like(src, -1)
    before_src = _shadow_cells_for_tensor(src[32768:])
    before_dst = _shadow_cells_for_tensor(dst[32768:])
    _gsan_gfx1250_tdm_indexed_kernel[(1, )](src, dst, True, 32, True, False, 32768)
    expected = torch.full_like(dst[32768:], -1)
    expected[:29:2, 32:96] = src[32768:32797:2, 32:96]
    torch.testing.assert_close(dst[32768:], expected)
    touched = torch.zeros((32, 144), dtype=torch.bool)
    touched[:29:2, 32:96] = True
    _assert_shadow_mask(before_src, _shadow_cells_for_tensor(src[32768:]), touched, access_kind="read")
    _assert_shadow_mask(before_dst, _shadow_cells_for_tensor(dst[32768:]), touched, access_kind="write")


@gluon.jit
def _gsan_gfx1250_tdm_unsigned32_broadcast_kernel(src, dst, nrows):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    parent: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    # Broadcast rows map to the same allocated row.  One indexed row avoids
    # conflicting scatter destinations, while exercising bit31 of the index.
    rows = gl.full([1], 0x80000000, gl.uint32, gl.SliceLayout(1, parent))
    nrows = nrows.to(gl.uint32)
    src_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [nrows, 64], [0, 1], [1, 64], shared)
    dst_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [nrows, 64], [0, 1], [1, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [1, 64], shared)
    gl.amd.cdna5.tdm.async_gather(src_desc, rows, smem)
    gl.amd.cdna5.tdm.async_wait(0)
    gl.amd.cdna5.tdm.async_scatter(dst_desc, rows, smem)
    gl.amd.cdna5.tdm.async_wait(0)


@pytest.mark.skipif(not is_hip_cdna5(), reason="TDM requires AMD gfx1250")
@pytest.mark.parametrize("sanitized", [False, True])
@pytest.mark.parametrize("in_bounds", [False, True])
def test_gsan_gfx1250_tdm_unsigned32_broadcast_footprint(with_gsan, sanitized, in_bounds):
    triton.knobs.compilation.instrumentation_mode = "gsan" if sanitized else ""
    src = torch.arange(64, dtype=torch.int32, device="cuda").reshape(1, 64)
    dst = torch.full_like(src, -1)
    before_src = _shadow_cells_for_tensor(src)
    before_dst = _shadow_cells_for_tensor(dst)
    _gsan_gfx1250_tdm_unsigned32_broadcast_kernel[(1, )](src, dst, 0x80000010 if in_bounds else 0x80000000)
    torch.testing.assert_close(dst, src if in_bounds else torch.full_like(src, -1))
    if sanitized:
        touched = torch.full((1, 64), in_bounds, dtype=torch.bool)
        _assert_shadow_mask(before_src, _shadow_cells_for_tensor(src), touched, access_kind="read")
        _assert_shadow_mask(before_dst, _shadow_cells_for_tensor(dst), touched, access_kind="write")


@gluon.jit
def _gsan_gfx1250_named_barrier_kernel(out, selector):
    bars = gl.allocate_shared_memory(gl.int64, [2, 1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    gl.amd.cdna5.mbarrier.init(bars.index(0), count=256)
    gl.amd.cdna5.mbarrier.init(bars.index(1), count=256)
    gl.amd.cdna5.mbarrier.wait(bars.index(selector), 1)
    for iteration in range(8):
        bar = bars.index((iteration + selector) % 2)
        gl.amd.cdna5.mbarrier.arrive(bar, count=1)
        phase = gl.amd.cdna5.mbarrier.arrive(bar, count=1)
        gl.amd.cdna5.mbarrier.wait(bar, phase)
        gl.amd.cdna5.mbarrier.wait(bar, phase)
        gl.store(out + iteration, phase)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Named barriers require AMD gfx1250")
@pytest.mark.parametrize("selector", [0, 1])
def test_gsan_gfx1250_named_barrier_dynamic_alias_phases(with_gsan, selector):
    out = torch.empty(8, dtype=torch.int32, device="cuda")
    _gsan_gfx1250_named_barrier_kernel[(1, )](out, selector, num_warps=4)
    torch.testing.assert_close(out, torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int32, device="cuda"))


@gluon.jit
def _gsan_gfx1250_async_completion_kernel(src, dst):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    rows = gl.arange(0, 16, gl.SliceLayout(1, layout))
    cols = gl.arange(0, 64, gl.SliceLayout(0, layout))
    indices = rows[:, None] * 64 + cols[None, :]
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    gl.amd.cdna5.mbarrier.init(bar, count=256)
    gl.amd.cdna5.async_copy.global_to_shared(smem, src + indices)
    gl.amd.cdna5.async_copy.mbarrier_arrive(bar)
    phase = gl.amd.cdna5.mbarrier.arrive(bar)
    gl.amd.cdna5.mbarrier.wait(bar, phase)
    gl.store(dst + indices, smem.load(layout))


@pytest.mark.skipif(not is_hip_cdna5(), reason="Completion barriers require AMD gfx1250")
def test_gsan_gfx1250_async_completion_barrier_payload(with_gsan):
    src = torch.arange(1024, dtype=torch.int32, device="cuda")
    dst = torch.empty_like(src)
    _gsan_gfx1250_async_completion_kernel[(1, )](src, dst, num_warps=4)
    torch.testing.assert_close(dst, src)


@gluon.jit
def _gsan_gfx1250_cluster_copy_kernel(src, dst, CTAS: gl.constexpr, REPLICATED: gl.constexpr):
    distributed: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0],
                                                 cga_layout=(((1, ), ) if CTAS == 2 else
                                                             ((1, ), (2, )) if CTAS == 4 else
                                                             ((1, ), (2, ), (4, )) if CTAS == 8 else
                                                             ((1, ), (2, ), (4, ), (8, ))))
    replicated: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=((0, ), ) if CTAS == 2 else
                                                ((0, ), (0, )))
    offsets = gl.program_id(0) * (128 * CTAS) + gl.arange(0, 128 * CTAS, distributed)
    if REPLICATED:
        read_offsets = gl.program_id(0) * (128 * CTAS) + gl.arange(0, 128 * CTAS, replicated)
        values = gl.load(src + read_offsets)
        values = gl.convert_layout(values, distributed)
    else:
        values = gl.load(src + offsets)
    # This rendezvous aligns execution; it does not publish global clocks.
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()
    gl.store(dst + offsets, values)


def _run_gsan_gfx1250_cluster_copy(ctas, replicated, sanitized, granularity):
    from triton.experimental.gsan import configure
    configure(num_slots=max(1024, 256 * ctas), clock_buffer_size=4)
    triton.knobs.compilation.instrumentation_mode = "gsan" if sanitized else ""
    pool = create_mem_pool(shadow_granularity=granularity)
    with torch.cuda.use_mem_pool(pool):
        src = torch.arange(128 * ctas, dtype=torch.int32, device="cuda")
        dst = torch.empty_like(src)
        triton.knobs.compilation.instrumentation_mode = "gsan" if sanitized else ""
        compiled = _gsan_gfx1250_cluster_copy_kernel[(1, )](src, dst, ctas, replicated, num_ctas=ctas,
                                                            instrumentation_mode="gsan" if sanitized else "")
        torch.testing.assert_close(dst, src)
        if not sanitized:
            return
        assert compiled.metadata.num_ctas == ctas
        from triton.backends.amd.compiler import get_static_lds_size
        assert compiled.metadata.shared + get_static_lds_size(compiled.asm["amdgcn"]) == 320 * 1024 // ctas
        writers = {
            shadow_cell_from_address(dst.data_ptr() + 128 * cta * 4).write_clock.thread_id
            for cta in range(ctas)
        }
        assert len(writers) == ctas
        for offset in [0, 127, 128 * ctas - 1]:
            cell = shadow_cell_from_address(src.data_ptr() + offset * 4)
            readers = {clock.thread_id for clock in cell.read_clocks if clock.epoch != 0}
            assert len(readers) == (ctas if replicated else 1)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Clusters require AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("sanitized", [False, True])
@pytest.mark.gsan_fine_granularity("White-box reader checks use 4-byte elements")
def test_gsan_gfx1250_cluster_copy_participants(shadow_granularity, ctas, replicated, sanitized, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_copy, args=(ctas, replicated, sanitized, shadow_granularity))
    assert result.exc is None, result.exc


@gluon.jit
def _gsan_gfx1250_cluster_tdm_kernel(src, dst, CTAS: gl.constexpr, MULTICAST: gl.constexpr,
                                     COMPLETION: gl.constexpr = False):
    cga: gl.constexpr = ((((0, 1), (0, 0)) if CTAS == 4 else ((0, 1), (0, 0), (0, 0)) if CTAS == 8 else
                          ((0, 1), (0, 2), (0, 0), (0, 0))) if MULTICAST == 2 else ((0, 1), ) if CTAS == 2 else
                         ((0, 1), (0, 2)))
    shared_cga: gl.constexpr = (cga if MULTICAST == 2 else (((0, 0), ) if CTAS == 2 else
                                                            ((0, 0), (0, 0))) if MULTICAST else cga)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=cga)
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], shared_cga)
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [16, 64], [64, 1], [16, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    if COMPLETION:
        bar_cga: gl.constexpr = ((0, ), ) if CTAS == 2 else ((0, ), (0, ))
        bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout(bar_cga))
        gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
        gl.amd.cdna5.tdm.async_load(desc, [0, 0], smem, mbarrier=bar)
        gl.amd.cdna5.mbarrier.wait(bar, 0)
        gl.amd.cdna5.mbarrier.wait(bar, 0)
    else:
        gl.amd.cdna5.tdm.async_load(desc, [0, 0], smem)
        gl.amd.cdna5.tdm.async_wait(0)
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()
    rows = gl.arange(0, 16, gl.SliceLayout(1, layout))
    cols = gl.arange(0, 64, gl.SliceLayout(0, layout))
    read_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], cga_layout=shared_cga)
    values = gl.convert_layout(smem.load(read_layout), layout)
    gl.store(dst + rows[:, None] * 64 + cols[None, :], values)


def _run_gsan_gfx1250_cluster_tdm(ctas, multicast, granularity, completion=False):
    from triton.experimental.gsan import configure
    configure(num_slots=max(1024, ctas * 256), clock_buffer_size=16)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool(shadow_granularity=granularity)
    with torch.cuda.use_mem_pool(pool):
        src = torch.arange(1024, dtype=torch.int32, device="cuda").reshape(16, 64)
        dst = torch.empty_like(src)
        _gsan_gfx1250_cluster_tdm_kernel[(1, )](src, dst, ctas, multicast, completion, num_ctas=ctas)
        torch.testing.assert_close(dst, src)
        for offset in [0, 63, 1023]:
            cell = shadow_cell_from_address(src.data_ptr() + offset * 4)
            readers = {clock.thread_id for clock in cell.read_clocks if clock.epoch != 0}
            assert len(readers) == ((2 if ctas == 4 else 4) if multicast == 2 else ctas if multicast else 1)

        if multicast == 2:
            first = shadow_cell_from_address(src.data_ptr())
            last = shadow_cell_from_address(src.data_ptr() + 63 * 4)
            first_readers = {clock.thread_id for clock in first.read_clocks if clock.epoch}
            last_readers = {clock.thread_id for clock in last.read_clocks if clock.epoch}
            assert first_readers.isdisjoint(last_readers)
            for reader in first_readers:
                state = thread_state_from_smid(reader)
                assert all(state.vector_clock[other] == 0 for other in last_readers)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Cluster TDM requires AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.gsan_fine_granularity("White-box reader checks use 4-byte elements")
def test_gsan_gfx1250_cluster_tdm_recipient_footprints(shadow_granularity, ctas, multicast, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_tdm, args=(ctas, multicast, shadow_granularity))
    assert result.exc is None, result.exc


@gluon.jit
def _gsan_gfx1250_cluster_atomic_handoff_kernel(payload, flag, out, CTAS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=((1, ), ) if CTAS == 2 else ((1, ), (2, )))
    offsets = gl.arange(0, 128 * CTAS, layout)
    consumer = (offsets > 0) & (offsets % 128 == 0)
    gl.store(payload + offsets * 0, 7, mask=offsets == 0)
    gl.atomic_add(flag + offsets * 0, 1, mask=offsets == 0, sem="release", scope="gpu")
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()
    gl.atomic_add(flag + offsets * 0, 0, mask=consumer, sem="acquire", scope="gpu")
    value = gl.load(payload + offsets * 0, mask=consumer, other=0)
    gl.store(out + offsets // 128, value, mask=consumer)


def _run_gsan_gfx1250_cluster_atomic_handoff(ctas):
    from triton.experimental.gsan import configure
    configure(num_slots=1024, clock_buffer_size=16)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        payload = torch.zeros(1, dtype=torch.int32, device="cuda")
        flag = torch.zeros_like(payload)
        out = torch.zeros(ctas, dtype=torch.int32, device="cuda")
        _gsan_gfx1250_cluster_atomic_handoff_kernel[(1, )](payload, flag, out, ctas, num_ctas=ctas)
        torch.testing.assert_close(out[1:], torch.full_like(out[1:], 7))
        cell = shadow_cell_from_address(payload.data_ptr())
        readers = [clock for clock in cell.read_clocks if clock.epoch != 0]
        assert len(readers) == ctas - 1
        for reader in readers:
            assert reader.thread_id != cell.write_clock.thread_id
            state = thread_state_from_smid(reader.thread_id)
            assert state.vector_clock[cell.write_clock.thread_id] >= cell.write_clock.epoch


@pytest.mark.skipif(not is_hip_cdna5(), reason="Cluster atomics require AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
def test_gsan_gfx1250_cluster_atomic_release_acquire(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_atomic_handoff, args=(ctas, ))
    assert result.exc is None, result.exc


@pytest.mark.skipif(not is_hip_cdna5(), reason="Cluster completion TDM requires AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
def test_gsan_gfx1250_cluster_tdm_local_completion(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_tdm, args=(ctas, True, 4, True))
    assert result.exc is None, result.exc


def _run_gsan_gfx1250_cluster_streams(ctas):
    from triton.experimental.gsan import configure
    configure(num_slots=1024, clock_buffer_size=16)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        src = torch.arange(300 * 128 * ctas, dtype=torch.int32, device="cuda")
        outputs = [torch.empty_like(src), torch.empty_like(src)]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for stream, out in zip(streams, outputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _gsan_gfx1250_cluster_copy_kernel[(300, )](src, out, ctas, False, num_ctas=ctas)
        for stream in streams:
            stream.synchronize()
        for out in outputs:
            torch.testing.assert_close(out, src)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Clusters require AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
def test_gsan_gfx1250_cluster_oversubscribed_streams(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_streams, args=(ctas, ))
    assert result.exc is None, result.exc


def _run_gsan_gfx1250_cluster_insufficient_slots():
    from triton.experimental.gsan import configure
    configure(num_slots=2, clock_buffer_size=16)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        src = torch.zeros(256, dtype=torch.int32, device="cuda")
        out = torch.empty_like(src)
        with pytest.raises(RuntimeError, match="GSan cluster launch needs at least .*runtime slots"):
            _gsan_gfx1250_cluster_copy_kernel[(1, )](src, out, 2, False, num_ctas=2)


@pytest.mark.skipif(not is_hip_cdna5(), reason="Clusters require AMD gfx1250")
def test_gsan_gfx1250_cluster_insufficient_slots_rejected(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_insufficient_slots)
    assert result.exc is None, result.exc


@pytest.mark.skipif(not is_hip_cdna5(), reason="Clusters require AMD gfx1250")
@pytest.mark.parametrize("ctas", [8, 16])
def test_gsan_gfx1250_cluster_large_launch_contract(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_copy, args=(ctas, False, True, 4))
    assert result.exc is None, result.exc


@gluon.jit
def _gsan_gfx1250_cluster_region_default(src, dst, layout: gl.constexpr):
    pass


@gluon.jit(noinline=True)
def _gsan_gfx1250_cluster_region_worker(src, dst, layout: gl.constexpr):
    offsets = gl.arange(0, 256, layout)
    gl.store(dst + offsets, gl.load(src + offsets))


@gluon.jit
def _gsan_gfx1250_cluster_region_kernel(src, dst):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=((1, ), ))
    gl.warp_specialize([(_gsan_gfx1250_cluster_region_default, (src, dst, layout)),
                        (_gsan_gfx1250_cluster_region_worker, (src, dst, layout))], [4], [24])
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()


def _run_gsan_gfx1250_cluster_region():
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        src = torch.arange(256, dtype=torch.int32, device="cuda")
        dst = torch.empty_like(src)
        compiled = _gsan_gfx1250_cluster_region_kernel[(1, )](src, dst, num_ctas=2)
        torch.testing.assert_close(dst, src)
        assert "tt.call" in compiled.asm["ttgir"]
        writers = {shadow_cell_from_address(dst.data_ptr() + cta * 128 * 4).write_clock.thread_id for cta in range(2)}
        assert len(writers) == 2


@pytest.mark.skipif(not is_hip_cdna5(), reason="Cluster regions require AMD gfx1250")
def test_gsan_gfx1250_cluster_noinline_worker_region(monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_region)
    assert result.exc is None, result.exc


@gluon.jit
def _gsan_gfx1250_cluster_indexed_kernel(src, dst, CTAS: gl.constexpr):
    cga: gl.constexpr = ((0, 1), ) if CTAS == 2 else ((0, 1), (0, 2))
    idx_cga: gl.constexpr = ((0, 0), ) if CTAS == 2 else ((0, 0), (0, 0))
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0], cga)
    parent: gl.constexpr = gl.BlockedLayout([16, 1], [1, 32], [1, 4], [1, 0], cga_layout=idx_cga)
    rows = 2 * gl.arange(0, 16, gl.SliceLayout(1, parent))
    src_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [31, 64], [80, 1], [16, 64], shared)
    dst_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [31, 64], [80, 1], [16, 64], shared)
    src_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(src_desc, add_offsets=[0, 8], clamp_bounds=True)
    dst_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(dst_desc, add_offsets=[0, 8], clamp_bounds=True)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    gl.amd.cdna5.tdm.async_gather(src_desc, rows, smem)
    gl.amd.cdna5.tdm.async_wait(0)
    gl.amd.cdna5.tdm.async_scatter(dst_desc, rows, smem)
    gl.amd.cdna5.tdm.async_wait(0)


def _run_gsan_gfx1250_cluster_indexed(ctas):
    from triton.experimental.gsan import configure
    configure(num_slots=1024, clock_buffer_size=16)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        src = (1 + torch.arange(32 * 80, dtype=torch.int32, device="cuda")).reshape(32, 80)
        dst = torch.zeros_like(src)
        _gsan_gfx1250_cluster_indexed_kernel[(1, )](src, dst, ctas, num_ctas=ctas)
        expected = torch.zeros_like(src)
        expected[:31:2, 8:64] = src[:31:2, 8:64]
        torch.testing.assert_close(dst, expected)
        writers = {
            shadow_cell_from_address(dst.data_ptr() + (8 + cta * (64 // ctas)) * 4).write_clock.thread_id
            for cta in range(ctas)
        }
        assert len(writers) == ctas
        for offset in [0, 64, 80, 31 * 80]:
            cell = shadow_cell_from_address(src.data_ptr() + offset * 4)
            assert all(clock.epoch == 0 for clock in cell.read_clocks)
        for offset in [8, 30 * 80 + 63]:
            cell = shadow_cell_from_address(src.data_ptr() + offset * 4)
            assert len({clock.thread_id for clock in cell.read_clocks if clock.epoch}) == 1


@pytest.mark.skipif(not is_hip_cdna5(), reason="Cluster indexed TDM requires AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
def test_gsan_gfx1250_cluster_indexed_column_footprints(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_indexed, args=(ctas, ))
    assert result.exc is None, result.exc


@pytest.mark.skipif(not is_hip_cdna5(), reason="Multicast subgroups require AMD gfx1250")
@pytest.mark.parametrize("ctas", [4, 8, 16])
def test_gsan_gfx1250_cluster_multicast_nonrecipient_independence(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_cluster_tdm, args=(ctas, 2, 4))
    assert result.exc is None, result.exc


@gluon.jit
def _gsan_gfx1250_acceptance_initialize_kernel(ptr):
    offsets = gl.arange(0, 2048, gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(ptr + offsets, offsets * 3 + 11)


@gluon.jit
def _gsan_gfx1250_acceptance_tdm_worker(src, dst):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    layout: gl.constexpr = gl.BlockedLayout([16, 1], [1, 32], [1, 4], [1, 0])
    rows = gl.arange(0, 16, gl.SliceLayout(1, layout))
    source = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [32, 64], [64, 1], [16, 64], shared)
    destination = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [16, 64], [64, 1], [16, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
    for iteration in range(3):
        gl.amd.cdna5.tdm.async_gather(source, rows * 2, smem, mbarrier=bar)
        gl.amd.cdna5.mbarrier.wait(bar, 0)
        gl.amd.cdna5.tdm.async_scatter(destination, 15 - rows, smem, mbarrier=bar)
        gl.amd.cdna5.mbarrier.wait(bar, 1)


@gluon.jit
def _gsan_gfx1250_acceptance_default(src, dst):
    pass


@gluon.jit
def _gsan_gfx1250_acceptance_region_kernel(src, dst):
    gl.warp_specialize([(_gsan_gfx1250_acceptance_default, (src, dst)),
                        (_gsan_gfx1250_acceptance_tdm_worker, (src, dst))], [4], [24])


@gluon.jit(noinline=True)
def _gsan_gfx1250_acceptance_rectangular_worker(src, dst):
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    source = gl.amd.cdna5.tdm.make_tensor_descriptor(src, [16, 64], [64, 1], [16, 64], shared)
    destination = gl.amd.cdna5.tdm.make_tensor_descriptor(dst, [16, 64], [64, 1], [16, 64], shared)
    smem = gl.allocate_shared_memory(gl.int32, [16, 64], shared)
    bar = gl.allocate_shared_memory(gl.int64, [1], gl.amd.cdna5.mbarrier.MBarrierLayout())
    gl.amd.cdna5.mbarrier.init(bar, count=gl.num_warps())
    gl.amd.cdna5.tdm.async_load(source, [0, 0], smem, mbarrier=bar)
    gl.amd.cdna5.mbarrier.wait(bar, 0)
    gl.amd.cdna5.tdm.async_store(destination, [0, 0], smem, mbarrier=bar)
    gl.amd.cdna5.mbarrier.wait(bar, 1)


@gluon.jit
def _gsan_gfx1250_acceptance_noinline_region_kernel(src, dst):
    gl.warp_specialize([(_gsan_gfx1250_acceptance_default, (src, dst)),
                        (_gsan_gfx1250_acceptance_rectangular_worker, (src, dst))], [4], [24])


@gluon.jit
def _gsan_gfx1250_acceptance_consumer_kernel(src, dst):
    offsets = gl.arange(0, 1024, gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(dst + offsets, gl.load(src + offsets) + 1)


def _run_gsan_gfx1250_composed_graph_cluster(ctas):
    from triton.experimental.gsan import configure, graph
    from triton.experimental.gsan._testing_utils import ExplicitGraph, shadow_tensor_for

    if ctas == 4:
        configure(num_slots=1024)
    triton.knobs.compilation.instrumentation_mode = "gsan"
    normal_pool = create_mem_pool()
    compact_pool = create_mem_pool(write_once=True)
    with torch.cuda.use_mem_pool(normal_pool):
        with torch.cuda.use_mem_pool(compact_pool):
            source = torch.empty((32, 64), dtype=torch.int32, device="cuda")
        middle, rectangular_result, graph_result, cluster_result, result = [
            torch.empty((16, 64), dtype=torch.int32, device="cuda") for _ in range(5)
        ]
        # Instrumented initialization publishes both write-once history and
        # the eager predecessor consumed by graph entry.
        _gsan_gfx1250_acceptance_initialize_kernel[(1, )](source)
        compact_shadow = shadow_tensor_for(source).cpu().clone()
        assert torch.all(compact_shadow.view(torch.int32) != 0)
        plan = graph.GraphPlan(torch.cuda.current_device(),
                               [graph.GraphNode(), graph.GraphNode(
                                   (0, )), graph.GraphNode((1, ))])
        worker = _gsan_gfx1250_acceptance_region_kernel.warmup(source, middle, grid=(1, ))
        rectangular = _gsan_gfx1250_acceptance_noinline_region_kernel.warmup(middle, rectangular_result, grid=(1, ))
        consumer = _gsan_gfx1250_acceptance_consumer_kernel.warmup(rectangular_result, graph_result, grid=(1, ))
        other_stream = torch.cuda.Stream()
        expected = ((torch.arange(2048, dtype=torch.int32).reshape(32, 64)[::2].flip(0) * 3 + 11) + 2).cuda()
        with ExplicitGraph(plan) as executable:
            executable.add_kernel(worker, (source, middle), 1)
            executable.add_kernel(rectangular, (middle, rectangular_result), 1)
            executable.add_kernel(consumer, (rectangular_result, graph_result), 1)
            executable.instantiate()
            for iteration in range(3):
                stream = other_stream if iteration % 2 else torch.cuda.current_stream()
                if stream != torch.cuda.current_stream():
                    stream.wait_stream(torch.cuda.current_stream())
                completion = executable.launch(stream)
            torch.cuda.current_stream().wait_stream(other_stream)
            graph.acquire_completions(torch.cuda.current_device(),
                                      torch.cuda.current_stream().cuda_stream, (completion, ))
            # Native multicast recipients execute after actual graph completion.
            # The raw cluster rendezvous inside this kernel is execution-only.
            _gsan_gfx1250_cluster_tdm_kernel[(1, )](graph_result, cluster_result, ctas, True, True, num_ctas=ctas)
            _gsan_gfx1250_acceptance_consumer_kernel[(1, )](cluster_result, result)
            torch.testing.assert_close(result, expected)
            assert torch.equal(shadow_tensor_for(source).cpu(), compact_shadow)
            cell = shadow_cell_from_address(graph_result.data_ptr())
            readers = [clock for clock in cell.read_clocks if clock.epoch]
            assert len({clock.thread_id for clock in readers}) == ctas
            for reader in readers:
                state = thread_state_from_smid(reader.thread_id)
                assert state.vector_clock[cell.write_clock.thread_id] >= cell.write_clock.epoch


@pytest.mark.skipif(not is_hip_cdna5(), reason="Composed TDM/graph/cluster acceptance requires AMD gfx1250")
@pytest.mark.parametrize("ctas", [2, 4])
def test_gsan_gfx1250_composed_write_once_graph_cluster(ctas, monkeypatch):
    monkeypatch.setenv("TRITON_TEST_PROCESS_TIMEOUT", "60")
    result = run_in_process(_run_gsan_gfx1250_composed_graph_cluster, args=(ctas, ))
    assert result.exc is None, (result.exc, result.driver_stderr_output)
