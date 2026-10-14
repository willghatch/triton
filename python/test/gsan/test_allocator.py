from __future__ import annotations

import os

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton._internal_testing import is_cuda, is_hip, run_in_process
from triton.experimental.gsan import ShareableHandleType, configure, create_mem_pool, freeze_config, get_allocator, has_live_allocations, reset
from triton.experimental.gsan import _stream_sync
from triton._C.libtriton.gsan_testing import (PER_DEVICE_STATE_STRIDE_BYTES, shadow_cell_address, shadow_granularity as
                                              get_shadow_granularity)
from triton.experimental.gsan._allocator import (
    export_allocation_handles,
    export_allocation_memhandle_regions,
    export_runtime_state_handle,
    free_allocation,
    get_device_rank,
    get_global_state_pointer,
    get_reserve_pointer,
    get_reserve_size,
    get_runtime_state_layout,
    gsan_free,
    gsan_malloc,
    import_allocation_handles,
    is_write_once_allocation,
    import_runtime_state_handle,
    supports_fabric_handles,
)
from triton.experimental.gsan._testing_utils import (global_state, shadow_cell_from_address, shadow_tensor_for,
                                                     store_one_i32, thread_state_from_smid)
from triton.experimental.gsan._utils import uint8_cuda_tensor_from_ptr


def has_gsan_allocator():
    return is_cuda() or is_hip()


# Allocator-only tests also run on HIP. Tests that launch GSan-instrumented
# kernels or use fabric handles require CUDA.
requires_gsan_allocator = pytest.mark.skipif(not has_gsan_allocator(), reason="requires CUDA or HIP backend")

# With 2 MiB pages, this rounds to a 6 MiB allocation inside an 8 MiB tree node.
# This tests cases where AllocNode.size != AllocNode.allocSize
_ODD_LARGE_ALLOCATION_SIZE = 4 * 1024 * 1024 + 1


def _run_configure_check(device_ranks: dict[int, int], num_devices: int) -> None:
    configure(device_ranks=device_ranks, num_devices=num_devices)
    assert get_device_rank(0) == device_ranks[0]
    assert get_device_rank(1) == device_ranks[1]


def _run_configure_runtime_fields_check(shadow_granularity: int) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345, clock_buffer_size=17)

    ptr = gsan_malloc(1, device, 0, shadow_granularity=shadow_granularity)
    try:
        state = global_state(device_index=device)
        assert state.rng_seed == 12345
        assert state.clock_buffer_size == 17
    finally:
        gsan_free(ptr, device, 0, 0)


def _run_default_clock_buffer_fits_check() -> None:
    device = torch.cuda.current_device()
    ptr = gsan_malloc(1, device)
    assert ptr != 0
    try:
        layout = get_runtime_state_layout(get_device_rank(device))
    finally:
        gsan_free(ptr, device)
    size = layout["clock_buffer_size"]
    num_sms, num_threads = layout["num_sms"], layout["num_threads"]
    state_end = layout["thread_state_base_ptr"] + layout["thread_state_stride_bytes"] * num_sms
    if "slot_table_ptr" in layout:
        # HIP: the slot table follows the thread states.
        state_end = layout["slot_table_ptr"] + num_sms * layout["slot_entry_bytes"]
    used = state_end - layout["global_state_ptr"]
    assert 0 < size <= 1024
    assert used <= PER_DEVICE_STATE_STRIDE_BYTES
    if size < 1024:
        # Only shrunk because one more entry per SM (plus alignment) would not fit.
        assert used + num_sms * (2 * num_threads + 16) > PER_DEVICE_STATE_STRIDE_BYTES


def _run_clock_buffer_size_must_fit_check() -> None:
    with pytest.raises(ValueError, match=r"clock_buffer_size must be in \[1, 65535\]"):
        configure(clock_buffer_size=65536)
    configure(clock_buffer_size=65535)
    assert gsan_malloc(1, torch.cuda.current_device()) == 0


def _run_freeze_config_check() -> None:
    configure(rng_seed=12345)
    freeze_config()
    try:
        configure(rng_seed=12345)
    except RuntimeError as exc:
        assert "configuration is already frozen" in str(exc)
    else:
        raise AssertionError("expected freeze_config() to reject later config changes")


def _run_allocator_freezes_config_check(shadow_granularity: int) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345)
    ptr = gsan_malloc(1, device, 0, shadow_granularity=shadow_granularity)
    try:
        try:
            configure(rng_seed=12345)
        except RuntimeError as exc:
            assert "configuration is already frozen" in str(exc)
        else:
            raise AssertionError("expected allocator initialization to freeze config")
    finally:
        gsan_free(ptr, device, 0, 0)


@gluon.jit
def _store_reset_target(ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([4], [32], [1], [0]))
    gl.store(ptr + offsets, 1)


def _run_has_live_allocations_check(shadow_granularity) -> None:
    assert has_live_allocations() is False
    configure(rng_seed=12345)
    device = torch.cuda.current_device()
    ptr = gsan_malloc(1, device, shadow_granularity=shadow_granularity)
    try:
        assert has_live_allocations() is True
    finally:
        gsan_free(ptr, device)
    assert has_live_allocations() is False


def _run_reset_rejects_live_allocations_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    try:
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
            torch.cuda.synchronize()

            stream_state = _stream_sync._launch_stream_state(device, stream)
            runtime_layout = _stream_sync._runtime_state_layout(get_device_rank(device), device)
            clocks_before = stream_state.clocks.cpu()
            shadow_before = shadow_tensor_for(target).cpu()
            thread_id = shadow_cell_from_address(ptr).write_clock.thread_id
            thread_state_before = thread_state_from_smid(thread_id)
            assert stream_state.next_kernel_id == 1

            with pytest.raises(AssertionError, match="GSan allocations are still live"):
                reset()

            assert has_live_allocations() is True
            assert _stream_sync._launch_stream_state(device, stream) is stream_state
            assert _stream_sync._runtime_state_layout(get_device_rank(device), device) is runtime_layout
            assert stream_state.next_kernel_id == 1
            assert torch.equal(stream_state.clocks.cpu(), clocks_before)
            assert torch.equal(shadow_tensor_for(target).cpu(), shadow_before)
            assert thread_state_from_smid(thread_id).vector_clock == thread_state_before.vector_clock

            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
            torch.cuda.synchronize()
            assert stream_state.next_kernel_id == 2
    finally:
        del target
        gsan_free(ptr, device)

    reset()


def _run_reset_collects_unreachable_allocations_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()

    class CyclicAllocation:

        def __init__(self):
            self.ptr = gsan_malloc(1, device, shadow_granularity=shadow_granularity)
            self.cycle = self

        def __del__(self):
            gsan_free(self.ptr, device)

    allocation = CyclicAllocation()
    del allocation
    reset()


def _run_reset_reinitializes_runtime_state_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    configure(rng_seed=12345, clock_buffer_size=17)
    original_state_pointer = get_global_state_pointer()

    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "gsan"
        _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
    torch.cuda.synchronize()

    thread_id = shadow_cell_from_address(ptr).write_clock.thread_id
    old_thread_state = thread_state_from_smid(thread_id)
    assert old_thread_state.globals_ptr != 0
    assert old_thread_state.vector_clock[thread_id] != 0
    assert _stream_sync._launch_stream_state.cache_info().currsize == 1
    assert _stream_sync._runtime_state_layout.cache_info().currsize == 1

    del target
    gsan_free(ptr, device)
    reset()

    assert get_global_state_pointer() == original_state_pointer
    assert _stream_sync._launch_stream_state.cache_info().currsize == 0
    assert _stream_sync._runtime_state_layout.cache_info().currsize == 0

    state = global_state(device_index=device)
    assert state.rng_seed == 12345
    assert state.clock_buffer_size == 17
    with pytest.raises(RuntimeError, match="configuration is already frozen"):
        configure(rng_seed=12345)

    new_thread_state = thread_state_from_smid(thread_id)
    assert new_thread_state.globals_ptr == 0
    assert all(epoch == 0 for epoch in new_thread_state.vector_clock)

    ptr = gsan_malloc(max(4, shadow_granularity), device, shadow_granularity=shadow_granularity)
    target = uint8_cuda_tensor_from_ptr(ptr, max(4, shadow_granularity), device).view(torch.int32)
    try:
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        assert _stream_sync._launch_stream_state(device, stream).next_kernel_id == 1
    finally:
        del target
        gsan_free(ptr, device)


def _run_reset_releases_cached_pool_clocks_check(num_pools: int, shadow_granularity) -> None:
    pools = [create_mem_pool(shadow_granularity=shadow_granularity) for _ in range(num_pools)]
    streams = [torch.cuda.Stream() for _ in pools]
    for pool, stream in zip(pools, streams):
        with torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool), triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            target = torch.zeros(max(1, shadow_granularity // 4), dtype=torch.int32, device="cuda")
            _store_reset_target[(1, )](target, WIDTH=target.numel(), num_warps=1)
        del target
    torch.cuda.synchronize()

    del pool
    del pools
    reset()


def _run_launch_stream_clocks_use_private_pool_check(shadow_granularity) -> None:
    device = torch.cuda.current_device()
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool):
        before = torch.empty(1, device=device)
        table, launch_index = _stream_sync.get_launch_state(device, stream.cuda_stream)
        clocks = _stream_sync._launch_stream_state(device, stream.cuda_stream).clocks
        after = torch.empty(1, device=device)
    stream.synchronize()

    reserve_begin = get_reserve_pointer()
    reserve_end = reserve_begin + get_reserve_size()
    assert reserve_begin <= before.data_ptr() < reserve_end
    assert not reserve_begin <= clocks.data_ptr() < reserve_end
    assert reserve_begin <= after.data_ptr() < reserve_end
    assert launch_index == 0
    assert not reserve_begin <= table.data_ptr() < reserve_end
    assert torch.count_nonzero(clocks).item() == 0

    del before, after, clocks, table, pool
    reset()


def _run_export_import_fabric_handles_check(explicit_config: bool, shadow_granularity: int, write_once: bool) -> None:
    device = torch.cuda.current_device()
    configure(
        device_ranks={device: 0},
        num_devices=2,
        handle_type=ShareableHandleType.FABRIC if explicit_config else None,
    )
    real_ptr = gsan_malloc(4096, device, shadow_granularity=shadow_granularity, write_once=write_once)
    imported_ptr = 0
    try:
        runtime_handle, runtime_alloc_size = export_runtime_state_handle(
            device,
            ShareableHandleType.FABRIC,
        )
        assert isinstance(runtime_handle, bytes)
        assert len(runtime_handle) == 64
        assert runtime_alloc_size > 0
        import_runtime_state_handle(
            runtime_handle,
            runtime_alloc_size,
            1,
            device,
            ShareableHandleType.FABRIC,
        )

        # Keep coverage of the legacy three-field export/default import API.
        if shadow_granularity == (1 if write_once else 4):
            real_handle, shadow_handle, alloc_size = export_allocation_handles(real_ptr, ShareableHandleType.FABRIC,
                                                                               write_once=write_once)
            import_kwargs = {"write_once": write_once}
        else:
            real_handle, shadow_handle, alloc_size, exported_granularity = export_allocation_handles(
                real_ptr, ShareableHandleType.FABRIC, include_granularity=True, write_once=write_once)
            assert exported_granularity == shadow_granularity
            import_kwargs = {"shadow_granularity": exported_granularity, "write_once": write_once}
        assert isinstance(real_handle, bytes)
        assert isinstance(shadow_handle, bytes)
        assert len(real_handle) == 64
        assert len(shadow_handle) == 64

        imported_ptr = import_allocation_handles(
            real_handle,
            shadow_handle,
            alloc_size,
            device,
            ShareableHandleType.FABRIC,
            **import_kwargs,
        )
        assert get_shadow_granularity(imported_ptr) == shadow_granularity
        assert is_write_once_allocation(imported_ptr) == write_once
        local_real = uint8_cuda_tensor_from_ptr(real_ptr, alloc_size, device)
        imported_real = uint8_cuda_tensor_from_ptr(imported_ptr, alloc_size, device)
        local_shadow = shadow_tensor_for(local_real)
        imported_shadow = shadow_tensor_for(imported_real)

        imported_real.fill_(11)
        assert torch.all(local_real == 11).item()
        imported_shadow.fill_(5)
        assert torch.all(local_shadow == 5).item()
    finally:
        if imported_ptr != 0:
            free_allocation(imported_ptr, device)
        gsan_free(real_ptr, device)


@pytest.fixture
def _direct_allocator(shadow_granularity):
    if not has_gsan_allocator():
        pytest.skip("requires CUDA or HIP backend")
    device = torch.cuda.current_device()
    stream = 0
    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    allocated = set()

    def malloc(size: int) -> int:
        ptr_int = gsan_malloc(size, device, stream, shadow_granularity=shadow_granularity)
        if ptr_int != 0:
            allocated.add(ptr_int)
        return ptr_int

    def free(ptr: int, size: int = 0) -> None:
        gsan_free(ptr, device, size, stream)
        if ptr in allocated:
            allocated.remove(ptr)

    try:
        yield malloc, free, reserve_ptr, reserve_size
    finally:
        # Cleanup any allocated pointers
        for ptr in list(allocated):
            gsan_free(ptr, device, 0, stream)


@requires_gsan_allocator
def test_malloc_edge_cases(_direct_allocator):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator

    # Invalid sizes are rejected.
    assert malloc(0) == 0
    assert malloc(-1) == 0
    assert malloc(reserve_size) == 0  # larger than the full real region

    # Null free is a no-op.
    free(0)


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not has_gsan_allocator() or torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_configure_supports_swapped_cuda_device_ids():
    device_ranks = {0: 1, 1: 0}
    result = run_in_process(_run_configure_check, args=(device_ranks, 2))
    assert result.exc is None


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not has_gsan_allocator() or torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_configure_supports_sparse_global_device_ids():
    device_ranks = {0: 2, 1: 3}
    result = run_in_process(_run_configure_check, args=(device_ranks, 4))
    assert result.exc is None


@requires_gsan_allocator
def test_configure_exposes_runtime_fields(shadow_granularity):
    result = run_in_process(_run_configure_runtime_fields_check, args=(shadow_granularity, ))
    assert result.exc is None


@requires_gsan_allocator
def test_default_clock_buffer_fits_runtime_state():
    result = run_in_process(_run_default_clock_buffer_fits_check)
    assert result.exc is None


# On HIP, an explicit clock buffer size shrinks the automatic slot count
# instead; see test_explicit_clock_buffer_size_shrinks_default_slots.
@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_clock_buffer_size_must_fit_runtime_state():
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_threads = num_sms * torch.cuda.device_count()
    if num_sms * num_threads * 2 * 65536 <= PER_DEVICE_STATE_STRIDE_BYTES:
        pytest.skip("a 65535-entry clock buffer fits on this topology")
    result = run_in_process(_run_clock_buffer_size_must_fit_check)
    assert result.exc is None
    assert "does not fit" in result.driver_stderr_output


@requires_gsan_allocator
def test_freeze_config_rejects_later_changes():
    result = run_in_process(_run_freeze_config_check)
    assert result.exc is None


@requires_gsan_allocator
def test_allocator_initialization_rejects_later_config(shadow_granularity):
    result = run_in_process(_run_allocator_freezes_config_check, args=(shadow_granularity, ))
    assert result.exc is None


@requires_gsan_allocator
def test_has_live_allocations(shadow_granularity):
    result = run_in_process(_run_has_live_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_reset_rejects_live_allocations(shadow_granularity):
    result = run_in_process(_run_reset_rejects_live_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@requires_gsan_allocator
def test_reset_collects_unreachable_allocations(shadow_granularity):
    result = run_in_process(_run_reset_collects_unreachable_allocations_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_reset_reinitializes_runtime_state(shadow_granularity):
    result = run_in_process(_run_reset_reinitializes_runtime_state_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize("num_pools", [1, 2])
def test_reset_releases_cached_pool_clocks(num_pools, shadow_granularity):
    result = run_in_process(_run_reset_releases_cached_pool_clocks_check, args=(num_pools, shadow_granularity))
    assert result.exc is None


@requires_gsan_allocator
def test_launch_stream_clocks_use_private_pool(shadow_granularity):
    result = run_in_process(_run_launch_stream_clocks_use_private_pool_check, args=(shadow_granularity, ))
    assert result.exc is None


@pytest.mark.xdist_group("gsan-multi-gpu")
@pytest.mark.skipif(not has_gsan_allocator() or torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_default_topology_uses_cuda_device_indices():
    assert get_device_rank(0) == 0
    assert get_device_rank(1) == 1


def _run_malloc_on_non_current_device_check() -> None:
    torch.cuda.set_device(0)
    ptrs = [gsan_malloc(4096, device) for device in (0, 1)]
    try:
        assert all(ptrs)
        assert torch.cuda.current_device() == 0
        with torch.cuda.device(1):
            real = uint8_cuda_tensor_from_ptr(ptrs[1], 4096, 1)
            real.fill_(5)
            assert torch.all(real == 5).item()
            del real
    finally:
        for device, ptr in zip((0, 1), ptrs):
            gsan_free(ptr, device)


@pytest.mark.xdist_group("gsan-multi-gpu")
# CUDA keeps the caller's current context during allocation (upstream behavior),
# so this is HIP-only.
@pytest.mark.skipif(not is_hip() or torch.cuda.device_count() < 2, reason="requires at least two HIP GPUs")
def test_malloc_on_non_current_device():
    result = run_in_process(_run_malloc_on_non_current_device_check)
    assert result.exc is None


@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_malloc_free(_direct_allocator, shadow_granularity):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    pool_index = 2 * (1, 2, 4, 8, 16).index(shadow_granularity)
    real_base = reserve_ptr + (2 * pool_index + 1) * reserve_size // 32

    # First valid allocation should come from the real base and be reusable.
    p0 = malloc(1)
    assert p0 == real_base
    free(p0)
    assert malloc(1) == p0

    p1 = malloc(1)
    _ = malloc(1)

    free(p1)
    p3 = malloc(1)
    assert p3 == p1


@requires_gsan_allocator
def test_malloc_fragmentation_reuse_and_coalesce(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    p0 = malloc(1)
    p1 = malloc(1)
    assert p0 != 0 and p1 != 0
    assert p0 < p1

    block = p1 - p0
    assert block > 0

    # Reuse exact freed block under fragmentation.
    free(p1)
    p1_reuse = malloc(1)
    assert p1_reuse == p1

    # Free two siblings and request a slightly larger block; should coalesce.
    free(p0)
    free(p1_reuse)
    parent = malloc(block + 1)
    assert parent == p0

    free(parent)
    torch.cuda.synchronize()


@requires_gsan_allocator
def test_malloc_free_large_odd_size(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    ptr = malloc(_ODD_LARGE_ALLOCATION_SIZE)
    assert ptr != 0

    free(ptr)
    torch.cuda.synchronize()


@requires_gsan_allocator
def test_free_invalid_pointer_and_double_free(_direct_allocator):
    malloc, free, _, _ = _direct_allocator

    p0 = malloc(1)
    assert p0 != 0

    free(p0 + 1)  # freeing an invalid pointer should not crash.

    free(p0)
    free(p0)  # double free must be a no-op

    # p0 should become reusable after the valid free above.
    p0_reuse = malloc(1)
    assert p0_reuse == p0

    free(p0_reuse)
    torch.cuda.synchronize()


@requires_gsan_allocator
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_mem_pool(shadow_granularity):
    pool = create_mem_pool(shadow_granularity=shadow_granularity)
    with torch.cuda.use_mem_pool(pool):
        real = torch.empty(4096, dtype=torch.uint8, device="cuda")

    reserve_ptr = get_reserve_pointer()
    reserve_size = get_reserve_size()
    assert reserve_ptr != 0
    assert reserve_size > 0

    assert reserve_ptr <= real.data_ptr() < reserve_ptr + reserve_size
    assert get_shadow_granularity(real.data_ptr()) == shadow_granularity
    assert shadow_cell_address(real.data_ptr() + shadow_granularity - 1) == shadow_cell_address(real.data_ptr())
    assert shadow_cell_address(real.data_ptr() + shadow_granularity) == shadow_cell_address(real.data_ptr()) + 24

    shadow = shadow_tensor_for(real)
    assert reserve_ptr <= shadow.data_ptr() < real.data_ptr()
    assert shadow.numel() == real.numel() // shadow_granularity * 24

    # Test that real and shadow allocation can be used
    real.zero_()
    real.add_(7)
    # Note: shadow memory is zero-initialized by the allocator
    shadow.add_(3)

    assert torch.all(real == 7).item()
    assert torch.all(shadow == 3).item()
    del pool
    del real
    del shadow
    torch.cuda.synchronize()


@requires_gsan_allocator
def test_export_allocation_memhandle_regions_identifies_real_and_shadow(_direct_allocator):
    malloc, free, _, _ = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(4096)
    assert real_ptr != 0

    try:
        exported_real_ptr, exported_real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(real_ptr)
        assert exported_real_ptr == real_ptr
        assert exported_real_size > 0
        assert shadow_ptr != 0
        assert shadow_size > 0

        real = uint8_cuda_tensor_from_ptr(exported_real_ptr, exported_real_size, device)
        shadow = shadow_tensor_for(real)
        assert shadow.data_ptr() == shadow_ptr
        assert shadow.numel() * shadow.element_size() == shadow_size
    finally:
        free(real_ptr)


@requires_gsan_allocator
def test_export_allocation_memhandle_regions_accepts_interior_pointer(_direct_allocator):
    malloc, free, _, _ = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(4096)
    assert real_ptr != 0

    try:
        exported_real_ptr, exported_real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(real_ptr +
                                                                                                             128)
        assert exported_real_ptr == real_ptr
        assert exported_real_size > 128
        assert shadow_ptr != 0
        assert shadow_size > 0

        real = uint8_cuda_tensor_from_ptr(exported_real_ptr, exported_real_size, device)
        shadow = shadow_tensor_for(real)
        assert shadow.data_ptr() == shadow_ptr
        assert shadow.numel() * shadow.element_size() == shadow_size
    finally:
        free(real_ptr)


# Check each granularity, and allocation-size handling independently at the default.
@requires_gsan_allocator
@pytest.mark.parametrize("shadow_granularity,size",
                         [(g, 4096) for g in (1, 2, 4, 8, 16)] + [(4, _ODD_LARGE_ALLOCATION_SIZE)],
                         indirect=["shadow_granularity"])
def test_export_import_allocation_handles_maps_real_and_shadow(_direct_allocator, size, shadow_granularity):
    malloc, free, reserve_ptr, reserve_size = _direct_allocator
    device = torch.cuda.current_device()

    real_ptr = malloc(size)
    assert real_ptr != 0

    imported_ptr = 0
    real_fd = -1
    shadow_fd = -1
    try:
        real_fd, shadow_fd, alloc_size, exported_granularity = export_allocation_handles(
            real_ptr,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
            include_granularity=True,
        )
        assert isinstance(real_fd, int)
        assert isinstance(shadow_fd, int)
        assert alloc_size > 0
        assert exported_granularity == shadow_granularity
        if shadow_granularity != 4:
            with pytest.raises(ValueError, match="include_granularity=True"):
                export_allocation_handles(real_ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR)

        imported_ptr = import_allocation_handles(
            real_fd,
            shadow_fd,
            alloc_size,
            device,
            ShareableHandleType.POSIX_FILE_DESCRIPTOR,
            shadow_granularity=exported_granularity,
        )
        assert imported_ptr != 0
        assert imported_ptr != real_ptr
        assert get_shadow_granularity(imported_ptr) == shadow_granularity

        local_real = uint8_cuda_tensor_from_ptr(real_ptr, alloc_size, device)
        imported_real = uint8_cuda_tensor_from_ptr(imported_ptr, alloc_size, device)

        local_shadow = shadow_tensor_for(local_real)
        imported_shadow = shadow_tensor_for(imported_real)
        assert local_shadow.numel() == imported_shadow.numel()

        imported_real.fill_(11)
        assert torch.all(local_real == 11).item()

        imported_shadow.fill_(5)
        assert torch.all(local_shadow == 5).item()
    finally:
        if real_fd >= 0:
            os.close(real_fd)
        if shadow_fd >= 0:
            os.close(shadow_fd)
        if imported_ptr != 0:
            free_allocation(imported_ptr, device)
        free(real_ptr)


@pytest.mark.parametrize("granularity", [0, 3, 32, -1, True, 1.0, "1"])
@pytest.mark.parametrize("factory", [get_allocator, create_mem_pool])
def test_mem_pool_rejects_invalid_granularity(factory, granularity):
    with pytest.raises(ValueError, match="shadow_granularity must be 1, 2, 4, 8, or 16"):
        factory(shadow_granularity=granularity)


@requires_gsan_allocator
def test_mem_pools_have_distinct_regions():
    tensors = []
    combinations = [(g, w) for g in (1, 2, 4, 8, 16) for w in (False, True)]
    pools = [create_mem_pool(shadow_granularity=g, write_once=w) for g, w in combinations]
    for pool in pools:
        with torch.cuda.use_mem_pool(pool):
            tensors.append(torch.empty(4096, dtype=torch.uint8, device="cuda"))
    assert [get_shadow_granularity(t.data_ptr()) for t in tensors] == [g for g, _ in combinations]
    assert len({shadow_cell_address(t.data_ptr()) for t in tensors}) == len(combinations)
    assert [is_write_once_allocation(t.data_ptr()) for t in tensors] == [w for _, w in combinations]
    assert get_allocator() is get_allocator(shadow_granularity=4)
    assert get_allocator(write_once=True) is get_allocator(write_once=True, shadow_granularity=1)


# Explicit configuration covers every pool. Environment-based configuration only
# needs the default granularity, still in both allocation modes.
@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize("shadow_granularity,explicit_config,allocator_config",
                         [(g, True, "fabric_handles:False")
                          for g in (1, 2, 4, 8, 16)] + [(4, False, "fabric_handles:True")],
                         indirect=["shadow_granularity"])
@pytest.mark.parametrize("write_once", [False, True])
def test_export_import_fabric_handles(explicit_config, allocator_config, write_once, shadow_granularity):
    if not supports_fabric_handles(torch.cuda.current_device()):
        pytest.skip("CUDA device does not support fabric handles")
    result = run_in_process(
        _run_export_import_fabric_handles_check,
        args=(explicit_config, shadow_granularity, write_once),
        env={"PYTORCH_CUDA_ALLOC_CONF": allocator_config},
    )
    if (isinstance(result.exc, RuntimeError) and str(result.exc) == "gsanExportRuntimeStateHandle failed."
            and "operation not permitted" in result.driver_stderr_output.lower()):
        pytest.skip("CUDA fabric handles require an accessible NVIDIA IMEX channel")
    assert result.exc is None


@requires_gsan_allocator
@pytest.mark.parametrize("write_once", [False, True])
@pytest.mark.parametrize("shadow_granularity,size",
                         [(g, 513) for g in (1, 2, 4, 8, 16)] + [(4, _ODD_LARGE_ALLOCATION_SIZE)],
                         indirect=["shadow_granularity"])
def test_allocation_mode_and_shadow_size(write_once, shadow_granularity, size):
    device = torch.cuda.current_device()
    ptr = gsan_malloc(size, device, write_once=write_once, shadow_granularity=shadow_granularity)
    assert ptr
    try:
        assert is_write_once_allocation(ptr) == write_once
        assert is_write_once_allocation(ptr + 512) == write_once
        real_ptr, real_size, shadow_ptr, shadow_size = export_allocation_memhandle_regions(ptr + 1)
        assert real_ptr == ptr
        assert shadow_size == real_size // shadow_granularity * (4 if write_once else 24)
        shadow = uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, device)
        assert torch.count_nonzero(shadow).item() == 0
        shadow.fill_(5)
        torch.cuda.synchronize()
    finally:
        gsan_free(ptr, device)
    with pytest.raises(ValueError, match="live GSan allocation"):
        is_write_once_allocation(ptr)
    reused = gsan_malloc(size, device, write_once=write_once, shadow_granularity=shadow_granularity)
    try:
        assert reused == ptr
        _, _, shadow_ptr, shadow_size = export_allocation_memhandle_regions(reused)
        shadow = uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, device)
        assert torch.count_nonzero(shadow).item() == 0
    finally:
        gsan_free(reused, device)


@gluon.jit
def _store_write_once_cell(ptr, WIDTH: gl.constexpr):
    offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([16], [32], [1], [0]))
    gl.store(ptr + offsets, 1)


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
@pytest.mark.parametrize("shadow_granularity", [1, 2, 4, 8, 16], indirect=True)
def test_write_once_export_import_preserves_shadow(fresh_knobs, shadow_granularity):
    triton.knobs.compilation.instrumentation_mode = "gsan"
    device = torch.cuda.current_device()
    ptr = gsan_malloc(513, device, write_once=True, shadow_granularity=shadow_granularity)
    imported = 0
    fds = []
    try:
        with pytest.raises(ValueError, match="allocation mode"):
            export_allocation_handles(ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR)
        if shadow_granularity != 1:
            with pytest.raises(ValueError, match="include_granularity=True"):
                export_allocation_handles(ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True)
        real_fd, shadow_fd, size, exported_granularity = export_allocation_handles(
            ptr, ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True, include_granularity=True)
        assert exported_granularity == shadow_granularity
        fds = [real_fd, shadow_fd]
        imported = import_allocation_handles(real_fd, shadow_fd, size, device,
                                             ShareableHandleType.POSIX_FILE_DESCRIPTOR, write_once=True,
                                             shadow_granularity=exported_granularity)
        assert is_write_once_allocation(imported + 1)
        original = uint8_cuda_tensor_from_ptr(ptr, 513, device)
        alias = uint8_cuda_tensor_from_ptr(imported, 513, device)
        _store_write_once_cell[(1, )](original, WIDTH=shadow_granularity, num_warps=1)
        assert alias[0].item() == 1
        assert shadow_cell_from_address(imported) == shadow_cell_from_address(ptr)
        assert shadow_cell_from_address(imported).epoch > 0
        assert shadow_cell_from_address(imported + shadow_granularity).epoch == 0
    finally:
        for fd in fds:
            os.close(fd)
        if imported:
            free_allocation(imported, device)
        gsan_free(ptr, device)


def _run_write_once_live_allocation_check(shadow_granularity):
    device = torch.cuda.current_device()
    ptr = gsan_malloc(1, device, write_once=True, shadow_granularity=shadow_granularity)
    assert has_live_allocations()
    with pytest.raises(AssertionError, match="allocations are still live"):
        reset()
    gsan_free(ptr, device)
    assert not has_live_allocations()
    reset()


@requires_gsan_allocator
def test_write_once_live_allocation_blocks_reset(shadow_granularity):
    result = run_in_process(_run_write_once_live_allocation_check, args=(shadow_granularity, ))
    assert result.exc is None, result.exc


@requires_gsan_allocator
def test_reserve_is_aligned_to_its_size():
    # The shadow translation masks addresses with the reserve size, so the
    # reservation must be size-aligned (HIP ignores the requested alignment).
    reserve_ptr = get_reserve_pointer()
    assert reserve_ptr != 0
    assert reserve_ptr % get_reserve_size() == 0


def _run_mem_pool_item_check() -> None:
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        x = torch.ones(4, device="cuda")
        total = x.sum()
        # ROCm PyTorch reads caching-allocator tensors directly from the host
        # on large-BAR GPUs, so GSan memory must be host-accessible there.
        assert total.item() == 4.0


@requires_gsan_allocator
def test_mem_pool_tensor_item():
    result = run_in_process(_run_mem_pool_item_check)
    assert result.exc is None


def _run_num_slots_layout_check(num_slots: int | None) -> None:
    device = torch.cuda.current_device()
    if num_slots is not None:
        configure(num_slots=num_slots, clock_buffer_size=8)
    ptr = gsan_malloc(1, device)
    assert ptr != 0
    slot_table = None
    try:
        expected_slots = num_slots if num_slots is not None else int(os.environ["TRITON_GSAN_NUM_SLOTS"])
        layout = get_runtime_state_layout(device)
        assert layout["num_slots"] == expected_slots
        assert layout["num_sms"] == expected_slots
        state = global_state(device_index=device)
        assert state.num_sms == expected_slots
        assert state.num_threads == expected_slots * state.num_devices

        thread_states_end = layout["thread_state_base_ptr"] + expected_slots * layout["thread_state_stride_bytes"]
        slot_table_ptr = layout["slot_table_ptr"]
        assert thread_states_end <= slot_table_ptr < thread_states_end + 8
        assert slot_table_ptr % 8 == 0
        slot_table_bytes = expected_slots * layout["slot_entry_bytes"]
        slot_table = uint8_cuda_tensor_from_ptr(slot_table_ptr, slot_table_bytes, device)
        assert torch.count_nonzero(slot_table).item() == 0

        slot_table.fill_(0xFF)
        torch.cuda.synchronize()
    finally:
        gsan_free(ptr, device)
    del slot_table
    reset()
    slot_table = uint8_cuda_tensor_from_ptr(slot_table_ptr, slot_table_bytes, device)
    assert torch.count_nonzero(slot_table).item() == 0


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
def test_num_slots_configures_slot_table():
    result = run_in_process(_run_num_slots_layout_check, args=(64, ))
    assert result.exc is None


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
def test_num_slots_from_environment():
    result = run_in_process(_run_num_slots_layout_check, args=(None, ), env={"TRITON_GSAN_NUM_SLOTS": "32"})
    assert result.exc is None


def _state_layout_after_malloc(device: int) -> dict:
    ptr = gsan_malloc(4096, device)
    assert ptr != 0
    try:
        layout = get_runtime_state_layout(device)
    finally:
        gsan_free(ptr, device)
    slot_table_end = layout["slot_table_ptr"] + layout["num_slots"] * layout["slot_entry_bytes"]
    assert slot_table_end - layout["global_state_ptr"] <= _PER_DEVICE_STATE_STRIDE
    return layout


def _run_explicit_num_slots_shrinks_default_clock_check() -> None:
    device = torch.cuda.current_device()
    configure(device_ranks={device: 0}, num_devices=1)
    layout = _state_layout_after_malloc(device)
    # Explicit slots are kept; the automatic clock buffer shrinks to fit them.
    assert layout["num_slots"] == 4096
    assert 0 < layout["clock_buffer_size"] < 1024


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
def test_explicit_num_slots_shrinks_default_clock_buffer():
    result = run_in_process(_run_explicit_num_slots_shrinks_default_clock_check, env={"TRITON_GSAN_NUM_SLOTS": "4096"})
    assert result.exc is None
    assert "clock buffer size" in result.driver_stderr_output


def _run_explicit_clock_buffer_size_shrinks_default_slots_check() -> None:
    device = torch.cuda.current_device()
    configure(device_ranks={device: 0}, num_devices=1, clock_buffer_size=65535)
    layout = _state_layout_after_malloc(device)
    # The explicit clock buffer is kept; the automatic slot count shrinks
    # (below the 512 default, and on gfx950 below the CU count) to fit it.
    assert layout["clock_buffer_size"] == 65535
    assert 0 < layout["num_slots"] < 512


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
def test_explicit_clock_buffer_size_shrinks_default_slots():
    result = run_in_process(_run_explicit_clock_buffer_size_shrinks_default_slots_check)
    assert result.exc is None


def _run_thread_state_lookup_unaligned_stride_check() -> None:
    device = torch.cuda.current_device()
    # 511 slots x 9 clocks x 2 B is not a multiple of 8, so thread states are
    # padded to their alignment. With several visible GPUs, the default slot
    # count gives such strides too (e.g. 2 GPUs: 511 slots, 1022 threads).
    # One device, so the thread count (and thus the stride) is as computed above.
    configure(device_ranks={device: 0}, num_devices=1, num_slots=511, clock_buffer_size=8)
    ptr = gsan_malloc(4, device)
    target = uint8_cuda_tensor_from_ptr(ptr, 4, device).view(torch.int32)
    try:
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "gsan"
            # Several launches, so blocks land in slots other than 0.
            for _ in range(8):
                store_one_i32[(1, )](target, num_warps=1)
                torch.cuda.synchronize()
                write = shadow_cell_from_address(ptr).write_clock
                # The writer's own vector-clock entry, read through the test
                # helpers, is the epoch the device recorded for the write.
                assert thread_state_from_smid(write.thread_id).vector_clock[write.thread_id] == write.epoch
    finally:
        del target
        gsan_free(ptr, device)


@pytest.mark.skipif(not is_hip(), reason="num_slots is HIP-only")
def test_thread_state_lookup_with_padded_thread_state_stride():
    result = run_in_process(_run_thread_state_lookup_unaligned_stride_check)
    assert result.exc is None, result.exc


_PER_DEVICE_STATE_STRIDE = 1 << 30


def _run_default_num_slots_check(num_devices: int) -> None:
    device = torch.cuda.current_device()
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    configure(device_ranks={device: 0}, num_devices=num_devices)
    layout = _state_layout_after_malloc(device)
    num_slots, clock_buffer_size = layout["num_slots"], layout["clock_buffer_size"]
    assert num_slots * num_devices <= 4096
    if num_devices == 1:
        assert (num_slots, clock_buffer_size) == (512, 1024)
    # Slots shrink first, but not below one per CU; only then does the clock
    # buffer shrink.
    assert num_slots >= min(num_cus, 4096 // num_devices)
    if clock_buffer_size < 1024:
        assert num_slots == min(num_cus, 4096 // num_devices)


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
@pytest.mark.parametrize("num_devices", [1, 2, 8])
def test_default_num_slots_fits_state_region(num_devices):
    result = run_in_process(_run_default_num_slots_check, args=(num_devices, ))
    assert result.exc is None


def _run_configure_rejects_num_slots_check(num_slots: int, num_devices: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        configure(device_ranks={torch.cuda.current_device(): 0}, num_devices=num_devices, num_slots=num_slots,
                  clock_buffer_size=1024)


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
@pytest.mark.parametrize(
    ("num_slots", "num_devices", "match"),
    [
        # 1024 slots x (1 + 1024) clocks x 1024 epochs x 2 B exceeds 1 GiB.
        pytest.param(1024, 1, "runtime state per device", id="state-size"),
        pytest.param(2048, 4, "thread-state limit", id="thread-ids"),
    ],
)
def test_configure_rejects_unusable_num_slots(num_slots, num_devices, match):
    result = run_in_process(_run_configure_rejects_num_slots_check, args=(num_slots, num_devices, match))
    assert result.exc is None


def _run_env_state_too_large_check() -> None:
    device = torch.cuda.current_device()
    configure(device_ranks={device: 0}, num_devices=1)
    assert gsan_malloc(4096, device) == 0


@pytest.mark.skipif(not is_hip(), reason="slot table is HIP-only")
def test_env_sizing_too_large_for_state_region_fails_allocation():
    # Both sizes are explicit, so neither may shrink to fit.
    result = run_in_process(_run_env_state_too_large_check, env={
        "TRITON_GSAN_NUM_SLOTS": "1024", "TRITON_GSAN_CLOCK_BUFFER_SIZE": "1024"
    })
    assert result.exc is None
    assert "does not fit" in result.driver_stderr_output


@pytest.mark.skipif(not is_hip(), reason="fabric handles are rejected only on HIP")
def test_fabric_handles_rejected_on_hip():
    with pytest.raises(ValueError, match="handle_type"):
        configure(handle_type=ShareableHandleType.FABRIC)


def _run_graph_capture_is_rejected_check() -> None:
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        target = torch.zeros(1, dtype=torch.int32, device="cuda")
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "gsan"
        store_one_i32[(1, )](target, num_warps=1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="GSan does not support (CUDA|HIP) graph capture"):
            with torch.cuda.graph(graph):
                store_one_i32[(1, )](target, num_warps=1)


@requires_gsan_allocator
def test_graph_capture_is_rejected():
    result = run_in_process(_run_graph_capture_is_rejected_check)
    assert result.exc is None


@pytest.mark.skipif(not is_cuda(), reason="requires CUDA backend")
def test_num_slots_rejected_on_cuda():
    with pytest.raises(ValueError, match="num_slots is only supported on the HIP backend"):
        configure(num_slots=64)
