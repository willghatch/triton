# **GSan \- Concurrency Sanitizer for Global Memory**

# **1\. Introduction**

This proposal outlines the design and implementation of **GSan**, a triton/gluon sanitizer for detecting data races in global memory, including remote accesses within a single node via NVLink. GSan will utilize shadow memory techniques and instrument existing memory operations to work near-seamlessly with existing kernels.

## **1.1. Goals**

* Detect unsynchronized read-after-write, write-after-write and write-after-read violations
* Support peer-to-peer accesses within a single node NVLink domain
* Cover all communication patterns needed by existing code
* Support all TritonGPU memory access operators including: ld/st, atom, cp.async, and cp.async.bulk (aka TMA) operations.

## **1.2. Non-goals**

* Proving correctness
  * This is very difficult as we have hundreds of independent threads, and tracking the full state of each memory cell for all threads is impractical. Thus, we will need to rely on stochastic detection.
* Support inline assembly comms
  * We aim to instrument the TritonGPUIR directly, and parsing inline assembly is out of scope.
  * This means we will need to ensure there are sufficient primitives within the language and library to support comms.
* Detect races with non-triton/gluon code.
  * We assume that all relevant code will be visible to the instrumentation.
  * This means we miss races outwith instrumented code, and may detect a false-positive if uninstrumented synchronization primitives are used.
* Detect races between the threads within the same CTA, e.g. no thread barrier between reading and writing to the same memory location.
  * This may be possible as an extension, but generally we will assume that the IR executes as a single synchronous program.

# **2\. Proposed Algorithm**

## **2.1 Vector clocks**

The proposed algorithm is based on the concept of **vector clocks**, which is a method to verify “happens-before” relationships in distributed systems.

Each thread maintains a vector clock (`VC[N]`) of time **epochs** for each thread in the system. `VC[tid]` represents the thread’s current epoch, and `VC[i]` for `i != tid` represents the most recent epoch of thread `i` which has been acquired by the current thread.

When two threads act on the same memory location in a way that might race, we can compare each thread's vector clocks to check if the two events are strictly ordered. So, if `VC(X)` is the vector clock associated with event `X`, then if `VC(A)[i] <= VC(B)[i]` for all `i`, then `A` strictly “happens-before” `B`, and so `A` does not race with `B`.

Note that “happens-before” is a term of art which implies not simply chronological ordering, but full visibility of the operations within the memory system. A writer can complete the write before a reader, but if the reader doesn’t explicitly synchronize with the writer through a (potentially transitive) acquire-release pattern, then we say the write did not happen-before the read.

## **2.2 Race Detection with Vector Clocks**

Using vector clocks, you can construct a fully general race-detection algorithm as follows:

For every memory location we record the vector clock of the last write into a shadow memory region. Any subsequent reads/writes can compare their local vector clock to the writer’s. If the write strictly happens-before the new access, then there is no race. We can also use this clock to update the reader’s vector clock when the reader forms a release-acquire pattern with the writer. This models acquiring both: the writes from that writer (in `VC[tid]`), but also any transitively acquired writes from other threads.

Similarly, for each location we can record the elementwise maximum of the vector clocks of all readers.  This elementwise maximum is the smallest clock value where all readers individually happened-before it. On write, you can compare with the elementwise-max read clock to ensure all previous reads happened-before the write.

Unfortunately for us, this fully general algorithm is not practical because we will be tracking accesses for each CTA separately, and so each vector clock will have hundreds of elements. That would require us to use hundreds of times the original memory usage as overhead to store the clock data. So, instead we will discuss some modifications to reduce memory overhead.

## **2.3 Scalar write clocks**

When you compare two writes by a single thread, you can find a strict ordering just by looking at that thread’s `VC[tid]`, since this gets increments for each ordered write. So, if we didn’t care about acquiring transitive dependencies, then we could just store a “scalar clock” value which is a pair of `(tid, VC[tid])` for the writing thread’s id. This is convenient for us, since most reads and writes in GPU programming are usually weakly ordered, and so cannot form acquire-release patterns.

Therefore, for non-atomic writes we will only store the local thread’s epoch along with the thread id. However, for atomic writes we will still need to store a full vector clock in order to allow matching atomic-acquires to update their vector clock appropriately.

As an implementation detail, I’m thinking that each thread can have a circular buffer of say 1024 vector clocks. On a GB200 system with 4 GPUs and 152 SMs, then this uses 722 MiB of extra memory which is perfectly reasonable compared to the significant overhead of other shadow memory metadata. We could also reduce this further if we limit the number of SMs that run in parallel, since the overhead scales with O(n^2) in the number of threads.

Also note that a circular buffer does run the risk of wrapping-around, which may lead to issues if it wraps around while a write is still yet to be read. We can, however, detect this edge case by storing the index into the circular buffer without-modulo, and tracking how many times the buffer has wrapped around. This at least means we can raise an error instead of reporting incorrect false-negatives.

## **2.4 Stochastic read clocks**

Similar to how write ordering can be checked by only comparing `VC[tid]`, a single reader thread can also be compared by only its local epoch value. However, we have the added complication that there may be many readers between writes, any of which might race with the writer even if the most recent chronological reader did not. This means in general we cannot simply store a scalar.

Instead, we can store a random sampling of `VC[tid]` from readers. We have a fixed buffer of `num_slots` (`VC[tid],tid)` pairs, and on each read the reader will replace 1 of the slots, with a probability proportional to `num_slots / num_readers`. This technique is known as reservoir sampling, and results in all past readers being equally likely to be sampled regardless of how long ago they were read in chronological order.

This is the first change that introduces potential false-negatives, as now readers can race but if they are not included in the stochastic clock value then it isn’t reported. However, this is mitigated somewhat by the fact that in practice every CTA is running the same program, so if one reader isn’t synchronized then it’s likely that many aren’t synchronized properly either. This gives us many attempts to randomly find the failing cases.

## **2.5 Asynchronous memory accesses (cp.async/cp.async.bulk.tensor)**

So far we’ve assumed that memory accesses happen at a fixed point in time for a given thread, however in the case of cp.async operations there is a window of time which may overlap with other memory interactions in an out of order way.

We can model this by considering two different vector clock values: the value upon issuing the async access (`VC_access`), and the value upon completion (`VC_completion`).  When comparing the existing shadow memory’s vector clock value, we compare it to `VC_access`. This ensures that those operations happened-before the earliest possible time the async access might happen. Then, in the shadow memory we actually store `VC_completion[tid]` as any future reads/writes must happen strictly after the completion of the async op.

In order to do this, we will need to instrument the async wait/mbarrier wait op that forms the actual completion, and do the shadow memory update then. For this we will need to record, for every asynchronous memory access: the associated `VC_access`, as well as metadata that allows us to map the async wait/mbarrier wait ops back to the operation.

## **2.6 Limitations**

False Negatives:

* Since we only track a random sampling of the true read vector clock, this means if there are many readers to a single memory location then we may miss races with reads that happened in the past.

False Positives:

* If the thread-local vector clock buffer wraps around before all references to the clock value are done with, then we will have to report a potential race even if none occurred.

These limitations can be mitigated by increasing the number of read samples, or increasing the vector clock buffer size. However, both of these come with increases in the respective memory overhead.

## **3\. Shadow Memory Details**

So far we’ve assumed that we can quickly and easily map between a given pointer and its corresponding shadow memory. To do this, we can take advantage of the CUDA driver API which has a number of primitives to control virtual memory addressing. `cuMemAddressReserve` allows us to reserve a large chunk of virtual address space, and `cuMemMap` allows us to map newly allocated memory pages anywhere we want within the reserved address space.

This gives us the tools we need to  implement a custom allocator that maps all tensor allocations into the higher address range with `cuMemMap`; and also creates a corresponding shadow allocation that gets mapped into the lower address range.

During kernel execution, we can tell if memory is managed by GSan by checking if it fits in the bounds of the reserved address space.

```
use_gsan = vmem_base_ptr <= ptr && ptr < vmem_base_ptr + VMEM_SIZE
```

Since we control the exact mapping within the address space, we can also place the shadow memory in a way that makes translation straight-forward. For example, if each 4-byte word has a corresponding shadow memory entry, then we can find the shadow memory by doing:

```c
// Mask out the high bit which indicates the real memory region
ALLOC_MASK = (VMEM_SIZE - 1) >> 1
byte_offset = (ptr - vmem_base_ptr) & ALLOC_MASK
word_offset = byte_offset / 4
shadow_ptr = vmem_base_ptr + word_offset * SHADOW_SIZE_BYTES
```

We also have the option to encode more information into the pointer addresses, since we have a full 64-bit address space to play with e.g. we might have one memory region for “bulk” data with course-grained shadow memory, and one for fine-grained word-level tracking.

# **4\. Triton API Changes**

Currently, we have atomic read-modify-write operations which have memory semantic and scope qualifiers in the frontend. Strictly speaking this is all we need as you can simply add 0 to load, or use `atomic_xchg` to store. However, these have overly strong synchronizing effects which may hurt performance of distributed comms.

We may want to consider adding more memory primitives:

1. Scoped memory fences e.g. `tl.fence(scope=”sys”)`.
2. Atomic loads and stores e.g. `tl.atomic_load(ptr, sem=”acquire”, scope=”sys”)`
3. Atomic reduce operations e.g. `tl.reduce_add(ptr, 1, sem=”release”, scope=”sys”)`

These would allow more efficient communication patterns by minimizing the synchronization effects of operations. For example, fence \+ relaxed atomic store allows more memory ops to be placed in-between without causing over-synchronization.

# **5\. AMD prototype**

GSan also runs, as a prototype, on AMD gfx950 (MI350-series, wave64) and gfx1250 (MI450-series, wave32) with a single GPU.
It supports ordinary global `tl.load`/`tl.store`, atomic RMW/CAS, atomic load/store/poll, AMD buffer loads and stores (including packed FP4 storage), and buffer atomics (from Triton's buffer-op conversion or explicit Gluon `buffer` ops), direct pointer-addressed global↔LDS copies (LDS-to-global on gfx1250), buffer-addressed global-to-LDS copies on gfx950, rectangular TDM copies on gfx1250, and launches on ordinary streams with memory from the GSan allocator.
On gfx1250, write-once pools support ordinary masked loads/stores, mixed normal/write-once pointer tensors, and direct asynchronous-copy reads at shadow granularities 1, 2, 4, 8, and 16.
Reads leave compact write-once shadow cells unchanged; unwritten reads, repeated writes, and active atomic accesses are rejected.
The complete aligned access requirement for 16-byte pools still applies.
Reset rejects live managed allocations and clears runtime and launch-stream clocks after their allocations are released.
On gfx1250, rectangular, fused-load, gather, and scatter TDM operations are instrumented, including named-barrier completion operands.
Checks decode the positioned native descriptor, including dynamic address, element strides, bounds, predicate, and unsigned gather/scatter indices.
Their checks happen when the transfer is issued, matching NVIDIA TMA checks; completion epochs are not tracked.
TDM instructions require gfx1250.
Native AMD tensor descriptors support 8-, 16-, 32-, and 64-bit elements; FP4 TDM remains a native compiler restriction.
Gfx1250 warp-specialized regions, noinline worker calls, and explicit warp-pipeline phase shifts are supported.
Rectangular TDM has noinline-worker execution coverage; indexed TDM inside a noinline worker can produce native LLVM register-copy errors and incorrect values with or without GSan, and remains outside this validated composition.
Each CTA remains one logical sanitizer participant shared by its regions; local named/completion barriers preserve hardware phases without adding global-memory ordering between different CTAs.
Automatic BlockPingpong scheduling remains disabled: its existing AMD compiler pass requires MFMA layouts, while gfx1250 uses WMMA.
Other AMD targets, combining GSan with AddressSanitizer, and implicit HIP graph capture remain rejected.
The wave32 runtime reserves 320 KiB of total LDS per cluster, divided among its CTAs; gfx950 reserves 160 KiB per workgroup.
MI450 cluster launches preserve independent sanitizer participants for every CTA, including replicated multicast readers.
Before launch, configured runtime slots must cover `device CU count * floor(device LDS per CU / emitted total LDS per CTA)`, using the actual HIP properties and static-plus-dynamic reservation.
This conservatively covers concurrent resident participants even when the cluster reservation is divided.
For example, on a 256-CU device a four-CTA cluster requires `configure(num_slots=1024, clock_buffer_size=16)` before creating a GSan pool; configured slot and clock-history sizes must fit the allocator state budget.
This is a physical-residency bound; suspended CWSR participants can still exhaust the fixed table and trigger its existing timeout diagnostic.
Two- and four-CTA launches have focused end-to-end memory, multicast, synchronization, and concurrent-stream coverage; eight- and sixteen-CTA disjoint launches also pass value and actor-identity checks, and grouped TDM multicast verifies four recipients per cell, disjoint groups, and nonrecipient clock independence.
AMD cluster arrive/wait are execution rendezvous and do not acquire other CTAs' global-memory clocks; release/acquire atomics provide their actual ordering.
AMD named and TDM completion barriers access each CTA's own LDS barrier object and retain their phase semantics.
Direct cross-CTA LDS transfers and remote LDS barrier-address mapping remain unavailable through the current AMD compiler; this is a compiler boundary, not a claim of absent hardware support.
Multi-CTA `atomic_poll` remains rejected by the native compiler because its current lowering requires cross-CTA shared memory.
A detected race prints `GSan device assertion failed: '<kind> race detected' at <file>:<line>` and aborts the process (device trap), so run failing cases in a subprocess.
The explicit graph interface (`triton.experimental.gsan.graph.GraphPlan`) supports HIP fork/join dependencies, repeated replay, immutable completion clocks, and executable storage lifetime.
An external launcher must install the plan's boundary kernels and actual node dependencies; importing completion clocks does not create a GPU execution dependency.
Repeated replay of one HIP plan establishes a real completion-event dependency before reusing its clock storage, including when switching streams.
The regression graph launcher covers single-CTA kernel nodes; cluster node attributes and scratch-buffer integration require support in the external launcher and are not validated by those graph tests.
NVIDIA programmatic dependent launches (PDL) have no HIP equivalent and are rejected; implicit graph capture remains unsupported on both backends.

Known limitations:

* Default pools have 4-byte shadow granularity: unsynchronized writes to different bytes of one 4-byte word are reported as a race. Pools from `create_mem_pool(shadow_granularity=1)` (or 2) track finer cells.
* Ordinary same-stream launch order and the explicit graph interface create modeled happens-before between kernels.
  General event waits (e.g. `wait_stream`) and host synchronization are not tracked, so cross-stream accesses ordered only by them are reported.
* Blocks are identified by slots of a fixed table (`num_slots` or `TRITON_GSAN_NUM_SLOTS`, default up to 512). When the runtime state would not fit in 1 GiB per device, the default slot count shrinks, but not below the CU count; then the default clock buffer shrinks instead. The allocator prints a note when it picks fewer slots than CUs or fewer than 1024 clock-buffer entries. A block that takes over a slot from an earlier block, in any kernel or stream, shares that block's identity (its epoch), so the earlier block's accesses look like its own and are treated as ordered. Slots are chosen by hash, so a race between two blocks that do not run concurrently is missed with probability about 1/num_slots per racing pair; grids larger than the slot count have many such pairs.
* Earlier prototype measurements found three to four orders of magnitude slowdown; this is not a performance estimate for every supported operation.
  Instrumented kernels reserve the maximum LDS per cluster, so a single-CTA launch admits one CTA per CU; multi-CTA clusters divide that reservation.
* Buffer loads and stores are checked on the addresses they access, `base + soffset + offset` in bytes, and only for elements the hardware does not drop (masked off or past the descriptor's `num_records`). An element wider than 4 bytes counts as accessed when its first 4 bytes are in bounds. A contiguous run of elements is checked as one access, which counts the whole run when its first element is in bounds, so the bytes of a run that `num_records` cuts are also counted (this can report a race on bytes just past the bound). A buffer descriptor must be built by `make_buffer_descriptor` in the same function as its use (not passed in, loop-carried, or selected); otherwise compilation fails.
* Under GSan, buffer atomics are emitted as global atomics.
  On gfx950, for elements wider than 4 bytes, an active element must be entirely in bounds: GSan reports a device assertion if the first 4 bytes fit but the whole element does not.
  On gfx1250, a partially bounded atomic element is dropped entirely, matching the hardware.
  This restriction also applies to dynamic bounds; masked-off and wholly out-of-bounds elements are still dropped.
  Partial atomic accesses need a lowering that preserves both their memory effects and synchronization before they can be supported.
* Direct async copies are checked at issue time, matching NVIDIA's checking approach; the interval through completion is not tracked, and async waits do not add happens-before edges.
* Buffer global-to-LDS copies check the independently wrapped unsigned byte offsets and descriptor byte bounds, including individual dwords of partially in-bounds wide elements.
  Sub-dword element types (such as f16 and i8) require a blocked layout whose per-lane dword groups have provably contiguous offsets and uniform masks; checks use the issued dword bounds, so a partial dword dropped by hardware leaves every contained element unchecked.
  A buffer descriptor must be defined directly by `amdg.make_buffer_desc`; passed-in, selected, or loop-carried descriptors fail compilation.
  Exact dynamic buffer bounds can prevent merging checks into complete 16-byte accesses, so use shadow granularity 1, 2, 4, or 8 for those copies; 16-byte pools retain the runtime's complete-access requirement.
* Pointer LDS-to-global copies (`amdg.async_copy_local_to_global`) on gfx1250 receive issue-time write checks.
* An unmatched timed-out `atomic_poll` does not check its read, matching the existing NVIDIA checker; a matched active poll on write-once memory is rejected.
* Single-CTA regions do not add intra-CTA race detection; the logical block programming model and existing sampled-reader/clock-history limits still apply.

## **5.1 Build and test from a fresh checkout**

These commands assume a gfx1250 machine with a ROCm build of PyTorch whose bundled HIP runtime Triton uses (validated with `torch 2.15.0a0+rocm10.1.0a20260822`).
Use the corresponding `device-gfx950` wheel extra on gfx950; the additional MI450 TDM, completion, region, and cluster coverage requires gfx1250.
The AMD build compiles the GSan device libraries (`third_party/amd/backend/lib/gsan.bc` for gfx950 and `gsan-gfx1250.bc` for gfx1250) with the `clang++` from Triton's LLVM; the host allocator is compiled on first use with the system C++ compiler.

```bash
git clone https://github.com/triton-lang/triton.git && cd triton
# In a fresh virtualenv; the validated PyTorch build came from:
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1250]==2.15.0a0+rocm10.1.0a20260822"
pip install -r python/requirements.txt
pip install -e . --no-build-isolation
pip install -r python/test-requirements.txt
export HIP_VISIBLE_DEVICES=0
# Without this, every reported race writes a large GPU core dump into the cwd.
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
# Single-GPU matrix; gpu-lock is required on shared GPU hosts.
TRITON_TEST_PROCESS_TIMEOUT=60 gpu-lock timeout 1200 pytest -s --tb=short -ra \
    python/test/gsan/test_allocator.py python/test/gsan/test_utils.py \
    python/test/gsan/test_write_once.py python/test/gsan/test_write_once_failures.py \
    python/test/gsan/test_gsan.py python/test/gsan/test_gsan_failures.py \
    -k 'not test_configure_supports_swapped_cuda_device_ids and not test_configure_supports_sparse_global_device_ids and not test_default_topology_uses_cuda_device_indices and not test_malloc_on_non_current_device'
# Compiler-only tests (no GPU); run from the CMake build directory.
cd build/cmake.linux-x86_64-cpython-3.*
lit -v test/TritonGPU/amd/amd-gsan-reject-unsupported.mlir test/Conversion/amd/tritongpu_to_llvm_gsan.mlir \
    test/TritonGPU/amd/amd-gsan-instrument-buffer-ops.mlir test/TritonGPU/gsan.mlir \
    test/Conversion/tritongpu_to_llvm_gsan.mlir
```

In the development container with `tritonbuild` and `gpu-lock` on `PATH`, use `MAX_JOBS=128 CMAKE_BUILD_PARALLEL_LEVEL=128 tritonbuild --rocm-gpu gfx1250 worktree-setup` and `tritonbuild --rocm-gpu gfx1250 run -- make` for CPU setup/build, without holding the GPU lock.
Run the matrix command above through `tritonbuild --rocm-gpu gfx1250 run -- gpu-lock timeout 1200 pytest ...`; the timeout starts after acquiring the GPU lock.
Use `tritonbuild --rocm-gpu gfx1250 lit -- -v ...` for the listed compiler-only tests.
Choose CPU parallelism appropriate to the host; 128 workers is half of the validated machine's 256 cores.
Expected device traps print ROCm queue-error diagnostics; a test passes only when its isolated child reports the expected sanitizer error at the user source location.
On a timeout or persistent GPU hang, stop further GPU runs and preserve the revision, command, and log for recovery.

To sanitize your own kernels, allocate their tensors from a GSan memory pool and enable the instrumentation mode:

```python
import torch, triton
from triton.experimental.gsan import create_mem_pool

triton.knobs.compilation.instrumentation_mode = "gsan"  # or TRITON_INSTRUMENTATION_MODE=gsan
with torch.cuda.use_mem_pool(create_mem_pool()):
    x = torch.zeros(1024, device="cuda")
    my_kernel[grid](x, ...)
    torch.cuda.synchronize()
```
