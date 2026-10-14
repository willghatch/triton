#pragma once

// Thin driver shim so GSanAllocator.cc can target either the CUDA driver API
// or HIP. Addresses are uintptr_t on both sides; HIP's void* addresses are
// produced only at the call boundary. Allocation properties are always the
// same (pinned, on `device`, exportable as `handleType`), so they are built
// here rather than at every call site.
//
// HIP differences handled here:
// - hipMemAddressReserve ignores its alignment argument, so addressReserve
//   over-reserves and aligns the result manually; Reservation remembers the
//   raw range so addressFree can release all of it.
// - Some device is always current on HIP, so DeviceScope selects the target
//   device around work on a device's memory (memsets, copies) and restores
//   the previous device afterwards; another device cannot access that memory.
// - A failing HIP call leaves a sticky "last error" that makes PyTorch's next
//   launch check fail. Every HIP wrapper clears it on failure and returns the
//   error instead. This also discards any earlier error the application had
//   not yet observed.
// - ROCm PyTorch may read user memory directly from the host, so memSetAccess
//   can also grant host access (kSupportsHostAccess).
// - Fabric handles are not supported.
//
// Not handled: the HIP 7.2.x VMM bugs found during GSan's AMD bring-up
// (hipMemSetAccess failing on reservations with mixed mapping sizes, and
// hipMemMap reporting success on failure). Use a HIP runtime with those fixes
// (e.g. 7.16).

#include <cstddef>
#include <cstdint>

#if defined(GSAN_BACKEND_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>
#else
#include <cuda.h>
#endif

namespace gsan::drv {

#if defined(GSAN_BACKEND_HIP)

using Result = hipError_t;
using Handle = hipMemGenericAllocationHandle_t;
using Stream = hipStream_t;

inline constexpr Result kSuccess = hipSuccess;
inline constexpr Result kErrorNotInitialized = hipErrorNotInitialized;
inline constexpr Result kErrorInvalidDevice = hipErrorInvalidDevice;
inline constexpr Result kErrorInvalidValue = hipErrorInvalidValue;
inline constexpr Result kErrorNoDevice = hipErrorNoDevice;
inline constexpr Result kErrorNotSupported = hipErrorNotSupported;

#else

using Result = CUresult;
using Handle = CUmemGenericAllocationHandle;
using Stream = CUstream;

inline constexpr Result kSuccess = CUDA_SUCCESS;
inline constexpr Result kErrorNotInitialized = CUDA_ERROR_NOT_INITIALIZED;
inline constexpr Result kErrorInvalidDevice = CUDA_ERROR_INVALID_DEVICE;
inline constexpr Result kErrorInvalidValue = CUDA_ERROR_INVALID_VALUE;
inline constexpr Result kErrorNoDevice = CUDA_ERROR_NO_DEVICE;
inline constexpr Result kErrorNotSupported = CUDA_ERROR_NOT_SUPPORTED;

#endif

// Values match ShareableHandleType in _allocator.py and the CUDA enum.
enum class HandleType : int {
  PosixFileDescriptor = 0x1,
  Fabric = 0x8,
};

#if defined(GSAN_BACKEND_HIP)
// HIP has no fabric handles. Keep the union in GSanAllocator.cc the same size.
struct FabricHandle {
  unsigned char data[64];
};
inline constexpr bool kHasFabricHandles = false;
#else
using FabricHandle = CUmemFabricHandle;
inline constexpr bool kHasFabricHandles = true;
#endif

inline bool isSupportedHandleType(HandleType type) {
  return type == HandleType::PosixFileDescriptor ||
         (kHasFabricHandles && type == HandleType::Fabric);
}

// A virtual address reservation. `base` is the usable, aligned address; the
// raw range is what the driver actually reserved and what addressFree
// releases. They differ on HIP, which over-reserves to align `base`.
struct Reservation {
  uintptr_t base = 0;
  uintptr_t rawBase = 0;
  size_t rawSize = 0;
};

#if defined(GSAN_BACKEND_HIP)

// Whether memSetAccess can also grant host access. ROCm PyTorch reads small
// caching-allocator tensors (e.g. Tensor.item()) directly from the host on
// large-BAR GPUs, assuming device memory is host-mapped as with hipMalloc.
inline constexpr bool kSupportsHostAccess = true;

// Clears HIP's sticky last error after a failed call; see the header comment.
inline Result check(Result err) {
  if (err != kSuccess)
    (void)hipGetLastError();
  return err;
}

inline void *ptr(uintptr_t addr) { return reinterpret_cast<void *>(addr); }

inline hipMemAllocationProp makeProp(int device, HandleType type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleType = static_cast<hipMemAllocationHandleType>(type);
  return prop;
}

inline const char *errorString(Result err) { return hipGetErrorString(err); }

inline Result deviceCount(int *count) {
  return check(hipGetDeviceCount(count));
}

inline Result multiprocessorCount(int device, int *count) {
  return check(hipDeviceGetAttribute(
      count, hipDeviceAttributeMultiprocessorCount, device));
}

inline Result supportsFabricHandles(int /*device*/, bool *supported) {
  *supported = false;
  return kSuccess;
}

inline Result allocationGranularity(int device, HandleType type,
                                    size_t *granularity) {
  auto prop = makeProp(device, type);
  return check(hipMemGetAllocationGranularity(
      granularity, &prop, hipMemAllocationGranularityMinimum));
}

// ROCm accepts but ignores the alignment argument, while GSan's address math
// relies on size-aligned reservations. Over-reserve and use the aligned
// subrange. `alignment` must be zero or a power of two.
inline Result addressReserve(Reservation *out, size_t size, size_t alignment) {
  void *raw = nullptr;
  Result err = check(
      hipMemAddressReserve(&raw, size + alignment, alignment, nullptr, 0));
  if (err != kSuccess)
    return err;
  auto rawBase = reinterpret_cast<uintptr_t>(raw);
  out->rawBase = rawBase;
  out->rawSize = size + alignment;
  out->base = alignment
                  ? (rawBase + alignment - 1) & ~(uintptr_t(alignment) - 1)
                  : rawBase;
  return kSuccess;
}

inline Result addressFree(const Reservation &reservation) {
  return check(
      hipMemAddressFree(ptr(reservation.rawBase), reservation.rawSize));
}

inline Result memCreate(Handle *handle, size_t size, int device,
                        HandleType type) {
  auto prop = makeProp(device, type);
  return check(hipMemCreate(handle, size, &prop, 0));
}

inline Result memMap(uintptr_t addr, size_t size, Handle handle) {
  return check(hipMemMap(ptr(addr), size, /*offset*/ 0, handle, /*flags*/ 0));
}

// Grants `device` read/write access, and also the host if `hostAccess`.
inline Result memSetAccess(uintptr_t addr, size_t size, int device,
                           bool hostAccess = false) {
  hipMemAccessDesc descs[2] = {};
  descs[0].location.type = hipMemLocationTypeDevice;
  descs[0].location.id = device;
  descs[0].flags = hipMemAccessFlagsProtReadWrite;
  descs[1] = descs[0];
  descs[1].location.type = hipMemLocationTypeHost;
  descs[1].location.id = 0;
  return check(hipMemSetAccess(ptr(addr), size, descs, hostAccess ? 2 : 1));
}

inline Result memUnmap(uintptr_t addr, size_t size) {
  return check(hipMemUnmap(ptr(addr), size));
}

inline Result memRelease(Handle handle) { return check(hipMemRelease(handle)); }

inline Result exportHandle(void *out, Handle handle, HandleType type) {
  return check(hipMemExportToShareableHandle(
      out, handle, static_cast<hipMemAllocationHandleType>(type), 0));
}

inline Result importHandle(Handle *handle, void *osHandle, HandleType type) {
  return check(hipMemImportFromShareableHandle(
      handle, osHandle, static_cast<hipMemAllocationHandleType>(type)));
}

inline Result memsetD8(uintptr_t addr, size_t size) {
  return check(hipMemsetD8(ptr(addr), 0, size));
}

inline Result memsetD8Async(uintptr_t addr, size_t size, Stream stream) {
  return check(hipMemsetD8Async(ptr(addr), 0, size, stream));
}

inline Result memcpyHtoD(uintptr_t dst, const void *src, size_t size) {
  return check(hipMemcpyHtoD(ptr(dst), const_cast<void *>(src), size));
}

inline Result streamSynchronize(Stream stream) {
  return check(hipStreamSynchronize(stream));
}

// Makes `device` current for the guard's lifetime and restores the previous
// device afterwards. HIP's context API is deprecated, so use the device API.
class DeviceGuard {
public:
  Result enter(int device) {
    Result err = check(hipGetDevice(&previous));
    if (err != kSuccess)
      return err;
    entered = true;
    return check(hipSetDevice(device));
  }
  Result synchronize() { return check(hipDeviceSynchronize()); }
  Result exit() {
    if (!entered)
      return kSuccess;
    entered = false;
    return check(hipSetDevice(previous));
  }
  ~DeviceGuard() { (void)exit(); }

private:
  int previous = 0;
  bool entered = false;
};

// Prepares `device` for allocator work (memsets, copies) until the scope ends.
// HIP always has a current device, which may be another one, so select
// `device` and restore the previous device afterwards.
class DeviceScope {
public:
  Result enter(int device) { return guard.enter(device); }

private:
  DeviceGuard guard;
};

#else // CUDA

// CUDA user memory is device-only; ROCm-specific host access is not needed.
inline constexpr bool kSupportsHostAccess = false;

inline CUmemAllocationProp makeProp(int device, HandleType type) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(type);
  return prop;
}

inline const char *errorString(Result err) {
  const char *msg = "<unknown error>";
  cuGetErrorString(err, &msg);
  return msg;
}

inline Result deviceCount(int *count) { return cuDeviceGetCount(count); }

inline Result multiprocessorCount(int device, int *count) {
  CUdevice cuDevice = 0;
  Result err = cuDeviceGet(&cuDevice, device);
  if (err != kSuccess)
    return err;
  return cuDeviceGetAttribute(count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                              cuDevice);
}

inline Result supportsFabricHandles(int device, bool *supported) {
  CUdevice cuDevice = 0;
  Result err = cuDeviceGet(&cuDevice, device);
  if (err != kSuccess)
    return err;
  int value = 0;
  err = cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cuDevice);
  *supported = value != 0;
  return err;
}

inline Result allocationGranularity(int device, HandleType type,
                                    size_t *granularity) {
  auto prop = makeProp(device, type);
  return cuMemGetAllocationGranularity(granularity, &prop,
                                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

inline Result addressReserve(Reservation *out, size_t size, size_t alignment) {
  CUdeviceptr base = 0;
  Result err = cuMemAddressReserve(&base, size, alignment, /*addr*/ 0, 0);
  if (err != kSuccess)
    return err;
  out->base = out->rawBase = static_cast<uintptr_t>(base);
  out->rawSize = size;
  return kSuccess;
}

inline Result addressFree(const Reservation &reservation) {
  return cuMemAddressFree(reservation.rawBase, reservation.rawSize);
}

inline Result memCreate(Handle *handle, size_t size, int device,
                        HandleType type) {
  auto prop = makeProp(device, type);
  return cuMemCreate(handle, size, &prop, 0);
}

inline Result memMap(uintptr_t addr, size_t size, Handle handle) {
  return cuMemMap(addr, size, /*offset*/ 0, handle, /*flags*/ 0);
}

// Grants `device` read/write access. Host access is not supported here.
inline Result memSetAccess(uintptr_t addr, size_t size, int device,
                           bool hostAccess = false) {
  if (hostAccess)
    return kErrorNotSupported;
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return cuMemSetAccess(addr, size, &desc, 1);
}

inline Result memUnmap(uintptr_t addr, size_t size) {
  return cuMemUnmap(addr, size);
}

inline Result memRelease(Handle handle) { return cuMemRelease(handle); }

inline Result exportHandle(void *out, Handle handle, HandleType type) {
  return cuMemExportToShareableHandle(
      out, handle, static_cast<CUmemAllocationHandleType>(type), 0);
}

inline Result importHandle(Handle *handle, void *osHandle, HandleType type) {
  return cuMemImportFromShareableHandle(
      handle, osHandle, static_cast<CUmemAllocationHandleType>(type));
}

inline Result memsetD8(uintptr_t addr, size_t size) {
  return cuMemsetD8(addr, 0, size);
}

inline Result memsetD8Async(uintptr_t addr, size_t size, Stream stream) {
  return cuMemsetD8Async(addr, 0, size, stream);
}

inline Result memcpyHtoD(uintptr_t dst, const void *src, size_t size) {
  return cuMemcpyHtoD(dst, src, size);
}

inline Result streamSynchronize(Stream stream) {
  return cuStreamSynchronize(stream);
}

// Retains `device`'s primary context and makes it current for the guard's
// lifetime, then restores the previous context and releases the retain.
class DeviceGuard {
public:
  Result enter(int device) {
    Result err = cuCtxGetCurrent(&previous);
    if (err != kSuccess)
      return err;
    err = cuDevicePrimaryCtxRetain(&context, device);
    if (err != kSuccess)
      return err;
    retainedDevice = device;
    return cuCtxSetCurrent(context);
  }
  Result synchronize() { return cuCtxSynchronize(); }
  Result exit() {
    if (retainedDevice < 0)
      return kSuccess;
    Result err = cuCtxSetCurrent(previous);
    Result releaseErr = cuDevicePrimaryCtxRelease(retainedDevice);
    retainedDevice = -1;
    return err != kSuccess ? err : releaseErr;
  }
  ~DeviceGuard() { (void)exit(); }

private:
  CUcontext previous = nullptr;
  CUcontext context = nullptr;
  int retainedDevice = -1;
};

// Prepares `device` for allocator work. Malloc may be called before anything
// has made a context current; adopt the primary context in that case and
// leave an existing context alone. The adopted context stays current after the
// scope ends.
class DeviceScope {
public:
  Result enter(int device) {
    CUcontext ctx = nullptr;
    Result err = cuCtxGetCurrent(&ctx);
    if (err != kSuccess || ctx)
      return err;
    err = cuDevicePrimaryCtxRetain(&ctx, device);
    if (err != kSuccess)
      return err;
    return cuCtxSetCurrent(ctx);
  }
};

#endif

} // namespace gsan::drv
