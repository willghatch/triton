include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  add_triton_plugin(TritonAMD
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dylib_utils.cc
    LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM TritonAMDMFMASchedule)
  target_link_libraries(TritonAMD PRIVATE Python3::Module nanobind-static lldCommon lldELF)

  # GSan device runtime as generic amdgcn bitcode (no -mcpu), linked into
  # instrumented kernels before ocml/ockl. Code object v5 matches the module
  # flag Triton sets; HIP mode would stamp v6 and fail to link.
  set(GSAN_RUNTIME_SRC "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSanLibrary.cu")
  set(GSAN_RUNTIME_HDRS
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/Compat.h"
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSan.h"
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/Hash.cuh"
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/TMA.cuh")
  set(GSAN_AMD_RUNTIME_BC "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/gsan.bc")
  find_program(TRITON_AMD_GSAN_CLANGXX NAMES clang++
               PATHS "${LLVM_SYSPATH}/bin"
               REQUIRED NO_DEFAULT_PATH)
  add_custom_command(
      OUTPUT "${GSAN_AMD_RUNTIME_BC}"
      COMMAND "${TRITON_AMD_GSAN_CLANGXX}"
              --target=amdgcn-amd-amdhsa
              -x c++
              -std=c++17 -O3 -emit-llvm -c
              -nogpulib
              -nostdinc
              -fno-exceptions
              -fno-rtti
              -mcode-object-version=5
              "${GSAN_RUNTIME_SRC}" -o "${GSAN_AMD_RUNTIME_BC}"
      DEPENDS "${GSAN_RUNTIME_SRC}" ${GSAN_RUNTIME_HDRS}
      COMMENT "Building AMD GSan runtime"
      VERBATIM)
  add_custom_target(TritonAMDGSanRuntime ALL DEPENDS "${GSAN_AMD_RUNTIME_BC}")
  add_dependencies(TritonAMD TritonAMDGSanRuntime)
endif()
add_subdirectory(test)
