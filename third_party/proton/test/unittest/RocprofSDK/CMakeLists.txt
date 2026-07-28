add_triton_ut(
  NAME RocprofSDKPCSamplingCorrection
  SRCS
    RocprofSDKPCSamplingCorrectionTest.cpp
    "${PROTON_SRC_DIR}/lib/Profiler/RocprofSDK/RocprofSDKPCSamplingCorrection.cpp"
)

target_include_directories(RocprofSDKPCSamplingCorrection
  PRIVATE
    "${ROCPROFILER_SDK_INCLUDE_DIR}"
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)
