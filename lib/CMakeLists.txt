# Generate protobuf C++ sources into the build directory.
# tracked_process.proto is shared by config + system + disk traces, so it
# must be generated first (others import its descriptor).
protobuf_generate_cpp(TP_PROTO_SRCS  TP_PROTO_HDRS  "${CMAKE_SOURCE_DIR}/proto/tracked_process.proto")
protobuf_generate_cpp(GPU_PROTO_SRCS GPU_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/gpu_metrics.proto")
protobuf_generate_cpp(SYS_PROTO_SRCS SYS_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/system_metrics.proto")
protobuf_generate_cpp(DISK_PROTO_SRCS DISK_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/disk_metrics.proto")
protobuf_generate_cpp(EVT_PROTO_SRCS EVT_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/events.proto")
protobuf_generate_cpp(SESS_PROTO_SRCS SESS_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/session_metadata.proto")
protobuf_generate_cpp(CFG_PROTO_SRCS CFG_PROTO_HDRS "${CMAKE_SOURCE_DIR}/proto/profiler_config.proto")

add_library(cupti_profiler SHARED
    # GPU profiler (existing)
    src/gpu_profiler.cpp
    src/cupti_pm_sampling.cpp
    src/profiler_host_internal.cpp
    src/decode_thread.cpp
    src/flush_thread.cpp
    # System profiler (CPU + memory)
    src/proc_readers.cpp
    src/system_flush_thread.cpp
    src/system_profiler.cpp
    # Disk profiler
    src/disk_readers.cpp
    src/disk_flush_thread.cpp
    src/disk_profiler.cpp
    # Events profiler (regions + instantaneous events, multi-domain)
    src/event_tracker.cpp
    src/event_flush_thread.cpp
    src/event_profiler.cpp
    # Session manifest writer
    src/session_metadata_writer.cpp
    # Orchestrator
    src/profiler_suite.cpp
    # Generated protobuf sources
    ${TP_PROTO_SRCS}
    ${GPU_PROTO_SRCS}
    ${SYS_PROTO_SRCS}
    ${DISK_PROTO_SRCS}
    ${EVT_PROTO_SRCS}
    ${SESS_PROTO_SRCS}
    ${CFG_PROTO_SRCS}
)

target_include_directories(cupti_profiler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}   # generated protobuf headers
)

target_link_libraries(cupti_profiler
    PUBLIC
        CUDA::cudart
    PRIVATE
        cupti
        CUDA::cuda_driver
        CUDA::nvml
        protobuf::libprotobuf
        pthread
)

target_compile_definitions(cupti_profiler PRIVATE CUPTI_PROFILER_EXPORTS)

set_target_properties(cupti_profiler PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Install libcupti_profiler.so alongside the pybind11 extension inside the
# Python package, so `pip install` produces a self-contained wheel where
# _native.so finds the C++ library via $ORIGIN.
install(TARGETS cupti_profiler
    LIBRARY DESTINATION cupti_profiler)
