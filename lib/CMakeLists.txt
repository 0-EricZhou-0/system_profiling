# Generate protobuf C++ sources into the build directory.
# metric_catalog.proto + metric_sample.proto define the type system and
# shared substructures (Sample, ProcessSample, TrackedProcessV2, ...)
# that the per-domain trace protos import.
protobuf_generate_cpp(MC_PROTO_SRCS  MC_PROTO_HDRS  "${CMAKE_SOURCE_DIR}/proto/metric_catalog.proto")
protobuf_generate_cpp(MS_PROTO_SRCS  MS_PROTO_HDRS  "${CMAKE_SOURCE_DIR}/proto/metric_sample.proto")
protobuf_generate_cpp(PN_PROTO_SRCS  PN_PROTO_HDRS  "${CMAKE_SOURCE_DIR}/proto/panels.proto")
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
    # Shared infrastructure
    src/metric_catalog.cpp
    src/metric_catalog_builtins.cpp
    src/builtin_catalog.cpp
    src/profiler_error.cpp
    src/sidecar_process.cpp
    src/process_tracking_probe.cpp
    # Orchestrator
    src/profiler_suite.cpp
    # Generated protobuf sources
    ${MC_PROTO_SRCS}
    ${MS_PROTO_SRCS}
    ${PN_PROTO_SRCS}
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

# The cupti-profiler-sidecar binary needs to parse ProfilerSuiteConfig
# from the pipe on startup — that requires the proto symbols to be
# visible in libcupti_profiler.so. The library's default hidden
# visibility hides them along with everything else; override that for
# just the generated proto TUs so the sidecar link resolves cleanly.
# We do NOT link the proto .cc's into the sidecar directly because
# protobuf's descriptor-database would double-register and abort.
set_source_files_properties(
    ${MC_PROTO_SRCS} ${MS_PROTO_SRCS} ${PN_PROTO_SRCS}
    ${GPU_PROTO_SRCS} ${SYS_PROTO_SRCS} ${DISK_PROTO_SRCS}
    ${EVT_PROTO_SRCS} ${SESS_PROTO_SRCS} ${CFG_PROTO_SRCS}
    PROPERTIES COMPILE_FLAGS "-fvisibility=default"
)

# Baked-in default path for the sidecar binary. Env var
# CUPTI_PROFILER_SIDECAR overrides at runtime; without either, the
# library reports SidecarNotFound if a caller requests SIDECAR mode.
if(NOT CUPTI_PROFILER_SIDECAR_PATH)
    set(CUPTI_PROFILER_SIDECAR_PATH
        "${CMAKE_BINARY_DIR}/tools/cupti-profiler-sidecar"
        CACHE PATH "Absolute path to cupti-profiler-sidecar (default = build tree)")
endif()
target_compile_definitions(cupti_profiler PRIVATE
    CUPTI_PROFILER_SIDECAR_PATH="${CUPTI_PROFILER_SIDECAR_PATH}")

set_target_properties(cupti_profiler PROPERTIES
    CXX_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Install libcupti_profiler.so alongside the pybind11 extension inside the
# Python package, so `pip install` produces a self-contained wheel where
# _native.so finds the C++ library via $ORIGIN.
install(TARGETS cupti_profiler
    LIBRARY DESTINATION cupti_profiler)
