cmake_minimum_required(VERSION 3.18)
project(cupti-profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA Toolkit (provides CUDA::cudart, CUDA::cublas, include paths)
find_package(CUDAToolkit REQUIRED)

# Find Protobuf — force system protoc to match libprotobuf version
set(Protobuf_PROTOC_EXECUTABLE "/usr/bin/protoc" CACHE FILEPATH "Path to protoc")
find_package(Protobuf REQUIRED)

# CUPTI — shipped with the CUDA Toolkit.
# Headers are in the standard CUDAToolkit include dir; library location varies.
find_library(CUPTI_LIBRARY cupti
    HINTS "${CUDAToolkit_LIBRARY_DIR}"
          "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
)
if(NOT CUPTI_LIBRARY)
    message(FATAL_ERROR "Could not find libcupti.so — is CUPTI installed?")
endif()
add_library(cupti SHARED IMPORTED)
set_target_properties(cupti PROPERTIES
    IMPORTED_LOCATION "${CUPTI_LIBRARY}"
)

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(tools/src)

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python wrapper" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Generate Python protobuf bindings into generated/proto/
# Uses grpc_tools.protoc (from conda) for version compatibility with the Python
# protobuf runtime, since system protoc may be too old.
set(PROTO_FILES
    "${CMAKE_SOURCE_DIR}/proto/metric_catalog.proto"
    "${CMAKE_SOURCE_DIR}/proto/metric_sample.proto"
    "${CMAKE_SOURCE_DIR}/proto/panels.proto"
    "${CMAKE_SOURCE_DIR}/proto/gpu_metrics.proto"
    "${CMAKE_SOURCE_DIR}/proto/system_metrics.proto"
    "${CMAKE_SOURCE_DIR}/proto/disk_metrics.proto"
    "${CMAKE_SOURCE_DIR}/proto/events.proto"
    "${CMAKE_SOURCE_DIR}/proto/session_metadata.proto"
    "${CMAKE_SOURCE_DIR}/proto/profiler_config.proto"
)
set(GENERATED_PROTO_DIR "${CMAKE_SOURCE_DIR}/generated/proto")
set(PY_PB2_FILES
    "${GENERATED_PROTO_DIR}/metric_catalog_pb2.py"
    "${GENERATED_PROTO_DIR}/metric_sample_pb2.py"
    "${GENERATED_PROTO_DIR}/panels_pb2.py"
    "${GENERATED_PROTO_DIR}/gpu_metrics_pb2.py"
    "${GENERATED_PROTO_DIR}/system_metrics_pb2.py"
    "${GENERATED_PROTO_DIR}/disk_metrics_pb2.py"
    "${GENERATED_PROTO_DIR}/events_pb2.py"
    "${GENERATED_PROTO_DIR}/session_metadata_pb2.py"
    "${GENERATED_PROTO_DIR}/profiler_config_pb2.py"
)
add_custom_command(
    OUTPUT ${PY_PB2_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_PROTO_DIR}"
    COMMAND python -m grpc_tools.protoc
        --python_out=${GENERATED_PROTO_DIR}
        -I${CMAKE_SOURCE_DIR}/proto
        ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Generating Python protobuf bindings into generated/proto/"
)
add_custom_target(proto_python ALL DEPENDS ${PY_PB2_FILES})
