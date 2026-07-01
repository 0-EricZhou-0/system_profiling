# Python bindings for cupti_profiler. Builds a pybind11 extension module
# named `_native` and stages it (plus the pure-Python wrapper code) into
# build/python/cupti_profiler/ so tests can import without installing.

# pybind11 ships its CMake config inside its Python package. If a vanilla
# find_package fails, fall back to asking the python interpreter where the
# pybind11 package put its share/cmake/pybind11 directory.

# Opt into the modern FindPython module before pybind11 includes it.
# Without this, recent pybind11 (3.x) emits a CMP0148 deprecation warning
# because it falls back to FindPythonInterp / FindPythonLibs.
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _PYBIND11_RC
    )
    if(NOT _PYBIND11_RC EQUAL 0 OR NOT _PYBIND11_CMAKE_DIR)
        message(FATAL_ERROR
            "Could not locate pybind11. Install it into the active Python "
            "environment with `pip install pybind11`, or pass "
            "-Dpybind11_DIR=/path/to/share/cmake/pybind11.")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${_PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(cupti_profiler_native MODULE binding.cpp)

set_target_properties(cupti_profiler_native PROPERTIES
    OUTPUT_NAME _native
    CXX_STANDARD 20
)

target_link_libraries(cupti_profiler_native
    PRIVATE
        cupti_profiler
        CUDA::cudart
)

target_include_directories(cupti_profiler_native PRIVATE
    "${CMAKE_SOURCE_DIR}/lib/include"
)

# Stage the importable layout under build/python/cupti_profiler/.
# The .so lands here as _native*.so and the pure-Python files alongside it,
# so `PYTHONPATH=build/python python -c "import cupti_profiler"` just works.
set(STAGED_PKG "${CMAKE_BINARY_DIR}/python/cupti_profiler")

add_custom_command(TARGET cupti_profiler_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${STAGED_PKG}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${STAGED_PKG}/proto"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/__init__.py"
        "${STAGED_PKG}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/_stream.py"
        "${STAGED_PKG}/_stream.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/proto/__init__.py"
        "${STAGED_PKG}/proto/__init__.py"
    # Stage generated *_pb2.py inside the package as cupti_profiler.proto.*
    # so `from cupti_profiler.proto import events_pb2` works in both dev
    # mode (PYTHONPATH=build/python) and after `pip install`.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/generated/proto/metric_catalog_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/metric_sample_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/panels_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/events_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/session_metadata_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/gpu_metrics_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/system_metrics_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/disk_metrics_pb2.py"
        "${CMAKE_SOURCE_DIR}/generated/proto/profiler_config_pb2.py"
        "${STAGED_PKG}/proto/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:cupti_profiler_native>
        "${STAGED_PKG}/$<TARGET_FILE_NAME:cupti_profiler_native>"
    COMMENT "Staging cupti_profiler Python package into build/python/"
)

# Make sure the generated *_pb2.py files exist before the staging step runs.
add_dependencies(cupti_profiler_native proto_python)

# Generate .pyi type stubs by introspecting the freshly-built extension.
# Stubs land alongside the .so in build/python/cupti_profiler/ so they
# travel with the package on PYTHONPATH and IDEs pick them up automatically.
# pybind11-stubgen is optional: if it isn't installed we just skip stub
# generation rather than fail the build.
find_program(PYBIND11_STUBGEN_EXE pybind11-stubgen)
if(PYBIND11_STUBGEN_EXE)
    add_custom_command(TARGET cupti_profiler_native POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_SOURCE_DIR}/generated/proto"
            "${PYBIND11_STUBGEN_EXE}" cupti_profiler
                --output-dir "${CMAKE_BINARY_DIR}/python"
                --exit-code
        COMMENT "Generating .pyi type stubs for cupti_profiler"
        VERBATIM
    )
else()
    message(STATUS
        "pybind11-stubgen not found — .pyi stubs will not be generated. "
        "Install it with `pip install pybind11-stubgen` to enable.")
endif()

# Make the staged package depend on the pure-Python sources so changes to
# them re-trigger the copy step on incremental builds.
set_property(TARGET cupti_profiler_native APPEND PROPERTY
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/_stream.py"
)

# Locate libcupti_profiler.so at runtime so the .so import doesn't need
# LD_LIBRARY_PATH. After `pip install`, libcupti_profiler.so is bundled
# alongside _native.so inside the cupti_profiler package directory, so
# INSTALL_RPATH is $ORIGIN. During development the .so lives in
# build/lib/, hence BUILD_RPATH points there.
set_target_properties(cupti_profiler_native PROPERTIES
    BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

# ----- install rules consumed by `pip install` (scikit-build-core) ----------
# Lay out everything inside the cupti_profiler/ package so a single wheel
# directory contains: pure-Python files, the pybind11 extension, the C++
# shared library it links against, the protobuf bindings, and .pyi stubs.
install(TARGETS cupti_profiler_native
    LIBRARY DESTINATION cupti_profiler)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/_stream.py"
    DESTINATION cupti_profiler)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cupti_profiler/proto/__init__.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/metric_catalog_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/metric_sample_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/panels_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/events_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/session_metadata_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/gpu_metrics_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/system_metrics_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/disk_metrics_pb2.py"
    "${CMAKE_SOURCE_DIR}/generated/proto/profiler_config_pb2.py"
    DESTINATION cupti_profiler/proto)

# .pyi stubs only exist if pybind11-stubgen was found at configure time.
if(PYBIND11_STUBGEN_EXE)
    install(FILES
        "${CMAKE_BINARY_DIR}/python/cupti_profiler/__init__.pyi"
        "${CMAKE_BINARY_DIR}/python/cupti_profiler/_native.pyi"
        "${CMAKE_BINARY_DIR}/python/cupti_profiler/_stream.pyi"
        DESTINATION cupti_profiler
        OPTIONAL)
endif()
