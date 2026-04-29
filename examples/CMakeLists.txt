add_executable(gemm_profiling gemm_profiling.cu)

target_link_libraries(gemm_profiling
    PRIVATE
        cupti_profiler
        CUDA::cublas
)

set_target_properties(gemm_profiling PROPERTIES
    CUDA_ARCHITECTURES 90
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)

# Full-system profiling example (GPU + CPU/Mem + Disk)
add_executable(full_system_profiling full_system_profiling.cu)

target_link_libraries(full_system_profiling
    PRIVATE
        cupti_profiler
        CUDA::cublas
)

set_target_properties(full_system_profiling PROPERTIES
    CUDA_ARCHITECTURES 90
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)

# Same example, but linked against the built libcupti_profiler.so by file
# path rather than via the in-tree CMake target. This mimics how a downstream
# consumer would build their app: they have only the public headers and the
# shared library binary, no access to the library's CMake graph.
#
# A successful build of this target is a smoke test that the .so's exported
# ABI is self-sufficient for building real programs.
add_executable(full_system_profiling_external full_system_profiling.cu)

target_include_directories(full_system_profiling_external PRIVATE
    "${CMAKE_SOURCE_DIR}/lib/include"
)

target_link_libraries(full_system_profiling_external
    PRIVATE
        "${CMAKE_BINARY_DIR}/lib/libcupti_profiler.so"
        CUDA::cublas
)

# Make sure the .so exists before this target tries to link to it.
add_dependencies(full_system_profiling_external cupti_profiler)

set_target_properties(full_system_profiling_external PROPERTIES
    CUDA_ARCHITECTURES 90
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
