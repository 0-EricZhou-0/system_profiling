add_executable(list_pm_metrics list_pm_metrics.cpp)

target_include_directories(list_pm_metrics PRIVATE
    "${CUDAToolkit_ROOT}/extras/CUPTI/include"
)

target_link_libraries(list_pm_metrics
    PRIVATE
        cupti
        CUDA::cudart
        CUDA::cuda_driver
)

set_target_properties(list_pm_metrics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)
