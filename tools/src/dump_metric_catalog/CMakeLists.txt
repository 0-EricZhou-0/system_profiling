add_executable(dump_metric_catalog dump_metric_catalog.cpp)

target_link_libraries(dump_metric_catalog
    PRIVATE
        cupti_profiler
        protobuf::libprotobuf
)

set_target_properties(dump_metric_catalog PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)
