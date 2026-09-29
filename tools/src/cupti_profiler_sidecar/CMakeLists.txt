add_executable(cupti_profiler_sidecar cupti_profiler_sidecar.cpp)
# The sidecar parses ProfilerSuiteConfig from the pipe, so it needs
# access to the proto symbols. Rather than duplicating the compiled
# .pb.cc files into this target (which causes protobuf's
# descriptor-database double-registration to abort at process start),
# we rely on the proto symbols exported from libcupti_profiler.so —
# see the visibility override applied to those .pb.cc TUs in
# lib/CMakeLists.txt.
add_dependencies(cupti_profiler_sidecar cupti_profiler)

target_link_libraries(cupti_profiler_sidecar
    PRIVATE
        cupti_profiler
        protobuf::libprotobuf
        pthread
)

target_include_directories(cupti_profiler_sidecar
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/src   # sidecar_protocol.h
        ${CMAKE_BINARY_DIR}/lib       # generated proto headers
)

set_target_properties(cupti_profiler_sidecar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
    OUTPUT_NAME "cupti-profiler-sidecar"
)

# POST_BUILD: check whether the freshly-linked binary carries
# CAP_NET_ADMIN. Emit a persistent reminder to stderr if not — the cap
# is required for the (future) taskstats backend when observing PIDs
# owned by other UIDs. The current /proc-based sidecar does not
# require any capability for same-UID observation, so the reminder is
# advisory. Non-fatal by design.
add_custom_command(TARGET cupti_profiler_sidecar POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
        SIDECAR=$<TARGET_FILE:cupti_profiler_sidecar>
        ${CMAKE_SOURCE_DIR}/cmake/check_sidecar_caps.sh
    VERBATIM
)
