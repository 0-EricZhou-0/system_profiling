add_executable(cupti_profiler_sidecar cupti_profiler_sidecar.cpp)

set_target_properties(cupti_profiler_sidecar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
    OUTPUT_NAME "cupti-profiler-sidecar"
)

# POST_BUILD: check whether the freshly-linked binary carries
# CAP_NET_ADMIN. Emit a persistent reminder to stderr if not — the cap
# is required to run in SIDECAR mode with the taskstats backend
# (untargeted UIDs), and every re-link strips the file capabilities so
# the reminder needs to fire on every rebuild. Non-fatal by design;
# LEGACY mode (in-process /proc) does not require any capability.
add_custom_command(TARGET cupti_profiler_sidecar POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env
        SIDECAR=$<TARGET_FILE:cupti_profiler_sidecar>
        ${CMAKE_SOURCE_DIR}/cmake/check_sidecar_caps.sh
    VERBATIM
)
