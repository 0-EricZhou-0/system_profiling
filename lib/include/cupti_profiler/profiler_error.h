// Error codes returned by ProfilerSuite::Configure() / Start().
//
// Introduced with the SIDECAR mode series: the sidecar spawn +
// capability handshake can fail in several distinct ways that
// callers should be able to distinguish (missing binary vs. missing
// caps vs. exec failure). The previous void-returning API forced
// std::cerr + std::exit(1) on any setup failure; this enum lets
// the workload handle setup errors as data.
//
// Ok = 0 is guaranteed and safe to compare against; other numeric
// values are stable across versions but callers should prefer
// symbolic names.
#pragma once

#if defined(_WIN32)
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __declspec(dllexport)
  #else
    #define CUPTI_PROFILER_API __declspec(dllimport)
  #endif
#else
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __attribute__((visibility("default")))
  #else
    #define CUPTI_PROFILER_API
  #endif
#endif

namespace cupti_profiler {

enum class ProfilerError {
    Ok = 0,

    // Generic setup problems
    NotConfigured        = 100,  // Configure() called before LoadConfig()

    // Sidecar-specific problems (SIDECAR mode only)
    SidecarNotFound      = 200,  // couldn't locate the sidecar binary
    SidecarSpawnFailed   = 201,  // fork() or execve() failed
    SidecarMissingCaps   = 202,  // sidecar started but lacks CAP_NET_ADMIN
    SidecarBadHandshake  = 203,  // sidecar returned an unexpected message
    SidecarExited        = 204,  // sidecar died before completing handshake
};

/// Stable human-readable identifier for a ProfilerError. Never
/// null; callers can log directly. Not localised.
CUPTI_PROFILER_API const char* ToString(ProfilerError err);

} // namespace cupti_profiler
