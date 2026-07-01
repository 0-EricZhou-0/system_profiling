#include <cupti_profiler/profiler_error.h>

namespace cupti_profiler {

const char* ToString(ProfilerError err) {
    switch (err) {
        case ProfilerError::Ok:                   return "Ok";
        case ProfilerError::NotConfigured:        return "NotConfigured";
        case ProfilerError::SidecarNotFound:      return "SidecarNotFound";
        case ProfilerError::SidecarSpawnFailed:   return "SidecarSpawnFailed";
        case ProfilerError::SidecarMissingCaps:   return "SidecarMissingCaps";
        case ProfilerError::SidecarBadHandshake:  return "SidecarBadHandshake";
        case ProfilerError::SidecarExited:        return "SidecarExited";
    }
    return "UnknownProfilerError";
}

} // namespace cupti_profiler
