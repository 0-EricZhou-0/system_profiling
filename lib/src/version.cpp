#include <cupti_profiler/version.h>

#include "build_version.h"  // generated at build time (cmake/WriteVersion.cmake)

namespace cupti_profiler {

const char* Version() { return CUPTI_PROFILER_VERSION; }
const char* GitCommit() { return CUPTI_PROFILER_GIT_COMMIT; }

} // namespace cupti_profiler
