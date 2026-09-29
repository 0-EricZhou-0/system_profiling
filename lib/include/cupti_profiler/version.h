#pragma once

#include <cupti_profiler/profiler_error.h>

namespace cupti_profiler {

// The version of this library: the package version in pyproject.toml,
// the one source of truth (read by CMake at configure time). Every
// session_metadata.pb records it (SessionMetadata.producer).
CUPTI_PROFILER_API const char* Version();

// The git commit the library was built from (12 hex digits, with
// "-dirty" when tracked files had uncommitted changes), or "" when the
// source tree was not a git checkout at build time.
CUPTI_PROFILER_API const char* GitCommit();

} // namespace cupti_profiler
