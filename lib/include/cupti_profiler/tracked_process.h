// Shared "one tracked PID, optionally with a display alias" struct used by
// SystemProfiler and DiskProfiler configs. Mirrors the TrackedProcessV2 proto
// message in proto/metric_sample.proto.

#pragma once

#include <cstdint>
#include <string>

namespace cupti_profiler {

struct TrackedProcess {
    uint32_t pid = 0;       // 0 = resolved to the current PID at config-load time
    std::string alias;      // optional display name; empty = no alias
};

} // namespace cupti_profiler
