// Internal: shared resolved-data types between the tracker and the events
// flush thread. EventTracker::Drain populates these via the public header.
#pragma once

#include <cstdint>
#include <string>

namespace cupti_profiler {
namespace internal {

struct ResolvedRegion {
    std::string name;
    uint64_t startNs = 0;
    uint64_t endNs   = 0;
};

struct ResolvedEvent {
    std::string name;
    uint64_t timestampNs = 0;
};

} // namespace internal
} // namespace cupti_profiler
