#include "flush_backlog.h"
#include "testing_hooks.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace cupti_profiler {
namespace internal {

FlushBacklog::FlushBacklog(std::string probe, uint64_t flushIntervalMs)
    : probe_(std::move(probe)), intervalNs_(flushIntervalMs * 1000000ull) {}

void FlushBacklog::Record(uint64_t durationNs, uint64_t bytes) {
    ++flushes_;
    if (durationNs <= intervalNs_) return;
    ++slowTotal_;
    ++slowSince_;
    longestTotalNs_    = std::max(longestTotalNs_, durationNs);
    longestSinceNs_    = std::max(longestSinceNs_, durationNs);
    largestSinceBytes_ = std::max(largestSinceBytes_, bytes);

    const uint64_t now = SteadyNowNs();
    const uint64_t periodNs = static_cast<uint64_t>(BacklogReportPeriodMs()) * 1000000ull;
    if (slowTotal_ == 1) {
        std::fprintf(stderr,
            "[cupti-profiler] warning: %s: a flush took %.0f ms, longer than the flush interval "
            "(%.0f ms): writing is not keeping up, buffered samples grow (nothing is dropped). "
            "Further occurrences are summed up every %.0f s\n",
            probe_.c_str(), durationNs / 1e6, intervalNs_ / 1e6, periodNs / 1e9);
    } else if (now - lastLineNs_ >= periodNs) {
        std::fprintf(stderr,
            "[cupti-profiler] warning: %s: %llu flush(es) slower than the %.0f ms interval in the "
            "last %.0f s (longest %.0f ms, largest %llu bytes); %llu so far\n",
            probe_.c_str(), static_cast<unsigned long long>(slowSince_), intervalNs_ / 1e6,
            (now - lastLineNs_) / 1e9, longestSinceNs_ / 1e6,
            static_cast<unsigned long long>(largestSinceBytes_),
            static_cast<unsigned long long>(slowTotal_));
    } else {
        return;
    }
    lastLineNs_ = now;
    slowSince_ = longestSinceNs_ = largestSinceBytes_ = 0;
}

void FlushBacklog::Summary() const {
    if (slowTotal_ == 0) return;
    std::fprintf(stderr,
        "[cupti-profiler] warning: %s flush summary: %llu of %llu flushes took longer than the "
        "%.0f ms interval (longest %.0f ms)\n",
        probe_.c_str(), static_cast<unsigned long long>(slowTotal_),
        static_cast<unsigned long long>(flushes_), intervalNs_ / 1e6, longestTotalNs_ / 1e6);
}

} // namespace internal
} // namespace cupti_profiler
