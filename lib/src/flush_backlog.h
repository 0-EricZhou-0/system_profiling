// Internal: detect a flush thread that cannot keep up with its probe,
// and say so on stderr without flooding it.
//
// A backlog occurrence is a periodic flush (drain + serialize + write)
// that takes longer than the flush interval. Then the next flush starts
// late and carries more than one interval of samples, so the buffered
// data grows for as long as the disk stays that slow. Nothing is
// dropped; the warning is so the user knows memory is growing and why.
//
// Reporting: the first occurrence at once; after that at most one
// summary line per report period (30 s), carrying the count, the longest
// flush and the largest one since the last line; and a final summary at
// Stop() if there was any. The running count goes into every trace frame
// (FlushStats.slow_flushes / EventFlushStats.slow_flushes).
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace cupti_profiler {
namespace internal {

inline uint64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class FlushBacklog {
public:
    FlushBacklog(std::string probe, uint64_t flushIntervalMs);

    /// After each periodic flush. `durationNs`: drain to written.
    void Record(uint64_t durationNs, uint64_t bytes);

    /// Slow flushes so far.
    uint64_t SlowFlushes() const { return slowTotal_; }

    /// At Stop(): one line if any flush was slow.
    void Summary() const;

private:
    std::string probe_;
    uint64_t intervalNs_;
    uint64_t flushes_ = 0;
    uint64_t slowTotal_ = 0;
    uint64_t longestTotalNs_ = 0;
    // Since the last line printed.
    uint64_t slowSince_ = 0;
    uint64_t longestSinceNs_ = 0;
    uint64_t largestSinceBytes_ = 0;
    uint64_t lastLineNs_ = 0;
};

} // namespace internal
} // namespace cupti_profiler
