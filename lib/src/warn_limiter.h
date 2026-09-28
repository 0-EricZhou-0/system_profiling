// Internal: per-process warnings, rate-limited per (tracked process,
// warning type). Owned by ProcessTrackingProbe; used by the System and
// Disk probes for their per-PID read failures.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace cupti_profiler {
namespace internal {

class WarnLimiter {
public:
    static constexpr uint64_t kPeriodNs = 1'000'000'000ull;   // at most one per key per second

    /// Warn `text` (one line, no newline) for the tracked process
    /// registered as `serial` and warning `type`, unless a warning for
    /// that key went out less than kPeriodNs before `nowNs`: then it is
    /// counted, and the next one that goes out says "(N suppressed)".
    /// Returns true if it was written. Thread-safe.
    bool Warn(uint64_t serial, int type, const std::string& text, uint64_t nowNs);

    /// The process registered as `serial` is no longer tracked: drop all
    /// its keys, first writing, for each with suppressed warnings, the
    /// last one with its count (once, whatever the rate). Thread-safe.
    void Remove(uint64_t serial);

    /// At stop: the same for every key; empty afterwards. Thread-safe.
    void Flush();

    /// Keys held (tracked processes x warning types with a warning).
    size_t Size() const;

    ~WarnLimiter();

private:
    struct State {
        uint64_t    lastNs = 0;       // when the last warning went out
        uint64_t    suppressed = 0;   // since then
        std::string text;             // the latest text for this key
    };
    void Summarize(const State& s, const char* when);

    mutable std::mutex mu_;
    std::map<std::pair<uint64_t, int>, State> keys_;
};

/// Keys held by every WarnLimiter in this process (testing::WarnStateSize).
size_t WarnLimiterKeysInProcess();

} // namespace internal
} // namespace cupti_profiler
