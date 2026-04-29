// Internal: periodic flush thread for the events profiler.
#pragma once

#include "event_tracker_internal.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>

class EventTrace;

namespace cupti_profiler {

class EventTracker;

namespace internal {

struct EventPendingFlushStats {
    uint64_t timestampNs   = 0;
    uint64_t bytesWritten  = 0;
    uint64_t intervalNs    = 0;
    bool     valid         = false;
};

/// Write `trace` length-delimited to `out`; returns serialized size in bytes.
size_t WriteDelimitedEventTraceSized(const EventTrace& trace, std::ofstream& out);

void EventFlushThreadFunc(EventTracker& generic,
                          EventTracker& gpu,
                          std::ofstream& outFile,
                          std::mutex& outMutex,
                          std::atomic<bool>& stop,
                          uint64_t flushIntervalMs,
                          uint64_t steadyClockRefNs,
                          uint64_t cuptiRefNs,
                          uint64_t wallClockEpochNs,
                          EventPendingFlushStats& pending,
                          std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
