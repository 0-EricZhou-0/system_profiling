#include "event_flush_thread.h"
#include "delimited_write.h"

#include <cupti_profiler/event_profiler.h>
#include "events.pb.h"

#include <chrono>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <vector>

namespace cupti_profiler {
namespace internal {

size_t WriteDelimitedEventTraceSized(const EventTrace& trace, std::ofstream& out) {
    return WriteDelimitedFrame(trace, out, "EventTrace");
}

namespace {

// Build an EventBuffer for a given domain by draining the tracker.
// Returns true if any data was added.
bool AppendBuffer(EventTrace& trace,
                  EventTracker& tracker,
                  TimeDomain domain,
                  bool forceResolve)
{
    std::vector<ResolvedRegion> regions;
    std::vector<ResolvedEvent>  events;
    tracker.Drain(forceResolve, regions, events);
    if (regions.empty() && events.empty()) return false;

    auto* buf = trace.add_buffers();
    buf->set_domain(domain);
    for (auto& r : regions) {
        auto* pr = buf->add_regions();
        pr->set_name(std::move(r.name));
        pr->set_start_timestamp_ns(r.startNs);
        pr->set_end_timestamp_ns(r.endNs);
    }
    for (auto& e : events) {
        auto* pe = buf->add_events();
        pe->set_name(std::move(e.name));
        pe->set_timestamp_ns(e.timestampNs);
    }
    return true;
}

} // namespace

void EventFlushThreadFunc(EventTracker& generic,
                          EventTracker& gpu,
                          std::ofstream& outFile,
                          std::mutex& outMutex,
                          StopSignal& stop,
                          uint64_t flushIntervalMs,
                          uint64_t steadyClockRefNs,
                          uint64_t cuptiRefNs,
                          uint64_t wallClockEpochNs,
                          EventPendingFlushStats& pending,
                          std::mutex& pendingMutex)
{
    ::pthread_setname_np(::pthread_self(), "cupti-evt-flush");
    uint64_t prevFlushNs = 0;
    // Stop() wakes the wait; the final flush is Stop()'s.
    while (!stop.WaitUntil(std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(flushIntervalMs))) {

        EventTrace trace;
        auto* meta = trace.mutable_metadata();
        meta->set_steady_clock_reference_ns(steadyClockRefNs);
        meta->set_cupti_reference_ns(cuptiRefNs);
        meta->set_wall_clock_epoch_ns(wallClockEpochNs);

        bool wrote = false;
        wrote |= AppendBuffer(trace, generic, TIME_DOMAIN_GENERIC, /*forceResolve=*/false);
        wrote |= AppendBuffer(trace, gpu,     TIME_DOMAIN_GPU,     /*forceResolve=*/false);

        // Attach previous flush's stats (known only in hindsight).
        {
            std::lock_guard<std::mutex> lk(pendingMutex);
            if (pending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_timestamp_ns(pending.timestampNs);
                fs->set_bytes_written(pending.bytesWritten);
                fs->set_interval_ns(pending.intervalNs);
                pending.valid = false;
            }
        }

        if (!wrote && trace.flush_stats_size() == 0) continue;

        size_t bytes = 0;
        {
            std::lock_guard<std::mutex> lk(outMutex);
            bytes = WriteDelimitedEventTraceSized(trace, outFile);
            outFile.flush();
        }
        uint64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t intervalNs = (prevFlushNs == 0) ? 0 : (nowNs - prevFlushNs);
        prevFlushNs = nowNs;

        {
            std::lock_guard<std::mutex> lk(pendingMutex);
            pending.timestampNs  = nowNs;
            pending.bytesWritten = bytes;
            pending.intervalNs   = intervalNs;
            pending.valid        = true;
        }

        size_t totalRegions = 0, totalEvents = 0;
        for (int i = 0; i < trace.buffers_size(); ++i) {
            totalRegions += trace.buffers(i).regions_size();
            totalEvents  += trace.buffers(i).events_size();
        }
        double kibPerSec = (intervalNs > 0)
            ? (double)bytes * 1e9 / ((double)intervalNs * 1024.0) : 0.0;
        std::cout << "[Events] Flushed " << totalRegions << " regions, "
                  << totalEvents << " events, " << bytes << " bytes in "
                  << (intervalNs / 1000000) << " ms ("
                  << kibPerSec << " KiB/s)\n";
    }
}

} // namespace internal
} // namespace cupti_profiler
