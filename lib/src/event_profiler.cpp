#include <cupti_profiler/event_profiler.h>

#include "event_tracker_internal.h"
#include "event_flush_thread.h"

#include "events.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <cupti_target.h>  // CUptiResult
extern "C" CUptiResult cuptiGetTimestamp(uint64_t *timestamp);

namespace cupti_profiler {

class EventProfiler::Impl {
public:
    EventProfilerConfig config;

    // Allocated lazily because EventTracker constructor is private; we use
    // unique_ptr + new (we're a friend) to avoid public ctor exposure.
    std::unique_ptr<EventTracker> generic;
    std::unique_ptr<EventTracker> gpu;

    std::ofstream outFile;
    std::mutex outMutex;

    std::thread flushThread;
    std::atomic<bool> stopFlush{false};

    internal::EventPendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;

    uint64_t steadyClockRefNs = 0;
    uint64_t cuptiRefNs = 0;
    uint64_t wallClockEpochNs = 0;

    bool configured = false;
    bool running = false;
};

EventProfiler::EventProfiler() : m_impl(std::make_unique<Impl>()) {
    m_impl->generic = std::unique_ptr<EventTracker>(new EventTracker(EventTracker::Domain::GENERIC));
    m_impl->gpu     = std::unique_ptr<EventTracker>(new EventTracker(EventTracker::Domain::GPU));
}
EventProfiler::~EventProfiler() {
    if (m_impl && m_impl->running) Stop();
}
EventProfiler::EventProfiler(EventProfiler&&) noexcept = default;
EventProfiler& EventProfiler::operator=(EventProfiler&&) noexcept = default;

EventTracker& EventProfiler::GetGenericTracker() { return *m_impl->generic; }
EventTracker& EventProfiler::GetGpuTracker()     { return *m_impl->gpu; }

void EventProfiler::Configure(const EventProfilerConfig& config) {
    m_impl->config = config;
    m_impl->configured = true;
}

void EventProfiler::Start() {
    if (!m_impl->configured) {
        std::cerr << "EventProfiler::Start() called before Configure()\n";
        return;
    }

    if (!m_impl->config.outputFile.empty()) {
        m_impl->outFile.open(m_impl->config.outputFile,
                             std::ios::binary | std::ios::trunc);
        if (!m_impl->outFile) {
            std::cerr << "Failed to open events output file: "
                      << m_impl->config.outputFile << "\n";
            return;
        }
    }

    // Sync anchors at Start.
    cuptiGetTimestamp(&m_impl->cuptiRefNs);
    m_impl->steadyClockRefNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_impl->wallClockEpochNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    m_impl->stopFlush = false;
    if (m_impl->config.flushIntervalMs > 0 && m_impl->outFile.is_open()) {
        m_impl->flushThread = std::thread(internal::EventFlushThreadFunc,
                                          std::ref(*m_impl->generic),
                                          std::ref(*m_impl->gpu),
                                          std::ref(m_impl->outFile),
                                          std::ref(m_impl->outMutex),
                                          std::ref(m_impl->stopFlush),
                                          m_impl->config.flushIntervalMs,
                                          m_impl->steadyClockRefNs,
                                          m_impl->cuptiRefNs,
                                          m_impl->wallClockEpochNs,
                                          std::ref(m_impl->flushStatsPending),
                                          std::ref(m_impl->flushStatsMutex));
    }

    m_impl->running = true;
    std::cout << "[Events] Profiler started\n";
}

void EventProfiler::SignalStop() {
    if (!m_impl->running) return;
    m_impl->stopFlush = true;
}

void EventProfiler::Stop() {
    if (!m_impl->running) return;
    m_impl->stopFlush = true;
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    // Final flush — block until any in-flight cudaEvents complete so the last
    // GPU regions are captured.
    if (m_impl->outFile.is_open()) {
        EventTrace trace;
        auto* meta = trace.mutable_metadata();
        meta->set_steady_clock_reference_ns(m_impl->steadyClockRefNs);
        meta->set_cupti_reference_ns(m_impl->cuptiRefNs);
        meta->set_wall_clock_epoch_ns(m_impl->wallClockEpochNs);

        auto append = [&](EventTracker& t, TimeDomain d) {
            std::vector<internal::ResolvedRegion> regions;
            std::vector<internal::ResolvedEvent> events;
            t.Drain(/*forceResolve=*/true, regions, events);
            if (regions.empty() && events.empty()) return;
            auto* buf = trace.add_buffers();
            buf->set_domain(d);
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
        };
        append(*m_impl->generic, TIME_DOMAIN_GENERIC);
        append(*m_impl->gpu,     TIME_DOMAIN_GPU);

        // Attach the last in-flight flush stats too.
        if (m_impl->flushStatsPending.valid) {
            auto* fs = trace.add_flush_stats();
            fs->set_timestamp_ns(m_impl->flushStatsPending.timestampNs);
            fs->set_bytes_written(m_impl->flushStatsPending.bytesWritten);
            fs->set_interval_ns(m_impl->flushStatsPending.intervalNs);
            m_impl->flushStatsPending.valid = false;
        }

        if (trace.buffers_size() > 0 || trace.flush_stats_size() > 0) {
            std::lock_guard<std::mutex> lk(m_impl->outMutex);
            internal::WriteDelimitedEventTraceSized(trace, m_impl->outFile);
            m_impl->outFile.flush();
        }
        m_impl->outFile.close();
        std::cout << "[Events] Wrote trace to "
                  << m_impl->config.outputFile << "\n";
    }

    m_impl->running = false;
}

} // namespace cupti_profiler
