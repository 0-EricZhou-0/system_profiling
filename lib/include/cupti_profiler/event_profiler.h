// EventProfiler — multi-domain region + event tracking with periodic flush.
// Records named time intervals (Regions) and instantaneous markers (Events)
// in either the host steady_clock domain (Generic) or the CUPTI clock
// domain (GPU), and writes them length-delimited to events.pb.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __declspec(dllexport)
  #else
    #define CUPTI_PROFILER_API __declspec(dllimport)
  #endif
#else
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __attribute__((visibility("default")))
  #else
    #define CUPTI_PROFILER_API
  #endif
#endif

namespace cupti_profiler {

namespace internal {
struct ResolvedRegion;
struct ResolvedEvent;
} // namespace internal

struct CUPTI_PROFILER_API EventProfilerConfig {
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;            // e.g. "events.pb"
};

/// Records regions and events in a single TimeDomain.
/// Constructed by EventProfiler — not directly by users.
///
/// Thread safety:
///   BeginRegion / EndRegion / MarkEvent / Drain are all safe to call
///   concurrently from any number of threads in the same process. The
///   tracker owns an internal mutex; lock-hold time is bounded by a single
///   map insert / map lookup. cudaEvent creation and steady_clock reads
///   happen outside the lock to keep contention minimal under load.
///
///   The opaque region id returned by BeginRegion is stable across
///   concurrent operations and may be passed between threads — e.g.
///   Begin on thread A, End on thread B is fully supported.
///
/// Multi-process:
///   This class is in-process only — the mutex protects in-process state.
///   For multi-process logging, give each process its own EventProfiler
///   writing to its own `events.pb` (e.g. `events_<pid>.pb`); the
///   visualizer can merge multiple files at view time.
class CUPTI_PROFILER_API EventTracker {
public:
    enum class Domain { GENERIC, GPU };

    EventTracker(EventTracker&&) noexcept;
    EventTracker& operator=(EventTracker&&) noexcept;
    ~EventTracker();

    Domain GetDomain() const;

    /// Mark the beginning of a named region. Returns an opaque id that
    /// can be passed to EndRegion later (from any thread).
    size_t BeginRegion(const std::string& name);

    /// Mark the end of a region previously started with BeginRegion().
    /// Safe to call from a thread other than the one that called Begin.
    void EndRegion(size_t idx);

    /// Record an instantaneous event.
    void MarkEvent(const std::string& name);

    /// GPU-domain only — register the CUDA stream that subsequent
    /// Begin/End/Mark calls will record cudaEvents on.
    /// Must be called once before any Begin/End/Mark; not thread-safe
    /// vs. those (call from setup code on a single thread).
    void SetStream(void* stream);

    /// Move all completed regions/events into the caller-supplied buffers.
    /// In-flight entries remain in the tracker for a subsequent drain. With
    /// `forceResolve = true`, blocks on outstanding cudaEvents so the final
    /// in-flight GPU regions/events are captured.
    void Drain(bool forceResolve,
               std::vector<internal::ResolvedRegion>& outRegions,
               std::vector<internal::ResolvedEvent>& outEvents);

private:
    friend class EventProfiler;
    explicit EventTracker(Domain);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

class CUPTI_PROFILER_API EventProfiler {
public:
    EventProfiler();
    ~EventProfiler();
    EventProfiler(EventProfiler&&) noexcept;
    EventProfiler& operator=(EventProfiler&&) noexcept;

    void Configure(const EventProfilerConfig&);
    void Start();
    void SignalStop();
    void Stop();

    EventTracker& GetGenericTracker();
    EventTracker& GetGpuTracker();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
