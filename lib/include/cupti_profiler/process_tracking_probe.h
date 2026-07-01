// Common base class for probes that sample per-PID quantities
// (currently SystemProfiler and DiskProfiler).
//
// Owns a thread-safe `tracked_processes_` vector and the two-stage
// removal protocol that lets the visualizer see exactly when a PID
// stops being tracked:
//
//   * Add(pid, alias)  — appends a new entry. The sample loop picks
//                        it up on the next iteration; the first
//                        emitted sample for the PID is one tick later
//                        (the first iteration only seeds /proc
//                        baselines so the second delta isn't garbage).
//
//   * Remove(pid)      — flips `pending_removal=true` on the entry.
//                        The entry stays in tracked_processes_ until
//                        CommitPendingRemovals() is called. This lets
//                        the writer emit one more flush that includes
//                        the PID with TrackedProcessV2.removed=true so
//                        the visualizer renders a removal marker, and
//                        the PID then disappears from subsequent
//                        flushes.
//
// Derived classes call SnapshotProcesses() from their sample loop and
// CommitPendingRemovals() from their flush thread after a successful
// flush.

#pragma once

#include <cstdint>
#include <mutex>
#include <shared_mutex>
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

class CUPTI_PROFILER_API ProcessTrackingProbe {
public:
    ProcessTrackingProbe() = default;
    virtual ~ProcessTrackingProbe() = default;

    // Non-copyable and non-movable: holds a shared_mutex. Derived
    // probes are owned by ProfilerSuite::Impl in place and never moved.
    ProcessTrackingProbe(const ProcessTrackingProbe&) = delete;
    ProcessTrackingProbe& operator=(const ProcessTrackingProbe&) = delete;
    ProcessTrackingProbe(ProcessTrackingProbe&&) = delete;
    ProcessTrackingProbe& operator=(ProcessTrackingProbe&&) = delete;

    /// Append a new tracked PID. Thread-safe; takes effect on the
    /// next sample tick of the derived probe.
    void AddTrackedProcess(uint32_t pid, std::string alias);

    /// Mark a tracked PID for removal. The PID remains in the next
    /// emitted flush (with `removed=true`) and is dropped after
    /// CommitPendingRemovals() is called. Thread-safe.
    void RemoveTrackedProcess(uint32_t pid);

    struct ProcessEntry {
        uint32_t    pid              = 0;
        std::string alias;
        bool        pending_removal  = false;
    };

    /// Replace the tracked process set in one shot. Called by derived
    /// classes from Configure() to seed config.processes.
    void SetInitialProcesses(std::vector<ProcessEntry> entries);

    /// Snapshot copy under shared_lock. Called from the sample loop
    /// (every tick) AND from the flush thread (every flush); the cost
    /// is one O(N) copy.
    std::vector<ProcessEntry> SnapshotProcesses() const;

    /// Drop every entry currently marked `pending_removal`. Called
    /// by the flush thread after emitting a successful flush whose
    /// trace.tracked_processes carried those entries with
    /// `removed=true`.
    void CommitPendingRemovals();

private:
    mutable std::shared_mutex   mutex_;
    std::vector<ProcessEntry>   processes_;
};

} // namespace cupti_profiler
