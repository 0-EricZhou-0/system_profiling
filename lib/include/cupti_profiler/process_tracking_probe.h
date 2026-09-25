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
//
// Descendant tracking (lib/src/process_discovery.h) registers what it
// finds through AddDiscoveredProcess(), which records the parent PID
// and marks the entry discovered, and publishes its own scan cost
// through SetDiscoveryStats() for the flush thread to emit.

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
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

// Descendant tracking's cumulative self-metrics (see the DiscoveryStats
// proto in proto/metric_sample.proto for field meanings).
struct DiscoveryStats {
    uint64_t scanIntervalNs = 0;
    uint64_t scans          = 0;
    uint64_t scanP50Ns      = 0;
    uint64_t scanP99Ns      = 0;
    uint64_t scanMaxNs      = 0;
    uint64_t discovered     = 0;
    uint64_t exited         = 0;
    uint64_t rejected       = 0;
};

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

    /// Register a process found by descendant tracking: like
    /// AddTrackedProcess, plus the parent it had when found and
    /// discovered=true. If the PID is already tracked as a listed
    /// root, the root entry is left as it is; if it is tracked as a
    /// discovered process, only its alias is refreshed. Thread-safe.
    void AddDiscoveredProcess(uint32_t pid, std::string alias, uint32_t parentPid);

    struct ProcessEntry {
        uint32_t    pid              = 0;
        std::string alias;
        bool        pending_removal  = false;
        uint32_t    parent_pid       = 0;      // discovered only
        bool        discovered       = false;
        // System probe, discovered only: process CPU clock (ns) at its
        // first sample — the CPU it used before it was found.
        uint64_t    cpu_before_discovery_ns = 0;
    };

    /// Record a discovered process's CPU before its first sample
    /// (ProcessEntry::cpu_before_discovery_ns). No-op if it is no longer
    /// tracked. Thread-safe.
    void SetCpuBeforeDiscovery(uint32_t pid, uint64_t ns);

    /// Replace the tracked process set in one shot. Called by derived
    /// classes from Configure() to seed config.processes.
    void SetInitialProcesses(std::vector<ProcessEntry> entries);

    /// Snapshot copy under shared_lock. Called from the sample loop
    /// (every tick) AND from the flush thread (every flush); the cost
    /// is one O(N) copy.
    std::vector<ProcessEntry> SnapshotProcesses() const;

    /// Drop the entries that `emitted` — the snapshot a flush just
    /// wrote — carried with `pending_removal=true`. Called by the flush
    /// thread after a successful flush. An entry marked for removal
    /// after that snapshot was taken is kept, so its removed=true marker
    /// goes out in the next flush instead of being lost.
    void CommitPendingRemovals(const std::vector<ProcessEntry>& emitted);

    /// Latest descendant-tracking self-metrics, emitted with every
    /// flush. nullopt until discovery has published once.
    void SetDiscoveryStats(const DiscoveryStats& stats);
    std::optional<DiscoveryStats> SnapshotDiscoveryStats() const;

private:
    mutable std::shared_mutex     mutex_;
    std::vector<ProcessEntry>     processes_;
    std::optional<DiscoveryStats> discoveryStats_;
};

} // namespace cupti_profiler
