// Common base class for probes that sample per-PID quantities
// (currently SystemProfiler and DiskProfiler).
//
// Owns a thread-safe tracked-process list — the trace's process table —
// and the two-stage removal protocol that lets the visualizer see
// exactly when a PID stops being tracked:
//
//   * Add(pid, alias)  — registers a new entry. The sample loop picks
//                        it up on the next iteration; the first
//                        emitted sample for the PID is one tick later
//                        (the first iteration only seeds /proc
//                        baselines so the second delta isn't garbage).
//
//   * Remove(pid)      — flips `pending_removal=true` on the entry.
//                        The entry stays in the list until
//                        CommitPendingRemovals() is called. This lets
//                        the writer emit one more flush that includes
//                        the PID with TrackedProcessV2.removed=true so
//                        the visualizer renders a removal marker, and
//                        the PID then disappears from subsequent
//                        flushes.
//
// Exit detection. Every registration — listed root or discovered, with
// descendant tracking on or off — holds a pidfd on the process it
// names. PollTracked(), called by the derived probe once per sample
// tick, polls them all; a process that has exited is removed exactly as
// by Remove(), with its end time recorded. Its entry is never sampled
// again: a later process that gets the same PID number is a different
// process and needs a new registration. The pidfd is opened at
// registration and checked still-alive AFTER the start time is read
// from /proc/<pid>/stat, so the recorded start time (and, for
// discovered processes, discovery's parent check) describes the process
// the pidfd pins, not an earlier owner of the number.
//
// Process table. Each entry also records the process's comm (followed
// across renames: PollTracked re-reads it every kCommRefreshNs), its
// start time (kernel, 10 ms resolution) and its end time (observed,
// within one sampling tick), so a reader can rebuild the tree and name
// every process. See TrackedProcessV2 in proto/metric_sample.proto.
//
// Derived classes, once per sample tick: SnapshotProcesses(), read each
// live entry's values by PID, then PollTracked() and discard the
// readings of the entries it reports exited (read-then-verify, as for
// registration). CommitPendingRemovals() runs from their flush thread
// after a successful flush.
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
#include <unordered_map>
#include <utility>
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

// The five /proc/<pid>/io counters of one reading, in bytes.
struct IoCounters {
    uint64_t rchar                 = 0;
    uint64_t wchar                 = 0;
    uint64_t read_bytes            = 0;
    uint64_t write_bytes           = 0;
    uint64_t cancelled_write_bytes = 0;
};

// One comm a tracked process had, and when the probe first saw it
// (trace clock, ns). The first entry is the comm at registration.
struct CommChange {
    uint64_t    timestamp_ns = 0;
    std::string comm;
};

class CUPTI_PROFILER_API ProcessTrackingProbe {
public:
    ProcessTrackingProbe() = default;
    virtual ~ProcessTrackingProbe();

    // Non-copyable and non-movable: holds a shared_mutex. Derived
    // probes are owned by ProfilerSuite::Impl in place and never moved.
    ProcessTrackingProbe(const ProcessTrackingProbe&) = delete;
    ProcessTrackingProbe& operator=(const ProcessTrackingProbe&) = delete;
    ProcessTrackingProbe(ProcessTrackingProbe&&) = delete;
    ProcessTrackingProbe& operator=(ProcessTrackingProbe&&) = delete;

    /// How often PollTracked() re-reads each process's comm.
    static constexpr uint64_t kCommRefreshNs = 100'000'000;   // 100 ms
    /// At most this many CommChange records are kept per process: the
    /// first, and the most recent ones.
    static constexpr size_t kMaxCommHistory = 16;

    /// Register a listed root. Thread-safe; takes effect on the next
    /// sample tick of the derived probe. If the PID is already tracked
    /// and alive, its entry is kept (a pending removal is cancelled, a
    /// non-empty alias replaces the old one, and a discovered entry
    /// becomes a root). If its entry is for a process that has exited,
    /// this names a new process: it is registered once the old entry's
    /// removal has been flushed. A PID that does not exist is recorded
    /// as already exited (removed=true in the next flush).
    void AddTrackedProcess(uint32_t pid, std::string alias);

    /// Mark a tracked PID for removal. The PID remains in the next
    /// emitted flush (with `removed=true`) and is dropped after
    /// CommitPendingRemovals() is called. Thread-safe.
    void RemoveTrackedProcess(uint32_t pid);

    /// Register a process found by descendant tracking, with the parent
    /// it had when found and discovered=true; its alias is
    /// "<label>/<comm>" and follows comm. `pidfd` is discovery's pidfd
    /// for it (duplicated here, so the entry pins the very process
    /// discovery checked); `startTimeTicks` is /proc/<pid>/stat field
    /// 22. If the PID is already tracked for a live process, nothing
    /// changes; if its entry is for a process that has exited, this one
    /// is registered once that entry's removal has been flushed.
    /// Thread-safe.
    void AddDiscoveredProcess(uint32_t pid, const std::string& label,
                              const std::string& comm, uint32_t parentPid,
                              int pidfd, uint64_t startTimeTicks);

    struct ProcessEntry {
        uint32_t    pid              = 0;
        std::string alias;
        bool        pending_removal  = false;
        // The parent the process had when it was registered (discovered
        // processes: the tracked parent it was found under).
        uint32_t    parent_pid       = 0;
        bool        discovered       = false;
        // System probe: process CPU clock (ns) at its first reading, the
        // tick after registration — the CPU it used before tracking began
        // (for a root started long before, its whole CPU up to attach).
        uint64_t    cpu_before_tracking_ns = 0;
        // Disk probe: the /proc/<pid>/io counters at its first reading —
        // the I/O it did before tracking began. nullopt until read.
        std::optional<IoCounters> io_before_tracking;
        // Identity of this registration within the probe; never reused.
        // Per-PID baselines are keyed by it, so a new process that got
        // an old one's number never inherits its baseline.
        uint64_t    serial           = 0;
        // Roots: the alias they were listed with. Discovered: their
        // root's alias (alias = label + "/" + comm).
        std::string label;
        std::string comm;                 // current /proc/<pid>/comm
        // Process start (kernel, /proc/<pid>/stat field 22) on the trace
        // clock (steady_clock, ns). USER_HZ resolution: 10 ms. 0 = unknown.
        uint64_t    start_time_ns    = 0;
        // Set with pending_removal when the process exited: the first
        // instant this probe saw it gone (trace clock, ns). It exited
        // within one sampling tick before. 0 = alive, or removed by
        // request while alive.
        uint64_t    end_time_ns      = 0;
        std::vector<CommChange> comm_history;
    };

    /// Record a process's CPU before its first sample
    /// (ProcessEntry::cpu_before_tracking_ns). No-op if it is no longer
    /// tracked. Thread-safe.
    void SetCpuBeforeTracking(uint32_t pid, uint64_t ns);

    /// Record a process's I/O before its first sample
    /// (ProcessEntry::io_before_tracking). No-op if it is no longer
    /// tracked. Thread-safe.
    void SetIoBeforeTracking(uint32_t pid, const IoCounters& io);

    /// Replace the tracked process set in one shot with these roots
    /// (pid and alias are used; each is registered as by
    /// AddTrackedProcess). Called by derived classes from Configure() to
    /// seed config.processes.
    void SetInitialProcesses(std::vector<ProcessEntry> entries);

    /// Once per sample tick, AFTER reading the tick's per-PID values:
    /// poll every pidfd and mark the processes that exited as removed
    /// (with their end time); every kCommRefreshNs also re-read each
    /// live process's comm. Returns the serials of the entries it just
    /// marked: their readings from this tick must be discarded, since
    /// the number may already name another process. A reading of an
    /// entry not returned here was taken while the process it pins was
    /// alive (or a zombie), so it is that process's. Thread-safe.
    std::vector<uint64_t> PollTracked();

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
    std::pair<ProcessEntry, int> MakeRoot(uint32_t pid, std::string alias);
    void InsertLocked(ProcessEntry e, int pidfd);
    void CloseLocked(uint64_t serial);
    bool GoneLocked(ProcessEntry& e);

    // A registration waiting for an exited entry with the same PID to be
    // flushed and dropped. Roots: pidfd = -1 (opened when applied).
    struct Deferred {
        ProcessEntry entry;
        int          pidfd = -1;
    };

    mutable std::shared_mutex     mutex_;
    std::vector<ProcessEntry>     processes_;
    std::unordered_map<uint64_t, int> pidfds_;         // serial -> pidfd
    std::vector<Deferred>         deferred_;
    uint64_t                      nextSerial_   = 1;
    uint64_t                      lastCommNs_   = 0;
    std::optional<DiscoveryStats> discoveryStats_;
};

} // namespace cupti_profiler
