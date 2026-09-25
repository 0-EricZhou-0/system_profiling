// Internal: descendant tracking for the System + Disk probes.
//
// One ProcessDiscovery per observer — in the workload process under
// LEGACY, inside the sidecar under SIDECAR — with a thread that, every
// scan interval:
//
//   1. For EVERY process it follows (listed roots whose descendants are
//      tracked, and every process already discovered under them — not
//      just the tree currently reachable from a root), reads
//      /proc/<pid>/task/*/children. Every thread's file, because a child
//      hangs off whichever thread forked it. Recursive mode also scans
//      the discovered processes; direct-only mode scans the roots alone.
//   2. For each new PID: pidfd_open (raw syscall), then re-reads its
//      parent from /proc/<pid>/stat and requires that parent to be in
//      the followed set — the PID-reuse guard. The pidfd is checked
//      still-alive AFTER the read, so the /proc data describes the
//      process the pidfd pins. Accepted PIDs are registered on the
//      probes (AddDiscoveredProcess, which duplicates the pidfd) with
//      label = the root's alias, parent_pid and discovered=true; the
//      probes name them "<label>/<comm>" and follow comm from then on.
//      Each is logged: "[discovery] + <pid> <comm> (parent <ppid>
//      <parent comm>)".
//   3. Polls the held pidfds, to stop following exited processes and
//      reap adopted ones. Removing an exited process from the trace is
//      the probes' job: they hold their own pidfds (for roots too, with
//      discovery on or off) and mark it removed at their next tick.
//   4. Records the scan's own wall time (p50/p99/max) and publishes it
//      to the probes as DiscoveryStats.
//
// Guarantee: a process that is a child of a followed process for at
// least one full scan interval is discovered; once discovered it is
// tracked until it exits, regardless of reparenting. A child that lives
// less than one interval may be missed.
//
// The discovery thread only starts once some root actually tracks its
// descendants, so with the feature off it costs nothing.
//
// PID reuse within one scan interval (a discovered process exits and its
// number is reused before the next scan) cannot misattribute samples:
// the probes watch every entry through their own pidfd and never sample
// an exited one again. See docs/system-guide.md "PID reuse".
//
// Test-only: CUPTI_PROFILER_PROC_ROOT replaces "/proc" for the reads in
// steps 1-2 (children, stat, comm) and for the probes' process-table
// reads (stat and comm at registration, comm refresh), so the algorithm
// can be exercised on a synthetic tree. pidfds and the probes' samples
// still use the real kernel, so the PIDs in a synthetic tree must be
// real, live processes. Not a supported configuration knob.
#pragma once

#include <cupti_profiler/process_tracking_probe.h>   // CUPTI_PROFILER_API

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cupti_profiler {
namespace internal {

struct DiscoverySettings {
    bool     enabled    = false;  // default for roots added without an override
    bool     recursive  = true;   // false = direct children of roots only
    uint64_t intervalMs = 100;
};

// What happens when a discovered process that the host (the launcher)
// adopted as a subreaper exits. See child_subreaper.h.
struct AdoptionReaping {
    enum class Mode {
        None,       // nobody reaps (no host, or no channel to it)
        InProcess,  // this process is the host: reap it here (LEGACY)
        Notify,     // write an AdoptedExitNotice to noticeFd (SIDECAR)
    };
    Mode     mode     = Mode::None;
    uint32_t hostPid  = 0;
    int      noticeFd = -1;
};

// Fixed-size log-linear histogram of scan durations: 16 buckets per
// power of two, so a percentile is within 6.25% of the true value.
class ScanHistogram {
public:
    void     Record(uint64_t ns);
    uint64_t Percentile(double q) const;
    uint64_t Count() const { return count_; }
    uint64_t Max() const { return max_; }
private:
    static constexpr int kSub = 16;
    static int      Index(uint64_t v);
    static uint64_t UpperBound(int idx);
    std::array<uint64_t, 64 * kSub> buckets_{};
    uint64_t count_ = 0;
    uint64_t max_   = 0;
};

class CUPTI_PROFILER_API ProcessDiscovery {
public:
    enum Sink : uint8_t { kSystemSink = 1, kDiskSink = 2 };

    // Either probe may be null (disabled, or served by another observer).
    ProcessDiscovery(DiscoverySettings settings,
                     ProcessTrackingProbe* system,
                     ProcessTrackingProbe* disk,
                     AdoptionReaping reaping);
    ~ProcessDiscovery();

    ProcessDiscovery(const ProcessDiscovery&) = delete;
    ProcessDiscovery& operator=(const ProcessDiscovery&) = delete;

    /// Declare a listed root. trackDescendants = nullopt inherits
    /// settings.enabled; a bool overrides it for this root. `sinks`
    /// selects which probes its descendants are registered on. Adding
    /// the same PID again updates its settings (sinks are OR-ed).
    /// Thread-safe; applied at the next scan. The root itself is NOT
    /// registered on the probes here — the caller already tracks it.
    void AddRoot(uint32_t pid, const std::string& alias,
                 std::optional<bool> trackDescendants,
                 uint8_t sinks = kSystemSink | kDiskSink);

    /// Stop following a listed root. Its already-discovered
    /// descendants stay tracked until they exit. Thread-safe.
    void RemoveRoot(uint32_t pid);

    /// Begin scanning (the thread starts once any root tracks its
    /// descendants). Stop() joins it and publishes final stats.
    void Start();
    void Stop();

private:
    struct Entry {
        int         pidfd       = -1;
        bool        isRoot      = false;
        bool        scan        = false;  // read this process's children
        bool        recursive   = false;  // its discovered children scan too
        uint8_t     sinks       = 0;
        std::string label;                // alias prefix: root alias or PID
        uint32_t    firstParent = 0;      // parent when discovered
        uint64_t    startTime   = 0;      // /proc/<pid>/stat field 22
        std::string comm;
    };
    struct Op {
        bool                remove = false;
        uint32_t            pid    = 0;
        std::string         alias;
        std::optional<bool> descend;
        uint8_t             sinks  = 0;
    };

    void Run();
    void ScanOnce(std::vector<Op>& ops);
    void ApplyOp(const Op& op);
    void HandleExits();
    void HandleExit(uint32_t pid, Entry& e);
    void MaybeReapAdopted(uint32_t pid, const Entry& e);
    void Scan();
    bool TryRegister(uint32_t child, uint32_t listedUnder);
    std::vector<uint32_t> ReadChildren(uint32_t pid) const;
    void PublishStats();
    template <class F> void ForEachSink(uint8_t mask, F&& f);

    const DiscoverySettings settings_;
    ProcessTrackingProbe* const system_;
    ProcessTrackingProbe* const disk_;
    const AdoptionReaping reaping_;
    const std::string procRoot_;
    const uint32_t selfPid_;

    // Scan-thread state (touched only by the scan thread once running).
    std::unordered_map<uint32_t, Entry> table_;
    ScanHistogram hist_;
    uint64_t discovered_ = 0;
    uint64_t exited_     = 0;
    uint64_t rejected_   = 0;

    // Shared with API callers.
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<Op>         ops_;
    bool                    started_       = false;
    bool                    stop_          = false;
    bool                    anyDescending_ = false;
    std::thread             thread_;
};

} // namespace internal
} // namespace cupti_profiler
