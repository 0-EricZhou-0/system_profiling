// Internal: one ProcessTrackingProbe entry -> one TrackedProcessV2 row of
// the trace's process table. Shared by the system and disk traces.
#pragma once

#include <cupti_profiler/process_tracking_probe.h>

#include "metric_sample.pb.h"

#include <vector>

namespace cupti_profiler {
namespace internal {

inline void FillTrackedProcess(TrackedProcessV2* tp,
                               const ProcessTrackingProbe::ProcessEntry& p) {
    tp->set_pid(p.pid);
    tp->set_alias(p.alias);
    tp->set_removed(p.pending_removal);
    tp->set_parent_pid(p.parent_pid);
    tp->set_discovered(p.discovered);
    tp->set_label(p.label);
    tp->set_comm(p.comm);
    tp->set_start_time_ns(p.start_time_ns);
    tp->set_end_time_ns(p.end_time_ns);
    for (const auto& c : p.comm_history) {
        auto* h = tp->add_comm_history();
        h->set_timestamp_ns(c.timestamp_ns);
        h->set_comm(c.comm);
    }
    tp->set_io_unreadable_since_ns(p.io_unreadable_since_ns);
    tp->set_io_unreadable_ticks(p.io_unreadable_ticks);
    tp->set_mem_unreadable_since_ns(p.mem_unreadable_since_ns);
    tp->set_mem_unreadable_ticks(p.mem_unreadable_ticks);
}

// Does this snapshot carry a removal marker (an entry with
// pending_removal)? A flush goes out for it even with no samples: a probe
// whose only process exited has none, and the marker would otherwise wait
// for the final flush at Stop().
inline bool HasRemovalMarker(const std::vector<ProcessTrackingProbe::ProcessEntry>& snapshot) {
    for (const auto& p : snapshot) if (p.pending_removal) return true;
    return false;
}

// Does a row say that a per-PID file of its process could not be read
// (io_ / mem_unreadable_*)? A flush goes out for it even with no
// samples: a process whose I/O is never readable has none, and without
// a flush the trace would not say why its I/O is missing.
inline bool HasUnreadableRecord(const std::vector<ProcessTrackingProbe::ProcessEntry>& snapshot) {
    for (const auto& p : snapshot)
        if (p.io_unreadable_ticks || p.mem_unreadable_ticks) return true;
    return false;
}

} // namespace internal
} // namespace cupti_profiler
