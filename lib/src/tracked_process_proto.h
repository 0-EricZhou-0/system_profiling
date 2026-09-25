// Internal: one ProcessTrackingProbe entry -> one TrackedProcessV2 row of
// the trace's process table. Shared by the system and disk traces.
#pragma once

#include <cupti_profiler/process_tracking_probe.h>

#include "metric_sample.pb.h"

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
}

} // namespace internal
} // namespace cupti_profiler
