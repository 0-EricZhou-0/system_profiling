// Internal: copy a probe's descendant-tracking self-metrics into the
// DiscoveryStats field of a SystemMetricsTrace / DiskMetricsTrace.
#pragma once

#include <cupti_profiler/process_tracking_probe.h>

#include "metric_sample.pb.h"

namespace cupti_profiler {
namespace internal {

template <class Trace>
void AttachDiscoveryStats(Trace& trace, const ProcessTrackingProbe& probe) {
    auto s = probe.SnapshotDiscoveryStats();
    if (!s) return;
    auto* d = trace.mutable_discovery_stats();
    d->set_scan_interval_ns(s->scanIntervalNs);
    d->set_scans(s->scans);
    d->set_scan_p50_ns(s->scanP50Ns);
    d->set_scan_p99_ns(s->scanP99Ns);
    d->set_scan_max_ns(s->scanMaxNs);
    d->set_discovered(s->discovered);
    d->set_exited(s->exited);
    d->set_rejected(s->rejected);
}

} // namespace internal
} // namespace cupti_profiler
