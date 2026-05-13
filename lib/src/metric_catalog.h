// Internal: in-memory wrapper around the loaded MetricCatalog pbtxt.
//
// Provides FQN-keyed lookup and scope-bucketed enumeration over the
// MetricDescriptor entries declared in configs/metric_catalog.pbtxt.
// The probes consult ByScope(SCOPE_X) at Configure() time to fix the
// per-scope FQN ordering (which is what gets emitted into
// ScopeMetricNames on every flush, and which determines the column
// order of samples' `values[]` arrays).
//
// GPU descriptors are appended at runtime via AppendGpuMetrics() —
// the GPU FQN catalog is chip-specific and discovered by walking
// cuptiProfilerHostGetSubMetrics(), so it can't live in the static
// pbtxt.
#pragma once

#include "metric_catalog.pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace cupti_profiler {
namespace internal {

class MetricCatalog {
public:
    /// Parse a MetricCatalog text-format pbtxt from `path`. Exits the
    /// process with an error message if the file is missing or
    /// malformed — the catalog is a hard dependency of the profiler.
    static MetricCatalog LoadFromPbtxt(const std::string& path);

    /// Look up a single descriptor by FQN. Returns nullptr if not found.
    const MetricDescriptor* Find(const std::string& fqn) const;

    /// Every descriptor whose scope == `scope`, in declared order.
    /// Stable across calls — probes rely on the order to fix their
    /// per-scope ScopeMetricNames registry.
    std::vector<const MetricDescriptor*> ByScope(Scope scope) const;

    /// Append descriptors for GPU FQNs that were discovered at
    /// Configure() by walking cuptiProfilerHostGetSubMetrics(). The
    /// descriptors are owned by this catalog after the call.
    void AppendGpuMetrics(const std::vector<MetricDescriptor>& descriptors);

    /// Read-only access to the underlying proto for inlining into
    /// SessionMetadata.catalog.
    const ::MetricCatalog& Proto() const { return proto_; }

private:
    ::MetricCatalog                                proto_;
    std::unordered_map<std::string, int>           index_by_fqn_;  // fqn -> proto_.metrics[index]

    void RebuildIndex();
};

} // namespace internal
} // namespace cupti_profiler
