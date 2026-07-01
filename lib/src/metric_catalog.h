// Internal: in-memory MetricCatalog assembled at ProfilerSuite::Configure().
//
// Sources, in the order they're applied:
//   1. RegisterBuiltinDescriptors() walks each probe's
//      MetricDescriptor<Tick> array (system, process, disk-device,
//      disk-process — see metric_catalog_builtins.cpp) and appends one
//      proto descriptor per emitted column. This is the authoritative
//      source for every non-GPU FQN; the descriptor sits in the same
//      TU as the value extractor, so wire FQN, wire column, and
//      catalog entry move together.
//   2. (optional) MergeOverridesFromPbtxt(path) overlays a user-
//      supplied pbtxt — descriptors with FQNs already present are
//      REPLACED; new FQNs are appended. Lets downstream forks tweak
//      a description / peak / smoothable without rebuilding.
//   3. AppendDescriptors(...) is called by the GPU probe with
//      descriptors discovered from cuptiProfilerHostGetSubMetrics()
//      (chip-specific, so can't be a static literal).
//
// The probes consult ByScope(SCOPE_X) at Configure() time to fix the
// per-scope FQN ordering that ScopeMetricNames will emit; the
// visualizer reads `values[]` positionally against that registry.
#pragma once

#include "metric_catalog.pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace cupti_profiler {
namespace internal {

class MetricCatalog {
public:
    /// Look up a single descriptor by FQN. Returns nullptr if not found.
    /// (`::MetricDescriptor` is the proto, distinct from the typed
    /// internal::MetricDescriptor<Tick> the probe TUs declare.)
    const ::MetricDescriptor* Find(const std::string& fqn) const;

    /// Every descriptor whose scope == `scope`, in declared order.
    /// Stable across calls — probes rely on the order to fix their
    /// per-scope ScopeMetricNames registry.
    std::vector<const ::MetricDescriptor*> ByScope(::Scope scope) const;

    /// Append descriptors to the catalog. Used by both
    /// metric_catalog_builtins (seeding non-GPU FQNs from the probe
    /// arrays) and the GPU probe (descriptors discovered from
    /// cuptiProfilerHostGetSubMetrics()). Descriptors are owned by
    /// this catalog after the call.
    void AppendDescriptors(const std::vector<::MetricDescriptor>& descriptors);

    /// Parse a MetricCatalog text-format pbtxt from `path` and overlay
    /// it onto this catalog: for each descriptor, if the FQN already
    /// exists it is REPLACED in place; otherwise the descriptor is
    /// appended. Exits the process on parse failure (the override
    /// path is opt-in; if a user pointed at a file, malformed content
    /// is a hard error worth surfacing).
    void MergeOverridesFromPbtxt(const std::string& path);

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
