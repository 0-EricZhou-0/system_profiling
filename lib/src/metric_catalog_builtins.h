// Internal: bridge between the typed per-probe MetricDescriptor<Tick>
// arrays (system_flush_thread.cpp, disk_flush_thread.cpp) and the
// proto-backed MetricCatalog the visualizer consumes.
//
// Called once from ProfilerSuite::Configure() to seed the catalog with
// every non-GPU FQN the runtime emits. GPU descriptors are appended
// separately by GpuProfiler after walking cuptiProfilerHostGetSubMetrics
// (chip-specific, so not part of a static literal).
#pragma once

namespace cupti_profiler {
namespace internal {

class MetricCatalog;

void RegisterBuiltinDescriptors(MetricCatalog& cat);

} // namespace internal
} // namespace cupti_profiler
