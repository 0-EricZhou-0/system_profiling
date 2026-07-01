// Public dump of the built-in MetricCatalog.
//
// The runtime's authoritative metadata source for non-GPU metrics is
// the typed MetricDescriptor<Tick> arrays declared inside each probe
// TU (system_flush_thread.cpp, disk_flush_thread.cpp). This header
// exposes a single function that materialises those arrays into
// text-format MetricCatalog pbtxt — useful for regenerating the
// human-readable reference at lib/data/metric_catalog.pbtxt, or as a
// starting template that downstream forks can edit and pass via
// ProfilerSuiteConfig.metric_catalog_path to overlay custom
// descriptions / peaks / smoothable flags onto the built-ins.
//
// GPU descriptors are not part of the dump — they're chip-specific and
// discovered at runtime from cuptiProfilerHostGetSubMetrics(), so they
// only exist after a live ProfilerSuite has configured a GPU probe.
#pragma once

#include <string>

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

/// Returns the built-in non-GPU MetricCatalog as text-format pbtxt.
/// Output is suitable for piping to `lib/data/metric_catalog.pbtxt`
/// (the human-readable reference copy regenerated from the registry).
CUPTI_PROFILER_API std::string BuiltinMetricCatalogPbtxt();

} // namespace cupti_profiler
