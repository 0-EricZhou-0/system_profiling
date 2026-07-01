// dump_metric_catalog — prints the built-in non-GPU MetricCatalog as
// text-format pbtxt to stdout. Use to regenerate the human-readable
// reference copy at lib/data/metric_catalog.pbtxt (the registry in
// the .cpp files is authoritative; this tool just materialises it as
// pbtxt for grep-ability / diff-ability) or as a starting template
// for --metric-catalog override files.
//
//   ./build/tools/dump_metric_catalog > lib/data/metric_catalog.pbtxt
#include <cupti_profiler/builtin_catalog.h>
#include <iostream>

int main() {
    std::cout
        << "# AUTO-GENERATED — do not edit by hand.\n"
        << "# Regenerate: ./build/tools/dump_metric_catalog > lib/data/metric_catalog.pbtxt\n"
        << "# Source of truth: kSystemMetrics / kProcessMetrics in\n"
        << "#   lib/src/system_flush_thread.cpp and lib/src/disk_flush_thread.cpp.\n"
        << "# This file is a human-readable reference only — the runtime seeds\n"
        << "# its in-memory MetricCatalog from RegisterBuiltinDescriptors() in\n"
        << "# lib/src/metric_catalog_builtins.cpp, not from this pbtxt.\n"
        << "#\n"
        << "# GPU descriptors are NOT included here — they are chip-specific and\n"
        << "# discovered at runtime via cuptiProfilerHostGetSubMetrics().\n"
        << "\n"
        << cupti_profiler::BuiltinMetricCatalogPbtxt();
    return 0;
}
