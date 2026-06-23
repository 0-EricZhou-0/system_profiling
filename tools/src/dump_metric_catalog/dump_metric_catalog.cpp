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
    std::cout << cupti_profiler::BuiltinMetricCatalogPbtxt();
    return 0;
}
