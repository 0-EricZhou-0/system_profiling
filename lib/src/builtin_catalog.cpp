#include <cupti_profiler/builtin_catalog.h>

#include "metric_catalog.h"
#include "metric_catalog_builtins.h"

#include <google/protobuf/text_format.h>

namespace cupti_profiler {

std::string BuiltinMetricCatalogPbtxt() {
    internal::MetricCatalog cat;
    internal::RegisterBuiltinDescriptors(cat);

    std::string out;
    google::protobuf::TextFormat::Printer printer;
    printer.SetUseShortRepeatedPrimitives(true);
    printer.PrintToString(cat.Proto(), &out);
    return out;
}

} // namespace cupti_profiler
