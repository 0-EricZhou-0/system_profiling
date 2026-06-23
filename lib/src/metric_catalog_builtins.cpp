#include "metric_catalog_builtins.h"

#include "disk_flush_thread.h"
#include "metric_catalog.h"
#include "metric_catalog.pb.h"
#include "metric_descriptor.h"
#include "system_flush_thread.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace cupti_profiler {
namespace internal {
namespace {

// Pour one typed MetricDescriptor<Tick> into a freshly-emplaced proto
// ::MetricDescriptor. Enum values were chosen to match numerically so
// the static_cast is correctness-preserving (see metric_descriptor.h).
template <typename Tick>
void ToProto(const MetricDescriptor<Tick>& d, ::MetricDescriptor& out) {
    out.set_fqn      (std::string(d.fqn));
    out.set_type     (static_cast<::MetricType>(d.type));
    out.set_entity   (std::string(d.entity));
    out.set_counter  (std::string(d.counter));
    if (!d.rollup   .empty()) out.set_rollup   (std::string(d.rollup));
    if (!d.submetric.empty()) out.set_submetric(std::string(d.submetric));
    out.set_unit     (static_cast<::Unit> (d.unit));
    out.set_scope    (static_cast<::Scope>(d.scope));
    out.set_smoothable(d.smoothable);
    if (!d.description.empty()) out.set_description(std::string(d.description));

    std::visit([&](const auto& p) {
        using T = std::decay_t<decltype(p)>;
        if      constexpr (std::is_same_v<T, PeakConstant>) out.set_peak_constant(p.value);
        else if constexpr (std::is_same_v<T, PeakRef>)      out.set_peak_ref     (std::string(p.fqn));
        else if constexpr (std::is_same_v<T, PeakExpr>)     out.set_peak_expr    (std::string(p.name));
        // std::monostate -> leave the peak oneof unset (panel auto-scales).
    }, d.peak);
}

template <typename Tick>
std::vector<::MetricDescriptor> ToProtos(
        std::span<const MetricDescriptor<Tick>> arr)
{
    std::vector<::MetricDescriptor> out;
    out.reserve(arr.size());
    for (const auto& d : arr) {
        if (d.read == nullptr) {
            std::cerr << "MetricCatalog: descriptor for '" << d.fqn
                      << "' has a null .read extractor — would crash at "
                         "emit time. Fix the literal in the probe's TU.\n";
            std::exit(1);
        }
        ToProto(d, out.emplace_back());
    }
    return out;
}

} // namespace

void RegisterBuiltinDescriptors(MetricCatalog& cat) {
    cat.AppendDescriptors(ToProtos(GetSystemMetrics()));
    cat.AppendDescriptors(ToProtos(GetProcessMetrics()));
    cat.AppendDescriptors(ToProtos(GetDiskDeviceMetrics()));
    cat.AppendDescriptors(ToProtos(GetDiskProcessMetrics()));
}

} // namespace internal
} // namespace cupti_profiler
