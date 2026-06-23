#include "metric_catalog.h"

#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace cupti_profiler {
namespace internal {

const ::MetricDescriptor* MetricCatalog::Find(const std::string& fqn) const {
    auto it = index_by_fqn_.find(fqn);
    if (it == index_by_fqn_.end()) return nullptr;
    return &proto_.metrics(it->second);
}

std::vector<const ::MetricDescriptor*> MetricCatalog::ByScope(::Scope scope) const {
    std::vector<const ::MetricDescriptor*> out;
    out.reserve(proto_.metrics_size());
    for (int i = 0; i < proto_.metrics_size(); ++i) {
        const auto& m = proto_.metrics(i);
        if (m.scope() == scope) out.push_back(&m);
    }
    return out;
}

void MetricCatalog::AppendDescriptors(
        const std::vector<::MetricDescriptor>& descriptors)
{
    for (const auto& d : descriptors) {
        *proto_.add_metrics() = d;
    }
    RebuildIndex();
}

void MetricCatalog::MergeOverridesFromPbtxt(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "MetricCatalog: failed to open override pbtxt " << path << "\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();

    ::MetricCatalog overlay;
    if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &overlay)) {
        std::cerr << "MetricCatalog: failed to parse override pbtxt " << path
                  << " — expected text-format MetricCatalog\n";
        std::exit(1);
    }

    int replaced = 0, added = 0;
    for (int i = 0; i < overlay.metrics_size(); ++i) {
        const auto& d = overlay.metrics(i);
        auto it = index_by_fqn_.find(d.fqn());
        if (it != index_by_fqn_.end()) {
            *proto_.mutable_metrics(it->second) = d;
            ++replaced;
        } else {
            *proto_.add_metrics() = d;
            ++added;
        }
    }
    RebuildIndex();
    std::cout << "MetricCatalog: merged override pbtxt " << path
              << " (" << replaced << " replaced, " << added << " added)\n";
}

void MetricCatalog::RebuildIndex() {
    index_by_fqn_.clear();
    index_by_fqn_.reserve(proto_.metrics_size());
    for (int i = 0; i < proto_.metrics_size(); ++i) {
        index_by_fqn_[proto_.metrics(i).fqn()] = i;
    }
}

} // namespace internal
} // namespace cupti_profiler
