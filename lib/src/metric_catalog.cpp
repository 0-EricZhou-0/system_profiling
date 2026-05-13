#include "metric_catalog.h"

#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace cupti_profiler {
namespace internal {

MetricCatalog MetricCatalog::LoadFromPbtxt(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "MetricCatalog: failed to open " << path << "\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();

    MetricCatalog out;
    if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &out.proto_)) {
        std::cerr << "MetricCatalog: failed to parse " << path
                  << " — expected text-format MetricCatalog\n";
        std::exit(1);
    }
    out.RebuildIndex();
    std::cout << "MetricCatalog: loaded " << out.proto_.metrics_size()
              << " descriptors from " << path << "\n";
    return out;
}

const MetricDescriptor* MetricCatalog::Find(const std::string& fqn) const {
    auto it = index_by_fqn_.find(fqn);
    if (it == index_by_fqn_.end()) return nullptr;
    return &proto_.metrics(it->second);
}

std::vector<const MetricDescriptor*> MetricCatalog::ByScope(Scope scope) const {
    std::vector<const MetricDescriptor*> out;
    out.reserve(proto_.metrics_size());
    for (int i = 0; i < proto_.metrics_size(); ++i) {
        const auto& m = proto_.metrics(i);
        if (m.scope() == scope) out.push_back(&m);
    }
    return out;
}

void MetricCatalog::AppendGpuMetrics(
        const std::vector<MetricDescriptor>& descriptors)
{
    for (const auto& d : descriptors) {
        *proto_.add_metrics() = d;
    }
    RebuildIndex();
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
