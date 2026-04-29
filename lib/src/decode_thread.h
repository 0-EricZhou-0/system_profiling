// Internal: Background decode thread that drains the GPU HW buffer.
#pragma once

#include "cupti_pm_sampling.h"
#include "profiler_host_internal.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include <cupti_pmsampling.h>

namespace cupti_profiler {
namespace internal {

void DecodeThreadFunc(std::vector<uint8_t>& counterDataImage,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      std::atomic<bool>& stop,
                      CUptiResult& result);

} // namespace internal
} // namespace cupti_profiler
