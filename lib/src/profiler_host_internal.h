// Internal: CUPTI host-side metric configuration and evaluation.
#pragma once

#include <cupti_profiler/gpu_profiler.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>

namespace cupti_profiler {
namespace internal {

class CuptiProfilerHost {
    std::string m_chipName;
    CUpti_Profiler_Host_Object* m_pHostObject = nullptr;
    std::mutex m_mutex;
    std::vector<SamplerRange> m_samplerRanges;

public:
    void SetUp(const std::string& chipName, std::vector<uint8_t>& counterAvailabilityImage);
    void TearDown();

    CUptiResult CreateConfigImage(const std::vector<const char*>& metricsList, std::vector<uint8_t>& configImage);

    CUptiResult EvaluateCounterData(CUpti_PmSampling_Object* pSamplingObject, size_t rangeIndex,
                                     const std::vector<const char*>& metricsList,
                                     std::vector<uint8_t>& counterDataImage);

    /// Atomically drain all accumulated samples.
    std::vector<SamplerRange> DrainSamples();
};

} // namespace internal
} // namespace cupti_profiler
