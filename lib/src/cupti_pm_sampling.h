// Internal: CUPTI PM Sampling target-side lifecycle management.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cupti_pmsampling.h>

namespace cupti_profiler {
namespace internal {

class CuptiPmSampling {
    CUpti_PmSampling_Object* m_pmSamplerObject = nullptr;

public:
    void SetUp(int deviceIndex);
    void TearDown();

    CUptiResult EnablePmSampling(int deviceIndex);
    CUptiResult DisablePmSampling();
    CUptiResult SetConfig(std::vector<uint8_t>& configImage, size_t hwBufferSize, uint64_t samplingIntervalNs);
    CUptiResult CreateCounterDataImage(uint64_t maxSamples, const std::vector<const char*>& metricsList,
                                       std::vector<uint8_t>& counterDataImage);
    CUptiResult ResetCounterDataImage(std::vector<uint8_t>& counterDataImage);
    CUptiResult Start();
    CUptiResult Stop();
    CUptiResult DecodeData(std::vector<uint8_t>& counterDataImage);

    CUpti_PmSampling_Object* GetPmSamplerObject() { return m_pmSamplerObject; }

    static void GetChipName(int deviceIndex, std::string& chipName);
    static void GetCounterAvailabilityImage(int deviceIndex, std::vector<uint8_t>& image);
};

} // namespace internal
} // namespace cupti_profiler
