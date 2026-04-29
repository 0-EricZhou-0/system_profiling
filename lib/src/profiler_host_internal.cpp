#include "profiler_host_internal.h"
#include "helper_cupti.h"

#include <iostream>

namespace cupti_profiler {
namespace internal {

void CuptiProfilerHost::SetUp(const std::string& chipName, std::vector<uint8_t>& counterAvailabilityImage) {
    m_chipName = chipName;
    CUpti_Profiler_Host_Initialize_Params params = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    params.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    params.pChipName = m_chipName.c_str();
    params.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerHostInitialize(&params));
    m_pHostObject = params.pHostObject;
}

void CuptiProfilerHost::TearDown() {
    CUpti_Profiler_Host_Deinitialize_Params params = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    params.pHostObject = m_pHostObject;
    CUPTI_API_CALL(cuptiProfilerHostDeinitialize(&params));
}

CUptiResult CuptiProfilerHost::CreateConfigImage(const std::vector<const char*>& metricsList, std::vector<uint8_t>& configImage) {
    {
        CUpti_Profiler_Host_ConfigAddMetrics_Params p = {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
        p.pHostObject = m_pHostObject;
        p.ppMetricNames = const_cast<const char**>(metricsList.data());
        p.numMetrics = metricsList.size();
        CUPTI_API_CALL(cuptiProfilerHostConfigAddMetrics(&p));
    }
    {
        CUpti_Profiler_Host_GetConfigImageSize_Params p = {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
        p.pHostObject = m_pHostObject;
        CUPTI_API_CALL(cuptiProfilerHostGetConfigImageSize(&p));
        configImage.resize(p.configImageSize);

        CUpti_Profiler_Host_GetConfigImage_Params p2 = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
        p2.pHostObject = m_pHostObject;
        p2.pConfigImage = configImage.data();
        p2.configImageSize = configImage.size();
        CUPTI_API_CALL(cuptiProfilerHostGetConfigImage(&p2));
    }
    {
        CUpti_Profiler_Host_GetNumOfPasses_Params p = {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
        p.pConfigImage = configImage.data();
        p.configImageSize = configImage.size();
        CUPTI_API_CALL(cuptiProfilerHostGetNumOfPasses(&p));
        std::cout << "Num of passes required: " << p.numOfPasses << " (must be 1 for PM sampling)\n";
    }
    return CUPTI_SUCCESS;
}

CUptiResult CuptiProfilerHost::EvaluateCounterData(CUpti_PmSampling_Object* pSamplingObject, size_t rangeIndex,
                                                     const std::vector<const char*>& metricsList,
                                                     std::vector<uint8_t>& counterDataImage) {
    SamplerRange sr;
    sr.rangeIndex = rangeIndex;

    CUpti_PmSampling_CounterData_GetSampleInfo_Params si = {CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
    si.pPmSamplingObject = pSamplingObject;
    si.pCounterDataImage = counterDataImage.data();
    si.counterDataImageSize = counterDataImage.size();
    si.sampleIndex = rangeIndex;
    CUPTI_API_CALL(cuptiPmSamplingCounterDataGetSampleInfo(&si));
    sr.startTimestamp = si.startTimestamp;
    sr.endTimestamp = si.endTimestamp;

    sr.metricValues.resize(metricsList.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params ev = {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    ev.pHostObject = m_pHostObject;
    ev.pCounterDataImage = counterDataImage.data();
    ev.counterDataImageSize = counterDataImage.size();
    ev.ppMetricNames = const_cast<const char**>(metricsList.data());
    ev.numMetrics = metricsList.size();
    ev.rangeIndex = rangeIndex;
    ev.pMetricValues = sr.metricValues.data();
    CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&ev));

    std::lock_guard<std::mutex> lock(m_mutex);
    m_samplerRanges.push_back(std::move(sr));
    return CUPTI_SUCCESS;
}

std::vector<SamplerRange> CuptiProfilerHost::DrainSamples() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<SamplerRange> drained;
    drained.swap(m_samplerRanges);
    return drained;
}

} // namespace internal
} // namespace cupti_profiler
