#include "decode_thread.h"

#include <chrono>
#include <thread>

namespace cupti_profiler {
namespace internal {

void DecodeThreadFunc(std::vector<uint8_t>& counterDataImage,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      std::atomic<bool>& stop,
                      CUptiResult& result)
{
    while (!stop) {
        result = target.DecodeData(counterDataImage);
        if (result != CUPTI_SUCCESS) return;

        CUpti_PmSampling_GetCounterDataInfo_Params info = {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        info.pCounterDataImage = counterDataImage.data();
        info.counterDataImageSize = counterDataImage.size();
        result = cuptiPmSamplingGetCounterDataInfo(&info);
        if (result != CUPTI_SUCCESS) return;

        for (size_t i = 0; i < info.numCompletedSamples; ++i) {
            host.EvaluateCounterData(target.GetPmSamplerObject(), i, metricsList, counterDataImage);
        }

        result = target.ResetCounterDataImage(counterDataImage);
        if (result != CUPTI_SUCCESS) return;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Final drain
    result = target.DecodeData(counterDataImage);
    if (result != CUPTI_SUCCESS) return;

    CUpti_PmSampling_GetCounterDataInfo_Params info = {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
    info.pCounterDataImage = counterDataImage.data();
    info.counterDataImageSize = counterDataImage.size();
    result = cuptiPmSamplingGetCounterDataInfo(&info);
    if (result != CUPTI_SUCCESS) return;

    for (size_t i = 0; i < info.numCompletedSamples; ++i) {
        host.EvaluateCounterData(target.GetPmSamplerObject(), i, metricsList, counterDataImage);
    }
}

} // namespace internal
} // namespace cupti_profiler
