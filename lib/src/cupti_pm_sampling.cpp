#include "cupti_pm_sampling.h"
#include "helper_cupti.h"

#include <cuda.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>

#include <iostream>

namespace cupti_profiler {
namespace internal {

void CuptiPmSampling::SetUp(int deviceIndex) {
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceIndex));

    int major = 0, minor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    std::cout << "Compute capability: " << major << "." << minor << "\n";
    if (major * 10 + minor < 75) {
        std::cerr << "PM sampling requires compute capability >= 7.5\n";
        exit(1);
    }

    CUpti_Profiler_Initialize_Params p = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&p));
}

void CuptiPmSampling::TearDown() {
    CUpti_Profiler_DeInitialize_Params p = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&p));
}

CUptiResult CuptiPmSampling::EnablePmSampling(int deviceIndex) {
    CUpti_PmSampling_Enable_Params p = {CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    p.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiPmSamplingEnable(&p));
    m_pmSamplerObject = p.pPmSamplingObject;
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::DisablePmSampling() {
    CUpti_PmSampling_Disable_Params p = {CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingDisable(&p));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::SetConfig(std::vector<uint8_t>& configImage, size_t hwBufferSize, uint64_t samplingIntervalNs) {
    CUpti_PmSampling_SetConfig_Params p = {CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    p.configSize = configImage.size();
    p.pConfig = configImage.data();
    p.hardwareBufferSize = hwBufferSize;
    p.samplingInterval = samplingIntervalNs;
    p.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    CUPTI_API_CALL(cuptiPmSamplingSetConfig(&p));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::CreateCounterDataImage(uint64_t maxSamples, const std::vector<const char*>& metricsList,
                                                      std::vector<uint8_t>& counterDataImage) {
    CUpti_PmSampling_GetCounterDataSize_Params p = {CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    p.numMetrics = metricsList.size();
    p.pMetricNames = const_cast<const char**>(metricsList.data());
    p.maxSamples = maxSamples;
    CUPTI_API_CALL(cuptiPmSamplingGetCounterDataSize(&p));

    counterDataImage.resize(p.counterDataSize);
    CUpti_PmSampling_CounterDataImage_Initialize_Params ip = {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    ip.pPmSamplingObject = m_pmSamplerObject;
    ip.counterDataSize = counterDataImage.size();
    ip.pCounterData = counterDataImage.data();
    CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&ip));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::ResetCounterDataImage(std::vector<uint8_t>& counterDataImage) {
    CUpti_PmSampling_CounterDataImage_Initialize_Params p = {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    p.counterDataSize = counterDataImage.size();
    p.pCounterData = counterDataImage.data();
    CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&p));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::Start() {
    CUpti_PmSampling_Start_Params p = {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingStart(&p));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::Stop() {
    CUpti_PmSampling_Stop_Params p = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    CUPTI_API_CALL(cuptiPmSamplingStop(&p));
    return CUPTI_SUCCESS;
}

CUptiResult CuptiPmSampling::DecodeData(std::vector<uint8_t>& counterDataImage) {
    CUpti_PmSampling_DecodeData_Params p = {CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
    p.pPmSamplingObject = m_pmSamplerObject;
    p.pCounterDataImage = counterDataImage.data();
    p.counterDataImageSize = counterDataImage.size();
    CUPTI_API_CALL(cuptiPmSamplingDecodeData(&p));
    return CUPTI_SUCCESS;
}

void CuptiPmSampling::GetChipName(int deviceIndex, std::string& chipName) {
    CUpti_Profiler_Initialize_Params ip = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&ip));
    CUpti_Device_GetChipName_Params p = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    p.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&p));
    chipName = p.pChipName;
}

void CuptiPmSampling::GetCounterAvailabilityImage(int deviceIndex, std::vector<uint8_t>& image) {
    CUpti_PmSampling_GetCounterAvailability_Params p = {CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    p.deviceIndex = deviceIndex;
    CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&p));
    image.resize(p.counterAvailabilityImageSize);
    p.pCounterAvailabilityImage = image.data();
    CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&p));
}

} // namespace internal
} // namespace cupti_profiler
