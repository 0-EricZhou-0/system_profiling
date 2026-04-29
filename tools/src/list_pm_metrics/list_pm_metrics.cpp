// List CUPTI PM-samplable base metrics on a given device.
// Usage: list_pm_metrics [-d device] [--sub] [filter_substring]
//
// The PM-sampling subset is a *subset* of the full CUPTI metric catalog:
// only counters that (a) fit in one pass and (b) the PM unit can stream
// at high rate are exposed here. Range-profiling metrics shown by
// `ncu --query-metrics` may not appear.

#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_pmsampling.h>
#include <cupti_target.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CUPTI_CALL(call) do {                                              \
    CUptiResult _s = (call);                                               \
    if (_s != CUPTI_SUCCESS) {                                             \
        const char* err = "?";                                             \
        cuptiGetResultString(_s, &err);                                    \
        fprintf(stderr, "CUPTI error %d (%s) at %s:%d\n", _s, err,         \
                __FILE__, __LINE__);                                       \
        return 1;                                                          \
    }                                                                      \
} while(0)

int main(int argc, char** argv) {
    int device = 0;
    bool showSub = false;
    const char* filter = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sub") == 0) {
            showSub = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-d device] [--sub] [filter_substring]\n"
                   "  -d  Device index (default: 0)\n"
                   "  --sub  Attempt to list submetrics (may be empty on some chips)\n"
                   "  filter  Only print base metrics whose name contains this substring\n",
                   argv[0]);
            return 0;
        } else {
            filter = argv[i];
        }
    }

    cuInit(0);
    cudaFree(0);
    cudaSetDevice(device);

    CUpti_Profiler_Initialize_Params pInit = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerInitialize(&pInit));

    CUpti_Device_GetChipName_Params chipP = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipP.deviceIndex = device;
    CUPTI_CALL(cuptiDeviceGetChipName(&chipP));
    std::string chipName = chipP.pChipName;
    printf("# Device %d, Chip: %s\n", device, chipName.c_str());

    CUpti_PmSampling_GetCounterAvailability_Params caP = {CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    caP.deviceIndex = device;
    CUPTI_CALL(cuptiPmSamplingGetCounterAvailability(&caP));
    std::vector<uint8_t> avail(caP.counterAvailabilityImageSize);
    caP.pCounterAvailabilityImage = avail.data();
    CUPTI_CALL(cuptiPmSamplingGetCounterAvailability(&caP));

    CUpti_Profiler_Host_Initialize_Params initP = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    initP.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    initP.pChipName = chipName.c_str();
    initP.pCounterAvailabilityImage = avail.data();
    CUPTI_CALL(cuptiProfilerHostInitialize(&initP));
    auto* host = initP.pHostObject;

    const char* typeNames[] = {"COUNTER", "RATIO", "THROUGHPUT"};
    size_t totals[3] = {0, 0, 0};
    for (int t = 0; t < 3; ++t) {
        CUpti_Profiler_Host_GetBaseMetrics_Params p = {CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE};
        p.pHostObject = host;
        p.metricType = (CUpti_MetricType)t;
        CUPTI_CALL(cuptiProfilerHostGetBaseMetrics(&p));

        totals[t] = p.numMetrics;
        for (size_t i = 0; i < p.numMetrics; ++i) {
            const char* name = p.ppMetricNames[i];
            if (filter && !strstr(name, filter)) continue;
            printf("[%s] %s\n", typeNames[t], name);
            if (showSub) {
                CUpti_Profiler_Host_GetMetricProperties_Params mp = {CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
                mp.pHostObject = host;
                mp.pMetricName = name;
                if (cuptiProfilerHostGetMetricProperties(&mp) != CUPTI_SUCCESS) continue;

                CUpti_Profiler_Host_GetSubMetrics_Params sp = {CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE};
                sp.pHostObject = host;
                sp.pMetricName = name;
                sp.metricType = mp.metricType;
                if (cuptiProfilerHostGetSubMetrics(&sp) != CUPTI_SUCCESS) continue;
                for (size_t j = 0; j < sp.numOfSubmetrics; ++j) {
                    printf("    %s.%s\n", name, sp.ppSubMetrics[j]);
                }
            }
        }
    }

    printf("# Totals: %zu counter, %zu ratio, %zu throughput (PM-samplable base metrics)\n",
           totals[0], totals[1], totals[2]);

    // Query theoretical max hardware metrics per pass (upper bound; practical may be lower)
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params maxP = {
        CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params_STRUCT_SIZE};
    maxP.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    maxP.pChipName = chipName.c_str();
    maxP.pCounterAvailabilityImage = avail.data();
    if (cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(&maxP) == CUPTI_SUCCESS) {
        printf("# Max HW metrics per pass (theoretical upper bound for PM sampling): %zu\n",
               maxP.maxMetricsPerPass);
    }

    CUpti_Profiler_Host_Deinitialize_Params deP = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    deP.pHostObject = host;
    CUPTI_CALL(cuptiProfilerHostDeinitialize(&deP));
    return 0;
}
