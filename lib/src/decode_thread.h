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

/// Decode health of one device, cumulative since Start(). Written by
/// its decode thread, read by the flush thread (GpuDecodeStats in
/// gpu_metrics.proto has the meaning of each counter).
struct DecodeStats {
    std::atomic<uint64_t> decodeCalls{0};
    std::atomic<uint64_t> samples{0};
    std::atomic<uint64_t> counterDataFull{0};
    std::atomic<uint64_t> invalidSamples{0};
    std::atomic<uint64_t> samplesLost{0};
    std::atomic<uint64_t> hwBufferOverflows{0};
};

/// What the decode thread needs to know about its device.
struct DecodeTarget {
    uint32_t gpuIndex = 0;
    uint64_t samplingIntervalNs = 0;
};

void DecodeThreadFunc(std::vector<uint8_t>& counterDataImage,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      DecodeTarget device,
                      DecodeStats& stats,
                      std::atomic<bool>& stop,
                      CUptiResult& result);

/// One stderr line summing up any decode trouble of a finished run
/// (image full, invalid samples, lost samples, overflow). Nothing when
/// the run was clean.
void ReportDecodeSummary(uint32_t gpuIndex, const DecodeStats& stats);

} // namespace internal
} // namespace cupti_profiler
