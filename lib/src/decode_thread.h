// Internal: Background decode thread that drains the GPU HW buffer.
#pragma once

#include "cupti_pm_sampling.h"
#include "profiler_host_internal.h"
#include "stop_signal.h"

#include <array>
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
    std::atomic<uint64_t> emptySamples{0};
    std::atomic<uint64_t> samplesLost{0};
    std::atomic<uint64_t> hwBufferOverflows{0};
    std::atomic<uint64_t> samplerRestarts{0};
    std::atomic<uint64_t> stretchedSamples{0};
    std::atomic<uint64_t> latePasses{0};
};

/// What the decode thread needs to know about its device.
struct DecodeTarget {
    uint32_t gpuIndex = 0;
    uint64_t samplingIntervalNs = 0;
    uint64_t decodeIntervalMs = 1000;   // one pass per interval
    // What it takes to re-enable the sampler after a hardware-buffer
    // overflow (see decode_thread.cpp).
    int deviceIndex = 0;
    const std::vector<uint8_t>* configImage = nullptr;
    size_t hwBufferSize = 0;
    uint64_t maxSamples = 0;
};

/// One pass per decodeIntervalMs into the two images (see
/// decode_thread.cpp); samples are evaluated on a worker thread. Stops the
/// sampler itself when `stop` is set, then decodes what is left.
void DecodeThreadFunc(std::array<std::vector<uint8_t>, 2>& counterDataImages,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      DecodeTarget device,
                      DecodeStats& stats,
                      StopSignal& stop,
                      CUptiResult& result);

/// One stderr line summing up any decode trouble of a finished run
/// (image full, invalid samples, lost samples, overflow). Nothing when
/// the run was clean.
void ReportDecodeSummary(uint32_t gpuIndex, const DecodeStats& stats);

} // namespace internal
} // namespace cupti_profiler
