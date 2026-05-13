// Internal: Periodic flush thread and protobuf serialization helpers
// for the GPU probe.
#pragma once

#include "profiler_host_internal.h"
#include <cupti_profiler/gpu_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Forward-declare generated protobuf class
class GPUMetricsTrace;

namespace cupti_profiler {
namespace internal {

/// Pending flush stats carried from one flush cycle to the next.
/// The size of a flush is known only after serialization, so it's
/// attached to the *next* Trace message rather than to its own.
struct PendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

/// Build a GPUMetricsTrace protobuf message from a batch of samples.
/// Until the multi-device commit, this writer emits a single
/// GPUDeviceInfo (gpu_index = 0) and tags every GPUSample.gpu_index = 0.
GPUMetricsTrace BuildTrace(const std::string& hostname,
                           uint64_t samplingFrequencyHz,
                           uint32_t hostCpuCount,
                           const std::string& deviceName,
                           const std::string& chipName,
                           const std::vector<const char*>& metricNames,
                           const std::vector<SamplerRange>& samples,
                           double peakDramBwBytesPerSec,
                           double peakPcieBwBytesPerSec,
                           double peakNvlinkBwBytesPerSec,
                           uint64_t steadyClockRefNs,
                           uint64_t cuptiRefNs,
                           uint64_t wallClockEpochNs);

/// Write a single length-delimited protobuf message to an open stream.
size_t WriteDelimitedToSized(const GPUMetricsTrace& trace,
                             std::ofstream& out);

/// Background flush thread function.
void FlushThreadFunc(CuptiProfilerHost& host,
                     std::ofstream& outFile,
                     std::mutex& outMutex,
                     const std::string& hostname,
                     uint64_t samplingFrequencyHz,
                     uint32_t hostCpuCount,
                     const std::string& deviceName,
                     const std::string& chipName,
                     const std::vector<const char*>& metricNames,
                     double* pPeakDramBwBytesPerSec,
                     double* pPeakPcieBwBytesPerSec,
                     double* pPeakNvlinkBwBytesPerSec,
                     std::atomic<bool>& stop,
                     uint64_t flushIntervalMs,
                     uint64_t steadyClockRefNs,
                     uint64_t cuptiRefNs,
                     uint64_t wallClockEpochNs,
                     PendingFlushStats& pending,
                     std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
