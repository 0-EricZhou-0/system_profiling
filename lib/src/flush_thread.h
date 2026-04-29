// Internal: Periodic flush thread and protobuf serialization helpers.
#pragma once

#include "profiler_host_internal.h"
#include <cupti_profiler/gpu_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Forward-declare generated protobuf class (defined in gpu_metrics.pb.h)
class GpuMetricsTrace;

namespace cupti_profiler {
namespace internal {

/// Pending flush stats carried from one flush cycle to the next.
/// The size of a flush is known only after serialization, so it's attached
/// to the *next* Trace message rather than to its own.
struct PendingFlushStats {
    uint64_t timestampNs = 0;
    uint64_t bytesWritten = 0;
    uint64_t intervalNs = 0;
    bool valid = false;
};

/// Build a GpuMetricsTrace protobuf message from samples.
GpuMetricsTrace BuildTrace(const std::string& deviceName,
                           const std::string& chipName,
                           uint64_t samplingFrequencyHz,
                           const std::vector<const char*>& metricNames,
                           const std::vector<SamplerRange>& samples,
                           double peakDramBwGbps,
                           double peakPcieBwBytesPerSec,
                           double peakNvlinkBwBytesPerSec,
                           uint64_t steadyClockRefNs,
                           uint64_t cuptiRefNs,
                           uint64_t wallClockEpochNs);

/// Write a single length-delimited protobuf message to an open stream.
void WriteDelimitedTo(const GpuMetricsTrace& trace, std::ofstream& out);

/// Same as WriteDelimitedTo, but returns the serialized size in bytes.
size_t WriteDelimitedToSized(const GpuMetricsTrace& trace, std::ofstream& out);

/// Background flush thread function.
void FlushThreadFunc(CuptiProfilerHost& host,
                     std::ofstream& outFile,
                     std::mutex& outMutex,
                     const std::string& deviceName,
                     const std::string& chipName,
                     uint64_t samplingFrequencyHz,
                     const std::vector<const char*>& metricNames,
                     double* pPeakDramBwGbps,
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
