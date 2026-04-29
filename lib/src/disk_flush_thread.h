// Internal: Flush thread for disk I/O metrics.
#pragma once

#include <cupti_profiler/disk_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

class DiskDeviceSample;
class DiskProcessSample;
class DiskMetricsTrace;

namespace cupti_profiler {
namespace internal {

struct DiskSampleBatch {
    std::vector<DiskDeviceSample> deviceSamples;
    std::vector<DiskProcessSample> procSamples;
};

// Pending flush stats carried across cycles (see flush_thread.h::PendingFlushStats).
struct DiskPendingFlushStats {
    uint64_t timestampNs = 0;
    uint64_t bytesWritten = 0;
    uint64_t intervalNs = 0;
    bool valid = false;
};

size_t WriteDelimitedDiskTraceSized(const DiskMetricsTrace& trace, std::ofstream& out);

void DiskFlushThreadFunc(DiskSampleBatch& batch,
                         std::mutex& batchMutex,
                         std::ofstream& outFile,
                         std::mutex& outMutex,
                         const std::string& hostname,
                         uint64_t samplingFrequencyHz,
                         const std::vector<std::string>& devices,
                         const std::vector<TrackedProcess>& Processes,
                         std::atomic<bool>& stop,
                         uint64_t flushIntervalMs,
                         uint64_t steadyClockRefNs,
                         uint64_t wallClockEpochNs,
                         DiskPendingFlushStats& pending,
                         std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
