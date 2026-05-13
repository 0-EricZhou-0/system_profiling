// Internal: Flush thread for the disk probe.
#pragma once

#include <cupti_profiler/disk_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

class DiskMetricsTrace;

namespace cupti_profiler {
namespace internal {

// One SCOPE_DEVICE sample at one tick. values[] order matches kDeviceFqns.
struct DiskDeviceTick {
    uint64_t timestamp_ns        = 0;
    std::string device_name;
    double   read_bytes_per_sec  = 0.0;
    double   write_bytes_per_sec = 0.0;
    uint32_t read_inflight       = 0;
    uint32_t write_inflight      = 0;
};

// One SCOPE_PROCESS sample at one tick. values[] order matches kProcessFqns.
struct DiskProcessTick {
    uint64_t timestamp_ns         = 0;
    uint32_t pid                  = 0;
    double   rchar_bytes_per_sec  = 0.0;
    double   wchar_bytes_per_sec  = 0.0;
};

struct DiskSampleBatch {
    std::vector<DiskDeviceTick>  deviceTicks;
    std::vector<DiskProcessTick> processTicks;
};

struct DiskPendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

DiskMetricsTrace BuildDiskTrace(const std::string& hostname,
                                uint64_t samplingFrequencyHz,
                                uint32_t hostCpuCount,
                                uint64_t steadyClockRefNs,
                                uint64_t wallClockEpochNs,
                                const std::vector<std::string>& devices,
                                const std::vector<TrackedProcess>& processes,
                                const DiskSampleBatch& drained);

size_t WriteDelimitedDiskTraceSized(const DiskMetricsTrace& trace,
                                    std::ofstream& out);

void DiskFlushThreadFunc(DiskSampleBatch& batch,
                         std::mutex& batchMutex,
                         std::ofstream& outFile,
                         std::mutex& outMutex,
                         const std::string& hostname,
                         uint64_t samplingFrequencyHz,
                         uint32_t hostCpuCount,
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
