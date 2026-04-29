// Internal: Flush thread for system (CPU + memory) metrics.
#pragma once

#include <cupti_profiler/system_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Forward-declare generated protobuf class
class SystemMetricsTrace;
class CPUSystemSample;
class CPUProcessSample;
class MemorySystemSample;
class MemoryProcessSample;

namespace cupti_profiler {
namespace internal {

// Accumulated samples from the sample thread, to be drained by the flush thread.
struct SystemSampleBatch {
    std::vector<CPUSystemSample> CPUSysSamples;
    std::vector<CPUProcessSample> CPUProcSamples;
    std::vector<MemorySystemSample> memSysSamples;
    std::vector<MemoryProcessSample> memProcSamples;
};

// Pending flush stats carried across cycles (see flush_thread.h::PendingFlushStats).
struct SystemPendingFlushStats {
    uint64_t timestampNs = 0;
    uint64_t bytesWritten = 0;
    uint64_t intervalNs = 0;
    bool valid = false;
};

size_t WriteDelimitedSystemTraceSized(const SystemMetricsTrace& trace, std::ofstream& out);

void SystemFlushThreadFunc(SystemSampleBatch& batch,
                           std::mutex& batchMutex,
                           std::ofstream& outFile,
                           std::mutex& outMutex,
                           const std::string& hostname,
                           uint64_t samplingFrequencyHz,
                           const std::vector<TrackedProcess>& Processes,
                           std::atomic<bool>& stop,
                           uint64_t flushIntervalMs,
                           uint64_t steadyClockRefNs,
                           uint64_t wallClockEpochNs,
                           SystemPendingFlushStats& pending,
                           std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
