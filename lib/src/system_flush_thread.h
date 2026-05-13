// Internal: Flush thread for the system probe (CPU + memory).
//
// The producer (system_profiler.cpp) reads /proc once per sample tick
// and emits one SystemTick (system-wide values) plus one ProcessTick
// per tracked PID. The flush thread drains those ticks and serializes
// them into a SystemMetricsTrace with the values[] columns ordered to
// match the per-scope FQN registry hardcoded in this file.
#pragma once

#include <cupti_profiler/system_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

class SystemMetricsTrace;

namespace cupti_profiler {
namespace internal {

// One SCOPE_SYSTEM sample (CPU utilization + system memory) at one tick.
// values[] column order matches kSystemFqns (see .cpp).
struct SystemTick {
    uint64_t timestamp_ns = 0;
    // CPU (%)
    double cpu_busy_pct   = 0.0;
    double cpu_user_pct   = 0.0;
    double cpu_kernel_pct = 0.0;
    double cpu_iowait_pct = 0.0;
    // Memory (bytes)
    uint64_t mem_capacity_bytes  = 0;
    uint64_t mem_used_bytes      = 0;
    uint64_t mem_available_bytes = 0;
    uint64_t mem_buffers_bytes   = 0;
    uint64_t mem_cached_bytes    = 0;
};

// One SCOPE_PROCESS sample (per-PID CPU + per-PID memory) at one tick.
// values[] column order matches kProcessFqns (see .cpp).
struct ProcessTick {
    uint64_t timestamp_ns   = 0;
    uint32_t pid            = 0;
    // CPU (% of one core)
    double cpu_user_pct     = 0.0;
    double cpu_kernel_pct   = 0.0;
    double cpu_iowait_pct   = 0.0;
    // Memory (bytes)
    uint64_t rss_bytes      = 0;
    uint64_t vms_bytes      = 0;
    uint64_t shared_bytes   = 0;
};

struct SystemSampleBatch {
    std::vector<SystemTick>  systemTicks;
    std::vector<ProcessTick> processTicks;
};

struct SystemPendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

/// Build a SystemMetricsTrace from a drained batch. Exposed so
/// SystemProfiler::Stop() can serialize the final batch on the calling
/// thread without re-implementing the conversion.
SystemMetricsTrace BuildSystemTrace(const std::string& hostname,
                                    uint64_t samplingFrequencyHz,
                                    uint32_t hostCpuCount,
                                    uint64_t steadyClockRefNs,
                                    uint64_t wallClockEpochNs,
                                    const std::vector<TrackedProcess>& processes,
                                    const SystemSampleBatch& drained);

size_t WriteDelimitedSystemTraceSized(const SystemMetricsTrace& trace,
                                      std::ofstream& out);

void SystemFlushThreadFunc(SystemSampleBatch& batch,
                           std::mutex& batchMutex,
                           std::ofstream& outFile,
                           std::mutex& outMutex,
                           const std::string& hostname,
                           uint64_t samplingFrequencyHz,
                           uint32_t hostCpuCount,
                           const std::vector<TrackedProcess>& Processes,
                           std::atomic<bool>& stop,
                           uint64_t flushIntervalMs,
                           uint64_t steadyClockRefNs,
                           uint64_t wallClockEpochNs,
                           SystemPendingFlushStats& pending,
                           std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
