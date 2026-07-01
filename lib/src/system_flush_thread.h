// Internal: Flush thread for the system probe (CPU + memory).
//
// The producer (system_profiler.cpp) reads /proc once per sample tick
// and emits one SystemTick (system-wide values) plus one ProcessTick
// per tracked PID. The flush thread drains those ticks and serializes
// them into a SystemMetricsTrace with the values[] columns ordered to
// match the per-scope FQN registry hardcoded in this file.
//
// Each flush also snapshots ProcessTrackingProbe — entries flagged
// `pending_removal=true` are emitted in this trace's tracked_processes
// with TrackedProcessV2.removed=true, then dropped via
// ProcessTrackingProbe::CommitPendingRemovals() so subsequent flushes
// no longer carry them.
#pragma once

#include "metric_descriptor.h"

#include <cupti_profiler/process_tracking_probe.h>
#include <cupti_profiler/system_profiler.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

class SystemMetricsTrace;

namespace cupti_profiler {
namespace internal {

// One SCOPE_SYSTEM sample (CPU utilization + system memory) at one tick.
// values[] column order is driven by GetSystemMetrics() iteration order
// (see .cpp). Adding a field here without adding a matching descriptor
// to kSystemMetrics is harmless — the value just won't be emitted; the
// reverse is a compile error at the descriptor's `.read` lambda.
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
// values[] column order is driven by GetProcessMetrics() iteration order.
struct ProcessTick {
    uint64_t timestamp_ns   = 0;
    uint32_t pid            = 0;
    // Total on-CPU time as % of one core. From /proc/<pid>/schedstat
    // sum_exec_runtime delta — nanosecond-precise (no CLK_TCK
    // quantization). >100% for multi-threaded tasks.
    double cpu_pct          = 0.0;
    // Memory (bytes)
    uint64_t rss_bytes      = 0;
    uint64_t vms_bytes      = 0;
    uint64_t shared_bytes   = 0;
};

struct SystemSampleBatch {
    std::vector<SystemTick>  systemTicks;
    std::vector<ProcessTick> processTicks;
};

// Accessors for the descriptor arrays owned by system_flush_thread.cpp.
// Same arrays drive trace emission (AppendSystemSample / AddScopeRegistry)
// and catalog registration (metric_catalog_builtins.cpp). Returning a
// std::span keeps the storage in one TU while letting other TUs iterate.
std::span<const MetricDescriptor<SystemTick>>  GetSystemMetrics();
std::span<const MetricDescriptor<ProcessTick>> GetProcessMetrics();

struct SystemPendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

/// Build a SystemMetricsTrace from a drained batch + a tracked-PID
/// snapshot. Exposed so SystemProfiler::Stop() can serialize the final
/// batch on the calling thread without re-implementing the conversion.
/// Entries with `pending_removal=true` are emitted with
/// TrackedProcessV2.removed=true.
SystemMetricsTrace BuildSystemTrace(
    const std::string& hostname,
    uint64_t samplingFrequencyHz,
    uint32_t hostCpuCount,
    uint64_t steadyClockRefNs,
    uint64_t wallClockEpochNs,
    const std::vector<ProcessTrackingProbe::ProcessEntry>& processes,
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
                           ProcessTrackingProbe& probe,
                           std::atomic<bool>& stop,
                           uint64_t flushIntervalMs,
                           uint64_t steadyClockRefNs,
                           uint64_t wallClockEpochNs,
                           SystemPendingFlushStats& pending,
                           std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
