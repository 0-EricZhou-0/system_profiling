// Internal: Flush thread for the disk probe.
//
// Each flush snapshots ProcessTrackingProbe — pending_removal entries
// are emitted with TrackedProcessV2.removed=true, then dropped via
// CommitPendingRemovals().
#pragma once

#include "metric_descriptor.h"

#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/process_tracking_probe.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

class DiskMetricsTrace;

namespace cupti_profiler {
namespace internal {

// One SCOPE_DEVICE sample at one tick. values[] column order is driven
// by GetDiskDeviceMetrics() iteration order (see .cpp).
struct DiskDeviceTick {
    uint64_t timestamp_ns        = 0;
    std::string device_name;
    double   read_bytes_per_sec  = 0.0;
    double   write_bytes_per_sec = 0.0;
    uint32_t read_inflight       = 0;
    uint32_t write_inflight      = 0;
};

// One SCOPE_PROCESS sample at one tick. values[] column order is driven
// by GetDiskProcessMetrics() iteration order.
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

// Accessors for the descriptor arrays owned by disk_flush_thread.cpp.
// Same arrays drive trace emission and catalog registration
// (metric_catalog_builtins.cpp), so wire FQN order and wire values[]
// order are structurally tied to the catalog.
std::span<const MetricDescriptor<DiskDeviceTick>>  GetDiskDeviceMetrics();
std::span<const MetricDescriptor<DiskProcessTick>> GetDiskProcessMetrics();

struct DiskPendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

DiskMetricsTrace BuildDiskTrace(
    const std::string& hostname,
    uint64_t samplingFrequencyHz,
    uint32_t hostCpuCount,
    uint64_t steadyClockRefNs,
    uint64_t wallClockEpochNs,
    const std::vector<std::string>& devices,
    const std::vector<ProcessTrackingProbe::ProcessEntry>& processes,
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
                         ProcessTrackingProbe& probe,
                         std::atomic<bool>& stop,
                         uint64_t flushIntervalMs,
                         uint64_t steadyClockRefNs,
                         uint64_t wallClockEpochNs,
                         DiskPendingFlushStats& pending,
                         std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
