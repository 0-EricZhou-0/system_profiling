// Internal: Flush thread for the disk probe.
//
// Each flush snapshots ProcessTrackingProbe — pending_removal entries
// are emitted with TrackedProcessV2.removed=true, then dropped via
// CommitPendingRemovals().
#pragma once

#include "metric_descriptor.h"
#include "stop_signal.h"

#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/process_tracking_probe.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <utility>
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
    // Rates (bytes/s over the actual interval) of the five
    // /proc/<pid>/io counters, each named for the counter it carries.
    double   rchar_bytes_per_sec                 = 0.0;
    double   wchar_bytes_per_sec                 = 0.0;
    double   read_bytes_per_sec                  = 0.0;
    double   write_bytes_per_sec                 = 0.0;
    double   cancelled_write_bytes_per_sec       = 0.0;
};

// The five /proc/<pid>/io counters of one reading, in bytes.
struct IoCounterValues {
    uint64_t rchar = 0, wchar = 0, readBytes = 0, writeBytes = 0, cancelledWriteBytes = 0;
};

// Reaped tracked children's I/O subtracted from their tracked parent's
// sample (IoReapAdjustment in disk_metrics.proto).
struct IoReapRecord {
    uint64_t timestamp_ns = 0;   // the parent's sample
    uint32_t parent_pid   = 0;
    struct Child {
        uint32_t        pid       = 0;
        IoCounterValues lastSeen;           // its last reading
        uint32_t        reapedBy  = 0;      // parent_pid, or a chain member
        bool            ambiguous = false;  // listed, not subtracted
        bool            autoreaped = false; // auto-reaped on the way: listed, not subtracted
    };
    std::vector<Child> children;
    bool     ambiguous    = false;   // some child is
    bool     autoreaped   = false;   // some child is
    // Raw parent delta minus the subtracted, per counter.
    int64_t  remainder[5] = {0, 0, 0, 0, 0};
};

// A reaped child whose reap chain stops at a traced parent that was never
// read (IoReapChainBreak in disk_metrics.proto).
struct IoChainBreakRecord {
    uint64_t timestamp_ns = 0;
    uint32_t pid          = 0;   // the reaped child
    uint32_t missing_pid  = 0;   // the traced parent with no reading
    uint32_t absorbed_by  = 0;   // missing_pid's parent (traced), which reaps it
};

struct DiskSampleBatch {
    std::vector<DiskDeviceTick>     deviceTicks;
    std::vector<DiskProcessTick>    processTicks;
    std::vector<IoReapRecord>       ioReaps;
    std::vector<IoChainBreakRecord> chainBreaks;
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
    uint64_t durationNs   = 0;   // drain to written
    uint64_t slowFlushes  = 0;   // so far (FlushBacklog)
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
                         StopSignal& stop,
                         uint64_t flushIntervalMs,
                         uint64_t steadyClockRefNs,
                         uint64_t wallClockEpochNs,
                         DiskPendingFlushStats& pending,
                         std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
