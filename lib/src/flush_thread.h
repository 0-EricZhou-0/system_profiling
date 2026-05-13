// Internal: Periodic flush thread and protobuf serialization helpers
// for the GPU probe. Supports multi-device runs — the flush thread
// drains every CUPTI session in `devices`, tags each sample with its
// `gpu_index`, and emits one merged GPUMetricsTrace per flush.
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

/// Per-device data fed to BuildTrace().
struct GpuDevicePayload {
    uint32_t                  gpu_index;
    std::string               device_name;
    std::string               chip_name;
    double                    peak_dram_bw_bytes_per_s;
    double                    peak_pcie_bw_bytes_per_s;
    double                    peak_nvlink_bw_bytes_per_s;
    std::vector<SamplerRange> samples;
};

/// Pending flush stats carried from one flush cycle to the next.
struct PendingFlushStats {
    uint64_t bytesWritten = 0;
    uint64_t intervalNs   = 0;
    bool     valid        = false;
};

/// Build a GPUMetricsTrace from per-device payloads.
GPUMetricsTrace BuildTrace(const std::string& hostname,
                           uint64_t samplingFrequencyHz,
                           uint32_t hostCpuCount,
                           const std::vector<const char*>& metricNames,
                           const std::vector<GpuDevicePayload>& devices,
                           uint64_t steadyClockRefNs,
                           uint64_t cuptiRefNs,
                           uint64_t wallClockEpochNs);

/// Write a single length-delimited protobuf message to an open stream.
size_t WriteDelimitedToSized(const GPUMetricsTrace& trace,
                             std::ofstream& out);

/// One per-device drain slot consumed by the multi-device flush thread.
/// `host`s are owned by GpuProfiler::Impl; the flush thread holds
/// pointers and calls DrainSamples() on each.
struct DeviceDrainSlot {
    uint32_t            gpu_index;
    std::string         device_name;
    std::string         chip_name;
    double*             peak_dram_bw_bytes_per_s;
    double*             peak_pcie_bw_bytes_per_s;
    double*             peak_nvlink_bw_bytes_per_s;
    CuptiProfilerHost*  host;
};

/// Background flush thread function. Iterates `devices` each cycle,
/// drains samples from each, and emits one merged trace.
void FlushThreadFunc(std::vector<DeviceDrainSlot> devices,
                     std::ofstream& outFile,
                     std::mutex& outMutex,
                     const std::string& hostname,
                     uint64_t samplingFrequencyHz,
                     uint32_t hostCpuCount,
                     const std::vector<const char*>& metricNames,
                     std::atomic<bool>& stop,
                     uint64_t flushIntervalMs,
                     uint64_t steadyClockRefNs,
                     uint64_t cuptiRefNs,
                     uint64_t wallClockEpochNs,
                     PendingFlushStats& pending,
                     std::mutex& pendingMutex);

} // namespace internal
} // namespace cupti_profiler
