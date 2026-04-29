#include "flush_thread.h"

#include "gpu_metrics.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

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
                           uint64_t wallClockEpochNs)
{
    GpuMetricsTrace trace;
    trace.set_device_name(deviceName);
    trace.set_chip_name(chipName);
    trace.set_sampling_frequency_hz(samplingFrequencyHz);
    trace.set_peak_dram_bw_gbps(peakDramBwGbps);
    trace.set_peak_pcie_bw_bytes_per_sec(peakPcieBwBytesPerSec);
    trace.set_peak_nvlink_bw_bytes_per_sec(peakNvlinkBwBytesPerSec);
    trace.set_steady_clock_reference_ns(steadyClockRefNs);
    trace.set_cupti_reference_ns(cuptiRefNs);
    trace.set_wall_clock_epoch_ns(wallClockEpochNs);

    for (const auto& name : metricNames) {
        trace.add_metric_names(name);
    }

    for (const auto& s : samples) {
        auto* sample = trace.add_samples();
        sample->set_start_timestamp_ns(s.startTimestamp);
        sample->set_end_timestamp_ns(s.endTimestamp);
        for (double v : s.metricValues) {
            sample->add_values(v);
        }
    }

    return trace;
}

void WriteDelimitedTo(const GpuMetricsTrace& trace, std::ofstream& out) {
    std::string serialized;
    if (!trace.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize protobuf message\n";
        exit(1);
    }
    google::protobuf::io::OstreamOutputStream raw(&out);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
    coded.WriteString(serialized);
}

// Variant that returns the serialized size (useful for FlushStats accounting).
size_t WriteDelimitedToSized(const GpuMetricsTrace& trace, std::ofstream& out) {
    std::string serialized;
    if (!trace.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize protobuf message\n";
        exit(1);
    }
    google::protobuf::io::OstreamOutputStream raw(&out);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
    coded.WriteString(serialized);
    return serialized.size();
}

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
                     std::mutex& pendingMutex)
{
    size_t totalFlushed = 0;
    uint64_t prevFlushNs = 0;
    while (!stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(flushIntervalMs));
        if (stop) break;

        auto samples = host.DrainSamples();
        if (samples.empty()) continue;

        GpuMetricsTrace trace = BuildTrace(deviceName, chipName, samplingFrequencyHz,
                                           metricNames, samples,
                                           *pPeakDramBwGbps, *pPeakPcieBwBytesPerSec,
                                           *pPeakNvlinkBwBytesPerSec,
                                           steadyClockRefNs, cuptiRefNs, wallClockEpochNs);

        // Attach previous flush's stats (known only in hindsight)
        {
            std::lock_guard<std::mutex> lock(pendingMutex);
            if (pending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_timestamp_ns(pending.timestampNs);
                fs->set_bytes_written(pending.bytesWritten);
                fs->set_interval_ns(pending.intervalNs);
                pending.valid = false;
            }
        }

        size_t bytes = 0;
        {
            std::lock_guard<std::mutex> lock(outMutex);
            bytes = WriteDelimitedToSized(trace, outFile);
            outFile.flush();
        }
        uint64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t intervalNs = (prevFlushNs == 0) ? 0 : (nowNs - prevFlushNs);
        prevFlushNs = nowNs;

        // Stash this flush's size for the next cycle (or Stop() to drain)
        {
            std::lock_guard<std::mutex> lock(pendingMutex);
            pending.timestampNs = nowNs;
            pending.bytesWritten = bytes;
            pending.intervalNs = intervalNs;
            pending.valid = true;
        }

        totalFlushed += samples.size();
        // bytes / (intervalNs / 1e9 sec) / (1024*1024) = MiB/s
        double mibPerSec = (intervalNs > 0)
            ? (double)bytes * 1e9 / ((double)intervalNs * 1024.0 * 1024.0) : 0.0;
        std::cout << "[GPU] Flushed " << samples.size() << " samples, "
                  << bytes << " bytes in "
                  << (intervalNs / 1000000) << " ms ("
                  << mibPerSec << " MiB/s), " << totalFlushed << " total\n";
    }
}

} // namespace internal
} // namespace cupti_profiler
