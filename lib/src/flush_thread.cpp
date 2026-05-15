#include "flush_thread.h"

#include "gpu_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

namespace {

void PopulateHeader(GPUMetricsTrace& trace,
                    const std::string& hostname,
                    uint64_t samplingFrequencyHz,
                    uint32_t hostCpuCount,
                    uint64_t steadyClockRefNs,
                    uint64_t cuptiRefNs,
                    uint64_t wallClockEpochNs)
{
    auto* hdr = trace.mutable_header();
    hdr->set_hostname(hostname);
    hdr->set_sampling_frequency_hz(samplingFrequencyHz);
    hdr->set_host_cpu_count(hostCpuCount);
    auto* anchors = hdr->mutable_anchors();
    anchors->set_steady_clock_reference_ns(steadyClockRefNs);
    anchors->set_wall_clock_epoch_ns(wallClockEpochNs);
    anchors->set_cupti_reference_ns(cuptiRefNs);
}

} // namespace

GPUMetricsTrace BuildTrace(const std::string& hostname,
                           uint64_t samplingFrequencyHz,
                           uint32_t hostCpuCount,
                           const std::vector<const char*>& metricNames,
                           const std::vector<GpuDevicePayload>& devices,
                           uint64_t steadyClockRefNs,
                           uint64_t cuptiRefNs,
                           uint64_t wallClockEpochNs)
{
    GPUMetricsTrace trace;
    PopulateHeader(trace, hostname, samplingFrequencyHz, hostCpuCount,
                   steadyClockRefNs, cuptiRefNs, wallClockEpochNs);

    auto* smn = trace.add_scope_metric_names();
    smn->set_scope(SCOPE_GPU);
    for (const auto& name : metricNames) {
        smn->add_fqns(name);
    }

    for (const auto& d : devices) {
        auto* dev = trace.add_tracked_gpus();
        dev->set_device_index(d.gpu_index);
        dev->set_device_name(d.device_name);
        dev->set_chip_name(d.chip_name);
        dev->set_peak_dram_bw_bytes_per_s(d.peak_dram_bw_bytes_per_s);
        dev->set_peak_pcie_bw_bytes_per_s(d.peak_pcie_bw_bytes_per_s);
        dev->set_peak_nvlink_bw_bytes_per_s(d.peak_nvlink_bw_bytes_per_s);
        dev->set_max_warps_per_sm(d.max_warps_per_sm);

        for (const auto& s : d.samples) {
            auto* sample = trace.add_samples();
            sample->set_timestamp_ns(s.endTimestamp);
            sample->set_gpu_index(d.gpu_index);
            for (double v : s.metricValues) {
                sample->add_values(v);
            }
        }
    }

    return trace;
}

size_t WriteDelimitedToSized(const GPUMetricsTrace& trace, std::ofstream& out) {
    std::string serialized;
    if (!trace.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize GPUMetricsTrace\n";
        return 0;
    }
    google::protobuf::io::OstreamOutputStream raw(&out);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
    coded.WriteString(serialized);
    return serialized.size();
}

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
                     std::mutex& pendingMutex)
{
    size_t totalFlushed = 0;
    uint64_t prevFlushNs = 0;
    while (!stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(flushIntervalMs));
        if (stop) break;

        // Drain every device. Skip the flush if no device produced
        // anything this cycle.
        std::vector<GpuDevicePayload> payloads;
        payloads.reserve(devices.size());
        size_t totalSamples = 0;
        for (auto& slot : devices) {
            GpuDevicePayload p;
            p.gpu_index                  = slot.gpu_index;
            p.device_name                = slot.device_name;
            p.chip_name                  = slot.chip_name;
            p.peak_dram_bw_bytes_per_s   = *slot.peak_dram_bw_bytes_per_s;
            p.peak_pcie_bw_bytes_per_s   = *slot.peak_pcie_bw_bytes_per_s;
            p.peak_nvlink_bw_bytes_per_s = *slot.peak_nvlink_bw_bytes_per_s;
            p.max_warps_per_sm           = *slot.max_warps_per_sm;
            p.samples                    = slot.host->DrainSamples();
            totalSamples += p.samples.size();
            payloads.push_back(std::move(p));
        }
        if (totalSamples == 0) continue;

        GPUMetricsTrace trace = BuildTrace(
            hostname, samplingFrequencyHz, hostCpuCount,
            metricNames, payloads,
            steadyClockRefNs, cuptiRefNs, wallClockEpochNs);

        // Attach previous flush's stats (known only in hindsight).
        {
            std::lock_guard<std::mutex> lock(pendingMutex);
            if (pending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_flush_byte_size(pending.bytesWritten);
                fs->set_flush_interval_ns(pending.intervalNs);
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

        {
            std::lock_guard<std::mutex> lock(pendingMutex);
            pending.bytesWritten = bytes;
            pending.intervalNs   = intervalNs;
            pending.valid        = true;
        }

        totalFlushed += totalSamples;
        double mibPerSec = (intervalNs > 0)
            ? (double)bytes * 1e9 / ((double)intervalNs * 1024.0 * 1024.0) : 0.0;
        std::cout << "[GPU] Flushed " << totalSamples
                  << " samples across " << devices.size() << " device(s), "
                  << bytes << " bytes in "
                  << (intervalNs / 1000000) << " ms ("
                  << mibPerSec << " MiB/s), " << totalFlushed << " total\n";
    }
}

} // namespace internal
} // namespace cupti_profiler
