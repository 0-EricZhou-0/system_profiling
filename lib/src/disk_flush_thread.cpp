#include "disk_flush_thread.h"

#include "disk_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

static const char* const kDeviceFqns[] = {
    "disk__read_bytes.sum.per_second",
    "disk__write_bytes.sum.per_second",
    "disk__read_inflight",
    "disk__write_inflight",
};
static const char* const kProcessFqns[] = {
    "proc__io_rchar.sum.per_second",
    "proc__io_wchar.sum.per_second",
};

namespace {

void PopulateHeader(DiskMetricsTrace& trace,
                    const std::string& hostname,
                    uint64_t samplingFrequencyHz,
                    uint32_t hostCpuCount,
                    uint64_t steadyClockRefNs,
                    uint64_t wallClockEpochNs)
{
    auto* hdr = trace.mutable_header();
    hdr->set_hostname(hostname);
    hdr->set_sampling_frequency_hz(samplingFrequencyHz);
    hdr->set_host_cpu_count(hostCpuCount);
    auto* anchors = hdr->mutable_anchors();
    anchors->set_steady_clock_reference_ns(steadyClockRefNs);
    anchors->set_wall_clock_epoch_ns(wallClockEpochNs);
}

void AddScopeRegistry(DiskMetricsTrace& trace) {
    auto* dev = trace.add_scope_metric_names();
    dev->set_scope(SCOPE_DEVICE);
    for (const char* f : kDeviceFqns) dev->add_fqns(f);

    auto* proc = trace.add_scope_metric_names();
    proc->set_scope(SCOPE_PROCESS);
    for (const char* f : kProcessFqns) proc->add_fqns(f);
}

void AppendDeviceSample(DiskMetricsTrace& trace, const DiskDeviceTick& t) {
    auto* s = trace.add_device_samples();
    s->set_timestamp_ns(t.timestamp_ns);
    s->set_device_name(t.device_name);
    s->add_values(t.read_bytes_per_sec);
    s->add_values(t.write_bytes_per_sec);
    s->add_values(static_cast<double>(t.read_inflight));
    s->add_values(static_cast<double>(t.write_inflight));
}

void AppendProcessSample(DiskMetricsTrace& trace, const DiskProcessTick& t) {
    auto* s = trace.add_process_samples();
    s->set_timestamp_ns(t.timestamp_ns);
    s->set_pid(t.pid);
    s->add_values(t.rchar_bytes_per_sec);
    s->add_values(t.wchar_bytes_per_sec);
}

} // namespace

DiskMetricsTrace BuildDiskTrace(
    const std::string& hostname,
    uint64_t samplingFrequencyHz,
    uint32_t hostCpuCount,
    uint64_t steadyClockRefNs,
    uint64_t wallClockEpochNs,
    const std::vector<std::string>& devices,
    const std::vector<ProcessTrackingProbe::ProcessEntry>& processes,
    const DiskSampleBatch& drained)
{
    DiskMetricsTrace trace;
    PopulateHeader(trace, hostname, samplingFrequencyHz, hostCpuCount,
                   steadyClockRefNs, wallClockEpochNs);
    AddScopeRegistry(trace);
    for (const auto& d : devices) trace.add_tracked_devices(d);
    for (const auto& p : processes) {
        auto* tp = trace.add_tracked_processes();
        tp->set_pid(p.pid);
        tp->set_alias(p.alias);
        tp->set_removed(p.pending_removal);
    }
    for (const auto& t : drained.deviceTicks)  AppendDeviceSample(trace, t);
    for (const auto& t : drained.processTicks) AppendProcessSample(trace, t);
    return trace;
}

size_t WriteDelimitedDiskTraceSized(const DiskMetricsTrace& trace,
                                    std::ofstream& out)
{
    std::string serialized;
    if (!trace.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize DiskMetricsTrace\n";
        return 0;
    }
    google::protobuf::io::OstreamOutputStream raw(&out);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
    coded.WriteString(serialized);
    return serialized.size();
}

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
                         std::mutex& pendingMutex)
{
    size_t totalFlushed = 0;
    uint64_t prevFlushNs = 0;
    while (!stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(flushIntervalMs));
        if (stop) break;

        DiskSampleBatch drained;
        {
            std::lock_guard<std::mutex> lock(batchMutex);
            drained.deviceTicks.swap(batch.deviceTicks);
            drained.processTicks.swap(batch.processTicks);
        }

        auto processSnapshot = probe.SnapshotProcesses();
        if (drained.deviceTicks.empty() && drained.processTicks.empty()) continue;

        DiskMetricsTrace trace = BuildDiskTrace(
            hostname, samplingFrequencyHz, hostCpuCount,
            steadyClockRefNs, wallClockEpochNs,
            devices, processSnapshot, drained);

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
            bytes = WriteDelimitedDiskTraceSized(trace, outFile);
            outFile.flush();
        }
        probe.CommitPendingRemovals();

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

        size_t n = drained.deviceTicks.size();
        totalFlushed += n;
        double kibPerSec = (intervalNs > 0)
            ? (double)bytes * 1e9 / ((double)intervalNs * 1024.0) : 0.0;
        std::cout << "[Disk] Flushed " << n << " device samples, "
                  << bytes << " bytes in "
                  << (intervalNs / 1000000) << " ms ("
                  << kibPerSec << " KiB/s), " << totalFlushed << " total\n";
    }
}

} // namespace internal
} // namespace cupti_profiler
