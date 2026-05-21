#include "system_flush_thread.h"

#include "system_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

// Hardcoded column orders matching the SystemTick / ProcessTick struct
// field orders. These match the descriptors declared in
// configs/metric_catalog.pbtxt; the catalog wiring commit will pull
// these from the loaded MetricCatalog instead.
static const char* const kSystemFqns[] = {
    "cpu__cycles_busy.avg.pct_of_peak_sustained_elapsed",
    "cpu__cycles_user.avg.pct_of_peak_sustained_elapsed",
    "cpu__cycles_kernel.avg.pct_of_peak_sustained_elapsed",
    "cpu__cycles_iowait.avg.pct_of_peak_sustained_elapsed",
    "mem__capacity_bytes",
    "mem__used_bytes",
    "mem__available_bytes",
    "mem__buffers_bytes",
    "mem__cached_bytes",
};
static const char* const kProcessFqns[] = {
    "proc__cycles_active.sum.per_second",
    "proc__rss_bytes",
    "proc__vms_bytes",
    "proc__shared_bytes",
};

namespace {

void PopulateHeader(SystemMetricsTrace& trace,
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

void AddScopeRegistry(SystemMetricsTrace& trace) {
    auto* sys = trace.add_scope_metric_names();
    sys->set_scope(SCOPE_SYSTEM);
    for (const char* f : kSystemFqns) sys->add_fqns(f);

    auto* proc = trace.add_scope_metric_names();
    proc->set_scope(SCOPE_PROCESS);
    for (const char* f : kProcessFqns) proc->add_fqns(f);
}

void AddTrackedProcesses(
    SystemMetricsTrace& trace,
    const std::vector<ProcessTrackingProbe::ProcessEntry>& processes)
{
    for (const auto& p : processes) {
        auto* tp = trace.add_tracked_processes();
        tp->set_pid(p.pid);
        tp->set_alias(p.alias);
        tp->set_removed(p.pending_removal);
    }
}

void AppendSystemSample(SystemMetricsTrace& trace, const SystemTick& t) {
    auto* s = trace.add_system_samples();
    s->set_timestamp_ns(t.timestamp_ns);
    s->add_values(t.cpu_busy_pct);
    s->add_values(t.cpu_user_pct);
    s->add_values(t.cpu_kernel_pct);
    s->add_values(t.cpu_iowait_pct);
    s->add_values(static_cast<double>(t.mem_capacity_bytes));
    s->add_values(static_cast<double>(t.mem_used_bytes));
    s->add_values(static_cast<double>(t.mem_available_bytes));
    s->add_values(static_cast<double>(t.mem_buffers_bytes));
    s->add_values(static_cast<double>(t.mem_cached_bytes));
}

void AppendProcessSample(SystemMetricsTrace& trace, const ProcessTick& t) {
    auto* s = trace.add_process_samples();
    s->set_timestamp_ns(t.timestamp_ns);
    s->set_pid(t.pid);
    s->add_values(t.cpu_pct);
    s->add_values(static_cast<double>(t.rss_bytes));
    s->add_values(static_cast<double>(t.vms_bytes));
    s->add_values(static_cast<double>(t.shared_bytes));
}

} // namespace

SystemMetricsTrace BuildSystemTrace(
    const std::string& hostname,
    uint64_t samplingFrequencyHz,
    uint32_t hostCpuCount,
    uint64_t steadyClockRefNs,
    uint64_t wallClockEpochNs,
    const std::vector<ProcessTrackingProbe::ProcessEntry>& processes,
    const SystemSampleBatch& drained)
{
    SystemMetricsTrace trace;
    PopulateHeader(trace, hostname, samplingFrequencyHz, hostCpuCount,
                   steadyClockRefNs, wallClockEpochNs);
    AddScopeRegistry(trace);
    AddTrackedProcesses(trace, processes);
    for (const auto& t : drained.systemTicks)  AppendSystemSample(trace, t);
    for (const auto& t : drained.processTicks) AppendProcessSample(trace, t);
    return trace;
}

size_t WriteDelimitedSystemTraceSized(const SystemMetricsTrace& trace,
                                      std::ofstream& out)
{
    std::string serialized;
    if (!trace.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize SystemMetricsTrace\n";
        return 0;
    }
    google::protobuf::io::OstreamOutputStream raw(&out);
    google::protobuf::io::CodedOutputStream coded(&raw);
    coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
    coded.WriteString(serialized);
    return serialized.size();
}

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
                           std::mutex& pendingMutex)
{
    size_t totalFlushed = 0;
    uint64_t prevFlushNs = 0;
    while (!stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(flushIntervalMs));
        if (stop) break;

        SystemSampleBatch drained;
        {
            std::lock_guard<std::mutex> lock(batchMutex);
            drained.systemTicks.swap(batch.systemTicks);
            drained.processTicks.swap(batch.processTicks);
        }

        auto processSnapshot = probe.SnapshotProcesses();
        if (drained.systemTicks.empty() && drained.processTicks.empty()) continue;

        SystemMetricsTrace trace = BuildSystemTrace(
            hostname, samplingFrequencyHz, hostCpuCount,
            steadyClockRefNs, wallClockEpochNs,
            processSnapshot, drained);

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
            bytes = WriteDelimitedSystemTraceSized(trace, outFile);
            outFile.flush();
        }
        // Now that the removed=true markers have been written, drop
        // those entries so subsequent flushes don't keep emitting them.
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

        size_t n = drained.systemTicks.size();
        totalFlushed += n;
        double kibPerSec = (intervalNs > 0)
            ? (double)bytes * 1e9 / ((double)intervalNs * 1024.0) : 0.0;
        std::cout << "[System] Flushed " << n << " samples, "
                  << bytes << " bytes in "
                  << (intervalNs / 1000000) << " ms ("
                  << kibPerSec << " KiB/s), " << totalFlushed << " total\n";
    }
}

} // namespace internal
} // namespace cupti_profiler
