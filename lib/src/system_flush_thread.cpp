#include "system_flush_thread.h"

#include "system_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <array>
#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

// One descriptor per emitted column. The position in the array is the
// column index in the SystemSample.values[] / ProcessSample.values[]
// repeated field — AppendSystemSample / AppendProcessSample below
// iterate this array in order, calling each `.read` lambda against the
// tick struct. AddScopeRegistry walks the same array to emit the
// matching ScopeMetricNames entry, so the wire FQN order can never
// drift from the wire value order.
//
// metric_catalog_builtins.cpp also walks these arrays (via the
// GetSystemMetrics / GetProcessMetrics accessors below) to seed the
// in-memory MetricCatalog at profiler startup — this is the
// "descriptor lives next to the producer" half of the registry
// pattern. Adding a metric here adds it to both the trace AND the
// catalog with no second edit anywhere.

namespace {

inline constexpr std::array kSystemMetrics = {
    MetricDescriptor<SystemTick>{
        .fqn         = "cpu__cycles_busy.avg.pct_of_peak_sustained_elapsed",
        .type        = MetricType::Throughput,
        .entity      = "cpu",  .counter = "cycles_busy",
        .rollup      = "avg",  .submetric = "pct_of_peak_sustained_elapsed",
        .unit        = Unit::Pct,  .scope = Scope::System,
        .peak        = PeakConstant{100.0},
        .description = "Fraction of wall time the CPU was not idle/iowait. From /proc/stat.",
        .read        = [](const SystemTick& t){ return t.cpu_busy_pct; },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "cpu__cycles_user.avg.pct_of_peak_sustained_elapsed",
        .type        = MetricType::Throughput,
        .entity      = "cpu",  .counter = "cycles_user",
        .rollup      = "avg",  .submetric = "pct_of_peak_sustained_elapsed",
        .unit        = Unit::Pct,  .scope = Scope::System,
        .peak        = PeakConstant{100.0},
        .description = "Fraction of wall time the CPU was in userspace (user + nice).",
        .read        = [](const SystemTick& t){ return t.cpu_user_pct; },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "cpu__cycles_kernel.avg.pct_of_peak_sustained_elapsed",
        .type        = MetricType::Throughput,
        .entity      = "cpu",  .counter = "cycles_kernel",
        .rollup      = "avg",  .submetric = "pct_of_peak_sustained_elapsed",
        .unit        = Unit::Pct,  .scope = Scope::System,
        .peak        = PeakConstant{100.0},
        .description = "Fraction of wall time the CPU was in kernel (system + irq + softirq).",
        .read        = [](const SystemTick& t){ return t.cpu_kernel_pct; },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "cpu__cycles_iowait.avg.pct_of_peak_sustained_elapsed",
        .type        = MetricType::Throughput,
        .entity      = "cpu",  .counter = "cycles_iowait",
        .rollup      = "avg",  .submetric = "pct_of_peak_sustained_elapsed",
        .unit        = Unit::Pct,  .scope = Scope::System,
        .peak        = PeakConstant{100.0},
        .description = "Fraction of wall time the CPU was idle waiting for I/O.",
        .read        = [](const SystemTick& t){ return t.cpu_iowait_pct; },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "mem__capacity_bytes",
        .type        = MetricType::Counter,
        .entity      = "mem",  .counter = "capacity_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::System,
        .description = "Total physical memory installed. /proc/meminfo MemTotal. Static across a run; serves as the peak for *_used / *_available.",
        .read        = [](const SystemTick& t){ return static_cast<double>(t.mem_capacity_bytes); },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "mem__used_bytes",
        .type        = MetricType::Counter,
        .entity      = "mem",  .counter = "used_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::System,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "MemTotal − MemAvailable. The 'committed-for-real-use' figure.",
        .read        = [](const SystemTick& t){ return static_cast<double>(t.mem_used_bytes); },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "mem__available_bytes",
        .type        = MetricType::Counter,
        .entity      = "mem",  .counter = "available_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::System,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "/proc/meminfo MemAvailable — what the kernel believes it can give to new allocations without swap.",
        .read        = [](const SystemTick& t){ return static_cast<double>(t.mem_available_bytes); },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "mem__buffers_bytes",
        .type        = MetricType::Counter,
        .entity      = "mem",  .counter = "buffers_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::System,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "/proc/meminfo Buffers — block-device cache.",
        .read        = [](const SystemTick& t){ return static_cast<double>(t.mem_buffers_bytes); },
    },
    MetricDescriptor<SystemTick>{
        .fqn         = "mem__cached_bytes",
        .type        = MetricType::Counter,
        .entity      = "mem",  .counter = "cached_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::System,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "/proc/meminfo Cached — page cache.",
        .read        = [](const SystemTick& t){ return static_cast<double>(t.mem_cached_bytes); },
    },
};

inline constexpr std::array kProcessMetrics = {
    MetricDescriptor<ProcessTick>{
        .fqn         = "proc__cycles_active.sum.per_second",
        .type        = MetricType::Counter,
        .entity      = "proc",  .counter = "cycles_active",
        .rollup      = "sum",   .submetric = "per_second",
        .unit        = Unit::PctOfCore,  .scope = Scope::Process,
        .peak        = PeakExpr{"ncpus_x_100"},
        .description = "Per-PID on-CPU time as % of one core, aggregated across every thread of the process. Sum of /proc/<pid>/task/*/schedstat sum_exec_runtime deltas (ns) over actual wall-clock between ticks — nanosecond-precise, no CLK_TCK quantization. >100% means multi-core use; the panel peak_expr caps the axis at ncpus × 100.",
        .read        = [](const ProcessTick& t){ return t.cpu_pct; },
    },
    MetricDescriptor<ProcessTick>{
        .fqn         = "proc__rss_bytes",
        .type        = MetricType::Counter,
        .entity      = "proc",  .counter = "rss_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::Process,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "Resident set size — /proc/<pid>/status VmRSS. Physical pages owned by the PID.",
        .read        = [](const ProcessTick& t){ return static_cast<double>(t.rss_bytes); },
    },
    MetricDescriptor<ProcessTick>{
        .fqn         = "proc__vms_bytes",
        .type        = MetricType::Counter,
        .entity      = "proc",  .counter = "vms_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::Process,
        .description = "Virtual memory size — /proc/<pid>/status VmSize. Can exceed physical RAM (file-backed, overcommit).",
        .read        = [](const ProcessTick& t){ return static_cast<double>(t.vms_bytes); },
    },
    MetricDescriptor<ProcessTick>{
        .fqn         = "proc__shared_bytes",
        .type        = MetricType::Counter,
        .entity      = "proc",  .counter = "shared_bytes",
        .unit        = Unit::Bytes,  .scope = Scope::Process,
        .peak        = PeakRef{"mem__capacity_bytes"},
        .description = "Resident shared memory — /proc/<pid>/status RssShmem.",
        .read        = [](const ProcessTick& t){ return static_cast<double>(t.shared_bytes); },
    },
};

} // namespace

std::span<const MetricDescriptor<SystemTick>>  GetSystemMetrics()  { return kSystemMetrics;  }
std::span<const MetricDescriptor<ProcessTick>> GetProcessMetrics() { return kProcessMetrics; }

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
    for (const auto& d : kSystemMetrics) sys->add_fqns(std::string(d.fqn));

    auto* proc = trace.add_scope_metric_names();
    proc->set_scope(SCOPE_PROCESS);
    for (const auto& d : kProcessMetrics) proc->add_fqns(std::string(d.fqn));
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
    for (const auto& d : kSystemMetrics) s->add_values(d.read(t));
}

void AppendProcessSample(SystemMetricsTrace& trace, const ProcessTick& t) {
    auto* s = trace.add_process_samples();
    s->set_timestamp_ns(t.timestamp_ns);
    s->set_pid(t.pid);
    for (const auto& d : kProcessMetrics) s->add_values(d.read(t));
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
