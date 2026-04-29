#include "system_flush_thread.h"

#include "system_metrics.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace cupti_profiler {
namespace internal {

size_t WriteDelimitedSystemTraceSized(const SystemMetricsTrace& trace, std::ofstream& out) {
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
                           const std::vector<TrackedProcess>& Processes,
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
            drained.CPUSysSamples.swap(batch.CPUSysSamples);
            drained.CPUProcSamples.swap(batch.CPUProcSamples);
            drained.memSysSamples.swap(batch.memSysSamples);
            drained.memProcSamples.swap(batch.memProcSamples);
        }

        if (drained.CPUSysSamples.empty() && drained.memSysSamples.empty()) continue;

        SystemMetricsTrace trace;
        trace.set_hostname(hostname);
        trace.set_sampling_frequency_hz(samplingFrequencyHz);
        trace.set_steady_clock_reference_ns(steadyClockRefNs);
        trace.set_wall_clock_epoch_ns(wallClockEpochNs);
        for (const auto& p : Processes) {
            auto* tp = trace.add_tracked_processes();
            tp->set_pid(p.pid);
            tp->set_alias(p.alias);
        }

        for (auto& s : drained.CPUSysSamples) *trace.add_cpu_system_samples() = std::move(s);
        for (auto& s : drained.CPUProcSamples) *trace.add_cpu_process_samples() = std::move(s);
        for (auto& s : drained.memSysSamples) *trace.add_memory_system_samples() = std::move(s);
        for (auto& s : drained.memProcSamples) *trace.add_memory_process_samples() = std::move(s);

        // Attach previous flush's stats
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
            bytes = WriteDelimitedSystemTraceSized(trace, outFile);
            outFile.flush();
        }
        uint64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t intervalNs = (prevFlushNs == 0) ? 0 : (nowNs - prevFlushNs);
        prevFlushNs = nowNs;

        {
            std::lock_guard<std::mutex> lock(pendingMutex);
            pending.timestampNs = nowNs;
            pending.bytesWritten = bytes;
            pending.intervalNs = intervalNs;
            pending.valid = true;
        }

        size_t n = drained.CPUSysSamples.size();
        totalFlushed += n;
        // bytes / (intervalNs / 1e9 sec) / 1024 = KiB/s
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
