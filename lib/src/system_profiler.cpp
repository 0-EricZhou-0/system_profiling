#include <cupti_profiler/system_profiler.h>

#include "proc_readers.h"
#include "system_flush_thread.h"

#include "system_metrics.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace cupti_profiler {

class SystemProfiler::Impl {
public:
    SystemProfilerConfig config;
    std::string hostname;

    // Sample accumulation
    internal::SystemSampleBatch batch;
    std::mutex batchMutex;

    // Output
    std::ofstream outFile;
    std::mutex outMutex;

    // Threads
    std::thread sampleThread;
    std::thread flushThread;
    std::atomic<bool> stopSample{false};
    std::atomic<bool> stopFlush{false};

    // Sync anchor
    uint64_t steadyClockRefNs = 0;
    uint64_t wallClockEpochNs = 0;

    bool configured = false;
    bool running = false;

    // Previous snapshots for delta computation
    internal::CPUStatSnapshot prevCPU;
    std::unordered_map<uint32_t, internal::PIDStatSnapshot> prevPID;

    // Per-flush write accounting
    internal::SystemPendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;
};

SystemProfiler::SystemProfiler() : m_impl(std::make_unique<Impl>()) {}
SystemProfiler::~SystemProfiler() {
    if (m_impl && m_impl->running) Stop();
}
SystemProfiler::SystemProfiler(SystemProfiler&&) noexcept = default;
SystemProfiler& SystemProfiler::operator=(SystemProfiler&&) noexcept = default;

void SystemProfiler::Configure(const SystemProfilerConfig& config) {
    m_impl->config = config;

    char buf[256];
    gethostname(buf, sizeof(buf));
    m_impl->hostname = buf;

    std::cout << "[System] Sampling frequency: " << config.samplingFrequencyHz << " Hz\n";
    std::cout << "[System] Tracking " << config.Processes.size() << " PID(s)\n";

    m_impl->configured = true;
}

void SystemProfiler::Start() {
    if (!m_impl->configured) {
        std::cerr << "SystemProfiler::Start() called before Configure()\n";
        return;
    }

    if (!m_impl->config.outputFile.empty()) {
        m_impl->outFile.open(m_impl->config.outputFile, std::ios::binary | std::ios::trunc);
        if (!m_impl->outFile) {
            std::cerr << "Failed to open system output file: " << m_impl->config.outputFile << "\n";
            return;
        }
    }

    // Record sync anchor
    m_impl->steadyClockRefNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_impl->wallClockEpochNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Take initial snapshots for delta computation
    m_impl->prevCPU = internal::ReadCPUStat();
    for (const auto& p : m_impl->config.Processes) {
        m_impl->prevPID[p.pid] = internal::ReadPIDStat(p.pid);
    }

    // Launch sample thread
    m_impl->stopSample = false;
    m_impl->sampleThread = std::thread([this]() {
        auto& impl = *m_impl;
        long clkTck = internal::GetCLKTCK();
        long pageSize = internal::GetPageSize();

        while (!impl.stopSample) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000000 / impl.config.samplingFrequencyHz));
            if (impl.stopSample) break;

            auto now = std::chrono::steady_clock::now().time_since_epoch();
            uint64_t tsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

            // CPU system-wide
            auto curCPU = internal::ReadCPUStat();
            {
                uint64_t dTotal = curCPU.Total() - impl.prevCPU.Total();
                if (dTotal > 0) {
                    double scale = 100.0 / dTotal;
                    CPUSystemSample s;
                    s.set_timestamp_ns(tsNs);
                    s.set_total_utilization_pct((double)(curCPU.Busy() - impl.prevCPU.Busy()) * scale);
                    s.set_user_pct((double)(curCPU.user + curCPU.nice - impl.prevCPU.user - impl.prevCPU.nice) * scale);
                    s.set_system_pct((double)(curCPU.system - impl.prevCPU.system) * scale);
                    s.set_iowait_pct((double)(curCPU.iowait - impl.prevCPU.iowait) * scale);

                    std::lock_guard<std::mutex> lock(impl.batchMutex);
                    impl.batch.CPUSysSamples.push_back(std::move(s));
                }
                impl.prevCPU = curCPU;
            }

            // CPU per-process
            for (const auto& proc : impl.config.Processes) {
                uint32_t pid = proc.pid;
                auto curPID = internal::ReadPIDStat(pid);
                auto& prev = impl.prevPID[pid];

                double dtSec = 1.0 / impl.config.samplingFrequencyHz;
                CPUProcessSample s;
                s.set_timestamp_ns(tsNs);
                s.set_pid(pid);
                s.set_user_pct((double)(curPID.utime - prev.utime) / (dtSec * clkTck) * 100.0);
                s.set_system_pct((double)(curPID.stime - prev.stime) / (dtSec * clkTck) * 100.0);
                s.set_iowait_pct((double)(curPID.blkioTicks - prev.blkioTicks) / (dtSec * clkTck) * 100.0);
                prev = curPID;

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.CPUProcSamples.push_back(std::move(s));
            }

            // Memory system-wide
            {
                auto mem = internal::ReadMemInfo();
                MemorySystemSample s;
                s.set_timestamp_ns(tsNs);
                s.set_total_bytes(mem.totalKB * 1024);
                s.set_used_bytes((mem.totalKB - mem.freeKB - mem.buffersKB - mem.cachedKB) * 1024);
                s.set_available_bytes(mem.availableKB * 1024);
                s.set_buffers_bytes(mem.buffersKB * 1024);
                s.set_cached_bytes(mem.cachedKB * 1024);

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.memSysSamples.push_back(std::move(s));
            }

            // Memory per-process
            for (const auto& proc : impl.config.Processes) {
                uint32_t pid = proc.pid;
                auto statm = internal::ReadPIDStatm(pid);
                MemoryProcessSample s;
                s.set_timestamp_ns(tsNs);
                s.set_pid(pid);
                s.set_rss_bytes(statm.RSSPages * pageSize);
                s.set_vms_bytes(statm.VMSPages * pageSize);
                s.set_shared_bytes(statm.sharedPages * pageSize);

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.memProcSamples.push_back(std::move(s));
            }
        }
    });

    // Launch flush thread
    m_impl->stopFlush = false;
    if (m_impl->config.flushIntervalMs > 0 && m_impl->outFile.is_open()) {
        m_impl->flushThread = std::thread(internal::SystemFlushThreadFunc,
                                           std::ref(m_impl->batch),
                                           std::ref(m_impl->batchMutex),
                                           std::ref(m_impl->outFile),
                                           std::ref(m_impl->outMutex),
                                           std::cref(m_impl->hostname),
                                           m_impl->config.samplingFrequencyHz,
                                           std::cref(m_impl->config.Processes),
                                           std::ref(m_impl->stopFlush),
                                           m_impl->config.flushIntervalMs,
                                           m_impl->steadyClockRefNs,
                                           m_impl->wallClockEpochNs,
                                           std::ref(m_impl->flushStatsPending),
                                           std::ref(m_impl->flushStatsMutex));
    }

    m_impl->running = true;
    std::cout << "[System] Profiler started\n";
}

void SystemProfiler::SignalStop() {
    if (!m_impl->running) return;
    m_impl->stopSample = true;
    m_impl->stopFlush = true;
}

void SystemProfiler::Stop() {
    if (!m_impl->running) return;

    // Signal if not already signaled
    m_impl->stopSample = true;
    m_impl->stopFlush = true;

    if (m_impl->sampleThread.joinable()) m_impl->sampleThread.join();
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    // Write remaining samples
    if (m_impl->outFile.is_open()) {
        internal::SystemSampleBatch drained;
        {
            std::lock_guard<std::mutex> lock(m_impl->batchMutex);
            drained.CPUSysSamples.swap(m_impl->batch.CPUSysSamples);
            drained.CPUProcSamples.swap(m_impl->batch.CPUProcSamples);
            drained.memSysSamples.swap(m_impl->batch.memSysSamples);
            drained.memProcSamples.swap(m_impl->batch.memProcSamples);
        }

        if (!drained.CPUSysSamples.empty() || !drained.memSysSamples.empty() ||
            m_impl->flushStatsPending.valid) {
            SystemMetricsTrace trace;
            trace.set_hostname(m_impl->hostname);
            trace.set_sampling_frequency_hz(m_impl->config.samplingFrequencyHz);
            trace.set_steady_clock_reference_ns(m_impl->steadyClockRefNs);
            trace.set_wall_clock_epoch_ns(m_impl->wallClockEpochNs);
            for (const auto& p : m_impl->config.Processes) {
                auto* tp = trace.add_tracked_processes();
                tp->set_pid(p.pid);
                tp->set_alias(p.alias);
            }
            for (auto& s : drained.CPUSysSamples) *trace.add_cpu_system_samples() = std::move(s);
            for (auto& s : drained.CPUProcSamples) *trace.add_cpu_process_samples() = std::move(s);
            for (auto& s : drained.memSysSamples) *trace.add_memory_system_samples() = std::move(s);
            for (auto& s : drained.memProcSamples) *trace.add_memory_process_samples() = std::move(s);
            // Attach any pending flush stats from the last background flush cycle.
            if (m_impl->flushStatsPending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_timestamp_ns(m_impl->flushStatsPending.timestampNs);
                fs->set_bytes_written(m_impl->flushStatsPending.bytesWritten);
                fs->set_interval_ns(m_impl->flushStatsPending.intervalNs);
                m_impl->flushStatsPending.valid = false;
            }

            std::string serialized;
            trace.SerializeToString(&serialized);
            google::protobuf::io::OstreamOutputStream raw(&m_impl->outFile);
            google::protobuf::io::CodedOutputStream coded(&raw);
            coded.WriteVarint32(static_cast<uint32_t>(serialized.size()));
            coded.WriteString(serialized);
            m_impl->outFile.flush();
        }
        m_impl->outFile.close();
        std::cout << "[System] Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    m_impl->running = false;
}

} // namespace cupti_profiler
