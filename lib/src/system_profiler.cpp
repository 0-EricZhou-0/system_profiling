#include <cupti_profiler/system_profiler.h>

#include "proc_readers.h"
#include "system_flush_thread.h"

#include "system_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace cupti_profiler {

class SystemProfiler::Impl {
public:
    SystemProfilerConfig config;
    std::string hostname;
    uint32_t hostCpuCount = 0;

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

void SystemProfiler::Configure(const SystemProfilerConfig& config) {
    m_impl->config = config;

    char buf[256];
    gethostname(buf, sizeof(buf));
    m_impl->hostname = buf;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    m_impl->hostCpuCount = (nproc > 0) ? static_cast<uint32_t>(nproc) : 0;

    // Seed the ProcessTrackingProbe with the configured PIDs. Mid-run
    // Add/Remove calls layer on top.
    std::vector<ProcessTrackingProbe::ProcessEntry> seed;
    seed.reserve(config.Processes.size());
    for (const auto& p : config.Processes) {
        seed.push_back({p.pid, p.alias, /*pending_removal=*/false});
    }
    SetInitialProcesses(std::move(seed));

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

    // Take initial CPU snapshot for delta computation. Per-PID
    // baselines are seeded lazily on the first iteration the PID is
    // seen — that way mid-run AddTrackedProcess() works without a
    // separate hook.
    m_impl->prevCPU = internal::ReadCPUStat();

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

            // System-wide tick — CPU + memory in one Sample.
            auto curCPU = internal::ReadCPUStat();
            auto mem    = internal::ReadMemInfo();
            {
                uint64_t dTotal = curCPU.Total() - impl.prevCPU.Total();
                if (dTotal > 0) {
                    double scale = 100.0 / dTotal;
                    internal::SystemTick t;
                    t.timestamp_ns        = tsNs;
                    t.cpu_busy_pct        = (double)(curCPU.Busy() - impl.prevCPU.Busy()) * scale;
                    t.cpu_user_pct        = (double)(curCPU.user + curCPU.nice - impl.prevCPU.user - impl.prevCPU.nice) * scale;
                    t.cpu_kernel_pct      = (double)(curCPU.system - impl.prevCPU.system) * scale;
                    t.cpu_iowait_pct      = (double)(curCPU.iowait - impl.prevCPU.iowait) * scale;
                    t.mem_capacity_bytes  = mem.totalKB * 1024;
                    t.mem_used_bytes      = (mem.totalKB - mem.freeKB - mem.buffersKB - mem.cachedKB) * 1024;
                    t.mem_available_bytes = mem.availableKB * 1024;
                    t.mem_buffers_bytes   = mem.buffersKB * 1024;
                    t.mem_cached_bytes    = mem.cachedKB * 1024;

                    std::lock_guard<std::mutex> lock(impl.batchMutex);
                    impl.batch.systemTicks.push_back(std::move(t));
                }
                impl.prevCPU = curCPU;
            }

            // Per-PID tick — CPU + memory in one ProcessSample. The
            // tracked-PID set is whatever ProcessTrackingProbe holds
            // right now (config + any mid-run Add/Remove). Entries with
            // pending_removal=true are skipped — they're awaiting the
            // next flush to be emitted as a removal marker.
            auto snapshot = this->SnapshotProcesses();
            std::unordered_set<uint32_t> snapshotPids;
            snapshotPids.reserve(snapshot.size());

            double dtSec = 1.0 / impl.config.samplingFrequencyHz;
            for (const auto& entry : snapshot) {
                uint32_t pid = entry.pid;
                snapshotPids.insert(pid);
                if (entry.pending_removal) continue;

                auto curPID = internal::ReadPIDStat(pid);
                auto it     = impl.prevPID.find(pid);
                if (it == impl.prevPID.end()) {
                    // Mid-run add — seed the baseline; skip this tick.
                    // First emitted sample is one tick later, so the
                    // delta isn't garbage.
                    impl.prevPID[pid] = curPID;
                    continue;
                }
                auto& prev  = it->second;
                auto statm  = internal::ReadPIDStatm(pid);

                internal::ProcessTick t;
                t.timestamp_ns   = tsNs;
                t.pid            = pid;
                t.cpu_user_pct   = (double)(curPID.utime - prev.utime) / (dtSec * clkTck) * 100.0;
                t.cpu_kernel_pct = (double)(curPID.stime - prev.stime) / (dtSec * clkTck) * 100.0;
                t.cpu_iowait_pct = (double)(curPID.blkioTicks - prev.blkioTicks) / (dtSec * clkTck) * 100.0;
                t.rss_bytes      = statm.RSSPages    * pageSize;
                t.vms_bytes      = statm.VMSPages    * pageSize;
                t.shared_bytes   = statm.sharedPages * pageSize;
                prev = curPID;

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.processTicks.push_back(std::move(t));
            }
            // Drop baselines for PIDs that are no longer in the
            // snapshot (committed removals).
            for (auto it = impl.prevPID.begin(); it != impl.prevPID.end(); ) {
                if (snapshotPids.find(it->first) == snapshotPids.end()) {
                    it = impl.prevPID.erase(it);
                } else {
                    ++it;
                }
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
                                           m_impl->hostCpuCount,
                                           std::ref(static_cast<ProcessTrackingProbe&>(*this)),
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
            drained.systemTicks.swap(m_impl->batch.systemTicks);
            drained.processTicks.swap(m_impl->batch.processTicks);
        }

        auto processSnapshot = SnapshotProcesses();
        if (!drained.systemTicks.empty() || !drained.processTicks.empty() ||
            m_impl->flushStatsPending.valid) {
            SystemMetricsTrace trace = internal::BuildSystemTrace(
                m_impl->hostname, m_impl->config.samplingFrequencyHz,
                m_impl->hostCpuCount,
                m_impl->steadyClockRefNs, m_impl->wallClockEpochNs,
                processSnapshot, drained);
            // Attach any pending flush stats from the last background flush cycle.
            if (m_impl->flushStatsPending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_flush_byte_size(m_impl->flushStatsPending.bytesWritten);
                fs->set_flush_interval_ns(m_impl->flushStatsPending.intervalNs);
                m_impl->flushStatsPending.valid = false;
            }

            internal::WriteDelimitedSystemTraceSized(trace, m_impl->outFile);
            m_impl->outFile.flush();
            CommitPendingRemovals();
        }
        m_impl->outFile.close();
        std::cout << "[System] Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    m_impl->running = false;
}

} // namespace cupti_profiler
