#include <cupti_profiler/disk_profiler.h>

#include "disk_readers.h"
#include "disk_flush_thread.h"

#include "disk_metrics.pb.h"
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

class DiskProfiler::Impl {
public:
    DiskProfilerConfig config;
    std::string hostname;

    internal::DiskSampleBatch batch;
    std::mutex batchMutex;

    std::ofstream outFile;
    std::mutex outMutex;

    std::thread sampleThread;
    std::thread flushThread;
    std::atomic<bool> stopSample{false};
    std::atomic<bool> stopFlush{false};

    uint64_t steadyClockRefNs = 0;
    uint64_t wallClockEpochNs = 0;

    bool configured = false;
    bool running = false;

    // Previous snapshots
    std::unordered_map<std::string, internal::DiskStatSnapshot> prevDisk;
    std::unordered_map<uint32_t, internal::PIDIOSnapshot> prevPIDIO;
    std::unordered_set<uint32_t> warnedPIDs; // PIDs we've already warned about EACCES

    // Per-flush write accounting
    internal::DiskPendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;
};

DiskProfiler::DiskProfiler() : m_impl(std::make_unique<Impl>()) {}
DiskProfiler::~DiskProfiler() {
    if (m_impl && m_impl->running) Stop();
}
DiskProfiler::DiskProfiler(DiskProfiler&&) noexcept = default;
DiskProfiler& DiskProfiler::operator=(DiskProfiler&&) noexcept = default;

void DiskProfiler::Configure(const DiskProfilerConfig& config) {
    m_impl->config = config;

    char buf[256];
    gethostname(buf, sizeof(buf));
    m_impl->hostname = buf;

    std::cout << "[Disk] Tracking " << config.devices.size() << " device(s): ";
    for (const auto& d : config.devices) std::cout << d << " ";
    std::cout << "\n";
    std::cout << "[Disk] Tracking " << config.Processes.size() << " PID(s)\n";

    m_impl->configured = true;
}

void DiskProfiler::Start() {
    if (!m_impl->configured) {
        std::cerr << "DiskProfiler::Start() called before Configure()\n";
        return;
    }

    if (!m_impl->config.outputFile.empty()) {
        m_impl->outFile.open(m_impl->config.outputFile, std::ios::binary | std::ios::trunc);
        if (!m_impl->outFile) {
            std::cerr << "Failed to open disk output file: " << m_impl->config.outputFile << "\n";
            return;
        }
    }

    // Record sync anchor
    m_impl->steadyClockRefNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_impl->wallClockEpochNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Initial snapshots
    auto initDisk = internal::ReadDiskStats(m_impl->config.devices);
    for (auto& ds : initDisk) m_impl->prevDisk[ds.device] = ds;
    for (const auto& proc : m_impl->config.Processes) {
        uint32_t pid = proc.pid;
        auto io = internal::ReadPIDIO(pid);
        if (!io.accessible) {
            std::cerr << "[Disk] Warning: cannot read /proc/" << pid << "/io (permission denied).\n"
                      << "  Per-process disk IO will be skipped for this PID.\n"
                      << "  To fix, either:\n"
                      << "    sudo setcap cap_sys_ptrace=ep <your_binary>\n"
                      << "    sudo sysctl -w kernel.yama.ptrace_scope=0\n";
            m_impl->warnedPIDs.insert(pid);
        }
        m_impl->prevPIDIO[pid] = io;
    }

    m_impl->stopSample = false;
    m_impl->sampleThread = std::thread([this]() {
        auto& impl = *m_impl;

        while (!impl.stopSample) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000000 / impl.config.samplingFrequencyHz));
            if (impl.stopSample) break;

            auto now = std::chrono::steady_clock::now().time_since_epoch();
            uint64_t tsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
            double dtSec = 1.0 / impl.config.samplingFrequencyHz;

            // Per-device stats
            auto curDisk = internal::ReadDiskStats(impl.config.devices);
            for (auto& ds : curDisk) {
                auto it = impl.prevDisk.find(ds.device);
                if (it == impl.prevDisk.end()) {
                    impl.prevDisk[ds.device] = ds;
                    continue;
                }
                auto& prev = it->second;

                auto inflight = internal::ReadDiskInflight(ds.device);

                DiskDeviceSample s;
                s.set_timestamp_ns(tsNs);
                s.set_device_name(ds.device);
                s.set_read_bytes_per_sec((double)(ds.sectorsRead - prev.sectorsRead) * 512.0 / dtSec);
                s.set_write_bytes_per_sec((double)(ds.sectorsWritten - prev.sectorsWritten) * 512.0 / dtSec);
                s.set_read_queue_depth(inflight.readInflight);
                s.set_write_queue_depth(inflight.writeInflight);
                prev = ds;

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.deviceSamples.push_back(std::move(s));
            }

            // Per-process I/O
            for (const auto& proc : impl.config.Processes) {
                uint32_t pid = proc.pid;
                auto curIO = internal::ReadPIDIO(pid);
                if (!curIO.accessible) {
                    if (impl.warnedPIDs.find(pid) == impl.warnedPIDs.end()) {
                        std::cerr << "[Disk] Warning: cannot read /proc/" << pid << "/io (permission denied). "
                                  << "Skipping per-process disk IO for this PID.\n";
                        impl.warnedPIDs.insert(pid);
                    }
                    continue;
                }

                auto it = impl.prevPIDIO.find(pid);
                if (it == impl.prevPIDIO.end()) {
                    impl.prevPIDIO[pid] = curIO;
                    continue;
                }
                auto& prev = it->second;

                DiskProcessSample s;
                s.set_timestamp_ns(tsNs);
                s.set_pid(pid);
                s.set_read_bytes_per_sec((double)(curIO.readBytes - prev.readBytes) / dtSec);
                s.set_write_bytes_per_sec((double)(curIO.writeBytes - prev.writeBytes) / dtSec);
                prev = curIO;

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.procSamples.push_back(std::move(s));
            }
        }
    });

    // Launch flush thread
    m_impl->stopFlush = false;
    if (m_impl->config.flushIntervalMs > 0 && m_impl->outFile.is_open()) {
        m_impl->flushThread = std::thread(internal::DiskFlushThreadFunc,
                                           std::ref(m_impl->batch),
                                           std::ref(m_impl->batchMutex),
                                           std::ref(m_impl->outFile),
                                           std::ref(m_impl->outMutex),
                                           std::cref(m_impl->hostname),
                                           m_impl->config.samplingFrequencyHz,
                                           std::cref(m_impl->config.devices),
                                           std::cref(m_impl->config.Processes),
                                           std::ref(m_impl->stopFlush),
                                           m_impl->config.flushIntervalMs,
                                           m_impl->steadyClockRefNs,
                                           m_impl->wallClockEpochNs,
                                           std::ref(m_impl->flushStatsPending),
                                           std::ref(m_impl->flushStatsMutex));
    }

    m_impl->running = true;
    std::cout << "[Disk] Profiler started\n";
}

void DiskProfiler::SignalStop() {
    if (!m_impl->running) return;
    m_impl->stopSample = true;
    m_impl->stopFlush = true;
}

void DiskProfiler::Stop() {
    if (!m_impl->running) return;

    m_impl->stopSample = true;
    m_impl->stopFlush = true;

    if (m_impl->sampleThread.joinable()) m_impl->sampleThread.join();
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    // Write remaining
    if (m_impl->outFile.is_open()) {
        internal::DiskSampleBatch drained;
        {
            std::lock_guard<std::mutex> lock(m_impl->batchMutex);
            drained.deviceSamples.swap(m_impl->batch.deviceSamples);
            drained.procSamples.swap(m_impl->batch.procSamples);
        }

        if (!drained.deviceSamples.empty() || !drained.procSamples.empty() ||
            m_impl->flushStatsPending.valid) {
            DiskMetricsTrace trace;
            trace.set_hostname(m_impl->hostname);
            trace.set_sampling_frequency_hz(m_impl->config.samplingFrequencyHz);
            trace.set_steady_clock_reference_ns(m_impl->steadyClockRefNs);
            trace.set_wall_clock_epoch_ns(m_impl->wallClockEpochNs);
            for (const auto& d : m_impl->config.devices) trace.add_tracked_devices(d);
            for (const auto& p : m_impl->config.Processes) {
                auto* tp = trace.add_tracked_processes();
                tp->set_pid(p.pid);
                tp->set_alias(p.alias);
            }
            for (auto& s : drained.deviceSamples) *trace.add_device_samples() = std::move(s);
            for (auto& s : drained.procSamples) *trace.add_process_samples() = std::move(s);
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
        std::cout << "[Disk] Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    m_impl->running = false;
}

} // namespace cupti_profiler
