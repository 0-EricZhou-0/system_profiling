#include <cupti_profiler/system_profiler.h>

#include "proc_readers.h"
#include "system_flush_thread.h"
#include "discovery_stats_proto.h"
#include "testing_hooks.h"

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
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

    // Per-tracked-PID baseline carried between sample ticks. tickTsNs
    // pins the actual wall-clock instant of the previous read (so the
    // %-of-core denominator uses real elapsed time, not a nominal
    // sample period). cpuNs is the process CPU clock at that read —
    // on-CPU time of the whole thread group, exited threads included.
    // serial is the tracked entry's (ProcessEntry::serial): a new
    // process that got an old one's PID never inherits its baseline.
    struct ProcessBaseline {
        uint64_t tickTsNs = 0;
        uint64_t cpuNs = 0;
        uint64_t serial = 0;
    };
    std::unordered_map<uint32_t, ProcessBaseline> prevPID;

    // Exit tails of discovered processes (CpuTail). Sample thread only.
    //   discoveredParent: every discovered PID seen in a snapshot and not
    //     yet exited -> the parent it was found under.
    //   awaitingTail: discovered PIDs that exited (or left the tracked
    //     set) and whose parent has not reaped them yet, with the CPU
    //     already accounted for: the clock at their last sample, and
    //     their own reaped children's CPU as last read.
    //   childrenCpuNs: last cutime+cstime (ns) of every tracked process
    //     that currently has discovered children — the baseline the
    //     reaping parent's growth is measured against — and of every
    //     discovered process, whose own tail must subtract the CPU of the
    //     children it reaped (its cutime), even once they are all gone.
    struct ExitedChild {
        uint32_t parent            = 0;
        uint64_t lastCpuNs         = 0;
        uint64_t lastChildrenCpuNs = 0;
    };
    struct ParentBaseline {
        uint64_t childrenCpuNs = 0;
        uint64_t startTime     = 0;   // guards against a reused PID
    };
    std::unordered_map<uint32_t, uint32_t>       discoveredParent;
    std::unordered_map<uint32_t, ExitedChild>    awaitingTail;
    std::unordered_map<uint32_t, ParentBaseline> childrenCpu;
    //   tailSettled: discovered PIDs whose tail was emitted or given up
    //     on. Such a PID stays in the snapshot until discovery's removal
    //     is flushed; it must not be noted as exiting (and tailed) again.
    std::unordered_set<uint32_t>                 tailSettled;

    void NoteExited(uint32_t pid);
    void AttributeTails(uint64_t tsNs,
                        const std::unordered_set<uint32_t>& trackedPids);

    // Per-flush write accounting
    internal::SystemPendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;
};

// A discovered process is gone from the samples (exited, unreadable, or
// dropped from the tracked set): remember what was accounted for, so the
// parent's cutime growth at reap time can be turned into its tail.
void SystemProfiler::Impl::NoteExited(uint32_t pid) {
    auto dp = discoveredParent.find(pid);
    if (dp == discoveredParent.end()) return;   // a root, or already noted
    ExitedChild c;
    c.parent = dp->second;
    discoveredParent.erase(dp);
    if (auto b = prevPID.find(pid); b != prevPID.end()) c.lastCpuNs = b->second.cpuNs;
    if (auto k = childrenCpu.find(pid); k != childrenCpu.end()) c.lastChildrenCpuNs = k->second.childrenCpuNs;
    // Reparented since discovery (its parent exited)? Then the new parent
    // reaps it; follow it if that one is tracked too.
    if (auto st = internal::ReadProcStat("/proc", pid); st && st->ppid != 0) c.parent = st->ppid;
    awaitingTail[pid] = c;
}

// Once per tick. For each tracked process with discovered children: read
// its cutime+cstime; if it grew and some of its exited children are now
// reaped (their /proc entry is gone), the growth minus what their samples
// already covered is their tail.
void SystemProfiler::Impl::AttributeTails(uint64_t tsNs,
                                          const std::unordered_set<uint32_t>& trackedPids) {
    for (auto it = tailSettled.begin(); it != tailSettled.end(); ) {
        it = trackedPids.count(*it) ? std::next(it) : tailSettled.erase(it);
    }
    if (discoveredParent.empty() && awaitingTail.empty()) {
        childrenCpu.clear();
        return;
    }
    const uint64_t nsPerTick = 1000000000ull / static_cast<uint64_t>(internal::GetCLKTCK());
    std::unordered_map<uint32_t, std::vector<uint32_t>> kids;   // parent -> awaiting children
    std::unordered_set<uint32_t> parents;
    // A discovered process's own cutime is read too: when its parent
    // reaps it, the parent's growth includes the CPU of every child it
    // reaped, which that process's cutime holds.
    for (const auto& [pid, parent] : discoveredParent) {
        parents.insert(parent);
        parents.insert(pid);
    }
    for (auto it = awaitingTail.begin(); it != awaitingTail.end(); ) {
        if (!trackedPids.count(it->second.parent)) {
            tailSettled.insert(it->first);
            it = awaitingTail.erase(it);
            continue;
        }
        kids[it->second.parent].push_back(it->first);
        parents.insert(it->second.parent);
        parents.insert(it->first);
        ++it;
    }
    // An exited child that was itself a parent: its own reaped children's
    // CPU keeps changing until it is reaped (a zombie's cutime is still
    // readable); use the latest reading.
    for (auto& [pid, e] : awaitingTail) {
        if (auto k = childrenCpu.find(pid); k != childrenCpu.end())
            e.lastChildrenCpuNs = k->second.childrenCpuNs;
    }
    for (auto it = childrenCpu.begin(); it != childrenCpu.end(); ) {
        it = parents.count(it->first) ? std::next(it) : childrenCpu.erase(it);
    }

    for (uint32_t parent : parents) {
        const auto& waiting = kids[parent];
        // Consistent reading: the parent's cutime is the same before and
        // after checking which children are reaped, so no reap was in
        // progress in between and every reaped child is in the value.
        std::optional<internal::ProcStat> st;
        std::vector<uint32_t> reaped;
        for (int attempt = 0; attempt < 3; ++attempt) {
            auto before = internal::ReadProcStat("/proc", parent);
            reaped.clear();
            for (uint32_t c : waiting) {
                if (!internal::ReadProcStat("/proc", c)) reaped.push_back(c);
            }
            st = internal::ReadProcStat("/proc", parent);
            if (!before || !st ||
                (before->cutime == st->cutime && before->cstime == st->cstime)) break;
        }
        if (!st) {                       // parent gone: nobody left to measure
            for (uint32_t c : waiting) { awaitingTail.erase(c); tailSettled.insert(c); }
            childrenCpu.erase(parent);
            continue;
        }
        const uint64_t cur = (st->cutime + st->cstime) * nsPerTick;
        auto base = childrenCpu.find(parent);
        const bool haveBase = base != childrenCpu.end() && base->second.startTime == st->startTime;
        if (haveBase && !reaped.empty()) {
            int64_t tail = static_cast<int64_t>(cur) - static_cast<int64_t>(base->second.childrenCpuNs);
            for (uint32_t c : reaped) {
                const auto& e = awaitingTail[c];
                tail -= static_cast<int64_t>(e.lastCpuNs + e.lastChildrenCpuNs);
            }
            internal::CpuTailRecord r;
            r.timestamp_ns = tsNs;
            r.parent_pid   = parent;
            r.pids         = reaped;
            r.cpu_ns       = tail > 0 ? static_cast<uint64_t>(tail) : 0;
            std::lock_guard<std::mutex> lock(batchMutex);
            batch.cpuTails.push_back(std::move(r));
        }
        // Reaped without a baseline (found and reaped within one tick of
        // its parent's first reading): no tail can be measured.
        for (uint32_t c : reaped) { awaitingTail.erase(c); tailSettled.insert(c); }
        childrenCpu[parent] = {cur, st->startTime};
    }
}

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
            //
            // Read-then-verify: values are read by PID number, then the
            // entries' pidfds are polled, and the readings of processes
            // found to have exited are dropped (the number may already
            // name another process). What is kept was read while the
            // process each entry pins was still there.
            auto snapshot = this->SnapshotProcesses();
            std::unordered_set<uint32_t> snapshotPids;
            snapshotPids.reserve(snapshot.size());
            struct Reading {
                const ProcessTrackingProbe::ProcessEntry* entry;
                uint64_t cpuNs;
                std::optional<internal::PIDStatmSnapshot> statm;   // set if a baseline exists
            };
            std::vector<Reading> readings;
            readings.reserve(snapshot.size());

            for (const auto& entry : snapshot) {
                uint32_t pid = entry.pid;
                snapshotPids.insert(pid);
                if (entry.discovered && !entry.pending_removal &&
                    !impl.awaitingTail.count(pid) && !impl.tailSettled.count(pid)) {
                    impl.discoveredParent[pid] = entry.parent_pid;
                }
                if (entry.pending_removal) { impl.NoteExited(pid); continue; }

                // A PID that has exited (or cannot be read) is skipped
                // for this tick; its baseline is kept, so no negative
                // or garbage delta can be emitted.
                auto curCpuNs = internal::ReadPIDCpuTimeNs(pid);
                if (!curCpuNs) { impl.NoteExited(pid); continue; }
                // Test-only: the process dies right after this read, and
                // the reading stands for its number's next owner.
                if (internal::PassReadHook(pid)) *curCpuNs += 1000'000'000'000ull;
                auto it = impl.prevPID.find(pid);
                const bool haveBase = it != impl.prevPID.end() && it->second.serial == entry.serial;
                readings.push_back({&entry, *curCpuNs,
                                    haveBase ? std::optional(internal::ReadPIDStatm(pid))
                                             : std::nullopt});
            }
            const auto goneList = this->PollTracked();
            const std::unordered_set<uint64_t> gone(goneList.begin(), goneList.end());

            for (const auto& r : readings) {
                const auto& entry = *r.entry;
                const uint32_t pid = entry.pid;
                if (gone.count(entry.serial)) continue;   // exited during this tick
                if (!r.statm) {
                    // Mid-run add — seed the baseline; skip this tick.
                    // First emitted sample is one tick later, so the
                    // delta isn't garbage.
                    auto& seed = impl.prevPID[pid];
                    seed.tickTsNs = tsNs;
                    seed.cpuNs    = r.cpuNs;
                    seed.serial   = entry.serial;
                    // The process's CPU so far — from its fork until
                    // this first reading — is recorded once as its head
                    // (cpu_before_tracking_ns), never as a first-interval
                    // spike. Same for roots and discovered processes.
                    this->SetCpuBeforeTracking(pid, r.cpuNs);
                    continue;
                }
                auto& prev = impl.prevPID[pid];
                const auto& statm = *r.statm;

                // On-CPU delta of the whole thread group. The process
                // CPU clock is monotonic for the life of the process,
                // and threads that exited since the previous tick have
                // already been folded into it.
                uint64_t deltaCpuNs = (r.cpuNs > prev.cpuNs) ? (r.cpuNs - prev.cpuNs) : 0;

                // Denominator: actual wall-clock elapsed between this
                // tick and the previous one, not the nominal sample
                // period — sleep_for/loop overhead/proc-read latency
                // make actual >= nominal, so dividing by the nominal
                // period would systematically overestimate %.
                uint64_t dtNs = (tsNs > prev.tickTsNs) ? (tsNs - prev.tickTsNs) : 0;

                internal::ProcessTick t;
                t.timestamp_ns = tsNs;
                t.pid          = pid;
                t.cpu_pct      = (dtNs > 0) ? (double)deltaCpuNs / (double)dtNs * 100.0
                                            : 0.0;
                t.rss_bytes    = statm.RSSPages    * pageSize;
                t.vms_bytes    = statm.VMSPages    * pageSize;
                t.shared_bytes = statm.sharedPages * pageSize;

                prev.tickTsNs = tsNs;
                prev.cpuNs    = r.cpuNs;

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.processTicks.push_back(std::move(t));
            }
            // Drop baselines for PIDs that are no longer in the
            // snapshot (committed removals).
            for (auto it = impl.prevPID.begin(); it != impl.prevPID.end(); ) {
                if (snapshotPids.find(it->first) == snapshotPids.end()) {
                    impl.NoteExited(it->first);
                    it = impl.prevPID.erase(it);
                } else {
                    ++it;
                }
            }
            // Discovered PIDs that left without ever being sampled.
            std::vector<uint32_t> left;
            for (const auto& [pid, parent] : impl.discoveredParent)
                if (!snapshotPids.count(pid)) left.push_back(pid);
            for (uint32_t pid : left) impl.NoteExited(pid);
            impl.AttributeTails(tsNs, snapshotPids);
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

bool SystemProfiler::IsRunning() const { return m_impl->running; }

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
            drained.cpuTails.swap(m_impl->batch.cpuTails);
        }

        auto processSnapshot = SnapshotProcesses();
        if (!drained.systemTicks.empty() || !drained.processTicks.empty() ||
            !drained.cpuTails.empty() ||
            m_impl->flushStatsPending.valid) {
            SystemMetricsTrace trace = internal::BuildSystemTrace(
                m_impl->hostname, m_impl->config.samplingFrequencyHz,
                m_impl->hostCpuCount,
                m_impl->steadyClockRefNs, m_impl->wallClockEpochNs,
                processSnapshot, drained);
            internal::AttachDiscoveryStats(trace, *this);
            // Attach any pending flush stats from the last background flush cycle.
            if (m_impl->flushStatsPending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_flush_byte_size(m_impl->flushStatsPending.bytesWritten);
                fs->set_flush_interval_ns(m_impl->flushStatsPending.intervalNs);
                m_impl->flushStatsPending.valid = false;
            }

            internal::WriteDelimitedSystemTraceSized(trace, m_impl->outFile);
            m_impl->outFile.flush();
            CommitPendingRemovals(processSnapshot);
        }
        m_impl->outFile.close();
        std::cout << "[System] Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    m_impl->running = false;
}

} // namespace cupti_profiler
