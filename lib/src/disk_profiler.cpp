#include <cupti_profiler/disk_profiler.h>

#include "lifecycle.h"
#include "stop_signal.h"
#include "disk_readers.h"
#include "proc_readers.h"
#include "disk_flush_thread.h"
#include "tracked_process_proto.h"
#include "discovery_stats_proto.h"
#include "testing_hooks.h"

#include "disk_metrics.pb.h"
#include "metric_sample.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <optional>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace cupti_profiler {

namespace {
// The /proc/<pid>/io counters, in IoCounters / IoReapRecord order.
constexpr uint64_t internal::PIDIOSnapshot::* kIoCounters[5] = {
    &internal::PIDIOSnapshot::rchar,
    &internal::PIDIOSnapshot::wchar,
    &internal::PIDIOSnapshot::readBytes,
    &internal::PIDIOSnapshot::writeBytes,
    &internal::PIDIOSnapshot::cancelledWriteBytes,
};
} // namespace

class DiskProfiler::Impl {
public:
    DiskProfilerConfig config;
    std::string hostname;
    uint32_t hostCpuCount = 0;

    internal::DiskSampleBatch batch;
    std::mutex batchMutex;

    std::ofstream outFile;
    std::mutex outMutex;

    std::thread sampleThread;
    std::thread flushThread;
    internal::StopSignal stopSample;
    internal::StopSignal stopFlush;

    uint64_t steadyClockRefNs = 0;
    uint64_t wallClockEpochNs = 0;

    bool configured = false;
    bool running = false;
    // Serializes Stop() (the host, the signal flusher and the exit hook
    // may all call it).
    std::mutex stopMutex;

    // Previous snapshots, each with the steady-clock instant it was read:
    // rates divide by the actual time since then, not the nominal sample
    // period (the loop always runs somewhat longer than 1/hz).
    // serial (per-PID baselines only) is the tracked entry's
    // ProcessEntry::serial: a new process that got an old one's PID
    // never inherits its baseline.
    // startNs / rootParent (per-PID baselines only) are set at the
    // first reading: see ReapWatch.
    template <typename Snapshot> struct Baseline {
        Snapshot s;
        uint64_t tickTsNs = 0;
        uint64_t serial = 0;
        uint64_t startNs = 0;
        uint32_t rootParent = 0;
    };
    std::unordered_map<std::string, Baseline<internal::DiskStatSnapshot>> prevDisk;
    std::unordered_map<uint32_t, Baseline<internal::PIDIOSnapshot>> prevPIDIO;

    // Reaped children's I/O (IoReapAdjustment in disk_metrics.proto).
    // A reaping parent's own /proc/<pid>/io counters grow by the child's
    // lifetime I/O, so a tracked child reaped by a tracked parent has its
    // last reading subtracted from the parent's delta. Sample thread only.
    //   reapWatch: sampled processes that exited (or left the tracked
    //     set) and have not been subtracted from their parent yet ->
    //     that parent, their last reading, and whether they are known
    //     to be reaped already.
    //   reapNoted: serials already handed to reapWatch (or found to need
    //     no watch), while they stay in the snapshot.
    // A watched process whose parent is itself watched (exited, or left
    // the tracked set) is part of a chain: its I/O reaches a tracked
    // ancestor only through that parent's reap, and only if that parent
    // reaped it (Resolve).
    struct ReapWatch {
        uint32_t parent     = 0;
        uint64_t startTicks = 0;   // /proc/<pid>/stat field 22, once seen
        internal::PIDIOSnapshot lastSeen;
        bool     reaped     = false;
        uint64_t startNs    = 0;   // ProcessEntry::start_time_ns: identity for `adopted`
        uint32_t rootParent = 0;   // parent of the listed root it descends from; 0 = unknown
        // Its parent ignored SIGCHLD when the reap was found, so it was
        // auto-reaped and its I/O never reached the parent; nullopt =
        // not known (not reaped yet, or the parent could not be read).
        std::optional<bool> autoreaped;
    };
    std::unordered_map<uint32_t, ReapWatch> reapWatch;
    std::unordered_set<uint64_t>            reapNoted;

    // The host's orphan reaper and its reaped set (the reap-chain rule of
    // ProcessTrackingProbe::WhoReaped), refreshed once per tick; the
    // reaped set is pruned once nothing can look a PID up.
    ProcessTrackingProbe::ReapRuleState reap;

    bool Reaped(uint32_t pid, ReapWatch& w);
    void MarkReaped(ReapWatch& w);
    void NoteGone(uint32_t pid, uint64_t serial, uint32_t parent);

    // Where a watched process's I/O ends up. At: in `anchor`'s next
    // reading (unsure: the anchor's reading this tick may or may not
    // include it; ambiguous: listed there but not subtracted). Wait: not
    // there yet (not reaped, or reaped by a watched process that is not
    // reaped yet). Drop: in no tracked process's reading. autoreaped: a
    // process on the way up was auto-reaped, so none of it reached the
    // anchor (listed there, not subtracted).
    struct Resolution {
        enum Kind { Wait, Drop, At } kind = Drop;
        uint32_t anchor     = 0;
        bool     unsure     = false;
        bool     ambiguous  = false;
        bool     autoreaped = false;
    };
    Resolution Resolve(uint32_t pid, const std::unordered_set<uint32_t>& liveTracked,
                       const std::unordered_set<uint32_t>& reapedAfter) const;

    // Per-flush write accounting
    internal::DiskPendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;
};

// Is the watched process reaped: its /proc entry gone, or its number now
// naming another process? If not (a zombie, or a process removed from
// tracking while alive), follow its current parent: a zombie whose
// parent exited has been reparented, and its new parent will reap it.
bool DiskProfiler::Impl::Reaped(uint32_t pid, ReapWatch& w) {
    auto st = internal::ReadProcStat("/proc", pid);
    if (!st || (w.startTicks != 0 && st->startTime != w.startTicks)) return true;
    w.startTicks = st->startTime;
    w.parent     = st->ppid;
    return false;
}

// The watch's reap was just found: did its parent reap it (wait_task_zombie
// folds its I/O into the parent), or was it auto-reaped because the parent
// ignores SIGCHLD (nothing folded)? One /proc/<parent>/status read per
// reap, never per tick. The disposition is the one at this reading, not
// at the child's exit; a parent already gone (and reaped) leaves it
// unknown, which is treated as a wait.
void DiskProfiler::Impl::MarkReaped(ReapWatch& w) {
    w.reaped = true;
    w.autoreaped = internal::IgnoresSigchld(w.parent);
}

// A tracked process exited or left the tracked set. If it was ever read,
// its I/O up to that reading is in its own samples: watch for its reap.
// One never read is not watched — its I/O is counted only in its parent.
void DiskProfiler::Impl::NoteGone(uint32_t pid, uint64_t serial, uint32_t parent) {
    if (!reapNoted.insert(serial).second) return;
    auto b = prevPIDIO.find(pid);
    if (b == prevPIDIO.end() || b->second.serial != serial) return;
    reapWatch[pid] = {parent, 0, b->second.s, false, b->second.startNs, b->second.rootParent};
}

// Follow the watched process up through watched parents to the first
// live tracked one. Each step up from a reaped process C to a watched
// parent P asks whether P reaped C (ProcessTrackingProbe::WhoReaped) —
// only then is C's I/O inside P's, and so inside whatever reaps P
// (kernel/exit.c, wait_task_zombie: the reaper gets P's own counters
// plus everything P had reaped). The host reaped it -> Drop; unknown ->
// ambiguous (listed, not subtracted).
// Bookkeeping only: no reads.
DiskProfiler::Impl::Resolution DiskProfiler::Impl::Resolve(
    uint32_t pid, const std::unordered_set<uint32_t>& liveTracked,
    const std::unordered_set<uint32_t>& reapedAfter) const
{
    Resolution r;
    const ReapWatch* w = &reapWatch.at(pid);
    bool pending = !w->reaped;
    r.unsure = reapedAfter.count(pid) > 0;
    uint32_t cur = pid;
    for (int depth = 0; depth < 64; ++depth) {
        const uint32_t par = w->parent;
        if (w->reaped && w->autoreaped.value_or(false)) r.autoreaped = true;
        if (liveTracked.count(par)) {
            r.kind   = pending ? Resolution::Wait : Resolution::At;
            r.anchor = par;
            return r;
        }
        auto pw = reapWatch.find(par);
        if (pw == reapWatch.end()) return r;   // Drop: an untracked parent counts it once
        if (!pending) {
            const auto by = DiskProfiler::WhoReaped(reap, cur, w->startNs, w->rootParent);
            if (by == ReapedBy::Host) return r;   // Drop
            if (by == ReapedBy::Unknown) r.ambiguous = true;
            if (!pw->second.reaped) pending = true;   // arrives with par's reap
            else if (reapedAfter.count(par)) r.unsure = true;
        }
        cur = par;
        w   = &pw->second;
    }
    return r;   // Drop (no cycle is expected; bounded anyway)
}

DiskProfiler::DiskProfiler() : m_impl(std::make_unique<Impl>()) {}
DiskProfiler::~DiskProfiler() {
    if (m_impl && m_impl->running) {
        internal::lifecycle::WarnNotStopped("DiskProfiler", "when it was destroyed");
        Stop();
    }
}

void DiskProfiler::Configure(const DiskProfilerConfig& config) {
    m_impl->config = config;
    m_impl->config.flushIntervalMs = ResolveFlushIntervalMs(config.flushIntervalMs);

    char buf[256];
    gethostname(buf, sizeof(buf));
    m_impl->hostname = buf;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    m_impl->hostCpuCount = (nproc > 0) ? static_cast<uint32_t>(nproc) : 0;

    // Seed the ProcessTrackingProbe with the configured PIDs.
    std::vector<ProcessTrackingProbe::ProcessEntry> seed;
    seed.reserve(config.Processes.size());
    for (const auto& p : config.Processes) {
        seed.push_back({p.pid, p.alias, /*pending_removal=*/false});
    }
    SetInitialProcesses(std::move(seed));

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

    // Initial disk snapshot. Per-PID I/O baselines are seeded lazily
    // on the first iteration each PID appears in SnapshotProcesses() —
    // that way mid-run AddTrackedProcess() works.
    auto initDisk = internal::ReadDiskStats(m_impl->config.devices);
    const uint64_t initTsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    for (auto& ds : initDisk) m_impl->prevDisk[ds.device] = {ds, initTsNs};

    m_impl->stopSample.Reset();
    m_impl->sampleThread = std::thread([this]() {
        internal::lifecycle::BlockSignalsInThisThread();
        ::pthread_setname_np(::pthread_self(), "cupti-dsk-samp");
        auto& impl = *m_impl;

        const auto period = std::chrono::microseconds(1000000 / impl.config.samplingFrequencyHz);
        while (!impl.stopSample.WaitUntil(std::chrono::steady_clock::now() + period)) {

            auto now = std::chrono::steady_clock::now().time_since_epoch();
            uint64_t tsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

            // Per-device stats
            auto curDisk = internal::ReadDiskStats(impl.config.devices);
            for (auto& ds : curDisk) {
                auto it = impl.prevDisk.find(ds.device);
                if (it == impl.prevDisk.end()) {
                    impl.prevDisk[ds.device] = {ds, tsNs};
                    continue;
                }
                auto& prev = it->second.s;
                const double dtSec = (double)(tsNs - it->second.tickTsNs) / 1e9;

                auto inflight = internal::ReadDiskInflight(ds.device);

                internal::DiskDeviceTick t;
                t.timestamp_ns        = tsNs;
                t.device_name         = ds.device;
                t.read_bytes_per_sec  = (double)(ds.sectorsRead - prev.sectorsRead) * 512.0 / dtSec;
                t.write_bytes_per_sec = (double)(ds.sectorsWritten - prev.sectorsWritten) * 512.0 / dtSec;
                t.read_inflight       = inflight.readInflight;
                t.write_inflight      = inflight.writeInflight;
                it->second = {ds, tsNs};

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.deviceTicks.push_back(std::move(t));
            }

            // Reaped children, before any per-process reading: a watched
            // process already reaped now is certainly inside its parent's
            // reading below. No reads while nothing is watched.
            std::unordered_set<uint32_t> reapedBefore;
            for (auto& [pid, w] : impl.reapWatch) {
                if (w.reaped || impl.Reaped(pid, w)) {
                    if (!w.reaped) impl.MarkReaped(w);
                    reapedBefore.insert(pid);
                }
            }

            // Per-process I/O — the tracked set is whatever
            // ProcessTrackingProbe holds right now. Entries with
            // pending_removal=true are skipped (they're awaiting the
            // next flush's removal marker).
            //
            // Read-then-verify, as in the system probe: the readings of
            // processes whose pidfd says they exited are dropped, since
            // the number may already name another process.
            auto snapshot = this->SnapshotProcesses();
            std::unordered_set<uint32_t> snapshotPids;
            snapshotPids.reserve(snapshot.size());
            std::vector<std::pair<const ProcessTrackingProbe::ProcessEntry*,
                                  internal::PIDIOSnapshot>> readings;
            readings.reserve(snapshot.size());
            for (const auto& entry : snapshot) {
                snapshotPids.insert(entry.pid);
                if (entry.pending_removal) continue;
                auto io = internal::ReadPIDIO(entry.pid);
                if (int e = internal::ReadErrorFor(entry.pid, testing::ReadProbe::Disk)) {   // test-only
                    io = {};
                    io.accessible = false;
                    io.error = e;
                }
                // Test-only: the process dies right after this read, and
                // the reading stands for its number's next owner.
                if (internal::PassReadHook(entry.pid, testing::ReadProbe::Disk)) {
                    constexpr uint64_t kForeign = 1000'000'000'000ull;
                    io.rchar += kForeign; io.wchar += kForeign;
                    io.readBytes += kForeign; io.writeBytes += kForeign;
                    io.cancelledWriteBytes += kForeign;
                }
                readings.emplace_back(&entry, io);
            }
            const auto goneList = this->PollTracked();
            const std::unordered_set<uint64_t> gone(goneList.begin(), goneList.end());

            // Reaped children, after the readings: processes that exited
            // or left the tracked set join the watch, and every watched
            // process not yet known to be reaped is looked at again. One
            // found reaped only now was reaped at some point during the
            // readings, so whether its parent's reading includes it is
            // unknown: that reading is not used (the parent keeps its
            // baseline, and its next sample spans both intervals and
            // certainly includes the reap).
            for (const auto& entry : snapshot) {
                if (entry.pending_removal || gone.count(entry.serial))
                    impl.NoteGone(entry.pid, entry.serial, entry.parent_pid);
            }
            // Baselines of PIDs no longer tracked are dropped here.
            for (auto it = impl.prevPIDIO.begin(); it != impl.prevPIDIO.end(); ) {
                if (snapshotPids.count(it->first)) { ++it; continue; }
                impl.NoteGone(it->first, it->second.serial, 0);
                it = impl.prevPIDIO.erase(it);
            }
            std::unordered_set<uint32_t> reapedAfter;
            for (auto& [pid, w] : impl.reapWatch) {
                if (!reapedBefore.count(pid) && impl.Reaped(pid, w)) {
                    impl.MarkReaped(w);
                    reapedAfter.insert(pid);
                }
            }
            this->RefreshReapRule(impl.reap);

            // Where each watched process's I/O is (Impl::Resolve): the
            // children whose reap a live tracked process's reading
            // includes, and the readings that may or may not include one
            // (an unsure reading is not used: the process keeps its
            // baseline, and its next sample spans both intervals and
            // certainly includes the reap). A watch ends when no tracked
            // process's reading can include it.
            std::unordered_set<uint32_t> unsureAnchors;
            std::unordered_map<uint32_t, std::vector<internal::IoReapRecord::Child>> reapedUnder;
            if (!impl.reapWatch.empty()) {
                std::unordered_set<uint32_t> liveTracked;
                for (const auto& entry : snapshot)
                    if (!entry.pending_removal && !gone.count(entry.serial)) liveTracked.insert(entry.pid);
                std::vector<uint32_t> ended;
                for (const auto& [pid, w] : impl.reapWatch) {
                    const auto r = impl.Resolve(pid, liveTracked, reapedAfter);
                    if (r.kind == Impl::Resolution::Drop) { ended.push_back(pid); continue; }
                    if (r.kind != Impl::Resolution::At) continue;
                    if (r.unsure) unsureAnchors.insert(r.anchor);
                    const auto& l = w.lastSeen;
                    reapedUnder[r.anchor].push_back({pid, {l.rchar, l.wchar, l.readBytes, l.writeBytes,
                                                           l.cancelledWriteBytes},
                                                     w.parent, r.ambiguous, r.autoreaped});
                }
                for (uint32_t pid : ended) impl.reapWatch.erase(pid);
            }

            for (const auto& [ep, curIO] : readings) {
                const auto& entry = *ep;
                const uint32_t pid = entry.pid;
                if (gone.count(entry.serial)) continue;   // exited during this tick
                if (!curIO.accessible) {
                    // No sample this tick; the baseline is kept. An exiting
                    // process's /proc/<pid>/io fails with EACCES until it is
                    // reaped: that is its exit, not a permission problem.
                    // A live one: warn once per tracked process and record
                    // it in the process table (missing I/O is not zero).
                    const auto kind = internal::ClassifyReadFailure(pid, curIO.error);
                    if (kind != internal::ReadFailure::Gone)
                        this->NoteUnreadable(entry.serial, UnreadableFile::Io, static_cast<int>(kind), tsNs,
                                             internal::UnreadableWarning("Disk", "io", pid, entry.comm,
                                                                         curIO.error, kind));
                    continue;
                }

                if (unsureAnchors.count(pid)) continue;   // see above

                // Tracked children whose reap this reading includes.
                std::vector<internal::IoReapRecord::Child> kids;
                if (auto k = reapedUnder.find(pid); k != reapedUnder.end()) kids = std::move(k->second);

                auto it = impl.prevPIDIO.find(pid);
                if (it == impl.prevPIDIO.end() || it->second.serial != entry.serial) {
                    // Mid-run add — seed baseline, skip this tick. A reap
                    // this reading includes is inside the baseline, so
                    // there is nothing to subtract.
                    // Its counters now are its I/O before tracking (the
                    // head); its tree's root's parent tells a later reap
                    // chain whether the host's reaper covers it.
                    const uint32_t rootParent = RootParentOf(entry, snapshot);
                    impl.prevPIDIO[pid] = {curIO, tsNs, entry.serial, entry.start_time_ns, rootParent};
                    this->SetIoBeforeTracking(pid, {curIO.rchar, curIO.wchar, curIO.readBytes,
                                                    curIO.writeBytes, curIO.cancelledWriteBytes});
                    for (const auto& c : kids) impl.reapWatch.erase(c.pid);
                    continue;
                }
                auto& base = it->second;
                auto& prev = base.s;
                // An unreadable tick keeps the baseline, so this spans it.
                const double dtSec = (double)(tsNs - it->second.tickTsNs) / 1e9;

                // Every counter is monotonic for the life of the process.
                // The process's own I/O: the delta minus the last reading
                // of each tracked child whose reap this interval brought
                // in (directly, or through a chain), unless ambiguous or
                // auto-reaped (then it never reached this process).
                int64_t own[5];
                for (int k = 0; k < 5; ++k) {
                    const auto c = kIoCounters[k];
                    own[k] = curIO.*c > prev.*c ? static_cast<int64_t>(curIO.*c - prev.*c) : 0;
                    for (const auto& kid : kids)
                        if (!kid.ambiguous && !kid.autoreaped)
                            own[k] -= static_cast<int64_t>(impl.reapWatch[kid.pid].lastSeen.*c);
                }
                auto rate = [&](int k) { return own[k] > 0 ? (double)own[k] / dtSec : 0.0; };
                internal::DiskProcessTick t;
                t.timestamp_ns                  = tsNs;
                t.pid                           = pid;
                t.rchar_bytes_per_sec           = rate(0);
                t.wchar_bytes_per_sec           = rate(1);
                t.read_bytes_per_sec            = rate(2);
                t.write_bytes_per_sec           = rate(3);
                t.cancelled_write_bytes_per_sec = rate(4);
                base.s = curIO;
                base.tickTsNs = tsNs;

                std::optional<internal::IoReapRecord> reap;
                if (!kids.empty()) {
                    reap.emplace();
                    reap->timestamp_ns = tsNs;
                    reap->parent_pid   = pid;
                    for (const auto& c : kids) {
                        reap->ambiguous  = reap->ambiguous || c.ambiguous;
                        reap->autoreaped = reap->autoreaped || c.autoreaped;
                        impl.reapWatch.erase(c.pid);
                    }
                    reap->children = std::move(kids);
                    std::copy(std::begin(own), std::end(own), reap->remainder);
                }

                std::lock_guard<std::mutex> lock(impl.batchMutex);
                impl.batch.processTicks.push_back(std::move(t));
                if (reap) impl.batch.ioReaps.push_back(std::move(*reap));
            }

            if (!impl.reapNoted.empty()) {
                std::unordered_set<uint64_t> serials;
                for (const auto& entry : snapshot) serials.insert(entry.serial);
                std::erase_if(impl.reapNoted, [&](uint64_t s) { return !serials.count(s); });
            }
            // A reaped PID nothing tracks or watches any more is never
            // looked up again.
            std::erase_if(impl.reap.adopted, [&](const auto& kv) {
                return !snapshotPids.count(kv.first) && !impl.reapWatch.count(kv.first);
            });

        }
    });

    // Launch flush thread
    m_impl->stopFlush.Reset();
    if (m_impl->outFile.is_open()) {
        m_impl->flushThread = std::thread(internal::DiskFlushThreadFunc,
                                           std::ref(m_impl->batch),
                                           std::ref(m_impl->batchMutex),
                                           std::ref(m_impl->outFile),
                                           std::ref(m_impl->outMutex),
                                           std::cref(m_impl->hostname),
                                           m_impl->config.samplingFrequencyHz,
                                           m_impl->hostCpuCount,
                                           std::cref(m_impl->config.devices),
                                           std::ref(static_cast<ProcessTrackingProbe&>(*this)),
                                           std::ref(m_impl->stopFlush),
                                           m_impl->config.flushIntervalMs,
                                           m_impl->steadyClockRefNs,
                                           m_impl->wallClockEpochNs,
                                           std::ref(m_impl->flushStatsPending),
                                           std::ref(m_impl->flushStatsMutex));
    }

    m_impl->running = true;
    internal::lifecycle::Register(m_impl.get(), internal::lifecycle::Order::Probe, "DiskProfiler",
                                  [this] { Stop(); });
    std::cout << "[Disk] Profiler started\n";
}

bool DiskProfiler::IsRunning() const { return m_impl->running; }

void DiskProfiler::SignalStop() {
    if (!m_impl->running) return;
    m_impl->stopSample.Set();
    m_impl->stopFlush.Set();
}

void DiskProfiler::Stop() {
    std::lock_guard<std::mutex> stopLock(m_impl->stopMutex);
    internal::lifecycle::StopScope stopping;
    if (!m_impl->running) return;

    m_impl->stopSample.Set();
    m_impl->stopFlush.Set();

    if (m_impl->sampleThread.joinable()) m_impl->sampleThread.join();
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    // Write remaining
    if (m_impl->outFile.is_open()) {
        internal::DiskSampleBatch drained;
        {
            std::lock_guard<std::mutex> lock(m_impl->batchMutex);
            drained.deviceTicks.swap(m_impl->batch.deviceTicks);
            drained.processTicks.swap(m_impl->batch.processTicks);
            drained.ioReaps.swap(m_impl->batch.ioReaps);
        }

        auto processSnapshot = SnapshotProcesses();
        if (!drained.deviceTicks.empty() || !drained.processTicks.empty() ||
            !drained.ioReaps.empty() ||
            internal::HasRemovalMarker(processSnapshot) ||
            internal::HasUnreadableRecord(processSnapshot) ||
            m_impl->flushStatsPending.valid) {
            DiskMetricsTrace trace = internal::BuildDiskTrace(
                m_impl->hostname, m_impl->config.samplingFrequencyHz,
                m_impl->hostCpuCount,
                m_impl->steadyClockRefNs, m_impl->wallClockEpochNs,
                m_impl->config.devices, processSnapshot, drained);
            internal::AttachDiscoveryStats(trace, *this);
            if (m_impl->flushStatsPending.valid) {
                auto* fs = trace.add_flush_stats();
                fs->set_flush_byte_size(m_impl->flushStatsPending.bytesWritten);
                fs->set_flush_interval_ns(m_impl->flushStatsPending.intervalNs);
                fs->set_flush_duration_ns(m_impl->flushStatsPending.durationNs);
                fs->set_slow_flushes(m_impl->flushStatsPending.slowFlushes);
                m_impl->flushStatsPending.valid = false;
            }

            internal::WriteDelimitedDiskTraceSized(trace, m_impl->outFile);
            m_impl->outFile.flush();
            CommitPendingRemovals(processSnapshot);
        }
        m_impl->outFile.close();
        std::cout << "[Disk] Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    internal::ReportWarnStateAtStop("disk", WarnStateSize());
    FlushWarnings();
    m_impl->running = false;
    internal::lifecycle::Unregister(m_impl.get());
}

} // namespace cupti_profiler
