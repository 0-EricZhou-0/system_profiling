#include <cupti_profiler/process_tracking_probe.h>

#include "proc_readers.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <unistd.h>
#include <unordered_set>

namespace cupti_profiler {

namespace {

uint64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string Trimmed(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

void AppendComm(ProcessTrackingProbe::ProcessEntry& e, uint64_t tsNs, std::string comm) {
    auto& h = e.comm_history;
    if (h.size() >= ProcessTrackingProbe::kMaxCommHistory) h.erase(h.begin() + 1);
    h.push_back({tsNs, comm});
    e.comm = std::move(comm);
}

} // namespace

ProcessTrackingProbe::~ProcessTrackingProbe() {
    for (auto& [serial, fd] : pidfds_) ::close(fd);
    for (auto& d : deferred_) if (d.pidfd >= 0) ::close(d.pidfd);
}

// A listed root: pin it with a pidfd, then read what the process table
// needs. The pidfd is checked still-alive AFTER the read, so the start
// time and parent describe the process it pins, not an earlier owner of
// the number. Runs without the lock (syscalls).
std::pair<ProcessTrackingProbe::ProcessEntry, int>
ProcessTrackingProbe::MakeRoot(uint32_t pid, std::string alias) {
    ProcessEntry e;
    e.pid   = pid;
    e.label = alias;
    e.alias = std::move(alias);
    const uint64_t now = SteadyNowNs();
    int fd = internal::PidfdOpen(pid);
    const int err = fd < 0 ? errno : 0;
    auto st = internal::ReadProcStat(internal::ProcRoot(), pid);
    const bool gone = (fd < 0 && err == ESRCH) || !st ||
                      (fd >= 0 && internal::PidfdExited(fd));
    if (gone) {
        if (fd >= 0) ::close(fd);
        std::cerr << "[tracking] pid " << pid << " does not exist; recorded as exited\n";
        e.pending_removal = true;
        e.end_time_ns = now;
        return {std::move(e), -1};
    }
    if (fd < 0) {
        // Not ESRCH (e.g. ENOSYS before Linux 5.3, EMFILE): tracked
        // without exit detection, as before pidfds.
        std::cerr << "[tracking] pid " << pid << ": pidfd_open: " << std::strerror(err)
                  << " — its exit will not be detected\n";
    }
    e.parent_pid    = st->ppid;
    e.start_time_ns = internal::BootTicksToSteadyNs(st->startTime);
    AppendComm(e, now, st->comm);
    return {std::move(e), fd};
}

void ProcessTrackingProbe::InsertLocked(ProcessEntry e, int pidfd) {
    e.serial = nextSerial_++;
    if (pidfd >= 0) pidfds_[e.serial] = pidfd;
    processes_.push_back(std::move(e));
}

void ProcessTrackingProbe::CloseLocked(uint64_t serial) {
    auto it = pidfds_.find(serial);
    if (it == pidfds_.end()) return;
    ::close(it->second);
    pidfds_.erase(it);
}

// True if the entry's process has exited — already marked, or its pidfd
// says so now (then it is marked here, as PollTracked would at the next
// tick). Lets a registration of the same number tell a new process from
// the one the entry pins.
bool ProcessTrackingProbe::GoneLocked(ProcessEntry& e) {
    if (e.end_time_ns != 0) return true;
    auto it = pidfds_.find(e.serial);
    if (it == pidfds_.end() || !internal::PidfdExited(it->second)) return false;
    e.pending_removal = true;
    e.end_time_ns = SteadyNowNs();
    CloseLocked(e.serial);
    return true;
}

void ProcessTrackingProbe::AddTrackedProcess(uint32_t pid, std::string alias) {
    {
        std::unique_lock<std::shared_mutex> lk(mutex_);
        for (auto& e : processes_) {
            if (e.pid != pid) continue;
            if (GoneLocked(e)) {
                // That process is gone; this PID now names another one.
                // Register it after the old entry's removal is flushed,
                // so one flush never carries the number twice.
                Deferred d;
                d.entry.pid   = pid;
                d.entry.alias = std::move(alias);
                deferred_.push_back(std::move(d));
                return;
            }
            // Idempotent: a re-Add after Remove un-removes.
            e.pending_removal = false;
            if (!alias.empty()) { e.alias = alias; e.label = std::move(alias); }
            // Listing a discovered process by hand makes it a root.
            e.discovered = false;
            return;
        }
    }
    auto [entry, fd] = MakeRoot(pid, std::move(alias));
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (const auto& e : processes_) {
        if (e.pid == pid) {   // registered concurrently
            if (fd >= 0) ::close(fd);
            return;
        }
    }
    InsertLocked(std::move(entry), fd);
}

void ProcessTrackingProbe::AddDiscoveredProcess(uint32_t pid, const std::string& label,
                                                const std::string& comm, uint32_t parentPid,
                                                int pidfd, uint64_t startTimeTicks) {
    ProcessEntry e;
    e.pid           = pid;
    e.label         = label;
    e.alias         = label + "/" + comm;
    e.parent_pid    = parentPid;
    e.discovered    = true;
    e.start_time_ns = internal::BootTicksToSteadyNs(startTimeTicks);
    AppendComm(e, SteadyNowNs(), comm);
    int fd = pidfd >= 0 ? ::fcntl(pidfd, F_DUPFD_CLOEXEC, 0) : -1;
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto& x : processes_) {
        if (x.pid != pid) continue;
        if (GoneLocked(x)) {
            // A new process with an old number: after the old one's
            // removal is flushed.
            deferred_.push_back({std::move(e), fd});
        } else if (fd >= 0) {
            ::close(fd);
        }
        return;
    }
    InsertLocked(std::move(e), fd);
}

void ProcessTrackingProbe::RemoveTrackedProcess(uint32_t pid) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto it = deferred_.begin(); it != deferred_.end(); ) {
        if (it->entry.pid != pid) { ++it; continue; }
        if (it->pidfd >= 0) ::close(it->pidfd);
        it = deferred_.erase(it);
    }
    for (auto& e : processes_) {
        if (e.pid == pid) {
            e.pending_removal = true;
            return;
        }
    }
    // PID wasn't tracked — silently no-op (matches Add idempotency).
}

void ProcessTrackingProbe::SetCpuBeforeTracking(uint32_t pid, uint64_t ns) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto& e : processes_) {
        if (e.pid == pid && e.end_time_ns == 0) { e.cpu_before_tracking_ns = ns; return; }
    }
}

void ProcessTrackingProbe::SetIoBeforeTracking(uint32_t pid, const IoCounters& io) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto& e : processes_) {
        if (e.pid == pid && e.end_time_ns == 0) { e.io_before_tracking = io; return; }
    }
}

void ProcessTrackingProbe::SetInitialProcesses(std::vector<ProcessEntry> entries) {
    std::vector<std::pair<ProcessEntry, int>> made;
    made.reserve(entries.size());
    for (auto& e : entries) made.push_back(MakeRoot(e.pid, std::move(e.alias)));
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto& [serial, fd] : pidfds_) ::close(fd);
    for (auto& d : deferred_) if (d.pidfd >= 0) ::close(d.pidfd);
    pidfds_.clear();
    processes_.clear();
    deferred_.clear();
    for (auto& [e, fd] : made) {
        bool dup = false;
        for (const auto& x : processes_) dup |= (x.pid == e.pid);
        if (dup) { if (fd >= 0) ::close(fd); continue; }
        InsertLocked(std::move(e), fd);
    }
}

std::vector<uint64_t> ProcessTrackingProbe::PollTracked() {
    struct Item { uint64_t serial; uint32_t pid; int fd; };
    std::vector<Item> items;
    {
        std::shared_lock<std::shared_mutex> lk(mutex_);
        items.reserve(processes_.size());
        for (const auto& e : processes_) {
            if (e.end_time_ns != 0) continue;
            auto it = pidfds_.find(e.serial);
            items.push_back({e.serial, e.pid, it == pidfds_.end() ? -1 : it->second});
        }
    }
    std::vector<uint64_t> gone;
    if (items.empty()) return gone;

    // comm first, pidfds second: a process whose pidfd is still not
    // readable after its comm was read is the process that comm is from.
    std::vector<std::optional<std::string>> comms(items.size());
    const uint64_t before = SteadyNowNs();
    const bool readComm = before - lastCommNs_ >= kCommRefreshNs;
    if (readComm) {
        lastCommNs_ = before;
        const std::string root = internal::ProcRoot();
        for (size_t i = 0; i < items.size(); ++i) {
            auto text = internal::ReadSmallFile(root + "/" + std::to_string(items[i].pid) + "/comm");
            if (text) comms[i] = Trimmed(*text);
        }
    }

    std::vector<struct pollfd> fds;
    std::vector<size_t> which;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].fd < 0) continue;
        fds.push_back({items[i].fd, POLLIN, 0});
        which.push_back(i);
    }
    std::vector<bool> exited(items.size(), false);
    if (!fds.empty()) {
        int r;
        do { r = ::poll(fds.data(), fds.size(), 0); } while (r < 0 && errno == EINTR);
        if (r > 0) {
            for (size_t k = 0; k < fds.size(); ++k) {
                if (fds[k].revents) exited[which[k]] = true;
            }
        }
    }
    // Taken after the poll: every exit it saw happened before this.
    const uint64_t now = SteadyNowNs();

    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (size_t i = 0; i < items.size(); ++i) {
        auto e = std::find_if(processes_.begin(), processes_.end(),
                              [&](const ProcessEntry& x) { return x.serial == items[i].serial; });
        if (e == processes_.end()) continue;           // committed away meanwhile
        if (e->end_time_ns != 0) {                     // marked by a registration meanwhile
            gone.push_back(e->serial);
            continue;
        }
        if (exited[i]) {
            e->pending_removal = true;
            e->end_time_ns = now;
            CloseLocked(e->serial);
            gone.push_back(e->serial);
            continue;
        }
        if (comms[i] && !comms[i]->empty() && *comms[i] != e->comm) {
            AppendComm(*e, now, *comms[i]);
            // A forked child carries its parent's comm until it execs or
            // renames itself (vLLM's EngineCore does, after fork), so a
            // discovered process's alias follows comm. A root keeps the
            // alias it was listed with.
            if (e->discovered) e->alias = e->label + "/" + e->comm;
        }
    }
    return gone;
}

std::vector<ProcessTrackingProbe::ProcessEntry>
ProcessTrackingProbe::SnapshotProcesses() const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return processes_;
}

void ProcessTrackingProbe::CommitPendingRemovals(const std::vector<ProcessEntry>& emitted) {
    std::unordered_set<uint64_t> marked;
    for (const auto& e : emitted) if (e.pending_removal) marked.insert(e.serial);
    std::vector<ProcessEntry> roots;        // deferred roots to register now
    {
        std::unique_lock<std::shared_mutex> lk(mutex_);
        if (marked.empty() && deferred_.empty()) return;
        processes_.erase(
            std::remove_if(processes_.begin(), processes_.end(),
                           [&](const ProcessEntry& e) {
                               if (!(e.pending_removal && marked.count(e.serial))) return false;
                               CloseLocked(e.serial);
                               return true;
                           }),
            processes_.end());
        // Registrations of a PID whose old entry is now gone.
        for (auto it = deferred_.begin(); it != deferred_.end(); ) {
            const uint32_t pid = it->entry.pid;
            const bool stillThere = std::any_of(processes_.begin(), processes_.end(),
                [&](const ProcessEntry& e) { return e.pid == pid; });
            if (stillThere) { ++it; continue; }
            if (it->entry.discovered) InsertLocked(std::move(it->entry), it->pidfd);
            else roots.push_back(std::move(it->entry));
            it = deferred_.erase(it);
        }
    }
    for (auto& e : roots) AddTrackedProcess(e.pid, std::move(e.alias));
}

void ProcessTrackingProbe::SetDiscoveryStats(const DiscoveryStats& stats) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    discoveryStats_ = stats;
}

std::optional<DiscoveryStats> ProcessTrackingProbe::SnapshotDiscoveryStats() const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return discoveryStats_;
}

} // namespace cupti_profiler
