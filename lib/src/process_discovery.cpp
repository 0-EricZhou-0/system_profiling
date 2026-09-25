#include "process_discovery.h"

#include "proc_readers.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <poll.h>
#include <unistd.h>

namespace cupti_profiler {
namespace internal {

namespace {

std::string ProcRootFromEnv() {
    // Test-only override; see process_discovery.h.
    const char* env = std::getenv("CUPTI_PROFILER_PROC_ROOT");
    std::string root = (env && *env) ? env : "/proc";
    while (root.size() > 1 && root.back() == '/') root.pop_back();
    return root;
}

uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool IsNumeric(const char* s) {
    if (!*s) return false;
    for (; *s; ++s) if (!std::isdigit(static_cast<unsigned char>(*s))) return false;
    return true;
}

std::string Trimmed(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// ScanHistogram

int ScanHistogram::Index(uint64_t v) {
    if (v < static_cast<uint64_t>(kSub)) return static_cast<int>(v);
    int e = 63 - __builtin_clzll(v);                 // e >= 4
    int sub = static_cast<int>((v >> (e - 4)) & (kSub - 1));
    return (e - 3) * kSub + sub;
}

uint64_t ScanHistogram::UpperBound(int idx) {
    if (idx < kSub) return static_cast<uint64_t>(idx);
    int e = idx / kSub + 3;
    int sub = idx % kSub;
    return (static_cast<uint64_t>(kSub + sub + 1) << (e - 4)) - 1;
}

void ScanHistogram::Record(uint64_t ns) {
    ++buckets_[Index(ns)];
    ++count_;
    if (ns > max_) max_ = ns;
}

uint64_t ScanHistogram::Percentile(double q) const {
    if (count_ == 0) return 0;
    uint64_t target = static_cast<uint64_t>(std::ceil(q * static_cast<double>(count_)));
    if (target == 0) target = 1;
    uint64_t seen = 0;
    for (int i = 0; i < static_cast<int>(buckets_.size()); ++i) {
        seen += buckets_[i];
        if (seen >= target) return std::min(UpperBound(i), max_);
    }
    return max_;
}

// ---------------------------------------------------------------------------
// ProcessDiscovery

ProcessDiscovery::ProcessDiscovery(DiscoverySettings settings,
                                   ProcessTrackingProbe* system,
                                   ProcessTrackingProbe* disk)
    : settings_([&] {
          if (settings.intervalMs == 0) settings.intervalMs = 100;
          return settings;
      }()),
      system_(system),
      disk_(disk),
      procRoot_(ProcRootFromEnv()),
      selfPid_(static_cast<uint32_t>(::getpid())) {}

ProcessDiscovery::~ProcessDiscovery() {
    Stop();
    for (auto& [pid, e] : table_) {
        if (e.pidfd >= 0) ::close(e.pidfd);
    }
}

template <class F>
void ProcessDiscovery::ForEachSink(uint8_t mask, F&& f) {
    if ((mask & kSystemSink) && system_) f(*system_);
    if ((mask & kDiskSink)   && disk_)   f(*disk_);
}

void ProcessDiscovery::AddRoot(uint32_t pid, const std::string& alias,
                               std::optional<bool> trackDescendants,
                               uint8_t sinks) {
    std::lock_guard<std::mutex> lk(mu_);
    ops_.push_back({/*remove=*/false, pid, alias, trackDescendants, sinks});
    if (trackDescendants.value_or(settings_.enabled)) anyDescending_ = true;
    if (started_ && !stop_ && anyDescending_ && !thread_.joinable()) {
        thread_ = std::thread(&ProcessDiscovery::Run, this);
    }
}

void ProcessDiscovery::RemoveRoot(uint32_t pid) {
    std::lock_guard<std::mutex> lk(mu_);
    ops_.push_back({/*remove=*/true, pid, {}, std::nullopt, 0});
}

void ProcessDiscovery::Start() {
    std::lock_guard<std::mutex> lk(mu_);
    started_ = true;
    stop_ = false;
    if (anyDescending_ && !thread_.joinable()) {
        thread_ = std::thread(&ProcessDiscovery::Run, this);
    }
}

void ProcessDiscovery::Stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void ProcessDiscovery::Run() {
    std::cerr << "[discovery] scanning every " << settings_.intervalMs << " ms ("
              << (settings_.recursive ? "recursive" : "direct children only")
              << ")\n";
    const auto interval = std::chrono::milliseconds(settings_.intervalMs);
    auto next = std::chrono::steady_clock::now();   // first scan right away
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        if (cv_.wait_until(lk, next, [&] { return stop_; })) break;
        std::vector<Op> ops;
        ops.swap(ops_);
        lk.unlock();
        ScanOnce(ops);
        lk.lock();
        next += interval;
        auto now = std::chrono::steady_clock::now();
        if (next < now) next = now + interval;   // overran: don't burst
    }
}

void ProcessDiscovery::ScanOnce(std::vector<Op>& ops) {
    const uint64_t t0 = NowNs();
    for (const auto& op : ops) ApplyOp(op);
    HandleExits();
    Scan();
    hist_.Record(NowNs() - t0);
    PublishStats();
}

void ProcessDiscovery::ApplyOp(const Op& op) {
    auto it = table_.find(op.pid);
    if (op.remove) {
        if (it != table_.end() && it->second.isRoot) {
            if (it->second.pidfd >= 0) ::close(it->second.pidfd);
            table_.erase(it);
        }
        return;
    }
    const bool descend = op.descend.value_or(settings_.enabled);
    const std::string label = op.alias.empty() ? std::to_string(op.pid) : op.alias;
    if (it == table_.end()) {
        int fd = PidfdOpen(op.pid);
        if (fd < 0) {
            std::cerr << "[discovery] root pid " << op.pid << ": pidfd_open: "
                      << std::strerror(errno) << " — its descendants are not tracked\n";
            return;
        }
        Entry e;
        e.pidfd = fd;
        if (auto st = ReadProcStat(procRoot_, op.pid)) {
            e.startTime = st->startTime;
            e.comm = st->comm;
        }
        it = table_.emplace(op.pid, std::move(e)).first;
    }
    Entry& e = it->second;
    e.isRoot    = true;
    e.scan      = descend;
    e.recursive = descend && settings_.recursive;
    e.sinks    |= op.sinks;
    e.label     = label;
}

void ProcessDiscovery::HandleExits() {
    std::vector<struct pollfd> fds;
    std::vector<uint32_t> pids;
    fds.reserve(table_.size());
    pids.reserve(table_.size());
    for (const auto& [pid, e] : table_) {
        if (e.pidfd < 0) continue;
        fds.push_back({e.pidfd, POLLIN, 0});
        pids.push_back(pid);
    }
    if (fds.empty()) return;
    int r;
    do { r = ::poll(fds.data(), fds.size(), 0); } while (r < 0 && errno == EINTR);
    if (r <= 0) return;
    for (size_t i = 0; i < fds.size(); ++i) {
        if (!fds[i].revents) continue;
        auto it = table_.find(pids[i]);
        HandleExit(it->first, it->second);
        table_.erase(it);
    }
}

void ProcessDiscovery::HandleExit(uint32_t pid, Entry& e) {
    if (!e.isRoot) {
        // Roots belong to the caller, who listed them; only processes
        // discovery registered are removed by it.
        ++exited_;
        ForEachSink(e.sinks, [&](ProcessTrackingProbe& p) { p.RemoveTrackedProcess(pid); });
        std::cerr << "[discovery] - " << pid << " exited\n";
    }
    if (e.pidfd >= 0) ::close(e.pidfd);
    e.pidfd = -1;
}

std::vector<uint32_t> ProcessDiscovery::ReadChildren(uint32_t pid) const {
    std::vector<uint32_t> out;
    const std::string taskDir = procRoot_ + "/" + std::to_string(pid) + "/task";
    DIR* d = ::opendir(taskDir.c_str());
    if (!d) return out;
    // Every thread's file: a child hangs off whichever thread forked it.
    while (struct dirent* ent = ::readdir(d)) {
        if (!IsNumeric(ent->d_name)) continue;
        auto text = ReadSmallFile(taskDir + "/" + ent->d_name + "/children");
        if (!text) continue;
        const char* p = text->c_str();
        while (*p) {
            char* end = nullptr;
            unsigned long v = std::strtoul(p, &end, 10);
            if (end == p) { ++p; continue; }
            if (v > 0) out.push_back(static_cast<uint32_t>(v));
            p = end;
        }
    }
    ::closedir(d);
    return out;
}

bool ProcessDiscovery::TryRegister(uint32_t child, uint32_t listedUnder) {
    int fd = PidfdOpen(child);
    if (fd < 0) return false;                     // already gone
    auto st = ReadProcStat(procRoot_, child);
    // Still alive AFTER the read => the /proc data describes the
    // process this pidfd pins, not an earlier owner of the PID.
    if (!st || PidfdExited(fd)) { ::close(fd); return false; }
    // PID-reuse guard: the parent must be a process we follow. A PID
    // listed in a children file can exit and be recycled by an
    // unrelated process before pidfd_open; its parent then is not ours.
    if (table_.find(st->ppid) == table_.end()) {
        ++rejected_;
        std::cerr << "[discovery] rejected " << child << " (" << st->comm
                  << "): parent " << st->ppid << " is not tracked (PID reused?)\n";
        ::close(fd);
        return false;
    }
    const Entry& via = table_.at(listedUnder);
    Entry e;
    e.pidfd       = fd;
    e.scan        = via.recursive;
    e.recursive   = via.recursive;
    e.sinks       = via.sinks;
    e.label       = via.label;
    e.firstParent = st->ppid;
    e.startTime   = st->startTime;
    e.comm        = st->comm;
    const std::string alias = e.label + "/" + e.comm;
    const uint32_t ppid = st->ppid;
    const uint8_t sinks = e.sinks;
    table_.emplace(child, std::move(e));
    ++discovered_;
    ForEachSink(sinks, [&](ProcessTrackingProbe& p) { p.AddDiscoveredProcess(child, alias, ppid); });
    std::cerr << "[discovery] + " << child << " " << alias << " (parent " << ppid << ")\n";
    return true;
}

void ProcessDiscovery::RefreshComm(uint32_t pid, Entry& e) {
    // A forked child carries its parent's comm until it execs or
    // renames itself (vLLM's EngineCore does, after fork), so the alias
    // follows comm rather than freezing the name seen at discovery.
    auto text = ReadSmallFile(procRoot_ + "/" + std::to_string(pid) + "/comm");
    if (!text) return;
    std::string comm = Trimmed(*text);
    if (comm.empty() || comm == e.comm) return;
    e.comm = comm;
    const std::string alias = e.label + "/" + comm;
    ForEachSink(e.sinks, [&](ProcessTrackingProbe& p) {
        p.AddDiscoveredProcess(pid, alias, e.firstParent);
    });
}

void ProcessDiscovery::Scan() {
    // Children of EVERY followed process, not only those reachable from
    // a root right now: a discovered process whose parent died has been
    // reparented out of the root's tree, and its future children are
    // still found here.
    std::vector<uint32_t> work;
    work.reserve(table_.size());
    for (auto& [pid, e] : table_) {
        if (!e.isRoot) RefreshComm(pid, e);
        if (e.scan) work.push_back(pid);
    }
    for (size_t i = 0; i < work.size(); ++i) {
        const uint32_t parent = work[i];
        for (uint32_t child : ReadChildren(parent)) {
            if (child == selfPid_ || table_.count(child)) continue;
            if (TryRegister(child, parent) && table_.at(child).scan) {
                work.push_back(child);   // recursive: descend this tick
            }
        }
    }
}

void ProcessDiscovery::PublishStats() {
    DiscoveryStats s;
    s.scanIntervalNs = settings_.intervalMs * 1000000ull;
    s.scans          = hist_.Count();
    s.scanP50Ns      = hist_.Percentile(0.50);
    s.scanP99Ns      = hist_.Percentile(0.99);
    s.scanMaxNs      = hist_.Max();
    s.discovered     = discovered_;
    s.exited         = exited_;
    s.rejected       = rejected_;
    ForEachSink(kSystemSink | kDiskSink, [&](ProcessTrackingProbe& p) { p.SetDiscoveryStats(s); });
}

} // namespace internal
} // namespace cupti_profiler
