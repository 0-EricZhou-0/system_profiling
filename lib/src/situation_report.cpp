#include "situation_report.h"

#include <cupti_profiler/child_subreaper.h>

#include "proc_readers.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <linux/genetlink.h>
#include <linux/netlink.h>
#include <linux/taskstats.h>
#include <sstream>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <sys/xattr.h>
#include <unistd.h>

namespace cupti_profiler {
namespace internal {

namespace {

constexpr int kCapDacReadSearch = 2;
constexpr int kCapNetAdmin      = 12;
constexpr int kCapSysPtrace     = 19;

std::optional<uint64_t> CapEff(pid_t pid) {
    auto text = ReadSmallFile("/proc/" + std::to_string(pid) + "/status");
    if (!text) return std::nullopt;
    std::istringstream in(*text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 7, "CapEff:") == 0)
            return std::strtoull(line.c_str() + 7, nullptr, 16);
    }
    return std::nullopt;
}

std::optional<uint32_t> Uid(uint32_t pid) {
    auto text = ReadSmallFile("/proc/" + std::to_string(pid) + "/status");
    if (!text) return std::nullopt;
    std::istringstream in(*text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 4, "Uid:") == 0)
            return static_cast<uint32_t>(std::strtoul(line.c_str() + 4, nullptr, 10));
    }
    return std::nullopt;
}

struct MountInfo { std::string point, fstype, options; };

// The mount holding `path`: longest mount point that prefixes it.
std::optional<MountInfo> MountOf(const std::string& path) {
    char real[PATH_MAX];
    if (!::realpath(path.c_str(), real)) return std::nullopt;
    const std::string p = real;
    auto text = ReadSmallFile("/proc/self/mountinfo");
    if (!text) return std::nullopt;
    std::optional<MountInfo> best;
    std::istringstream in(*text);
    std::string line;
    while (std::getline(in, line)) {
        size_t sep = line.find(" - ");
        if (sep == std::string::npos) continue;
        std::istringstream pre(line.substr(0, sep)), post(line.substr(sep + 3));
        std::string id, parent, dev, root, point, opts, fstype;
        pre >> id >> parent >> dev >> root >> point >> opts;
        post >> fstype;
        const bool under = (p == point) ||
            (p.compare(0, point.size(), point) == 0 &&
             (point == "/" || p[point.size()] == '/'));
        if (under && (!best || point.size() > best->point.size()))
            best = MountInfo{point, fstype, opts};
    }
    return best;
}

bool IsNetworkFs(const std::string& fstype) {
    static const char* kNet[] = {
        "nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "afs", "ceph",
        "glusterfs", "lustre", "gpfs", "beegfs", "fuse.sshfs", "fuse.glusterfs",
    };
    for (const char* n : kNet) if (fstype == n) return true;
    return false;
}

// Per-PID taskstats query for our own PID over generic netlink. Returns
// 0 if allowed, else the errno the kernel replied with (EPERM without
// CAP_NET_ADMIN), or -1 if the family itself is unavailable.
int TaskstatsQueryError() {
    int s = ::socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_GENERIC);
    if (s < 0) return -1;
    struct timeval tv{1, 0};
    ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    struct sockaddr_nl local{};
    local.nl_family = AF_NETLINK;
    if (::bind(s, reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0) { ::close(s); return -1; }

    struct Req {
        nlmsghdr n;
        genlmsghdr g;
        char buf[256];
    };
    auto send = [&](uint16_t type, uint8_t cmd, uint16_t attrType,
                    const void* data, uint16_t len, uint32_t seq) {
        Req r{};
        r.n.nlmsg_type  = type;
        r.n.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
        r.n.nlmsg_seq   = seq;
        r.g.cmd     = cmd;
        r.g.version = 1;
        auto* na = reinterpret_cast<nlattr*>(r.buf);
        na->nla_type = attrType;
        na->nla_len  = static_cast<uint16_t>(NLA_HDRLEN + len);
        std::memcpy(reinterpret_cast<char*>(na) + NLA_HDRLEN, data, len);
        r.n.nlmsg_len = NLMSG_LENGTH(GENL_HDRLEN) + NLA_ALIGN(na->nla_len);
        struct sockaddr_nl kernel{};
        kernel.nl_family = AF_NETLINK;
        return ::sendto(s, &r, r.n.nlmsg_len, 0,
                        reinterpret_cast<sockaddr*>(&kernel), sizeof(kernel)) >= 0;
    };

    // 1. Resolve the TASKSTATS family id.
    const char name[] = TASKSTATS_GENL_NAME;
    int familyId = -1;
    if (send(GENL_ID_CTRL, CTRL_CMD_GETFAMILY, CTRL_ATTR_FAMILY_NAME, name, sizeof(name), 1)) {
        char buf[8192];
        int len = static_cast<int>(::recv(s, buf, sizeof(buf), 0));
        for (auto* h = reinterpret_cast<nlmsghdr*>(buf);
             len > 0 && NLMSG_OK(h, len); h = NLMSG_NEXT(h, len)) {
            if (h->nlmsg_type == NLMSG_ERROR) break;
            auto* g = static_cast<genlmsghdr*>(NLMSG_DATA(h));
            auto* a = reinterpret_cast<nlattr*>(reinterpret_cast<char*>(g) + GENL_HDRLEN);
            int rem = static_cast<int>(h->nlmsg_len) - NLMSG_LENGTH(GENL_HDRLEN);
            while (rem >= NLA_HDRLEN && a->nla_len >= NLA_HDRLEN && a->nla_len <= rem) {
                if (a->nla_type == CTRL_ATTR_FAMILY_ID)
                    familyId = *reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(a) + NLA_HDRLEN);
                rem -= NLA_ALIGN(a->nla_len);
                a = reinterpret_cast<nlattr*>(reinterpret_cast<char*>(a) + NLA_ALIGN(a->nla_len));
            }
            break;
        }
    }
    if (familyId < 0) { ::close(s); return -1; }

    // 2. Query our own PID; the kernel answers with data + ACK, or an
    // error. Only replies to THIS request (seq 2) count: step 1's ACK
    // (error 0, seq 1) can still be queued and would read as "allowed".
    uint32_t pid = static_cast<uint32_t>(::getpid());
    int result = -1;
    if (send(static_cast<uint16_t>(familyId), TASKSTATS_CMD_GET, TASKSTATS_CMD_ATTR_PID,
             &pid, sizeof(pid), 2)) {
        char buf[8192];
        for (int tries = 0; tries < 8 && result < 0; ++tries) {
            int len = static_cast<int>(::recv(s, buf, sizeof(buf), 0));
            if (len <= 0) break;
            for (auto* h = reinterpret_cast<nlmsghdr*>(buf);
                 NLMSG_OK(h, len); h = NLMSG_NEXT(h, len)) {
                if (h->nlmsg_seq != 2) continue;
                if (h->nlmsg_type == NLMSG_ERROR) {
                    result = -static_cast<nlmsgerr*>(NLMSG_DATA(h))->error;
                    break;
                }
            }
        }
    }
    ::close(s);
    return result;
}

std::string CapNames(uint64_t eff) {
    std::string out;
    auto add = [&](int bit, const char* n) {
        if (eff >> bit & 1) { if (!out.empty()) out += ", "; out += n; }
    };
    add(kCapDacReadSearch, "CAP_DAC_READ_SEARCH");
    add(kCapNetAdmin,      "CAP_NET_ADMIN");
    add(kCapSysPtrace,     "CAP_SYS_PTRACE");
    return out.empty() ? "none of CAP_DAC_READ_SEARCH/NET_ADMIN/SYS_PTRACE" : out;
}

} // namespace

std::vector<SituationLine> ProbeSituation(const SituationInputs& in) {
    std::vector<SituationLine> out;
    auto row = [&](std::string c, std::string o, std::string q, bool bad = false) {
        out.push_back({std::move(c), std::move(o), std::move(q), bad});
    };

    // Observer placement.
    if (in.sidecar) {
        row("observer", "sidecar pid " + std::to_string(in.sidecarPid) + " (" + in.observerBinary + ")",
            "system/disk sampling and descendant tracking run outside the workload; "
            "their CPU is not charged to any tracked process");
    } else {
        row("observer", "in-process (pid " + std::to_string(::getpid()) + ")",
            "system/disk sampling and descendant tracking run as threads of this "
            "process; if it is tracked, their CPU is charged to it");
    }

    // Kernel features.
    const long tid = ::syscall(SYS_gettid);
    const bool children = ::access(("/proc/self/task/" + std::to_string(tid) + "/children").c_str(),
                                   R_OK) == 0;
    row("/proc/<pid>/task/<tid>/children", children ? "present" : "ABSENT (CONFIG_PROC_CHILDREN off)",
        children ? "descendant tracking can enumerate children"
                 : "descendant tracking cannot work: only listed PIDs are traced",
        !children);
    int fd = PidfdOpen(static_cast<uint32_t>(::getpid()));
    const int pidfdErr = fd < 0 ? errno : 0;
    if (fd >= 0) ::close(fd);
    row("pidfd_open (syscall)", fd >= 0 ? "available" : std::string("ABSENT: ") + std::strerror(pidfdErr),
        fd >= 0 ? "discovered processes are pinned by pidfd: PID-reuse guard and exit detection work"
                : "descendant tracking cannot run (it requires pidfd_open, Linux >= 5.3)",
        fd < 0);

    // Descendant tracking setting.
    if (in.discovery.enabled) {
        const std::string iv = std::to_string(in.discovery.intervalMs) + " ms";
        row("descendant tracking",
            std::string("on by default, ") + (in.discovery.recursive ? "recursive" : "direct children only")
                + ", every " + iv,
            "a child of a tracked process that stays its child for >= " + iv +
                " is discovered and then tracked until it exits; shorter-lived children may be missed");
    } else {
        row("descendant tracking", "off by default",
            "only listed PIDs are traced, except roots added with track_descendants=True");
    }

    // Yama.
    auto yama = ReadSmallFile("/proc/sys/kernel/yama/ptrace_scope");
    std::string scope = yama ? *yama : std::string("not loaded");
    while (!scope.empty() && scope.back() == '\n') scope.pop_back();
    row("yama ptrace_scope", scope,
        "restricts ptrace ATTACH only; /proc/<pid>/io reads are unaffected (the profiler never attaches)");

    // Observer capabilities.
    const pid_t obs = in.sidecar ? in.sidecarPid : ::getpid();
    auto eff = CapEff(obs);
    const bool readOthers = eff && (*eff >> kCapDacReadSearch & 1) && (*eff >> kCapSysPtrace & 1);
    row("observer effective capabilities", eff ? CapNames(*eff) : "unknown",
        readOthers ? "can read other users' /proc/<pid>/io"
                   : "per-PID I/O readable for same-uid, dumpable targets only");

    // Secure-exec: the loader strips LD_LIBRARY_PATH for setuid / file-cap binaries.
    bool secure = false;
    std::string why;
    if (in.sidecar) {
        struct stat st{};
        const bool setid = ::stat(in.observerBinary.c_str(), &st) == 0 && (st.st_mode & (S_ISUID | S_ISGID));
        const bool fcaps = ::getxattr(in.observerBinary.c_str(), "security.capability", nullptr, 0) >= 0;
        secure = setid || fcaps;
        why = fcaps ? " (file capabilities)" : setid ? " (setuid/setgid)" : "";
    } else {
        secure = ::getauxval(AT_SECURE) != 0;
    }
    row("secure-exec (AT_SECURE) of the observer", secure ? "yes" + why : "no",
        secure ? "LD_LIBRARY_PATH is STRIPPED by the loader: CUDA forward compat and any "
                 "library found only via LD_LIBRARY_PATH are unavailable to the observer"
               : "LD_LIBRARY_PATH honoured",
        secure);
    const char* ldp = std::getenv("LD_LIBRARY_PATH");
    std::string compat;
    if (ldp) {
        std::istringstream parts(ldp);
        std::string part;
        while (std::getline(parts, part, ':'))
            if (compat.empty() && part.find("compat") != std::string::npos) compat = part;
    }
    row("CUDA forward-compat on LD_LIBRARY_PATH", compat.empty() ? "none" : compat,
        compat.empty() ? "CUDA runtime must be supported by the installed driver"
                       : "a newer CUDA runtime can run on the older driver");

    // Observer binary's filesystem.
    auto m = MountOf(in.observerBinary);
    if (m) {
        std::string opts = "," + m->options + ",";
        const bool nosuid = opts.find(",nosuid,") != std::string::npos;
        const bool net = IsNetworkFs(m->fstype);
        std::string o = m->fstype + " on " + m->point + (nosuid ? " (nosuid)" : "");
        std::string q = net ? "network filesystem: cannot hold file capabilities (setcap fails "
                              "or is ignored); install the observer on a local filesystem to grant it any"
                      : nosuid ? "nosuid mount: file capabilities AND setuid are IGNORED; "
                                 "install the observer elsewhere to grant it any"
                               : "file capabilities / setuid on the observer would be honoured";
        row("filesystem of " + in.observerBinary, o, q, net || nosuid);
    } else {
        row("filesystem of " + in.observerBinary, "unknown", "could not resolve the mount");
    }

    out.push_back(ProbeSubreaper());

    // taskstats.
    int ts = TaskstatsQueryError();
    std::string tsObs = ts == 0 ? "allowed" : ts < 0 ? "unavailable" : std::string(std::strerror(ts));
    std::string tsQ = ts == 0 ? "taskstats backend usable from this process"
                    : ts < 0 ? "no taskstats backend on this kernel"
                             : "needs CAP_NET_ADMIN: /proc backend only";
    if (in.sidecar && eff && (*eff >> kCapNetAdmin & 1)) {
        tsObs += " here; the sidecar holds CAP_NET_ADMIN";
        tsQ = "taskstats backend usable from the sidecar";
    }
    row("taskstats per-PID query", tsObs, tsQ);
    return out;
}

SituationLine ProbeSubreaper() {
    int v = 0;
    ::prctl(PR_GET_CHILD_SUBREAPER, &v, 0, 0, 0);
    const bool helper = ChildSubreaperEnabled();
    SituationLine l;
    l.check = "child subreaper (this process)";
    if (v && helper) {
        l.observed = "set by adopt_orphans()";
        l.consequence = "orphans of spawned roots re-parent here and stay under this process; "
                        "descendant tracking reaps the ones it saw adopted; their CPU and storage "
                        "I/O fold into this process's rusage. Orphans adopted before the first scan "
                        "are neither tracked nor reaped";
    } else if (v) {
        l.observed = "set, but not by adopt_orphans()";
        l.consequence = "orphans re-parent here; this library does not reap them";
    } else {
        l.observed = "not set";
        l.consequence = "an orphan whose parent exits re-parents to init or the nearest subreaper; "
                        "it stays tracked only if it was discovered before that";
    }
    return l;
}

SituationLine ProbeRoot(uint32_t pid, bool tracksDescendants, uint64_t scanIntervalMs) {
    SituationLine l;
    l.check = "target " + std::to_string(pid);
    auto uid = Uid(pid);
    if (!uid) {
        l.observed = "not found";
        l.consequence = "nothing to trace";
        l.degraded = true;
        return l;
    }
    // Spawned = this process is one of its ancestors.
    const uint32_t me = static_cast<uint32_t>(::getpid());
    bool spawned = false;
    uint32_t p = pid;
    for (int hops = 0; hops < 4096 && p > 1; ++hops) {
        auto st = ReadProcStat("/proc", p);
        if (!st) break;
        p = st->ppid;
        if (p == me) { spawned = true; break; }
    }
    auto io = ReadSmallFile("/proc/" + std::to_string(pid) + "/io");
    const bool same = *uid == ::getuid();
    l.observed = "uid " + std::to_string(*uid) + (same ? " (same)" : " (OTHER user)") + "; " +
                 (spawned ? "spawned by this process" : "ATTACHED (not a descendant of this process)") +
                 "; io " + (io ? "readable" : "DENIED");
    l.consequence = spawned
        ? "its exit status and final CPU/storage I/O reach this process when it is reaped"
        : "this process is not its reaper: its exit tail is not visible here";
    if (tracksDescendants)
        l.consequence += "; a child whose parent lives < " + std::to_string(scanIntervalMs) +
                         " ms can escape discovery";
    if (!io) {
        l.consequence += "; per-PID I/O reads as zero (needs same uid + dumpable target, "
                         "or CAP_DAC_READ_SEARCH + CAP_SYS_PTRACE)";
        l.degraded = true;
    }
    return l;
}

void LogSituation(const std::string& heading, const std::vector<SituationLine>& lines) {
    if (lines.empty()) return;
    std::ostringstream o;
    o << "[cupti-profiler] " << heading << "\n";
    size_t w1 = 0, w2 = 0;
    for (const auto& l : lines) { w1 = std::max(w1, l.check.size()); w2 = std::max(w2, l.observed.size()); }
    w1 = std::min<size_t>(w1, 40);
    w2 = std::min<size_t>(w2, 60);
    for (const auto& l : lines) {
        o << "  " << (l.degraded ? "! " : "  ") << l.check
          << std::string(l.check.size() < w1 ? w1 - l.check.size() : 0, ' ') << "  "
          << l.observed << std::string(l.observed.size() < w2 ? w2 - l.observed.size() : 0, ' ')
          << "  -> " << l.consequence << "\n";
    }
    std::cerr << o.str();
}

} // namespace internal
} // namespace cupti_profiler
