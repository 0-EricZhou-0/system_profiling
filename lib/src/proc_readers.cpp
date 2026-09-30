#include "proc_readers.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434   // same number on every architecture
#endif

namespace cupti_profiler {
namespace internal {

CPUStatSnapshot ReadCPUStat() {
    CPUStatSnapshot s;
    std::ifstream f("/proc/stat");
    if (!f) return s;

    std::string line;
    std::getline(f, line);
    // Format: "cpu  user nice system idle iowait irq softirq steal ..."
    if (line.substr(0, 3) != "cpu") return s;

    std::istringstream iss(line.substr(3)); // skip "cpu"
    iss >> s.user >> s.nice >> s.system >> s.idle
        >> s.iowait >> s.irq >> s.softirq >> s.steal;
    return s;
}

std::optional<uint64_t> ReadPIDCpuTimeNs(uint32_t pid) {
    // A process CPU clock covers the whole thread group, including
    // threads that have already exited. Per-thread
    // /proc/<pid>/task/*/schedstat would miss those, and
    // /proc/<pid>/schedstat reports only the group leader.
    clockid_t clk;
    if (clock_getcpuclockid(static_cast<pid_t>(pid), &clk) != 0) return std::nullopt;
    struct timespec ts;
    if (clock_gettime(clk, &ts) != 0) return std::nullopt;
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull
         + static_cast<uint64_t>(ts.tv_nsec);
}

MemInfoSnapshot ReadMemInfo() {
    MemInfoSnapshot s;
    std::ifstream f("/proc/meminfo");
    if (!f) return s;

    std::string line;
    while (std::getline(f, line)) {
        uint64_t val = 0;
        if (line.compare(0, 9, "MemTotal:") == 0) {
            std::istringstream(line.substr(9)) >> val;
            s.totalKB = val;
        } else if (line.compare(0, 8, "MemFree:") == 0) {
            std::istringstream(line.substr(8)) >> val;
            s.freeKB = val;
        } else if (line.compare(0, 13, "MemAvailable:") == 0) {
            std::istringstream(line.substr(13)) >> val;
            s.availableKB = val;
        } else if (line.compare(0, 8, "Buffers:") == 0) {
            std::istringstream(line.substr(8)) >> val;
            s.buffersKB = val;
        } else if (line.compare(0, 7, "Cached:") == 0) {
            std::istringstream(line.substr(7)) >> val;
            s.cachedKB = val;
        }
    }
    return s;
}

std::optional<PIDStatmSnapshot> ReadPIDStatm(uint32_t pid, int* err) {
    errno = 0;
    auto text = ReadSmallFile("/proc/" + std::to_string(pid) + "/statm");
    if (!text) {
        if (err) *err = errno ? errno : EIO;
        return std::nullopt;
    }
    // Format: size resident shared text lib data dt
    PIDStatmSnapshot s;
    std::istringstream f(*text);
    if (!(f >> s.VMSPages >> s.RSSPages >> s.sharedPages)) {
        if (err) *err = ENODATA;
        return std::nullopt;
    }
    return s;
}

bool IsExiting(uint32_t pid) {
    constexpr uint64_t kPfExiting = 0x4;
    auto st = ReadProcStat("/proc", pid);
    return !st || st->state == 'Z' || st->state == 'X' || (st->flags & kPfExiting);
}

ReadFailure ClassifyReadFailure(uint32_t pid, int err) {
    if (err == ENOENT || err == ESRCH) return ReadFailure::Gone;
    if (IsExiting(pid)) return ReadFailure::Gone;
    if (err == EACCES || err == EPERM) return ReadFailure::Unreadable;
    return ReadFailure::Other;
}

std::optional<std::string> ReadSmallFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::nullopt;
    std::string out;
    char buf[4096];
    for (;;) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {                      // not a partial file: errno kept
            int e = errno;
            ::close(fd);
            errno = e;
            return std::nullopt;
        }
        if (n == 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return out;
}

std::optional<bool> IgnoresSigchld(uint32_t pid) {
    auto text = ReadSmallFile("/proc/" + std::to_string(pid) + "/status");
    if (!text) return std::nullopt;
    const auto at = text->find("\nSigIgn:");
    if (at == std::string::npos) return std::nullopt;
    char* end = nullptr;
    const char* hex = text->c_str() + at + 8;
    errno = 0;
    const unsigned long long mask = std::strtoull(hex, &end, 16);
    if (end == hex || errno != 0) return std::nullopt;
    return (mask >> (SIGCHLD - 1)) & 1;
}

std::optional<ProcStat> ReadProcStat(const std::string& procRoot, uint32_t pid) {
    auto text = ReadSmallFile(procRoot + "/" + std::to_string(pid) + "/stat");
    if (!text) return std::nullopt;
    // "pid (comm) state ppid ..." — comm may itself contain spaces and
    // parentheses, so split at the LAST ')'.
    size_t open = text->find('(');
    size_t close = text->rfind(')');
    if (open == std::string::npos || close == std::string::npos || close < open)
        return std::nullopt;
    ProcStat st;
    st.comm = text->substr(open + 1, close - open - 1);
    std::istringstream rest(text->substr(close + 1));
    std::string field;
    // Fields after comm: index 0 = state (field 3), 1 = ppid (4), ...,
    // 6 = flags (9), 13 = cutime (16), 14 = cstime (17), 19 = starttime (22).
    for (int i = 0; i <= 19 && (rest >> field); ++i) {
        if (i == 0) st.state = field.empty() ? '?' : field[0];
        else if (i == 1) st.ppid = static_cast<uint32_t>(std::strtoul(field.c_str(), nullptr, 10));
        else if (i == 6) st.flags = std::strtoull(field.c_str(), nullptr, 10);
        else if (i == 13) st.cutime = std::strtoull(field.c_str(), nullptr, 10);
        else if (i == 14) st.cstime = std::strtoull(field.c_str(), nullptr, 10);
        else if (i == 19) st.startTime = std::strtoull(field.c_str(), nullptr, 10);
    }
    if (st.state == '?') return std::nullopt;   // truncated or malformed
    return st;
}

std::string UnreadableWarning(const char* probeTag, const char* file, uint32_t pid,
                              const std::string& comm, int err, ReadFailure kind) {
    const bool io = std::string(file) == "io";
    std::string m = std::string("[") + probeTag + "] Warning: cannot read /proc/" +
                    std::to_string(pid) + "/" + file + " of " + (comm.empty() ? "process" : comm) +
                    " (pid " + std::to_string(pid) + "): " + std::strerror(err);
    if (kind == ReadFailure::Unreadable)
        m += io ? " -- not readable by this process: it runs as another uid or is not dumpable "
                  "(reading needs the same uid and a dumpable process, or CAP_SYS_PTRACE)"
                : " -- /proc is mounted with hidepid, or the process is another uid's";
    m += io ? ". Its I/O is missing from the trace (not zero) while this lasts: "
              "io_unreadable_since_ns / io_unreadable_ticks in the process table."
            : ". Its memory values are NaN (missing, not zero) while this lasts: "
              "mem_unreadable_since_ns / mem_unreadable_ticks in the process table.";
    m += " At most once a second per process while it lasts.";
    return m;
}

int PidfdOpen(uint32_t pid) {
    return static_cast<int>(::syscall(SYS_pidfd_open, static_cast<pid_t>(pid), 0));
}

bool PidfdExited(int pidfd) {
    struct pollfd p{pidfd, POLLIN, 0};
    int r;
    do { r = ::poll(&p, 1, 0); } while (r < 0 && errno == EINTR);
    return r > 0;
}

long GetPageSize() {
    static long ps = sysconf(_SC_PAGESIZE);
    return ps;
}

std::string ProcRoot() {
    const char* env = std::getenv("CUPTI_PROFILER_PROC_ROOT");   // test-only
    std::string root = (env && *env) ? env : "/proc";
    while (root.size() > 1 && root.back() == '/') root.pop_back();
    return root;
}

uint64_t BootTicksToSteadyNs(uint64_t ticks) {
    if (ticks == 0) return 0;
    // Field 22 counts from boot on CLOCK_BOOTTIME, which runs on through
    // suspend; CLOCK_MONOTONIC does not. Shift by their current offset.
    struct timespec boot{}, mono{};
    ::clock_gettime(CLOCK_BOOTTIME, &boot);
    ::clock_gettime(CLOCK_MONOTONIC, &mono);
    const int64_t offset =
        (static_cast<int64_t>(boot.tv_sec) - mono.tv_sec) * 1000000000LL +
        (static_cast<int64_t>(boot.tv_nsec) - mono.tv_nsec);
    const int64_t ns = static_cast<int64_t>(ticks) * (1000000000LL / GetCLKTCK()) - offset;
    return ns > 0 ? static_cast<uint64_t>(ns) : 0;
}

long GetCLKTCK() {
    static long clk = sysconf(_SC_CLK_TCK);
    return clk;
}

} // namespace internal
} // namespace cupti_profiler
