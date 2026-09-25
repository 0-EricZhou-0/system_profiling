#include "proc_readers.h"

#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
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

PIDStatmSnapshot ReadPIDStatm(uint32_t pid) {
    PIDStatmSnapshot s;
    std::string path = "/proc/" + std::to_string(pid) + "/statm";
    std::ifstream f(path);
    if (!f) return s;

    // Format: size resident shared text lib data dt
    f >> s.VMSPages >> s.RSSPages >> s.sharedPages;
    return s;
}

std::optional<std::string> ReadSmallFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::nullopt;
    std::string out;
    char buf[4096];
    for (;;) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return out;
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
    // 19 = starttime (22).
    for (int i = 0; i <= 19 && (rest >> field); ++i) {
        if (i == 0) st.state = field.empty() ? '?' : field[0];
        else if (i == 1) st.ppid = static_cast<uint32_t>(std::strtoul(field.c_str(), nullptr, 10));
        else if (i == 19) st.startTime = std::strtoull(field.c_str(), nullptr, 10);
    }
    if (st.state == '?') return std::nullopt;   // truncated or malformed
    return st;
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

long GetCLKTCK() {
    static long clk = sysconf(_SC_CLK_TCK);
    return clk;
}

} // namespace internal
} // namespace cupti_profiler
