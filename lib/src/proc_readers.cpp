#include "proc_readers.h"

#include <ctime>
#include <fstream>
#include <sstream>
#include <unistd.h>

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
