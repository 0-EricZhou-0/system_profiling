#include "proc_readers.h"

#include <cstring>
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

PIDSchedStatSnapshot ReadPIDSchedStat(uint32_t pid) {
    PIDSchedStatSnapshot s;
    std::string path = "/proc/" + std::to_string(pid) + "/schedstat";
    std::ifstream f(path);
    if (!f) return s;

    // Format (Documentation/scheduler/sched-stats.rst):
    //   <sum_exec_runtime> <run_delay> <pcount>
    // We only consume field 1 — the nanoseconds the task has spent
    // on a CPU. The remaining two fields (runqueue wait, schedule
    // count) are gated by the kernel.sched_schedstats sysctl and not
    // used by this profiler.
    f >> s.cpuTimeNs;
    return s;
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
