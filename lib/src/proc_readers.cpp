#include "proc_readers.h"

#include <cctype>
#include <cstring>
#include <dirent.h>
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

PIDThreadCpuMap ReadPIDSchedStatPerThread(uint32_t pid) {
    PIDThreadCpuMap m;
    std::string taskDir = "/proc/" + std::to_string(pid) + "/task";
    DIR* d = opendir(taskDir.c_str());
    if (!d) return m;

    // Each /proc/<pid>/task/<tid>/schedstat format
    // (Documentation/scheduler/sched-stats.rst):
    //   <sum_exec_runtime> <run_delay> <pcount>
    // We only consume field 1 — the nanoseconds this thread has spent
    // on a CPU. The two trailing fields (runqueue wait, schedule
    // count) are gated by the kernel.sched_schedstats sysctl and not
    // used by this profiler.
    while (struct dirent* e = readdir(d)) {
        const char* n = e->d_name;
        if (!std::isdigit(static_cast<unsigned char>(n[0]))) continue;
        uint32_t tid = static_cast<uint32_t>(std::strtoul(n, nullptr, 10));
        if (tid == 0) continue;

        std::string path = taskDir + "/" + n + "/schedstat";
        std::ifstream f(path);
        if (!f) continue;     // thread exited mid-walk
        uint64_t ns = 0;
        f >> ns;
        if (!f) continue;
        m.emplace(tid, ns);
    }
    closedir(d);
    return m;
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
