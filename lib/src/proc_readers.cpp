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

PIDStatSnapshot ReadPIDStat(uint32_t pid) {
    PIDStatSnapshot s;
    std::string path = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream f(path);
    if (!f) return s;

    std::string line;
    std::getline(f, line);

    // Field 2 (comm) can contain spaces and parentheses.
    // Find the last ')' to anchor field parsing.
    auto lastParen = line.rfind(')');
    if (lastParen == std::string::npos) return s;

    // Fields after comm start at lastParen+2 (skip ") ").
    // These are fields 3..N (1-indexed). We need:
    //   field 14 = utime  → index 11 after comm (14 - 3 = 11)
    //   field 15 = stime  → index 12
    //   field 42 = delayacct_blkio_ticks → index 39
    std::istringstream iss(line.substr(lastParen + 2));
    std::string token;
    std::vector<std::string> fields;
    while (iss >> token) {
        fields.push_back(token);
    }

    // fields[0] = field 3 (state), fields[11] = field 14 (utime), etc.
    if (fields.size() > 11) s.utime = std::stoull(fields[11]);
    if (fields.size() > 12) s.stime = std::stoull(fields[12]);
    if (fields.size() > 39) s.blkioTicks = std::stoull(fields[39]);

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
