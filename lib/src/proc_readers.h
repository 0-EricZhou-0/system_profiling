// Internal: /proc filesystem readers for CPU and memory metrics.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cupti_profiler {
namespace internal {

struct CPUStatSnapshot {
    uint64_t user = 0, nice = 0, system = 0, idle = 0;
    uint64_t iowait = 0, irq = 0, softirq = 0, steal = 0;
    uint64_t Total() const {
        return user + nice + system + idle + iowait + irq + softirq + steal;
    }
    uint64_t Busy() const {
        return Total() - idle - iowait;
    }
};

struct PIDSchedStatSnapshot {
    // /proc/<pid>/schedstat field 1: sum_exec_runtime (nanoseconds the
    // task has spent on a CPU). Always tracked by the kernel scheduler
    // regardless of the kernel.sched_schedstats sysctl, so we get
    // nanosecond-precision per-PID CPU time without the 10 ms quantization
    // that /proc/<pid>/stat's utime/stime impose.
    uint64_t cpuTimeNs = 0;
};

struct MemInfoSnapshot {
    uint64_t totalKB = 0;
    uint64_t freeKB = 0;
    uint64_t availableKB = 0;
    uint64_t buffersKB = 0;
    uint64_t cachedKB = 0;
};

struct PIDStatmSnapshot {
    uint64_t VMSPages = 0;        // field 1: total program size
    uint64_t RSSPages = 0;        // field 2: resident set size
    uint64_t sharedPages = 0;     // field 3: shared pages
};

/// Read aggregate CPU stats from /proc/stat (first "cpu" line).
CPUStatSnapshot ReadCPUStat();

/// Read per-process CPU time from /proc/[pid]/schedstat.
/// Returns zero-initialized snapshot if the process does not exist.
PIDSchedStatSnapshot ReadPIDSchedStat(uint32_t pid);

/// Read system memory info from /proc/meminfo.
MemInfoSnapshot ReadMemInfo();

/// Read per-process memory from /proc/[pid]/statm.
PIDStatmSnapshot ReadPIDStatm(uint32_t pid);

/// Get the system page size in bytes (typically 4096).
long GetPageSize();

/// Get CLK_TCK (jiffies per second, typically 100).
long GetCLKTCK();

} // namespace internal
} // namespace cupti_profiler
