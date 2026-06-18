// Internal: /proc filesystem readers for CPU and memory metrics.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
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

// Per-thread sum_exec_runtime baselines for a process. Keyed by TID;
// value is field 1 of /proc/<pid>/task/<tid>/schedstat — nanoseconds
// the thread has spent on a CPU. Aggregating across the whole thread
// group is necessary because /proc/<tgid>/schedstat reports only the
// group leader's task_struct (unlike /proc/<tgid>/stat's utime+stime
// which the kernel aggregates with whole=1).
using PIDThreadCpuMap = std::unordered_map<uint32_t, uint64_t>;

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

/// Read per-thread sum_exec_runtime for every thread in the given
/// process by walking /proc/[pid]/task/. Returns an empty map if the
/// process directory cannot be opened. Threads that disappear
/// mid-walk are silently skipped; the caller is expected to diff
/// successive maps to compute on-CPU deltas (delta = sum over visible
/// TIDs of cur - prev, treating absent prev as 0). This naturally
/// absorbs spawned threads and ignores the small slice of CPU time an
/// exited thread accrued between its last appearance and exit.
PIDThreadCpuMap ReadPIDSchedStatPerThread(uint32_t pid);

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
