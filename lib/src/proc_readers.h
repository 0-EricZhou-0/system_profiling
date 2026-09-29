// Internal: /proc filesystem readers for CPU and memory metrics.
#pragma once

#include <cstdint>
#include <optional>
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

/// Read the total on-CPU time of the whole process, in nanoseconds,
/// from its process CPU clock (clock_getcpuclockid + clock_gettime).
/// The kernel sums sum_exec_runtime over every live thread plus the
/// time already folded into the thread group by threads that have
/// exited, so it is monotonic for the life of the process and loses
/// nothing to short-lived threads. One syscall, independent of the
/// thread count, and readable for other users' processes without
/// capabilities. Returns nullopt if the process no longer exists
/// (ESRCH/EINVAL) or cannot be read.
std::optional<uint64_t> ReadPIDCpuTimeNs(uint32_t pid);

/// Read system memory info from /proc/meminfo.
MemInfoSnapshot ReadMemInfo();

/// Read per-process memory from /proc/[pid]/statm. nullopt if it cannot
/// be opened or parsed, with *err = the errno (ENODATA: malformed).
std::optional<PIDStatmSnapshot> ReadPIDStatm(uint32_t pid, int* err = nullptr);

/// The parts of /proc/<pid>/stat that process discovery needs.
struct ProcStat {
    std::string comm;          // field 2, without the parentheses
    char        state = '?';   // field 3 ('Z' = zombie)
    uint32_t    ppid  = 0;     // field 4
    uint64_t    cutime = 0;    // field 16, clock ticks: reaped children's user time
    uint64_t    cstime = 0;    // field 17, clock ticks: reaped children's system time
    uint64_t    startTime = 0; // field 22, clock ticks since boot
    uint64_t    flags = 0;     // field 9, PF_* (PF_EXITING = 0x4)
};

/// Parse <procRoot>/<pid>/stat. procRoot is "/proc" except in tests
/// (see process_discovery.h). nullopt if missing or malformed.
std::optional<ProcStat> ReadProcStat(const std::string& procRoot, uint32_t pid);

/// Why a per-PID /proc read of a tracked process failed (err = errno):
///   Gone       ENOENT/ESRCH, or any error while the process is exiting:
///              /proc/<pid>/stat is missing, its state is Z/X, or
///              PF_EXITING is set. /proc/<pid>/io of an exiting process
///              fails with EACCES from exit_mm until it is reaped
///              (measured, kernel 5.15: ~0.5 s for 8 GB mapped, while its
///              pidfd still reports it alive), so EACCES alone is no
///              permission failure.
///   Unreadable EACCES/EPERM of a live process: another uid, or not
///              dumpable, and no CAP_SYS_PTRACE (or hidepid).
///   Other      anything else.
enum class ReadFailure { Gone, Unreadable, Other };

/// Is the process exiting (or gone)? /proc/<pid>/stat missing, state Z/X,
/// or PF_EXITING set — the exit evidence ClassifyReadFailure uses.
bool IsExiting(uint32_t pid);
ReadFailure ClassifyReadFailure(uint32_t pid, int err);

/// The stderr warning (one line, no newline) for an Unreadable / Other
/// read failure. probeTag "Disk" or "System"; file "io" or "statm".
std::string UnreadableWarning(const char* probeTag, const char* file, uint32_t pid,
                              const std::string& comm, int err, ReadFailure kind);

/// Whole small file (e.g. a /proc entry) as a string; nullopt, with
/// errno set, if it cannot be opened or read.
std::optional<std::string> ReadSmallFile(const std::string& path);

/// pidfd_open(2) via the raw syscall (conda Pythons and older glibcs
/// lack a wrapper; the syscall itself is in Linux >= 5.3). Returns the
/// fd (O_CLOEXEC) or -1 with errno set.
int PidfdOpen(uint32_t pid);

/// True if the process a pidfd refers to has exited (the pidfd polls
/// readable once the whole thread group is gone, reaped or not).
bool PidfdExited(int pidfd);

/// "/proc", or the test-only CUPTI_PROFILER_PROC_ROOT override (see
/// process_discovery.h). Used for the process-table reads (children,
/// stat, comm) of discovery and of the probes' tracked-process lists.
std::string ProcRoot();

/// A /proc/<pid>/stat start time (field 22: clock ticks since boot, on
/// CLOCK_BOOTTIME) converted to the trace clock (steady_clock =
/// CLOCK_MONOTONIC, ns). Resolution is one clock tick (10 ms at
/// USER_HZ=100). 0 if ticks is 0.
uint64_t BootTicksToSteadyNs(uint64_t ticks);

/// Get the system page size in bytes (typically 4096).
long GetPageSize();

/// Get CLK_TCK (jiffies per second, typically 100).
long GetCLKTCK();

} // namespace internal
} // namespace cupti_profiler
