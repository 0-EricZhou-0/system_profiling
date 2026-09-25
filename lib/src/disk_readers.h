// Internal: /proc and /sys readers for disk I/O metrics.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cupti_profiler {
namespace internal {

struct DiskStatSnapshot {
    std::string device;
    uint64_t sectorsRead = 0;     // field 6 in /proc/diskstats (× 512 = bytes)
    uint64_t sectorsWritten = 0;  // field 10 in /proc/diskstats
};

struct DiskInflightSnapshot {
    uint32_t readInflight = 0;
    uint32_t writeInflight = 0;
};

// The five byte counters of /proc/<pid>/io (see docs/metric-model.md,
// "Per-PID I/O counters", for what each one sees).
struct PIDIOSnapshot {
    uint64_t rchar = 0;                // syscall layer: bytes read() et al. returned
    uint64_t wchar = 0;                // syscall layer: bytes write() et al. accepted
    uint64_t readBytes = 0;            // storage layer: bytes fetched from storage
    uint64_t writeBytes = 0;           // storage layer: bytes dirtied in the page cache
    uint64_t cancelledWriteBytes = 0;  // dirtied bytes discarded before writeback
    bool accessible = true;            // false if the file cannot be opened
};

/// Read disk stats for specified devices from /proc/diskstats.
std::vector<DiskStatSnapshot> ReadDiskStats(const std::vector<std::string>& devices);

/// Read in-flight I/O counts from /sys/block/<dev>/inflight.
DiskInflightSnapshot ReadDiskInflight(const std::string& device);

/// Read per-process I/O from /proc/[pid]/io.
/// Sets accessible=false if it cannot be opened (EACCES, or the PID is gone).
PIDIOSnapshot ReadPIDIO(uint32_t pid);

} // namespace internal
} // namespace cupti_profiler
