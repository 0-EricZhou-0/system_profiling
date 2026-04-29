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

struct PIDIOSnapshot {
    uint64_t readBytes = 0;       // physical read bytes
    uint64_t writeBytes = 0;      // physical write bytes
    bool accessible = true;       // false if EACCES
};

/// Read disk stats for specified devices from /proc/diskstats.
std::vector<DiskStatSnapshot> ReadDiskStats(const std::vector<std::string>& devices);

/// Read in-flight I/O counts from /sys/block/<dev>/inflight.
DiskInflightSnapshot ReadDiskInflight(const std::string& device);

/// Read per-process I/O from /proc/[pid]/io.
/// Sets accessible=false on EACCES.
PIDIOSnapshot ReadPIDIO(uint32_t pid);

} // namespace internal
} // namespace cupti_profiler
