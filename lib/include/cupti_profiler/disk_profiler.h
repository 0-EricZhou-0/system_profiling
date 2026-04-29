// Disk Profiler — per-device throughput/queue depth + per-process I/O.
#pragma once

#include <cupti_profiler/tracked_process.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __declspec(dllexport)
  #else
    #define CUPTI_PROFILER_API __declspec(dllimport)
  #endif
#else
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __attribute__((visibility("default")))
  #else
    #define CUPTI_PROFILER_API
  #endif
#endif

namespace cupti_profiler {

struct CUPTI_PROFILER_API DiskProfilerConfig {
    uint64_t samplingFrequencyHz = 10;              // 10 Hz
    std::vector<std::string> devices;   // block device names (e.g. "nvme0n1")
    // Processes to track per-process I/O (with optional display aliases —
    // see SystemProfilerConfig::Processes).
    std::vector<TrackedProcess> Processes;
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;
};

class CUPTI_PROFILER_API DiskProfiler {
public:
    DiskProfiler();
    ~DiskProfiler();
    DiskProfiler(DiskProfiler&&) noexcept;
    DiskProfiler& operator=(DiskProfiler&&) noexcept;

    void Configure(const DiskProfilerConfig& config);
    void Start();

    /// Signal sampling to stop (non-blocking). Call Stop() after to join and flush.
    void SignalStop();

    /// Join threads, flush remaining data, close file.
    void Stop();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
