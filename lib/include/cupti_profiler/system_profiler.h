// System Profiler — CPU utilization + memory usage (system-wide and per-process).
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

struct CUPTI_PROFILER_API SystemProfilerConfig {
    uint64_t samplingFrequencyHz = 100;             // 100 Hz
    // Processes to track per-process. Empty = system-wide only.
    // Each entry carries a PID and an optional display alias; visualizers
    // render labels as "<alias> (PID xxx)" when alias is non-empty,
    // otherwise plain "PID xxx".
    std::vector<TrackedProcess> Processes;
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;
};

class CUPTI_PROFILER_API SystemProfiler {
public:
    SystemProfiler();
    ~SystemProfiler();
    SystemProfiler(SystemProfiler&&) noexcept;
    SystemProfiler& operator=(SystemProfiler&&) noexcept;

    void Configure(const SystemProfilerConfig& config);
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
