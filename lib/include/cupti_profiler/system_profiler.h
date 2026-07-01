// System Profiler — CPU utilization + memory usage (system-wide and per-process).
#pragma once

#include <cupti_profiler/process_tracking_probe.h>
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

// Where the sampler + flush threads live. LEGACY runs them in the
// workload's address space (existing behavior); SIDECAR moves them
// to an out-of-process cupti-profiler-sidecar so they don't inflate
// the workload's per-PID CPU accounting. See
// proto/profiler_config.proto :: SystemProbeMode for the wire enum.
enum class SystemProbeMode {
    Legacy  = 1,
    Sidecar = 2,
};

struct CUPTI_PROFILER_API SystemProfilerConfig {
    uint64_t samplingFrequencyHz = 100;             // 100 Hz
    // Processes to track per-process. Empty = system-wide only.
    // Each entry carries a PID and an optional display alias; visualizers
    // render labels as "<alias> (PID xxx)" when alias is non-empty,
    // otherwise plain "PID xxx".
    std::vector<TrackedProcess> Processes;
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;
    SystemProbeMode mode = SystemProbeMode::Legacy;
};

class CUPTI_PROFILER_API SystemProfiler : public ProcessTrackingProbe {
public:
    SystemProfiler();
    ~SystemProfiler();
    // Non-movable: ProcessTrackingProbe owns a shared_mutex. The suite
    // constructs this in place inside its Impl.
    SystemProfiler(SystemProfiler&&) = delete;
    SystemProfiler& operator=(SystemProfiler&&) = delete;

    void Configure(const SystemProfilerConfig& config);
    void Start();

    /// Signal sampling to stop (non-blocking). Call Stop() after to join and flush.
    void SignalStop();

    /// Join threads, flush remaining data, close file.
    void Stop();

    // AddTrackedProcess / RemoveTrackedProcess are inherited from
    // ProcessTrackingProbe — call them between Start() and Stop() to
    // adjust the tracked PID set mid-run.

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
