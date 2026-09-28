// Disk Profiler — per-device throughput/queue depth + per-process I/O.
#pragma once

#include <cupti_profiler/process_tracking_probe.h>
#include <cupti_profiler/system_profiler.h>   // SystemProbeMode
#include <cupti_profiler/tracked_process.h>

#include <cupti_profiler/defaults.h>

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
    uint64_t samplingFrequencyHz = 50;              // 50 Hz
    std::vector<std::string> devices;   // block device names (e.g. "nvme0n1")
    // Processes to track per-process I/O (with optional display aliases —
    // see SystemProfilerConfig::Processes).
    std::vector<TrackedProcess> Processes;
    // Periodic flush to outputFile. 0 = kDefaultFlushIntervalMs (5 s).
    uint64_t flushIntervalMs = kDefaultFlushIntervalMs;
    std::string outputFile;
    SystemProbeMode mode = SystemProbeMode::Legacy;
};

class CUPTI_PROFILER_API DiskProfiler : public ProcessTrackingProbe {
public:
    DiskProfiler();
    ~DiskProfiler();
    // Non-movable (see SystemProfiler — same shared_mutex constraint).
    DiskProfiler(DiskProfiler&&) = delete;
    DiskProfiler& operator=(DiskProfiler&&) = delete;

    void Configure(const DiskProfilerConfig& config);
    void Start();

    /// True between a Start() that succeeded and Stop(). Start() fails
    /// (logged, returns with nothing running) when called before
    /// Configure() or when the output file cannot be opened.
    bool IsRunning() const;

    /// Signal sampling to stop (non-blocking). Call Stop() after to join and flush.
    void SignalStop();

    /// Join threads, flush remaining data, close file.
    void Stop();

    /// The host (the launcher) with this PID is a child subreaper whose
    /// orphan reaper (adopt_orphans()) reaps the orphans descendant
    /// tracking reports. Lets a reap chain be resolved: see
    /// IoReapAdjustment in proto/disk_metrics.proto. Latched; 0 = no.
    /// Thread-safe.
    void SetHostReaper(uint32_t hostPid);

    void NoteAdoptedExit(uint32_t pid, uint64_t startTimeTicks) override;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
