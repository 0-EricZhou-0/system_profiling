// ProfilerSuite — orchestrates GPU, System, and Disk profilers from a .pbtxt config.
#pragma once

#include <cupti_profiler/gpu_profiler.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/event_profiler.h>
#include <cupti_profiler/profiler_error.h>

#include <memory>
#include <string>

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

class CUPTI_PROFILER_API ProfilerSuite {
public:
    ProfilerSuite();
    ~ProfilerSuite();
    ProfilerSuite(ProfilerSuite&&) noexcept;
    ProfilerSuite& operator=(ProfilerSuite&&) noexcept;

    /// Load configuration from a protobuf text format (.pbtxt) file.
    void LoadConfig(const std::string& pbtxtPath);

    /// Load configuration from a serialized ProfilerSuiteConfig protobuf
    /// (binary wire format). Used by language bindings that build the
    /// config in their own runtime and want to skip the .pbtxt round-trip.
    void LoadConfigFromBytes(const std::string& serializedProto);

    /// Access individual profilers (available after LoadConfig).
    GpuProfiler& GetGPUProfiler();
    SystemProfiler& GetSystemProfiler();
    DiskProfiler& GetDiskProfiler();
    EventProfiler& GetEventProfiler();

    /// Configure all enabled profilers.
    ///
    /// Under SystemProbeMode::Sidecar, this is where the sidecar
    /// process is spawned and the initial handshake (config +
    /// sync anchor) runs — capability failures surface here as
    /// ProfilerError::SidecarMissingCaps rather than as a silent
    /// no-sample run. Under Legacy, always returns Ok.
    /// Returns ProfilerError::NotConfigured if LoadConfig has not
    /// been called yet.
    ProfilerError Configure();

    /// Start all enabled profilers.
    void Start();

    /// Stop all enabled profilers.
    void Stop();

    /// Begin tracking a PID mid-run. Fans out to every enabled probe
    /// that supports per-PID sampling (currently System + Disk).
    /// Thread-safe.
    void AddTrackedProcess(uint32_t pid, std::string alias = {});

    /// Stop tracking a PID mid-run. The PID appears one more time in
    /// the next flush of each affected probe (with TrackedProcessV2.
    /// removed=true) before being dropped. Thread-safe.
    void RemoveTrackedProcess(uint32_t pid);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
