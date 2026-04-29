// ProfilerSuite — orchestrates GPU, System, and Disk profilers from a .pbtxt config.
#pragma once

#include <cupti_profiler/gpu_profiler.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/event_profiler.h>

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
    void Configure();

    /// Start all enabled profilers.
    void Start();

    /// Stop all enabled profilers.
    void Stop();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
