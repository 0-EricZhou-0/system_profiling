// CUPTI PM Sampling Profiler — Public API
// Link against libcupti_profiler.so to use.

#pragma once

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

/// A single PM sampling data point.
struct CUPTI_PROFILER_API SamplerRange {
    size_t rangeIndex;
    uint64_t startTimestamp;  // nanoseconds (CUPTI clock)
    uint64_t endTimestamp;
    std::vector<double> metricValues;
};

/// Configuration for GpuProfiler.
struct CUPTI_PROFILER_API ProfilerConfig {
    int deviceIndex = 0;
    uint64_t samplingFrequencyHz = 10000;          // 10 kHz
    size_t hwBufferSize = 512 * 1024 * 1024;    // 512 MB
    uint64_t maxSamples = 50000;
    std::vector<std::string> metrics;

    // Periodic flush. 0 = disabled (single write at end).
    uint64_t flushIntervalMs = 10000;

    // Output file path. Empty = no file output.
    std::string outputFile;
};

/// High-level GPU profiler using CUPTI PM Sampling.
///
/// Usage:
///   GpuProfiler profiler;
///   profiler.Configure(config);
///   profiler.Start();
///   // ... run your CUDA workload ...
///   profiler.Stop();
///
/// For region/event annotations, use the EventProfiler instead — see
/// cupti_profiler/event_profiler.h.
class CUPTI_PROFILER_API GpuProfiler {
public:
    GpuProfiler();
    ~GpuProfiler();
    GpuProfiler(GpuProfiler&&) noexcept;
    GpuProfiler& operator=(GpuProfiler&&) noexcept;

    /// Initialize the profiler. Must be called before Start().
    /// The caller must have an active CUDA context on config.deviceIndex.
    void Configure(const ProfilerConfig& config);

    /// Start PM sampling, background decode thread, and optional flush thread.
    void Start();

    /// Stop PM sampling, join threads, write remaining data.
    void Stop();

    /// Atomically drain all collected samples. Can be called during or after profiling.
    std::vector<SamplerRange> DrainSamples();

    // Device info (available after Configure()).
    std::string GetDeviceName() const;
    std::string GetChipName() const;
    double GetPeakDramBwGbps() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
