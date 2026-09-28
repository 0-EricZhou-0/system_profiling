// Defaults shared by the probes' config structs, the suite and the sidecar.
#pragma once

#include <cstdint>

namespace cupti_profiler {

// Every probe (GPU, System, Disk, Events) writes its buffered samples to
// its trace this often; flushIntervalMs = 0 means this. There is no
// "never flush" mode: buffered data would grow without bound.
inline constexpr uint64_t kDefaultFlushIntervalMs = 5000;

// Sampling rates when the configured one is 0 (unset). GPU PM sampling
// costs the same at 100-1000 Hz (phase 6b); pick the rate for time
// resolution and trace size (~4 kB/s per 100 Hz with 4 metrics).
inline constexpr uint64_t kDefaultGpuSamplingHz    = 100;
inline constexpr uint64_t kDefaultSystemSamplingHz = 100;
inline constexpr uint64_t kDefaultDiskSamplingHz   = 100;

// GPU: one decode pass (hardware buffer -> host) per interval.
inline constexpr uint64_t kDefaultDecodeIntervalMs = 1000;

/// flushIntervalMs as configured, with 0 meaning the default.
inline constexpr uint64_t ResolveFlushIntervalMs(uint64_t ms) {
    return ms == 0 ? kDefaultFlushIntervalMs : ms;
}

} // namespace cupti_profiler
