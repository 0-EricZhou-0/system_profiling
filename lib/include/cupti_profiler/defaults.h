// Defaults shared by the probes' config structs, the suite and the sidecar.
#pragma once

#include <cstdint>

namespace cupti_profiler {

// Every probe (GPU, System, Disk, Events) writes its buffered samples to
// its trace this often; flushIntervalMs = 0 means this. There is no
// "never flush" mode: buffered data would grow without bound.
inline constexpr uint64_t kDefaultFlushIntervalMs = 5000;

// GPU: one decode pass (hardware buffer -> host) per interval.
inline constexpr uint64_t kDefaultDecodeIntervalMs = 1000;

/// flushIntervalMs as configured, with 0 meaning the default.
inline constexpr uint64_t ResolveFlushIntervalMs(uint64_t ms) {
    return ms == 0 ? kDefaultFlushIntervalMs : ms;
}

} // namespace cupti_profiler
