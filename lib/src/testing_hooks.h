// Internal side of <cupti_profiler/testing.h>: the call sites.
#pragma once

#include <cstdint>

#include <cupti_profiler/testing.h>

namespace cupti_profiler {
namespace internal {

/// Called by the System/Disk flush threads after writing a flush and
/// before committing its removals. No-op unless a test armed the gate.
void PassFlushGate();

/// Called by the System and Disk probes right after reading `pid`'s
/// values. If a test armed testing::KillAfterNextRead(pid, probe) (or
/// the environment variable) and this is the read it named, kills
/// `pid`, waits for its exit and returns true: the caller then treats
/// its reading as another process's. Disarmed: one relaxed atomic load.
bool PassReadHook(uint32_t pid, testing::ReadProbe probe);

} // namespace internal
} // namespace cupti_profiler
