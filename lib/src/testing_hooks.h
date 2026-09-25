// Internal side of <cupti_profiler/testing.h>: the call sites.
#pragma once

#include <cstdint>

namespace cupti_profiler {
namespace internal {

/// Called by the System/Disk flush threads after writing a flush and
/// before committing its removals. No-op unless a test armed the gate.
void PassFlushGate();

/// Called by the System probe right after reading `pid`'s values. If a
/// test armed testing::KillAfterNextRead(pid), kills it, waits for its
/// exit and returns true: the caller then treats its reading as another
/// process's. Disarmed: one relaxed atomic load.
bool PassReadHook(uint32_t pid);

} // namespace internal
} // namespace cupti_profiler
