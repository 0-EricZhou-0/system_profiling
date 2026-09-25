// Internal side of <cupti_profiler/testing.h>: the call sites.
#pragma once

namespace cupti_profiler {
namespace internal {

/// Called by the System/Disk flush threads after writing a flush and
/// before committing its removals. No-op unless a test armed the gate.
void PassFlushGate();

} // namespace internal
} // namespace cupti_profiler
