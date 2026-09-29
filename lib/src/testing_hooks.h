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

/// The errno a test wants `probe`'s read of `pid`'s per-PID file to fail
/// with (testing::SetReadError), 0 = read it. Disarmed: one relaxed
/// atomic load.
int ReadErrorFor(uint32_t pid, testing::ReadProbe probe);

/// At a System/Disk probe's Stop(): with
/// CUPTI_PROFILER_TEST_REPORT_WARN_STATE set, print its warning-key
/// count before it drops them (testing::WarnStateSize for the sidecar).
void ReportWarnStateAtStop(const char* probe, size_t n);

/// Called by every flush thread after its write: sleeps for the delay a
/// test set with testing::SetFlushDelayMs. Off: one relaxed atomic load.
void PassFlushDelay();

/// Called by ProfilerSuite::Stop() once it has begun.
void PassStopDelay();

/// Called by the GPU decode thread before each pass: sleeps once for the
/// stall a test set with testing::StallNextDecodeMs.
void PassDecodeStall();

/// testing::SetBacklogReportPeriodMs, 30000 by default.
unsigned BacklogReportPeriodMs();

} // namespace internal
} // namespace cupti_profiler
