// TEST-ONLY hooks. Not a supported API: names and behaviour may change
// without notice, and nothing outside the test suite should call them.
//
// Flush gate: lets a test stop a System/Disk flush thread between the
// point where it has taken its tracked-process snapshot and written the
// flush, and the point where it commits the removals that flush emitted
// (ProcessTrackingProbe::CommitPendingRemovals). A RemoveTrackedProcess()
// made while the thread is held there lands exactly in the window that
// cad56b2 fixed, so the race can be driven deterministically instead of
// hoped for. Applies to in-process (LEGACY) probes only; the sidecar is
// another process.
//
// Disarmed, the gate costs one relaxed atomic load per flush.
#pragma once

#include <cstdint>

#include <cupti_profiler/process_tracking_probe.h>   // CUPTI_PROFILER_API

namespace cupti_profiler {
namespace testing {

/// Arm the gate: the next flush (of any in-process System/Disk probe)
/// that reaches it blocks there until ReleaseFlushGate(). One-shot.
CUPTI_PROFILER_API void ArmFlushGate();

/// Wait up to timeoutMs for a flush to be held at the armed gate.
/// Returns true once one is.
CUPTI_PROFILER_API bool WaitFlushHeld(unsigned timeoutMs);

/// Let the held flush continue (or disarm a gate nothing reached yet).
CUPTI_PROFILER_API void ReleaseFlushGate();

/// Which probe's per-PID read the kill-after-read hook follows.
enum class ReadProbe : uint8_t { System = 0, Disk = 1 };

/// Death between a read and the liveness check (gap A). One-shot: the
/// next time an in-process `probe` reads `pid` (System: its CPU clock;
/// Disk: its /proc/<pid>/io), the hook SIGKILLs `pid` right after that
/// read, waits until it has exited, and marks the reading as another
/// process's (System adds 1000 s to the CPU value; Disk adds 1 TB to
/// each of the five I/O counters), as if a new owner of the number had
/// been read. The probe polls its pidfds after the reads, so such a
/// reading must be discarded and never reach the trace. In-process
/// (LEGACY) probes; for the sidecar see ArmKillAfterReadFromEnv().
CUPTI_PROFILER_API void KillAfterNextRead(uint32_t pid, ReadProbe probe = ReadProbe::System);

/// The same hook, armed from the environment, for probes in another
/// process (the sidecar calls this at startup; the variable reaches it
/// through the environment it inherits from the host):
///
///     CUPTI_PROFILER_TEST_KILL_AFTER_READ=<system|disk>:<pid>:<n>
///
/// kills `pid` right after that probe's n-th read of it (n >= 1).
/// Unset: does nothing. Returns true if it armed the hook.
CUPTI_PROFILER_API bool ArmKillAfterReadFromEnv();

/// Slow writer: every periodic flush of an in-process probe (GPU,
/// System, Disk, Events) takes `ms` longer, as if the disk were slow
/// (the flush thread sleeps after its write). 0 = off.
CUPTI_PROFILER_API void SetFlushDelayMs(unsigned ms);

/// Stalled decode thread: the next GPU decode pass of every in-process
/// GPU probe starts `ms` late (the thread sleeps first), as if the host
/// were starved. One-shot.
CUPTI_PROFILER_API void StallNextDecodeMs(unsigned ms);

/// Period of the rate-limited flush-backlog summary lines (default
/// 30000 ms), so a test does not have to run for minutes.
CUPTI_PROFILER_API void SetBacklogReportPeriodMs(unsigned ms);

} // namespace testing
} // namespace cupti_profiler
