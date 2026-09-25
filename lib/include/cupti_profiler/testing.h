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

/// Death between a read and the liveness check (gap A). One-shot: the
/// next time an in-process System probe reads `pid`'s CPU clock, the
/// hook SIGKILLs `pid` right after that read, waits until it has
/// exited, and marks the reading as another process's (it adds 1000 s
/// to the CPU value, as if a new owner of the number had been read).
/// The probe polls its pidfds after the reads, so such a reading must
/// be discarded and never reach the trace. Applies to in-process
/// (LEGACY) probes only; the sidecar's code path is the same.
CUPTI_PROFILER_API void KillAfterNextRead(uint32_t pid);

} // namespace testing
} // namespace cupti_profiler
