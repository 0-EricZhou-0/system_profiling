// Opt-in subreaper helper for launchers that spawn the workload.
//
// The library never sets PR_SET_CHILD_SUBREAPER by itself. A launcher
// that wants orphaned descendants of its workload to re-home to it
// (instead of init) calls EnableChildSubreaper() once, before spawning.
//
// What changes when it is enabled (measured or verified on kernel 5.15,
// 2026-09-24; full table in docs/system-guide.md, "Subreaper helper"):
//
//   * Who is marked: the calling process (the launcher) only.
//   * Inherited by its children? No — neither fork nor Popen children
//     get it, so the workload itself is never a subreaper.
//   * Survives the launcher execve-ing? Yes: a program the launcher
//     execs is still a subreaper.
//   * Where orphaned descendants go: to the launcher (their PPid becomes
//     the launcher's PID) instead of init / the nearest subreaper such
//     as systemd --user or slurmstepd.
//   * Signals: SIGCHLD for every adopted orphan that exits. Code that
//     handles SIGCHLD or calls waitpid(-1) will see children it never
//     started.
//   * Zombies: owned by the launcher until reaped. Descendant tracking
//     reaps the orphans it saw being adopted — one by one, through their
//     pidfd, never with waitpid(-1), so the exit status of the
//     launcher's own children (e.g. Popen.wait() on the workload) is
//     never taken. Orphans adopted before the first scan are not reaped
//     and stay zombies (a PID and a process-table slot, no memory or
//     CPU) until the launcher exits.
//   * Accounting: adopted orphans' CPU and storage I/O fold into the
//     launcher's getrusage(RUSAGE_CHILDREN) / os.times() child fields
//     (measured 0.001 s -> 0.501 s for a 0.500 s orphan; 32.0 MiB
//     written -> 32.0 MiB folded).
//   * Process group, session, signal delivery: unchanged.
//   * Attached (not spawned) targets: no effect.
//   * Cost: zero at steady state; ~41 us of launcher CPU per adopted
//     orphan that exits, never on the target (SIDECAR, 2026-09-25:
//     41.4 us, 95% CI +/-3.0 us, 5 x 600 orphans; under LEGACY it is
//     too small to separate from the in-process sampler's own CPU).
//   * Turning it off: prctl(PR_SET_CHILD_SUBREAPER, 0); already-adopted
//     orphans stay adopted.
//
// What it does NOT buy: an orphan whose intermediate parent died before
// the first scan shows up as the launcher's child, indistinguishable
// from the launcher's other children, so it is not auto-tracked.
#pragma once

#include <cupti_profiler/profiler_error.h>   // CUPTI_PROFILER_API

namespace cupti_profiler {

/// Mark the calling process as a child subreaper
/// (prctl(PR_SET_CHILD_SUBREAPER, 1)) and let descendant tracking reap
/// the orphans it sees adopted. Returns false with errno set on failure.
CUPTI_PROFILER_API bool EnableChildSubreaper();

/// True once EnableChildSubreaper() has succeeded in this process.
CUPTI_PROFILER_API bool ChildSubreaperEnabled();

} // namespace cupti_profiler
