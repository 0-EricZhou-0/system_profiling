// Internal: reaping adopted orphans without stealing anyone's exit status.
//
// A process that holds PR_SET_CHILD_SUBREAPER must reap the orphans it
// adopts, or they stay zombies. The obvious reaper — waitpid(-1) —
// would also reap the launcher's OWN children, and Popen.wait() on the
// workload would then fail with ECHILD (Python turns that into
// returncode 0). So an orphan is reaped only when descendant tracking
// saw it being adopted: it was discovered with a parent other than the
// launcher, and at exit its parent is the launcher. It is then reaped
// alone, through its pidfd.
//
// Under LEGACY the discovery thread runs in the launcher and reaps
// directly. Under SIDECAR it runs in the sidecar, which cannot wait on
// the launcher's children, so it writes an AdoptedExitNotice to fd
// kSidecarNoticeFd; a thread in the launcher reads it and reaps.
#pragma once

#include <cstdint>

namespace cupti_profiler {
namespace internal {

struct AdoptedExitNotice {
    uint32_t pid       = 0;
    uint32_t reserved  = 0;
    uint64_t startTime = 0;   // /proc/<pid>/stat field 22 when discovered
};

/// Reap one adopted orphan of THIS process. Opens a pidfd if `pidfd`
/// is -1. Verifies, from /proc, that the process is a zombie whose
/// parent is this process and (if startTime != 0) that its start time
/// matches — so a recycled PID is never waited on — then
/// waitid(P_PIDFD, WEXITED | WNOHANG) on that process alone. Returns
/// true if it was reaped.
bool ReapAdoptedChild(int pidfd, uint32_t pid, uint64_t startTime);

} // namespace internal
} // namespace cupti_profiler
