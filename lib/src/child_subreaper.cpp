#include <cupti_profiler/child_subreaper.h>

#include "child_subreaper_internal.h"
#include "proc_readers.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef P_PIDFD
#define P_PIDFD 3   // Linux >= 5.4; older glibc headers lack the name
#endif

namespace cupti_profiler {

namespace {
std::atomic<bool> g_subreaperEnabled{false};
}

bool EnableChildSubreaper() {
    if (::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0) {
        int err = errno;
        std::cerr << "[cupti-profiler] PR_SET_CHILD_SUBREAPER failed: "
                  << std::strerror(err) << "\n";
        errno = err;
        return false;
    }
    g_subreaperEnabled = true;
    std::cerr << "[cupti-profiler] pid " << ::getpid()
              << " is now a child subreaper: orphaned descendants re-parent "
                 "here, and descendant tracking reaps the ones it saw adopted\n";
    return true;
}

bool ChildSubreaperEnabled() { return g_subreaperEnabled.load(); }

namespace internal {

bool ReapAdoptedChild(int pidfd, uint32_t pid, uint64_t startTime) {
    int fd = pidfd;
    if (fd < 0) {
        fd = PidfdOpen(pid);
        if (fd < 0) return false;   // already reaped by someone else
    }
    bool reaped = false;
    // The pidfd pins one process, so this check and the waitid below
    // are about the same process even if the PID number is recycled in
    // between (waitid would then return ECHILD, not another child).
    auto st = ReadProcStat("/proc", pid);
    const bool ours = st && (st->state == 'Z' || st->state == 'X')
                      && st->ppid == static_cast<uint32_t>(::getpid())
                      && (startTime == 0 || st->startTime == startTime);
    if (ours) {
        siginfo_t si{};
        int r;
        do {
            r = ::waitid(static_cast<idtype_t>(P_PIDFD), static_cast<id_t>(fd),
                         &si, WEXITED | WNOHANG);
        } while (r < 0 && errno == EINTR);
        reaped = (r == 0 && si.si_pid != 0);
    }
    if (fd != pidfd) ::close(fd);
    return reaped;
}

} // namespace internal
} // namespace cupti_profiler
