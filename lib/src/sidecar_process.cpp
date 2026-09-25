#include "sidecar_process.h"
#include "sidecar_protocol.h"

#include <cupti_profiler/child_subreaper.h>

#include "child_subreaper_internal.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <string>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

// Build-time default; can be overridden via -DCUPTI_PROFILER_SIDECAR_PATH.
#ifndef CUPTI_PROFILER_SIDECAR_PATH
#define CUPTI_PROFILER_SIDECAR_PATH ""
#endif

namespace cupti_profiler {
namespace internal {

namespace {

bool IsExecutableRegularFile(const std::string& path) {
    if (path.empty()) return false;
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    if (!S_ISREG(st.st_mode)) return false;
    return ::access(path.c_str(), X_OK) == 0;
}

std::string ResolveSidecarPath() {
    if (const char* env = std::getenv("CUPTI_PROFILER_SIDECAR")) {
        if (IsExecutableRegularFile(env)) return env;
    }
    const char* baked = CUPTI_PROFILER_SIDECAR_PATH;
    if (baked && *baked && IsExecutableRegularFile(baked)) return baked;
    return {};
}

// Fully read/write across short returns; return true on complete success.
bool ReadAll(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;  // EOF before we got everything
        p += r; n -= static_cast<size_t>(r);
    }
    return true;
}

bool WriteAll(int fd, const void* buf, size_t n) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w; n -= static_cast<size_t>(w);
    }
    return true;
}

// WriteAll without SIGPIPE. The sidecar exits on its own when it gets
// SIGTERM/SIGINT (a terminal Ctrl-C reaches it too), so a later write
// from this process can hit a pipe with no reader; the default SIGPIPE
// action would kill a host that has not ignored it (C++ programs; Python
// ignores it). SIGPIPE for a pipe write goes to the writing thread, so
// blocking it on this thread and discarding the one this write raised
// leaves the rest of the process's signal handling untouched.
bool WriteAllNoSigpipe(int fd, const void* buf, size_t n) {
    sigset_t pipeSet, old;
    sigemptyset(&pipeSet);
    sigaddset(&pipeSet, SIGPIPE);
    sigset_t pending;
    sigemptyset(&pending);
    ::pthread_sigmask(SIG_BLOCK, &pipeSet, &old);
    sigpending(&pending);
    const bool alreadyPending = sigismember(&pending, SIGPIPE);
    const bool ok = WriteAll(fd, buf, n);
    const int err = errno;
    if (!ok && err == EPIPE && !alreadyPending) {
        struct timespec zero{0, 0};
        while (::sigtimedwait(&pipeSet, nullptr, &zero) < 0 && errno == EINTR) {}
    }
    ::pthread_sigmask(SIG_SETMASK, &old, nullptr);
    errno = err;
    return ok;
}

} // namespace

SidecarProcess::~SidecarProcess() {
    if (pipe_to_child_   >= 0) ::close(pipe_to_child_);
    if (pipe_from_child_ >= 0) ::close(pipe_from_child_);
    if (child_pid_ > 0) {
        // Normally the sidecar has exited after MSG_STOP. If Stop() was
        // never reached, closing the control pipe (EOF) and SIGTERM both
        // make it stop gracefully — final flush included — then reap.
        ::kill(child_pid_, SIGTERM);
        int status = 0;
        ::waitpid(child_pid_, &status, 0);
    }
    // The sidecar held the only write end of the notice pipe, so the
    // reader has seen EOF by now.
    if (notice_reader_.joinable()) notice_reader_.join();
    if (notice_from_child_ >= 0) ::close(notice_from_child_);
}

ProfilerError SidecarProcess::Spawn() {
    std::string sidecar = ResolveSidecarPath();
    if (sidecar.empty()) {
        std::cerr << "[ProfilerSuite] Sidecar binary not found. Set "
                     "CUPTI_PROFILER_SIDECAR or build the "
                     "cupti_profiler_sidecar target.\n";
        return ProfilerError::SidecarNotFound;
    }

    int down[2];  // parent → child
    int up[2];    // child → parent
    int note[2];  // child → parent, adopted-orphan exit notices
    // All O_CLOEXEC: a program this process later execs (the workload
    // it launches, say) must not inherit its ends. An inherited write
    // end of `down` would hold the sidecar's control pipe open past
    // this process's exit, so the sidecar would never see EOF. The
    // sidecar's own copies are dup2'ed onto fixed fds in the child
    // below, which clears the flag on them.
    if (::pipe2(down, O_CLOEXEC) != 0) {
        std::cerr << "[ProfilerSuite] pipe2() failed: " << ::strerror(errno) << "\n";
        return ProfilerError::SidecarSpawnFailed;
    }
    if (::pipe2(up, O_CLOEXEC) != 0) {
        std::cerr << "[ProfilerSuite] pipe2() failed: " << ::strerror(errno) << "\n";
        ::close(down[0]); ::close(down[1]);
        return ProfilerError::SidecarSpawnFailed;
    }
    if (::pipe2(note, O_CLOEXEC) != 0) {
        std::cerr << "[ProfilerSuite] pipe2() failed: " << ::strerror(errno) << "\n";
        ::close(down[0]); ::close(down[1]);
        ::close(up[0]);   ::close(up[1]);
        return ProfilerError::SidecarSpawnFailed;
    }

    // The sidecar watches THIS process (a pidfd on it) and shuts down
    // gracefully when it exits. Passed explicitly rather than read with
    // getppid(), which after an early death of this process would name
    // whoever adopted the sidecar.
    const std::string host_arg = "--host-pid=" + std::to_string(::getpid());

    pid_t child = ::fork();
    if (child < 0) {
        std::cerr << "[ProfilerSuite] fork() failed: " << ::strerror(errno) << "\n";
        ::close(down[0]); ::close(down[1]);
        ::close(up[0]);   ::close(up[1]);
        ::close(note[0]); ::close(note[1]);
        return ProfilerError::SidecarSpawnFailed;
    }

    if (child == 0) {
        // Child ---------------------------------------------------------
        // No PR_SET_PDEATHSIG: it fires when the forking THREAD exits,
        // so a Configure() called from a short-lived thread would kill
        // the sidecar mid-run. The sidecar watches this process with a
        // pidfd instead (see cupti_profiler_sidecar.cpp).

        // Remap down[0] → kSidecarInFd, up[1] → kSidecarOutFd,
        // note[1] → kSidecarNoticeFd. Any pipe fd may already sit on
        // one of the targets (3/4/5), so first move the three we keep
        // above them, close every original, then dup2 into place.
        int in  = ::fcntl(down[0], F_DUPFD, 10);
        int out = ::fcntl(up[1],   F_DUPFD, 10);
        int nt  = ::fcntl(note[1], F_DUPFD, 10);
        if (in < 0 || out < 0 || nt < 0) _exit(2);
        for (int fd : {down[0], down[1], up[0], up[1], note[0], note[1]}) ::close(fd);
        if (::dup2(in,  kSidecarInFd)     < 0) _exit(2);
        if (::dup2(out, kSidecarOutFd)    < 0) _exit(2);
        if (::dup2(nt,  kSidecarNoticeFd) < 0) _exit(2);
        ::close(in); ::close(out); ::close(nt);

        char* const argv[] = {
            const_cast<char*>(sidecar.c_str()),
            const_cast<char*>(host_arg.c_str()),
            nullptr
        };
        ::execve(sidecar.c_str(), argv, environ);
        // Only reached on execve failure.
        std::fprintf(stderr, "[ProfilerSuite] execve(%s) failed: %s\n",
                     sidecar.c_str(), ::strerror(errno));
        _exit(127);
    }

    // Parent --------------------------------------------------------------
    ::close(down[0]);
    ::close(up[1]);
    ::close(note[1]);
    pipe_to_child_     = down[1];
    pipe_from_child_   = up[0];
    notice_from_child_ = note[0];
    child_pid_         = child;

    // Reap adopted orphans the sidecar reports — only if this process
    // opted in with EnableChildSubreaper(); otherwise drain and ignore.
    notice_reader_ = std::thread([fd = notice_from_child_] {
        AdoptedExitNotice n;
        while (ReadAll(fd, &n, sizeof(n))) {
            if (ChildSubreaperEnabled()) ReapAdoptedChild(-1, n.pid, n.startTime);
        }
    });

    // Grant the sidecar ptrace-mode access to us so it can read
    // /proc/<workload>/io. Under Yama ptrace_scope >= 1 (Ubuntu's
    // default), descendants can't trace their ancestors without
    // this hint — the sidecar's DiskProfiler reads of the workload's
    // I/O counters would return EPERM and silently zero out.
    // PR_SET_PTRACER is per-target-PID; failure is not fatal, just
    // means per-PID I/O for the workload won't appear in disk_metrics.pb.
    if (::prctl(PR_SET_PTRACER, static_cast<unsigned long>(child), 0, 0, 0) != 0) {
        std::cerr << "[ProfilerSuite] PR_SET_PTRACER(" << child
                  << ") failed: " << ::strerror(errno)
                  << " — per-PID I/O may be zero in the sidecar's trace.\n";
    }

    return ProfilerError::Ok;
}

ProfilerError SidecarProcess::WriteMsg(uint32_t type,
                                       const void* payload,
                                       uint32_t length)
{
    MsgHeader hdr{ type, length };
    if (!WriteAllNoSigpipe(pipe_to_child_, &hdr, sizeof(hdr))) return ProfilerError::SidecarExited;
    if (length > 0 && !WriteAllNoSigpipe(pipe_to_child_, payload, length)) {
        return ProfilerError::SidecarExited;
    }
    return ProfilerError::Ok;
}

std::string SidecarProcess::DescribeExit() {
    if (child_pid_ <= 0) return "not running";
    int status = 0;
    pid_t r;
    do { r = ::waitpid(child_pid_, &status, WNOHANG); } while (r < 0 && errno == EINTR);
    if (r == 0) return "still running";
    if (r < 0) return std::string("waitpid: ") + ::strerror(errno);
    child_pid_ = -1;   // reaped here; the destructor must not wait again
    if (WIFEXITED(status)) {
        return "exited with status " + std::to_string(WEXITSTATUS(status)) +
               (WEXITSTATUS(status) == 0
                    ? " (it stops cleanly, final flush included, on SIGTERM/SIGINT "
                      "or when its control pipe closes)"
                    : "");
    }
    if (WIFSIGNALED(status)) {
        return "killed by signal " + std::to_string(WTERMSIG(status)) +
               " (its trace may lack the data since its last flush)";
    }
    return "status " + std::to_string(status);
}

ProfilerError SidecarProcess::ReadStatus() {
    MsgHeader hdr{};
    if (!ReadAll(pipe_from_child_, &hdr, sizeof(hdr))) return ProfilerError::SidecarExited;
    if (hdr.type != MSG_STATUS || hdr.length != sizeof(StatusPayload)) {
        return ProfilerError::SidecarBadHandshake;
    }
    StatusPayload sp{};
    if (!ReadAll(pipe_from_child_, &sp, sizeof(sp))) return ProfilerError::SidecarExited;
    return static_cast<ProfilerError>(sp.error_code);
}

ProfilerError SidecarProcess::SendConfig(const std::string& serialized_config) {
    std::lock_guard<std::mutex> lk(send_mutex_);
    if (auto e = WriteMsg(MSG_CONFIG, serialized_config.data(),
                          static_cast<uint32_t>(serialized_config.size()));
        e != ProfilerError::Ok)
    {
        return e;
    }
    return ReadStatus();
}

ProfilerError SidecarProcess::SendStart() {
    std::lock_guard<std::mutex> lk(send_mutex_);
    if (auto e = WriteMsg(MSG_START, nullptr, 0); e != ProfilerError::Ok) return e;
    return ReadStatus();
}

ProfilerError SidecarProcess::SignalStop() {
    std::lock_guard<std::mutex> lk(send_mutex_);
    return WriteMsg(MSG_STOP, nullptr, 0);
}

ProfilerError SidecarProcess::JoinStopAck() {
    std::lock_guard<std::mutex> lk(send_mutex_);
    return ReadStatus();
}

ProfilerError SidecarProcess::SendAddPid(uint32_t pid, const std::string& alias,
                                         std::optional<bool> trackDescendants) {
    // Payload: [uint32 pid][uint32 alias_len][alias bytes][AddPidDescend?]
    // Serialise ourselves rather than pulling in a proto — one message,
    // one write, no framing complications on the sidecar side.
    std::string buf;
    buf.reserve(8 + alias.size());
    uint32_t alias_len = static_cast<uint32_t>(alias.size());
    buf.append(reinterpret_cast<const char*>(&pid),       sizeof(pid));
    buf.append(reinterpret_cast<const char*>(&alias_len), sizeof(alias_len));
    buf.append(alias);
    if (trackDescendants) {
        buf.push_back(static_cast<char>(*trackDescendants ? ADD_PID_DESCEND_ON
                                                          : ADD_PID_DESCEND_OFF));
    }
    std::lock_guard<std::mutex> lk(send_mutex_);
    if (auto e = WriteMsg(MSG_ADD_PID, buf.data(),
                          static_cast<uint32_t>(buf.size()));
        e != ProfilerError::Ok)
    {
        return e;
    }
    return ReadStatus();
}

ProfilerError SidecarProcess::SendRemovePid(uint32_t pid) {
    std::lock_guard<std::mutex> lk(send_mutex_);
    if (auto e = WriteMsg(MSG_REMOVE_PID, &pid, sizeof(pid));
        e != ProfilerError::Ok)
    {
        return e;
    }
    return ReadStatus();
}

} // namespace internal
} // namespace cupti_profiler
