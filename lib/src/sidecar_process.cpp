#include "sidecar_process.h"
#include "sidecar_protocol.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
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

} // namespace

SidecarProcess::~SidecarProcess() {
    if (pipe_to_child_   >= 0) ::close(pipe_to_child_);
    if (pipe_from_child_ >= 0) ::close(pipe_from_child_);
    if (child_pid_ > 0) {
        // Best-effort: give the child SIGTERM if it's still alive
        // (later commits add MSG_STOP), then reap.
        ::kill(child_pid_, SIGTERM);
        int status = 0;
        ::waitpid(child_pid_, &status, 0);
    }
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
    if (::pipe(down) != 0 || ::pipe(up) != 0) {
        std::cerr << "[ProfilerSuite] pipe() failed: " << ::strerror(errno) << "\n";
        return ProfilerError::SidecarSpawnFailed;
    }

    // Capture parent PID *before* fork so the child can detect the
    // rare parent-died-between-fork-and-prctl race.
    const pid_t parent_pid = ::getpid();

    pid_t child = ::fork();
    if (child < 0) {
        std::cerr << "[ProfilerSuite] fork() failed: " << ::strerror(errno) << "\n";
        ::close(down[0]); ::close(down[1]);
        ::close(up[0]);   ::close(up[1]);
        return ProfilerError::SidecarSpawnFailed;
    }

    if (child == 0) {
        // Child ---------------------------------------------------------
        // Get SIGTERM if parent dies. Handles crash/orphan cleanup so
        // the sidecar doesn't outlive the workload as a zombie helper.
        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (::getppid() != parent_pid) {
            // Parent died between fork and prctl — bail before exec.
            _exit(1);
        }

        // Remap down[0] → kSidecarInFd, up[1] → kSidecarOutFd. dup2
        // no-ops if the source is already at the target; we still
        // close the source afterwards unless it was the target.
        if (::dup2(down[0], kSidecarInFd)  < 0) _exit(2);
        if (::dup2(up[1],   kSidecarOutFd) < 0) _exit(2);
        if (down[0] != kSidecarInFd)  ::close(down[0]);
        if (up[1]   != kSidecarOutFd) ::close(up[1]);
        ::close(down[1]);
        ::close(up[0]);

        char* const argv[] = {
            const_cast<char*>(sidecar.c_str()),
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
    pipe_to_child_   = down[1];
    pipe_from_child_ = up[0];
    child_pid_       = child;
    return ProfilerError::Ok;
}

ProfilerError SidecarProcess::WriteMsg(uint32_t type,
                                       const void* payload,
                                       uint32_t length)
{
    MsgHeader hdr{ type, length };
    if (!WriteAll(pipe_to_child_, &hdr, sizeof(hdr))) return ProfilerError::SidecarExited;
    if (length > 0 && !WriteAll(pipe_to_child_, payload, length)) {
        return ProfilerError::SidecarExited;
    }
    return ProfilerError::Ok;
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

ProfilerError SidecarProcess::SendSyncAnchor(uint64_t steady_clock_ref_ns,
                                             uint64_t wall_clock_epoch_ns)
{
    std::lock_guard<std::mutex> lk(send_mutex_);
    SyncAnchorPayload sa{ steady_clock_ref_ns, wall_clock_epoch_ns };
    if (auto e = WriteMsg(MSG_SYNC_ANCHOR, &sa, sizeof(sa));
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

ProfilerError SidecarProcess::SendAddPid(uint32_t pid, const std::string& alias) {
    // Payload: [uint32 pid][uint32 alias_len][alias bytes]
    // Serialise ourselves rather than pulling in a proto — one message,
    // one write, no framing complications on the sidecar side.
    std::string buf;
    buf.reserve(8 + alias.size());
    uint32_t alias_len = static_cast<uint32_t>(alias.size());
    buf.append(reinterpret_cast<const char*>(&pid),       sizeof(pid));
    buf.append(reinterpret_cast<const char*>(&alias_len), sizeof(alias_len));
    buf.append(alias);
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
