// Parent-side sidecar handle: spawns cupti-profiler-sidecar as a
// child process, holds the control pipes, and provides RAII-guarded
// send/recv helpers for the protocol messages defined in
// sidecar_protocol.h.
//
// Lifecycle:
//   Spawn()      — locates sidecar binary, fork/execve, installs
//                  PR_SET_PDEATHSIG, sets up the two pipes on the
//                  well-known fds; returns Ok or one of the
//                  SidecarNotFound / SidecarSpawnFailed / SidecarExited
//                  errors from ProfilerError.
//   SendConfig() — serialize the system+disk config subset and post
//                  MSG_CONFIG; wait for MSG_STATUS reply.
//   SendSyncAnchor()  — post the workload's steady_clock/wall_clock
//                       reference so sidecar timestamps line up.
//   SendStart() / SendStop() — trivial control messages, no payload.
//   ~SidecarProcess()  — close pipes, waitpid.
//
// This TU stays synchronous — the sidecar's sample loop is not
// running until MSG_START, so handshake messages block waiting for
// the reply without any coordination beyond the pipes.
#pragma once

#include <cupti_profiler/profiler_error.h>

#include <cstdint>
#include <string>
#include <sys/types.h>

namespace cupti_profiler {
namespace internal {

class SidecarProcess {
public:
    SidecarProcess() = default;
    ~SidecarProcess();

    // Non-copyable, non-movable — holds live pipe fds and a child PID.
    SidecarProcess(const SidecarProcess&) = delete;
    SidecarProcess& operator=(const SidecarProcess&) = delete;

    /// Locate and spawn the sidecar binary. Search order:
    ///   1. $CUPTI_PROFILER_SIDECAR env var (dev override).
    ///   2. Build-time CUPTI_PROFILER_SIDECAR_PATH baked in via CMake.
    ///   3. Falls through to SidecarNotFound if neither resolves to an
    ///      executable regular file.
    /// Sends nothing yet — call SendConfig() / SendSyncAnchor() next.
    ProfilerError Spawn();

    /// Post serialized-proto config bytes and block for status reply.
    ProfilerError SendConfig(const std::string& serialized_config);

    /// Post steady_clock + wall_clock references and block for status.
    ProfilerError SendSyncAnchor(uint64_t steady_clock_ref_ns,
                                 uint64_t wall_clock_epoch_ns);

    /// Tell the sidecar to build its probes and start sampling.
    /// Blocks for the sidecar's ack.
    ProfilerError SendStart();

    /// Tell the sidecar to stop sampling, flush pending trace bytes,
    /// and exit. Blocks for the sidecar's ack. The destructor's
    /// SIGTERM path is the fallback if this fails.
    ProfilerError SendStop();

    bool is_running() const { return child_pid_ > 0; }
    pid_t child_pid() const { return child_pid_; }

private:
    // Parent's ends of the two pipes after Spawn().
    // Sidecar sees them at fds kSidecarInFd / kSidecarOutFd.
    int   pipe_to_child_   = -1;    // parent writes here
    int   pipe_from_child_ = -1;    // parent reads here
    pid_t child_pid_       = -1;

    // Blocking write of a MsgHeader + payload on pipe_to_child_.
    ProfilerError WriteMsg(uint32_t type, const void* payload, uint32_t length);
    // Blocking read of one MSG_STATUS reply. Returns the sidecar's
    // reported ProfilerError verbatim (Ok on success).
    ProfilerError ReadStatus();
};

} // namespace internal
} // namespace cupti_profiler
