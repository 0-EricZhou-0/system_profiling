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
#include <mutex>
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

    /// Write MSG_STOP to the sidecar WITHOUT waiting for the ack.
    /// Lets the parent kick off the sidecar's shutdown early so its
    /// sample threads can wind down in parallel with any slow
    /// in-process Stop paths (e.g. GPU flush thread joining an
    /// uninterruptible sleep_for). Follow up with JoinStopAck() once
    /// the local work is done to reap the STATUS reply.
    ProfilerError SignalStop();

    /// Block on the sidecar's MSG_STATUS reply to a preceding
    /// SignalStop(). Returns the reported error code.
    ProfilerError JoinStopAck();

    /// Add a PID + alias to the sidecar's tracked set. Thread-safe;
    /// serialised against other Send* calls by an internal mutex so
    /// the workload can call AddTrackedProcess concurrently from
    /// unrelated threads.
    ProfilerError SendAddPid(uint32_t pid, const std::string& alias);

    /// Remove a PID from the sidecar's tracked set. Same thread-
    /// safety guarantee as SendAddPid.
    ProfilerError SendRemovePid(uint32_t pid);

    bool is_running() const { return child_pid_ > 0; }
    pid_t child_pid() const { return child_pid_; }

private:
    // Parent's ends of the two pipes after Spawn().
    // Sidecar sees them at fds kSidecarInFd / kSidecarOutFd.
    int   pipe_to_child_   = -1;    // parent writes here
    int   pipe_from_child_ = -1;    // parent reads here
    pid_t child_pid_       = -1;
    // Serialises Send* calls — the workload can call AddTrackedProcess
    // from unrelated threads while another Send* is in flight.
    mutable std::mutex send_mutex_;

    // Blocking write of a MsgHeader + payload on pipe_to_child_.
    ProfilerError WriteMsg(uint32_t type, const void* payload, uint32_t length);
    // Blocking read of one MSG_STATUS reply. Returns the sidecar's
    // reported ProfilerError verbatim (Ok on success).
    ProfilerError ReadStatus();
};

} // namespace internal
} // namespace cupti_profiler
