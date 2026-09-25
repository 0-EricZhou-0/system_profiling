// Wire format for the parent-workload ↔ sidecar handshake and
// runtime control channel. Two pipes, fd 3 (parent → sidecar) and
// fd 4 (sidecar → parent), remapped by the parent during fork()
// so the sidecar can find them at well-known numbers.
//
// Every message is a MsgHeader followed by `length` payload bytes.
// Payload interpretation is per MsgType. No streaming/framing beyond
// length prefixing; a partial read is an error.
#pragma once

#include <cstdint>

namespace cupti_profiler {
namespace internal {

// Well-known fd numbers the sidecar reads/writes on. Parent dup2's
// the pipe endpoints onto these before execve(); sidecar picks them
// up from argv-independent constants.
inline constexpr int kSidecarInFd  = 3;   // parent → sidecar
inline constexpr int kSidecarOutFd = 4;   // sidecar → parent

// Message types on either channel. Numeric values are frozen —
// parent and sidecar are linked separately, so we treat this as a
// wire protocol rather than an internal ABI.
enum MsgType : uint32_t {
    // Parent → sidecar
    MSG_CONFIG      = 0x01,   // serialized SystemProfilerConfig proto blob
    MSG_SYNC_ANCHOR = 0x02,   // 2× uint64: steady_clock_ref_ns, wall_clock_epoch_ns
    MSG_START       = 0x03,   // no payload — begin sampling
    MSG_STOP        = 0x04,   // no payload — signal sample loop to exit
    MSG_ADD_PID     = 0x05,   // uint32 pid, uint32 alias_len, alias bytes,
                              // then OPTIONALLY one AddPidDescend byte.
                              // Length-checked: without the byte the
                              // payload is exactly the original format,
                              // and the root inherits the config.
    MSG_REMOVE_PID  = 0x06,   // uint32 pid

    // Sidecar → parent
    MSG_STATUS      = 0x80,   // uint32 ProfilerError code; sent after each
                              // handshake step to confirm success/failure
};

struct MsgHeader {
    uint32_t type;
    uint32_t length;   // bytes of payload following this header
};

struct SyncAnchorPayload {
    uint64_t steady_clock_ref_ns;
    uint64_t wall_clock_epoch_ns;
};

// Optional trailing byte of MSG_ADD_PID: the per-root override of
// ProcessDiscoveryConfig.enabled. Sent only when the caller passed an
// explicit trackDescendants, so a plain add is byte-identical to the
// original message (an older sidecar still accepts it; given the extra
// byte, an older sidecar fails the length check and replies
// SidecarBadHandshake rather than silently ignoring the override).
enum AddPidDescend : uint8_t {
    ADD_PID_DESCEND_OFF = 1,
    ADD_PID_DESCEND_ON  = 2,
};

struct StatusPayload {
    uint32_t error_code;   // corresponds to ProfilerError; 0 = Ok
};

} // namespace internal
} // namespace cupti_profiler
