// cupti-profiler sidecar — external observer for CPU / memory / disk
// probes.
//
// Runs as a child of the workload (fork+exec'd by
// ProfilerSuite::Configure() when SystemProbeMode::Sidecar is
// requested). Reads its config + PID list + sync anchor from the
// workload over pipe fd 3 (kSidecarInFd), replies with MSG_STATUS
// codes over pipe fd 4 (kSidecarOutFd).
//
// Current scope: handshake only.
//   - Read MSG_CONFIG, verify at least one enabled probe → reply Ok
//     (or a specific error code if the wire is malformed).
//   - After receiving MSG_CONFIG, do the CAP_NET_ADMIN self-check by
//     parsing /proc/self/status. If missing, reply SidecarMissingCaps
//     and exit.
//   - Read MSG_SYNC_ANCHOR, stash it, reply Ok.
//   - Exit 0 (later commits add the MSG_START sample loop + the
//     mid-run Add/Remove PID Unix socket).

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

// Wire protocol shared with the library side.
#include "../../../lib/src/sidecar_protocol.h"

using namespace cupti_profiler::internal;

namespace {

// ProfilerError values must stay in sync with
// lib/include/cupti_profiler/profiler_error.h — duplicated here to
// avoid pulling the whole cupti_profiler public header into the
// sidecar target's translation unit.
enum : uint32_t {
    ERR_OK                    = 0,
    ERR_SIDECAR_MISSING_CAPS  = 202,
    ERR_SIDECAR_BAD_HANDSHAKE = 203,
};

bool ReadAll(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        p += r; n -= static_cast<size_t>(r);
    }
    return true;
}

bool WriteAll(int fd, const void* buf, size_t n) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += w; n -= static_cast<size_t>(w);
    }
    return true;
}

void SendStatus(uint32_t code) {
    MsgHeader hdr{ MSG_STATUS, sizeof(StatusPayload) };
    StatusPayload sp{ code };
    WriteAll(kSidecarOutFd, &hdr, sizeof(hdr));
    WriteAll(kSidecarOutFd, &sp,  sizeof(sp));
}

// Bit 12 (CAP_NET_ADMIN) in the CapEff mask from /proc/self/status.
// Doing this directly rather than pulling in libcap because the
// sidecar target should stay dependency-lean; libcap would drag in
// libcap-dev at build time on every host that builds this project.
bool HasCapNetAdmin() {
    std::ifstream f("/proc/self/status");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 7, "CapEff:") == 0) {
            // Format: "CapEff:\t<hex_mask>"
            try {
                uint64_t mask = std::stoull(line.substr(7), nullptr, 16);
                return (mask & (1ULL << 12)) != 0;   // CAP_NET_ADMIN = 12
            } catch (...) {
                return false;
            }
        }
    }
    return false;
}

// Read one message; returns false on EOF/short read/malformed length.
// Payload is drained into `payload` (resized to hdr.length).
bool ReadMsg(MsgHeader& hdr, std::string& payload) {
    if (!ReadAll(kSidecarInFd, &hdr, sizeof(hdr))) return false;
    payload.resize(hdr.length);
    if (hdr.length > 0 && !ReadAll(kSidecarInFd, payload.data(), hdr.length)) {
        return false;
    }
    return true;
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
    std::cerr << "[sidecar] up, pid=" << ::getpid()
              << " parent=" << ::getppid() << "\n";

    // 1. MSG_CONFIG — the workload's suite config. Sidecar-side
    //    proto parsing lands in commit 4 (TaskstatsClient); for now
    //    we just accept any non-empty payload as valid so the
    //    handshake framing is exercised end-to-end.
    MsgHeader hdr{};
    std::string payload;
    if (!ReadMsg(hdr, payload) || hdr.type != MSG_CONFIG) {
        SendStatus(ERR_SIDECAR_BAD_HANDSHAKE);
        return 1;
    }
    std::cerr << "[sidecar] got MSG_CONFIG, " << payload.size() << " bytes\n";

    // 2. CAP_NET_ADMIN self-check. Do this AFTER MSG_CONFIG so the
    //    caller sees a specific SidecarMissingCaps reply rather than
    //    an ambiguous handshake failure — the config was fine, the
    //    caps aren't.
    if (!HasCapNetAdmin()) {
        std::cerr << "[sidecar] CAP_NET_ADMIN missing — replying "
                     "SidecarMissingCaps and exiting.\n"
                     "  fix: sudo setcap cap_net_admin=ep " << "$SIDECAR" << "\n";
        SendStatus(ERR_SIDECAR_MISSING_CAPS);
        return 2;
    }
    SendStatus(ERR_OK);
    std::cerr << "[sidecar] config accepted; CAP_NET_ADMIN present\n";

    // 3. MSG_SYNC_ANCHOR — steady_clock + wall_clock references so
    //    sidecar timestamps line up with the workload's traces.
    if (!ReadMsg(hdr, payload) || hdr.type != MSG_SYNC_ANCHOR ||
        payload.size() != sizeof(SyncAnchorPayload))
    {
        SendStatus(ERR_SIDECAR_BAD_HANDSHAKE);
        return 3;
    }
    SyncAnchorPayload sa{};
    std::memcpy(&sa, payload.data(), sizeof(sa));
    std::cerr << "[sidecar] got MSG_SYNC_ANCHOR"
                 " steady=" << sa.steady_clock_ref_ns
              << " wall=" << sa.wall_clock_epoch_ns << "\n";
    SendStatus(ERR_OK);

    // Later commits: block on MSG_START, run sample loop, honour
    // MSG_STOP + MSG_ADD_PID + MSG_REMOVE_PID. For now the handshake
    // is complete; exit cleanly.
    std::cerr << "[sidecar] handshake complete; exiting until sample "
                 "loop lands in later commits\n";
    return 0;
}
