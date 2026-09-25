// cupti-profiler sidecar — external observer for CPU / memory / disk
// probes.
//
// Runs as a child of the workload (fork+exec'd by
// ProfilerSuite::Configure() when SystemProbeMode::Sidecar is
// requested). Reuses libcupti_profiler.so's SystemProfiler and
// DiskProfiler internally — the same /proc-based sample loops the
// legacy path runs, only in a separate process so the sampler +
// flush threads don't inflate the workload's per-PID CPU accounting.
//
// Protocol (see lib/src/sidecar_protocol.h):
//   1. Parent sends MSG_CONFIG (serialized ProfilerSuiteConfig).
//      Sidecar parses, replies Ok / SidecarBadHandshake.
//   2. Parent sends MSG_SYNC_ANCHOR (steady_clock + wall_clock).
//      Sidecar stashes; replies Ok. Currently unused — later commits
//      wire it into sample timestamps.
//   3. Parent sends MSG_START. Sidecar configures + starts probes,
//      replies Ok. Sample loops now run.
//   4. Parent may send MSG_ADD_PID / MSG_REMOVE_PID at any time.
//      MSG_ADD_PID may carry a trailing AddPidDescend byte: the per-root
//      override of descendant tracking.
//   5. Parent sends MSG_STOP. Sidecar stops probes (flushes final
//      trace), replies Ok, exits.
//
// A CAP_NET_ADMIN self-check runs during MSG_CONFIG for future
// taskstats-backend readiness; the current /proc backend needs no
// caps for same-UID observation, so a missing cap is currently
// advisory (logged, not an error).

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>

#include <cupti_profiler/profiler_error.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/tracked_process.h>

#include "profiler_config.pb.h"

// Wire protocol shared with the library side (relative include so the
// sidecar target doesn't need lib/src on its default include path).
#include "sidecar_protocol.h"
// Descendant tracking runs here, on the observer side, under SIDECAR.
#include "process_discovery.h"

using namespace cupti_profiler;
using namespace cupti_profiler::internal;

namespace {

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

void SendStatus(ProfilerError err) {
    MsgHeader hdr{ MSG_STATUS, sizeof(StatusPayload) };
    StatusPayload sp{ static_cast<uint32_t>(err) };
    WriteAll(kSidecarOutFd, &hdr, sizeof(hdr));
    WriteAll(kSidecarOutFd, &sp,  sizeof(sp));
}

bool ReadMsg(MsgHeader& hdr, std::string& payload) {
    if (!ReadAll(kSidecarInFd, &hdr, sizeof(hdr))) return false;
    payload.resize(hdr.length);
    if (hdr.length > 0 && !ReadAll(kSidecarInFd, payload.data(), hdr.length)) {
        return false;
    }
    return true;
}

// Advisory only under the /proc backend. Kept as diagnostic so a
// deployer flipping the (future) taskstats backend on can tell up
// front whether the sidecar has the cap it will need.
bool HasCapNetAdmin() {
    std::ifstream f("/proc/self/status");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 7, "CapEff:") == 0) {
            try {
                uint64_t mask = std::stoull(line.substr(7), nullptr, 16);
                return (mask & (1ULL << 12)) != 0;
            } catch (...) { return false; }
        }
    }
    return false;
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
    std::cerr << "[sidecar] up, pid=" << ::getpid()
              << " parent=" << ::getppid() << "\n";

    // 1. MSG_CONFIG — parse the workload's ProfilerSuiteConfig proto.
    MsgHeader hdr{};
    std::string payload;
    if (!ReadMsg(hdr, payload) || hdr.type != MSG_CONFIG) {
        SendStatus(ProfilerError::SidecarBadHandshake);
        return 1;
    }
    ProfilerSuiteConfig cfg;
    if (!cfg.ParseFromString(payload)) {
        std::cerr << "[sidecar] failed to parse ProfilerSuiteConfig ("
                  << payload.size() << " bytes)\n";
        SendStatus(ProfilerError::SidecarBadHandshake);
        return 1;
    }
    std::cerr << "[sidecar] got MSG_CONFIG, "
              << payload.size() << " bytes, "
              << "system=" << (cfg.has_system() && cfg.system().enabled())
              << " disk="   << (cfg.has_disk()   && cfg.disk().enabled())
              << "\n";
    if (!HasCapNetAdmin()) {
        std::cerr << "[sidecar] note: CAP_NET_ADMIN not held. Fine for the "
                     "current /proc backend + same-UID observation; the "
                     "future taskstats backend or cross-UID observation "
                     "will need it (setcap cap_net_admin=ep on this binary).\n";
    }
    SendStatus(ProfilerError::Ok);

    // 2. MSG_SYNC_ANCHOR — stashed but not yet consumed. Future commits
    //    thread it into ProfilerSuite::Start's WallClockEpochNs so
    //    sidecar samples align 1:1 with the workload's traces.
    if (!ReadMsg(hdr, payload) || hdr.type != MSG_SYNC_ANCHOR ||
        payload.size() != sizeof(SyncAnchorPayload))
    {
        SendStatus(ProfilerError::SidecarBadHandshake);
        return 1;
    }
    SyncAnchorPayload sa{};
    std::memcpy(&sa, payload.data(), sizeof(sa));
    std::cerr << "[sidecar] MSG_SYNC_ANCHOR steady=" << sa.steady_clock_ref_ns << "\n";
    SendStatus(ProfilerError::Ok);

    // 3. MSG_START — build local SystemProfiler + DiskProfiler from
    //    the parsed config, drive them from this process's threads.
    if (!ReadMsg(hdr, payload) || hdr.type != MSG_START) {
        SendStatus(ProfilerError::SidecarBadHandshake);
        return 1;
    }

    // Build C++ configs from proto. Same shape as ProfilerSuite's
    // ApplyParsedConfig, minus the pid=0 resolution (already done
    // parent-side before serialisation — the sidecar's getpid()
    // would resolve pid=0 to itself, wrong PID).
    auto build_output_path = [&](const std::string& file) {
        const std::string& dir = cfg.output_dir();
        if (dir.empty()) return file;
        if (file.empty()) return file;
        if (dir.back() == '/') return dir + file;
        return dir + "/" + file;
    };

    std::unique_ptr<SystemProfiler> sys;
    std::unique_ptr<DiskProfiler>   dsk;

    if (cfg.has_system() && cfg.system().enabled() &&
        cfg.system().mode() == SYSTEM_PROBE_MODE_SIDECAR)
    {
        cupti_profiler::SystemProfilerConfig sc;
        const auto& s = cfg.system();
        sc.samplingFrequencyHz = s.sampling_frequency_hz() > 0 ? s.sampling_frequency_hz() : 100;
        sc.flushIntervalMs     = s.flush_interval_ms()     > 0 ? s.flush_interval_ms()     : 5000;
        sc.outputFile          = build_output_path(s.output_file());
        // sc.mode stays Legacy here — from the sidecar's POV, running
        // in-process is the only path (this IS the sidecar).
        for (const auto& p : s.processes()) {
            TrackedProcess tp;
            tp.pid   = p.pid();
            tp.alias = p.alias();
            sc.Processes.push_back(std::move(tp));
        }
        sys = std::make_unique<SystemProfiler>();
        sys->Configure(sc);
        sys->Start();
        std::cerr << "[sidecar] SystemProfiler started, output="
                  << sc.outputFile << ", tracking "
                  << sc.Processes.size() << " PID(s)\n";
    }

    if (cfg.has_disk() && cfg.disk().enabled() &&
        cfg.disk().mode() == SYSTEM_PROBE_MODE_SIDECAR)
    {
        cupti_profiler::DiskProfilerConfig dc;
        const auto& d = cfg.disk();
        dc.samplingFrequencyHz = d.sampling_frequency_hz() > 0 ? d.sampling_frequency_hz() : 10;
        dc.flushIntervalMs     = d.flush_interval_ms()     > 0 ? d.flush_interval_ms()     : 5000;
        dc.outputFile          = build_output_path(d.output_file());
        for (const auto& dev : d.devices()) dc.devices.push_back(dev);
        for (const auto& p : d.processes()) {
            TrackedProcess tp;
            tp.pid   = p.pid();
            tp.alias = p.alias();
            dc.Processes.push_back(std::move(tp));
        }
        dsk = std::make_unique<DiskProfiler>();
        dsk->Configure(dc);
        dsk->Start();
        std::cerr << "[sidecar] DiskProfiler started, output="
                  << dc.outputFile << ", "
                  << dc.devices.size() << " device(s), "
                  << dc.Processes.size() << " PID(s)\n";
    }

    // Descendant tracking for both probes, fed the listed PIDs of each.
    // Adopted orphans that exit are reported to the parent (the
    // launcher, which may be a subreaper) over kSidecarNoticeFd; it
    // reaps them — we cannot wait on its children.
    std::unique_ptr<ProcessDiscovery> discovery;
    if (sys || dsk) {
        DiscoverySettings ds;
        const auto& pd = cfg.process_discovery();
        ds.enabled    = pd.enabled();
        ds.recursive  = !pd.direct_children_only();
        ds.intervalMs = pd.scan_interval_ms() > 0 ? pd.scan_interval_ms() : 100;
        AdoptionReaping reaping;
        if (::fcntl(kSidecarNoticeFd, F_SETFL, O_NONBLOCK) == 0) {
            reaping.mode     = AdoptionReaping::Mode::Notify;
            reaping.hostPid  = static_cast<uint32_t>(::getppid());
            reaping.noticeFd = kSidecarNoticeFd;
        }
        discovery = std::make_unique<ProcessDiscovery>(ds, sys.get(), dsk.get(), reaping);
        if (sys) for (const auto& p : cfg.system().processes())
            discovery->AddRoot(p.pid(), p.alias(), std::nullopt, ProcessDiscovery::kSystemSink);
        if (dsk) for (const auto& p : cfg.disk().processes())
            discovery->AddRoot(p.pid(), p.alias(), std::nullopt, ProcessDiscovery::kDiskSink);
        discovery->Start();
    }

    SendStatus(ProfilerError::Ok);

    // 4. Message loop until MSG_STOP.
    while (true) {
        if (!ReadMsg(hdr, payload)) {
            std::cerr << "[sidecar] pipe closed by parent — shutting down\n";
            break;
        }
        if (hdr.type == MSG_STOP) {
            std::cerr << "[sidecar] MSG_STOP received\n";
            break;
        }
        if (hdr.type == MSG_ADD_PID) {
            // Payload: [uint32 pid][uint32 alias_len][alias bytes]
            //          [optional AddPidDescend byte]
            if (payload.size() < 2 * sizeof(uint32_t)) {
                SendStatus(ProfilerError::SidecarBadHandshake);
                continue;
            }
            uint32_t pid = 0, alias_len = 0;
            std::memcpy(&pid,       payload.data(),                     sizeof(pid));
            std::memcpy(&alias_len, payload.data() + sizeof(pid),       sizeof(alias_len));
            const size_t base = 2 * sizeof(uint32_t) + static_cast<size_t>(alias_len);
            std::optional<bool> descend;
            if (payload.size() == base + 1) {
                const uint8_t b = static_cast<uint8_t>(payload[base]);
                if (b != ADD_PID_DESCEND_OFF && b != ADD_PID_DESCEND_ON) {
                    SendStatus(ProfilerError::SidecarBadHandshake);
                    continue;
                }
                descend = (b == ADD_PID_DESCEND_ON);
            } else if (payload.size() != base) {
                SendStatus(ProfilerError::SidecarBadHandshake);
                continue;
            }
            std::string alias(
                payload.data() + 2 * sizeof(uint32_t), alias_len);
            std::cerr << "[sidecar] MSG_ADD_PID pid=" << pid
                      << " alias=\"" << alias << "\""
                      << (descend ? (*descend ? " descendants=on" : " descendants=off") : "")
                      << "\n";
            if (sys) sys->AddTrackedProcess(pid, alias);
            if (dsk) dsk->AddTrackedProcess(pid, alias);
            if (discovery) discovery->AddRoot(pid, alias, descend);
            SendStatus(ProfilerError::Ok);
            continue;
        }
        if (hdr.type == MSG_REMOVE_PID) {
            if (payload.size() != sizeof(uint32_t)) {
                SendStatus(ProfilerError::SidecarBadHandshake);
                continue;
            }
            uint32_t pid = 0;
            std::memcpy(&pid, payload.data(), sizeof(pid));
            std::cerr << "[sidecar] MSG_REMOVE_PID pid=" << pid << "\n";
            if (sys) sys->RemoveTrackedProcess(pid);
            if (dsk) dsk->RemoveTrackedProcess(pid);
            if (discovery) discovery->RemoveRoot(pid);
            SendStatus(ProfilerError::Ok);
            continue;
        }
        std::cerr << "[sidecar] unexpected msg type=" << hdr.type
                  << " len="  << hdr.length << "; ignoring\n";
    }

    // Discovery first, so nothing is registered during teardown and its
    // final stats reach the probes' last flush. Then SignalStop both
    // probes so their sample threads see the flag in parallel while
    // their flush threads finish their current sleep_for; Stop() joins.
    if (discovery) discovery->Stop();
    if (sys) sys->SignalStop();
    if (dsk) dsk->SignalStop();
    if (sys) sys->Stop();
    if (dsk) dsk->Stop();
    SendStatus(ProfilerError::Ok);
    std::cerr << "[sidecar] clean shutdown, exit 0\n";
    return 0;
}
