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
//   2. Parent sends MSG_START. Sidecar configures + starts probes,
//      replies Ok. Sample loops now run. Timestamps are steady_clock
//      (CLOCK_MONOTONIC, system-wide), so they line up with the host's
//      traces without a clock handshake.
//   3. Parent may send MSG_ADD_PID / MSG_REMOVE_PID at any time.
//      MSG_ADD_PID may carry a trailing AddPidDescend byte: the per-root
//      override of descendant tracking.
//   4. Parent sends MSG_STOP. Sidecar stops probes (flushes final
//      trace), replies Ok, exits.
//
// It also stops gracefully — probes stopped, final trace flushed, exit
// 0 — without MSG_STOP when:
//   * the host exits: argv[1] is --host-pid=<pid>, and the sidecar holds
//     a pidfd on it. By PROCESS, not thread: PR_SET_PDEATHSIG, used
//     before, fired when the host's forking thread exited. The pidfd
//     also works when a process the host forked still holds the control
//     pipe open, so EOF never comes;
//   * the control pipe reaches EOF;
//   * it receives SIGTERM, SIGINT, SIGHUP, SIGQUIT or another signal
//     whose default action terminates it (SIGPIPE aside). They are
//     blocked in every thread and read from a signalfd in the control
//     loop, so the handling runs on the main thread, outside signal
//     context.
// SIGPIPE is ignored: a status write to a host that is gone fails with
// EPIPE instead of killing the sidecar before it has flushed.
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
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <sys/syscall.h>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434   // same number on every architecture
#endif

#include <cupti_profiler/defaults.h>
#include <cupti_profiler/profiler_error.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/testing.h>
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

// What woke the control loop.
enum class Wake { Message, Eof, HostExited, Signal };

const char* WakeName(Wake w, int signo) {
    switch (w) {
        case Wake::Message:    return "message";
        case Wake::Eof:        return "control pipe closed by the host";
        case Wake::HostExited: return "host process exited";
        case Wake::Signal:     return ::strsignal(signo);
    }
    return "?";
}

struct Control {
    int hostFd = -1;   // pidfd on the host; -1 = EOF is the only exit signal
    int sigFd  = -1;   // signalfd for the stop signals (kStopSignals)
    int signo  = 0;    // the signal that ended the loop
};

// Block until the next control message, the host's exit, or a stop
// signal. A signal wins over a pending message, and the host's exit
// over a message it may have left in the pipe.
Wake NextMsg(Control& c, MsgHeader& hdr, std::string& payload) {
    struct pollfd fds[3] = {
        {kSidecarInFd, POLLIN, 0},
        {c.hostFd,     POLLIN, 0},   // ignored by poll() when -1
        {c.sigFd,      POLLIN, 0},
    };
    for (;;) {
        int r = ::poll(fds, 3, -1);
        if (r < 0) {
            if (errno == EINTR) continue;
            return Wake::Eof;
        }
        if (fds[2].revents) {
            struct signalfd_siginfo si{};
            if (::read(c.sigFd, &si, sizeof(si)) == static_cast<ssize_t>(sizeof(si)))
                c.signo = static_cast<int>(si.ssi_signo);
            return Wake::Signal;
        }
        if (fds[1].revents) return Wake::HostExited;
        if (fds[0].revents) return ReadMsg(hdr, payload) ? Wake::Message : Wake::Eof;
    }
}

// --host-pid=<pid> from argv; getppid() for an older host that does not
// pass it.
pid_t HostPid(int argc, char** argv) {
    const std::string flag = "--host-pid=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.compare(0, flag.size(), flag) == 0)
            return static_cast<pid_t>(std::strtol(a.c_str() + flag.size(), nullptr, 10));
    }
    return ::getppid();
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

int main(int argc, char** argv) {
    std::cerr << "[sidecar] up, pid=" << ::getpid()
              << " parent=" << ::getppid() << "\n";

    ::signal(SIGPIPE, SIG_IGN);

    // Test-only (CUPTI_PROFILER_TEST_KILL_AFTER_READ); unset does nothing.
    cupti_profiler::testing::ArmKillAfterReadFromEnv();

    // Every signal whose default action terminates the process (except
    // SIGPIPE, ignored above) -> graceful stop, final flush included.
    // Blocked before any thread exists, so every probe thread inherits
    // the mask and the signal is only ever consumed here, through the
    // signalfd. SIGKILL cannot be caught: up to one flush interval of
    // samples is lost then.
    Control ctl;
    {
        sigset_t stopSignals;
        sigemptyset(&stopSignals);
        for (int s : {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM, SIGXCPU,
                      SIGXFSZ, SIGVTALRM, SIGPROF, SIGIO, SIGPWR})
            sigaddset(&stopSignals, s);
        ::pthread_sigmask(SIG_BLOCK, &stopSignals, nullptr);
        ctl.sigFd = ::signalfd(-1, &stopSignals, SFD_CLOEXEC);
        if (ctl.sigFd < 0) {
            std::cerr << "[sidecar] signalfd: " << std::strerror(errno)
                      << " — stop signals will not flush\n";
            ::pthread_sigmask(SIG_UNBLOCK, &stopSignals, nullptr);
        }
    }

    // Watch the host by process. Opened before checking getppid(), so a
    // host that already died is caught either way.
    const pid_t hostPid = HostPid(argc, argv);
    ctl.hostFd = static_cast<int>(::syscall(SYS_pidfd_open, hostPid, 0));
    if (ctl.hostFd < 0) {
        std::cerr << "[sidecar] pidfd_open(host " << hostPid << "): " << std::strerror(errno)
                  << " — host exit is detected only through control-pipe EOF\n";
    } else {
        ::fcntl(ctl.hostFd, F_SETFD, FD_CLOEXEC);
    }
    if (::getppid() != hostPid) {
        std::cerr << "[sidecar] host " << hostPid << " exited before the handshake — exit\n";
        return 0;
    }

    // Before MSG_START nothing runs, so any other wake-up just exits.
    auto handshake = [&](MsgHeader& hdr, std::string& payload) {
        Wake w = NextMsg(ctl, hdr, payload);
        if (w != Wake::Message) {
            std::cerr << "[sidecar] " << WakeName(w, ctl.signo)
                      << " during the handshake — exit\n";
            std::exit(0);
        }
    };

    // 1. MSG_CONFIG — parse the workload's ProfilerSuiteConfig proto.
    MsgHeader hdr{};
    std::string payload;
    handshake(hdr, payload);
    if (hdr.type != MSG_CONFIG) {
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
    // Optional pinning, before any thread exists so every probe thread
    // inherits it.
    if (cfg.sidecar_cpus_size() > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        bool inRange = true;
        std::string list;
        for (uint32_t cpu : cfg.sidecar_cpus()) {
            if (cpu >= CPU_SETSIZE) { inRange = false; break; }
            CPU_SET(cpu, &set);
            list += (list.empty() ? "" : ",") + std::to_string(cpu);
        }
        if (!inRange || ::sched_setaffinity(0, sizeof(set), &set) != 0) {
            std::cerr << "[sidecar] sidecar_cpus {" << list << "}: "
                      << (inRange ? std::strerror(errno) : "CPU number out of range")
                      << " — reporting SidecarAffinityFailed, exit\n";
            SendStatus(ProfilerError::SidecarAffinityFailed);
            return 1;
        }
        std::cerr << "[sidecar] pinned to CPU(s) " << list << "\n";
    }
    if (!HasCapNetAdmin()) {
        std::cerr << "[sidecar] note: CAP_NET_ADMIN not held. Fine for the "
                     "current /proc backend + same-UID observation; the "
                     "future taskstats backend or cross-UID observation "
                     "will need it (setcap cap_net_admin=ep on this binary).\n";
    }
    SendStatus(ProfilerError::Ok);

    // 2. MSG_START — build local SystemProfiler + DiskProfiler from
    //    the parsed config, drive them from this process's threads.
    handshake(hdr, payload);
    if (hdr.type != MSG_START) {
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
        sc.samplingFrequencyHz = s.sampling_frequency_hz() > 0 ? s.sampling_frequency_hz() : kDefaultSystemSamplingHz;
        sc.flushIntervalMs     = s.flush_interval_ms();   // 0 = default
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
        dc.samplingFrequencyHz = d.sampling_frequency_hz() > 0 ? d.sampling_frequency_hz() : kDefaultDiskSamplingHz;
        dc.flushIntervalMs     = d.flush_interval_ms();   // 0 = default
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

    // A probe that did not start (output file cannot be opened, ...) is
    // reported to the host as an error rather than a silent run with
    // no data. Nothing else is left running: stop the other one, exit.
    if ((sys && !sys->IsRunning()) || (dsk && !dsk->IsRunning())) {
        std::cerr << "[sidecar] a probe failed to start — reporting ProbeStartFailed, exit\n";
        if (sys) sys->Stop();
        if (dsk) dsk->Stop();
        SendStatus(ProfilerError::ProbeStartFailed);
        return 1;
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

    // 3. Message loop until MSG_STOP, the host's exit, EOF, or a signal.
    bool ackStop = false;
    while (true) {
        Wake w = NextMsg(ctl, hdr, payload);
        if (w != Wake::Message) {
            std::cerr << "[sidecar] " << WakeName(w, ctl.signo)
                      << " — stopping, final flush\n";
            break;
        }
        if (hdr.type == MSG_STOP) {
            std::cerr << "[sidecar] MSG_STOP received\n";
            ackStop = true;
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
        if (hdr.type == MSG_HOST_REAPER) {
            std::cerr << "[sidecar] MSG_HOST_REAPER: parent " << ::getppid()
                      << " reaps the adopted orphans reported to it\n";
            if (dsk) dsk->SetHostReaper(static_cast<uint32_t>(::getppid()));
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
    // their flush threads wake from their wait; Stop() joins.
    if (discovery) discovery->Stop();
    if (sys) sys->SignalStop();
    if (dsk) dsk->SignalStop();
    if (sys) sys->Stop();
    if (dsk) dsk->Stop();
    // Only MSG_STOP has a host waiting for the reply.
    if (ackStop) SendStatus(ProfilerError::Ok);
    std::cerr << "[sidecar] clean shutdown, exit 0\n";
    return 0;
}
