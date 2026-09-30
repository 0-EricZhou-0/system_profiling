#include <cupti_profiler/profiler_suite.h>

#include "lifecycle.h"
#include "testing_hooks.h"
#include "metric_catalog.h"
#include "metric_catalog_builtins.h"
#include "process_discovery.h"
#include "profiler_config.pb.h"
#include "session_metadata.pb.h"
#include "session_metadata_writer.h"
#include "sidecar_process.h"
#include <cupti_profiler/child_subreaper.h>
#include <cupti_profiler/defaults.h>
#include <cupti_profiler/version.h>
#include "situation_report.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace cupti_profiler {

class ProfilerSuite::Impl {
public:
    GpuProfiler gpuProfiler;
    SystemProfiler systemProfiler;
    DiskProfiler diskProfiler;
    EventProfiler eventProfiler;

    ProfilerConfig gpuConfig;
    SystemProfilerConfig sysConfig;
    DiskProfilerConfig diskConfig;
    EventProfilerConfig eventConfig;

    bool gpuEnabled = false;
    bool sysEnabled = false;
    bool diskEnabled = false;
    bool eventEnabled = false;

    // Every file name of the trace (field, effective name: the one given,
    // or the default when unset): each must be a plain file name in
    // output_dir, and no two may be the same (checked at Configure()).
    std::vector<std::pair<std::string, std::string>> fileNames;
    std::string sessionMetadataPath;
    std::string metricCatalogPath;

    // Loaded MetricCatalog. nullopt before Configure().
    std::unique_ptr<internal::MetricCatalog> catalog;

    // Sidecar handle, populated when either system or disk config
    // requested SystemProbeMode::Sidecar. Owned by the suite; joins
    // + reaps in the destructor.
    std::unique_ptr<internal::SidecarProcess> sidecar;
    // Serialized suite config bytes cached across Configure(), used
    // to feed the sidecar on first handshake. Kept as std::string
    // (which the proto's SerializeToString hands us) rather than a
    // std::vector<uint8_t> for zero-copy through the SendConfig API.
    std::string cachedConfigBytes;

    uint64_t startWallClockEpochNs = 0;

    // ProfilerSuiteConfig.sidecar_cpus is non-empty (applied by the
    // sidecar itself; remembered here only to warn when unused).
    bool sidecarCpusSet = false;

    // Descendant tracking (ProfilerSuiteConfig.process_discovery). The
    // in-process instance serves LEGACY probes; under SIDECAR the
    // sidecar runs its own from the serialized config.
    internal::DiscoverySettings discoverySettings;
    std::unique_ptr<internal::ProcessDiscovery> discovery;

    // Startup situation report: environment lines from Configure(),
    // plus one line per root added (config-listed or mid-run).
    std::mutex situationMutex;
    std::vector<internal::SituationLine> situation;
    std::vector<internal::SituationLine> rootSituation;

    bool loaded = false;

    bool SysLegacy()  const { return sysEnabled  && sysConfig.mode  == SystemProbeMode::Legacy; }
    bool DiskLegacy() const { return diskEnabled && diskConfig.mode == SystemProbeMode::Legacy; }

    void AddTrackedProcess(uint32_t pid, const std::string& alias,
                           std::optional<bool> trackDescendants);
    void RecordRootSituation(uint32_t pid, std::optional<bool> trackDescendants);

    // Once this process is a child subreaper (adopt_orphans()), tell the
    // disk probe — in-process, or in the sidecar — so it can resolve
    // reap chains (ProcessTrackingProbe::SetHostReaper: CPU tails, reaped I/O). adopt_orphans() comes
    // before the workload is spawned, so checking at Start() and at
    // every AddTrackedProcess() catches it before any root's tree runs.
    bool hostReaperSent = false;
    bool started        = false;   // the sidecar takes messages only after MSG_START
    void SyncHostReaper();

    // Stop() may come from the host, the signal flusher or the exit
    // hook; stopMutex + stopped make it run once. running: between
    // Start() and Stop().
    bool running = false;
    bool stopped = false;
    bool installSignalHandlers = true;   // ProfilerSuiteConfig.disable_signal_handlers
    bool signalHandlers = false;         // installed by this suite's Start()
    std::mutex stopMutex;
    void Stop();

    // Walk a parsed ProfilerSuiteConfig and populate this Impl. Shared
    // between the .pbtxt path (LoadConfig) and the serialized-bytes path
    // used by language bindings (LoadConfigFromBytes).
    void ApplyParsedConfig(const ProfilerSuiteConfig& proto);

    // Build the SessionMetadata message from the parsed config + the
    // captured wall-clock anchor, and write it (atomically) to disk.
    // Called from both Start() — so live tailers see the manifest from
    // second one — and Stop(), which re-emits identical content.
    void WriteSessionManifest();
};

ProfilerSuite::ProfilerSuite() : m_impl(std::make_unique<Impl>()) {}
ProfilerSuite::~ProfilerSuite() {
    if (!m_impl) return;
    if (m_impl->running && !m_impl->stopped) {
        internal::lifecycle::WarnNotStopped("ProfilerSuite", "when it was destroyed");
        m_impl->Stop();
    }
    internal::lifecycle::Unregister(m_impl.get());
}

void StopRunningAtExit() { internal::lifecycle::StopAtExit(); }
ProfilerSuite::ProfilerSuite(ProfilerSuite&&) noexcept = default;
ProfilerSuite& ProfilerSuite::operator=(ProfilerSuite&&) noexcept = default;

/// Recursively create directories (like mkdir -p).
static void MakeDirs(const std::string& path) {
    if (path.empty()) return;
    // Build each component
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

/// Prepend directory to filename if both are non-empty.
static std::string JoinPath(const std::string& dir, const std::string& file) {
    if (dir.empty() || file.empty()) return file;
    if (dir.back() == '/') return dir + file;
    return dir + "/" + file;
}

/// A probe file's path as session_metadata.pb records it: relative to the
/// directory holding session_metadata.pb (both are plain file names in
/// output_dir, so this is the file name), so a trace directory stays
/// readable after it is copied or moved. Both paths are as the probes
/// open them (relative ones against the current directory).
static std::string RelativeToManifest(const std::string& file, const std::string& manifest) {
    namespace fs = std::filesystem;
    std::error_code ec1, ec2;
    const fs::path f   = fs::absolute(file, ec1).lexically_normal();
    const fs::path dir = fs::absolute(manifest, ec2).lexically_normal().parent_path();
    if (ec1 || ec2) return file;
    const fs::path rel = f.lexically_relative(dir);
    return rel.empty() ? file : rel.generic_string();
}

static void ResolvePIDZero(std::vector<TrackedProcess>& processes) {
    uint32_t myPID = static_cast<uint32_t>(getpid());
    for (auto& p : processes) {
        if (p.pid == 0) p.pid = myPID;
    }
}

// Resolve pid:0 → workload getpid() in the proto's per-probe process
// lists, so the serialised bytes we cache for the sidecar carry the
// real PID rather than the sentinel. The sidecar (a different PID)
// otherwise ends up tracking pid=0 (nonexistent /proc/0) and reports
// all-zero per-PID CPU.
static void ResolvePID0InProto(ProfilerSuiteConfig& proto) {
    uint32_t myPID = static_cast<uint32_t>(getpid());
    if (proto.has_system()) {
        for (auto& p : *proto.mutable_system()->mutable_processes()) {
            if (p.pid() == 0) p.set_pid(myPID);
        }
    }
    if (proto.has_disk()) {
        for (auto& p : *proto.mutable_disk()->mutable_processes()) {
            if (p.pid() == 0) p.set_pid(myPID);
        }
    }
}

void ProfilerSuite::LoadConfig(const std::string& pbtxtPath) {
    std::ifstream f(pbtxtPath);
    if (!f) {
        std::cerr << "Failed to open config file: " << pbtxtPath << "\n";
        exit(1);
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    ProfilerSuiteConfig proto;
    if (!google::protobuf::TextFormat::ParseFromString(content, &proto)) {
        std::cerr << "Failed to parse config file: " << pbtxtPath << "\n";
        exit(1);
    }

    std::cout << "Loaded config from " << pbtxtPath << "\n";
    // Resolve pid:0 sentinel to the workload's actual PID in the
    // proto BEFORE we cache the wire bytes — the sidecar (a different
    // PID) can't do this resolution itself, and would end up tracking
    // pid=0 (/proc/0 doesn't exist → all-zero cpu_pct). The parent-
    // side ApplyParsedConfig does the same to the C++ config it
    // builds for the in-process (Legacy) probes.
    ResolvePID0InProto(proto);
    proto.SerializeToString(&m_impl->cachedConfigBytes);
    m_impl->ApplyParsedConfig(proto);
}

void ProfilerSuite::LoadConfigFromBytes(const std::string& serializedProto) {
    ProfilerSuiteConfig proto;
    if (!proto.ParseFromString(serializedProto)) {
        std::cerr << "Failed to parse serialized ProfilerSuiteConfig "
                  << "(" << serializedProto.size() << " bytes)\n";
        exit(1);
    }
    std::cout << "Loaded config from serialized protobuf ("
              << serializedProto.size() << " bytes)\n";
    // Same pid:0 → workload PID resolution as the pbtxt path; then
    // re-serialise so the sidecar gets the resolved bytes.
    ResolvePID0InProto(proto);
    proto.SerializeToString(&m_impl->cachedConfigBytes);
    m_impl->ApplyParsedConfig(proto);
}

void ProfilerSuite::Impl::ApplyParsedConfig(const ProfilerSuiteConfig& proto) {
    auto* m_impl = this;

    // GPU config
    if (proto.has_gpu() && proto.gpu().enabled()) {
        m_impl->gpuEnabled = true;
        const auto& g = proto.gpu();
        m_impl->gpuConfig.deviceIndices.clear();
        for (int idx : g.device_indices()) {
            m_impl->gpuConfig.deviceIndices.push_back(idx);
        }
        m_impl->gpuConfig.samplingFrequencyHz = g.sampling_frequency_hz() > 0 ? g.sampling_frequency_hz() : kDefaultGpuSamplingHz;
        m_impl->gpuConfig.hwBufferSize = g.hw_buffer_size() > 0 ? g.hw_buffer_size() : 512 * 1024 * 1024;
        m_impl->gpuConfig.maxSamples = g.max_samples();   // 0 = auto
        m_impl->gpuConfig.flushIntervalMs = g.flush_interval_ms();    // 0 = default
        m_impl->gpuConfig.decodeIntervalMs = g.decode_interval_ms();   // 0 = default
        m_impl->gpuConfig.outputFile = !g.output_file().empty() ? g.output_file() : "gpu_metrics.pb";
        for (const auto& m : g.metrics()) {
            m_impl->gpuConfig.metrics.push_back(m);
        }
    }

    // System config
    if (proto.has_system() && proto.system().enabled()) {
        m_impl->sysEnabled = true;
        const auto& s = proto.system();
        m_impl->sysConfig.samplingFrequencyHz = s.sampling_frequency_hz() > 0 ? s.sampling_frequency_hz() : kDefaultSystemSamplingHz;
        m_impl->sysConfig.flushIntervalMs = s.flush_interval_ms();   // 0 = default
        m_impl->sysConfig.outputFile = !s.output_file().empty() ? s.output_file() : "system_metrics.pb";
        m_impl->sysConfig.mode =
            (s.mode() == SYSTEM_PROBE_MODE_SIDECAR)
                ? SystemProbeMode::Sidecar
                : SystemProbeMode::Legacy;
        for (const auto& p : s.processes()) {
            TrackedProcess tp;
            tp.pid   = p.pid();
            tp.alias = p.alias();
            m_impl->sysConfig.Processes.push_back(std::move(tp));
        }
        ResolvePIDZero(m_impl->sysConfig.Processes);
    }

    // Disk config
    if (proto.has_disk() && proto.disk().enabled()) {
        m_impl->diskEnabled = true;
        const auto& d = proto.disk();
        m_impl->diskConfig.samplingFrequencyHz = d.sampling_frequency_hz() > 0 ? d.sampling_frequency_hz() : kDefaultDiskSamplingHz;
        m_impl->diskConfig.flushIntervalMs = d.flush_interval_ms();  // 0 = default
        m_impl->diskConfig.outputFile = !d.output_file().empty() ? d.output_file() : "disk_metrics.pb";
        m_impl->diskConfig.mode =
            (d.mode() == SYSTEM_PROBE_MODE_SIDECAR)
                ? SystemProbeMode::Sidecar
                : SystemProbeMode::Legacy;
        for (const auto& dev : d.devices()) {
            m_impl->diskConfig.devices.push_back(dev);
        }
        for (const auto& p : d.processes()) {
            TrackedProcess tp;
            tp.pid   = p.pid();
            tp.alias = p.alias();
            m_impl->diskConfig.Processes.push_back(std::move(tp));
        }
        ResolvePIDZero(m_impl->diskConfig.Processes);
    }

    // Events config
    if (proto.has_events() && proto.events().enabled()) {
        m_impl->eventEnabled = true;
        const auto& e = proto.events();
        m_impl->eventConfig.flushIntervalMs = e.flush_interval_ms(); // 0 = default
        m_impl->eventConfig.outputFile = !e.output_file().empty() ? e.output_file() : "events.pb";
    }

    // Session metadata path
    m_impl->sessionMetadataPath = proto.session_metadata_file().empty()
        ? std::string("session_metadata.pb")
        : proto.session_metadata_file();

    // Effective file names, before output_dir is joined on (Configure()
    // rejects one that is not a plain file name, and duplicates). An
    // unset name has its default, so no empty name reaches the metadata.
    m_impl->fileNames.clear();
    if (m_impl->gpuEnabled)   m_impl->fileNames.emplace_back("gpu.output_file", m_impl->gpuConfig.outputFile);
    if (m_impl->sysEnabled)   m_impl->fileNames.emplace_back("system.output_file", m_impl->sysConfig.outputFile);
    if (m_impl->diskEnabled)  m_impl->fileNames.emplace_back("disk.output_file", m_impl->diskConfig.outputFile);
    if (m_impl->eventEnabled) m_impl->fileNames.emplace_back("events.output_file", m_impl->eventConfig.outputFile);
    m_impl->fileNames.emplace_back("session_metadata_file", m_impl->sessionMetadataPath);

    // Metric catalog path (loaded at Configure()). Empty = default
    // location next to the binary.
    m_impl->metricCatalogPath = proto.metric_catalog_path();

    m_impl->sidecarCpusSet = proto.sidecar_cpus_size() > 0;
    m_impl->installSignalHandlers = !proto.disable_signal_handlers();

    // Descendant tracking — one setting for both system and disk.
    {
        const auto& pd = proto.process_discovery();
        m_impl->discoverySettings.enabled    = pd.enabled();
        m_impl->discoverySettings.recursive  = !pd.direct_children_only();
        m_impl->discoverySettings.intervalMs = pd.scan_interval_ms() > 0 ? pd.scan_interval_ms() : 100;
    }

    // Apply output_dir: prepend to each component's output_file, create dir if needed
    std::string outputDir = proto.output_dir();
    if (!outputDir.empty()) {
        MakeDirs(outputDir);
        std::cout << "Output directory: " << outputDir << "\n";
        if (m_impl->gpuEnabled)
            m_impl->gpuConfig.outputFile = JoinPath(outputDir, m_impl->gpuConfig.outputFile);
        if (m_impl->sysEnabled)
            m_impl->sysConfig.outputFile = JoinPath(outputDir, m_impl->sysConfig.outputFile);
        if (m_impl->diskEnabled)
            m_impl->diskConfig.outputFile = JoinPath(outputDir, m_impl->diskConfig.outputFile);
        if (m_impl->eventEnabled)
            m_impl->eventConfig.outputFile = JoinPath(outputDir, m_impl->eventConfig.outputFile);
        m_impl->sessionMetadataPath = JoinPath(outputDir, m_impl->sessionMetadataPath);
    }

    m_impl->loaded = true;
}

GpuProfiler& ProfilerSuite::GetGPUProfiler() { return m_impl->gpuProfiler; }
SystemProfiler& ProfilerSuite::GetSystemProfiler() { return m_impl->systemProfiler; }
DiskProfiler& ProfilerSuite::GetDiskProfiler() { return m_impl->diskProfiler; }
EventProfiler& ProfilerSuite::GetEventProfiler() { return m_impl->eventProfiler; }

ProfilerError ProfilerSuite::Configure() {
    if (!m_impl->loaded) {
        std::cerr << "ProfilerSuite::Configure() called before LoadConfig()\n";
        return ProfilerError::NotConfigured;
    }
    // Every file of a trace lives directly in output_dir, so the trace
    // directory is self-contained (session_metadata.pb names its probe
    // files by name alone, and the directory can be copied or moved).
    for (size_t i = 0; i < m_impl->fileNames.size(); ++i) {
        const auto& [field, name] = m_impl->fileNames[i];
        if (name.empty() || name.find('/') != std::string::npos || name == "." || name == "..") {
            std::cerr << "[cupti-profiler] error: " << field << " \"" << name
                      << "\" must be a plain file name: subdirectories are not allowed; "
                         "use output_dir\n";
            return ProfilerError::InvalidConfig;
        }
        for (size_t j = 0; j < i; ++j) {
            if (m_impl->fileNames[j].second == name) {
                std::cerr << "[cupti-profiler] error: " << field << " and " << m_impl->fileNames[j].first
                          << " are both \"" << name << "\": every file of a trace needs its own name\n";
                return ProfilerError::InvalidConfig;
            }
        }
    }

    // Assemble the MetricCatalog before any probe configures.
    //
    //   builtins (every probe's MetricDescriptor<Tick> array)
    //   + optional pbtxt overrides (merge-by-FQN, opt-in via
    //     ProfilerSuiteConfig.metric_catalog_path)
    //   + (later, GPU descriptors from CUPTI enumeration via
    //     MetricCatalog::AppendDescriptors())
    //
    // The seeded catalog is what gets inlined into session_metadata.pb
    // for the visualizer, so a downstream user that wants to tweak a
    // description / peak / smoothable only needs to ship a small
    // override pbtxt — no rebuild, no full catalog copy.
    m_impl->catalog = std::make_unique<internal::MetricCatalog>();
    internal::RegisterBuiltinDescriptors(*m_impl->catalog);
    if (!m_impl->metricCatalogPath.empty()) {
        m_impl->catalog->MergeOverridesFromPbtxt(m_impl->metricCatalogPath);
    }

    if (m_impl->gpuEnabled) {
        try {
            m_impl->gpuProfiler.Configure(m_impl->gpuConfig);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[cupti-profiler] error: " << e.what() << "\n";
            return ProfilerError::InvalidConfig;
        }
    }
    // Sidecar spawn + handshake, done BEFORE the sys/disk probes
    // configure so a cap failure surfaces here rather than after
    // the probes have already started allocating thread state.
    // Runs when either probe requested Sidecar mode; the sidecar
    // itself services both when both are enabled.
    const bool wantSidecar =
        (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Sidecar) ||
        (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Sidecar);
    if (wantSidecar) {
        m_impl->sidecar = std::make_unique<internal::SidecarProcess>();
        if (auto e = m_impl->sidecar->Spawn(); e != ProfilerError::Ok) {
            std::cerr << "[ProfilerSuite] sidecar Spawn: " << ToString(e) << "\n";
            m_impl->sidecar.reset();
            return e;
        }
        if (auto e = m_impl->sidecar->SendConfig(m_impl->cachedConfigBytes);
            e != ProfilerError::Ok)
        {
            std::cerr << "[ProfilerSuite] sidecar SendConfig: " << ToString(e) << "\n";
            m_impl->sidecar.reset();
            return e;
        }
        // No clock handshake: the sidecar stamps samples with
        // CLOCK_MONOTONIC (steady_clock), which is system-wide, and its
        // traces carry their own anchors, like the in-process probes'.
    }
    if (!wantSidecar && m_impl->sidecarCpusSet) {
        std::cerr << "[ProfilerSuite] note: sidecar_cpus ignored — no probe runs in "
                     "SIDECAR mode, and this process's own affinity is left alone\n";
    }
    // Legacy mode configures in-process. Sidecar mode: the sidecar
    // has already been sent the config over the pipe and will build
    // its own SystemProfiler / DiskProfiler when it receives
    // MSG_START (see ProfilerSuite::Start below).
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.Configure(m_impl->sysConfig);
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.Configure(m_impl->diskConfig);
    if (m_impl->eventEnabled) m_impl->eventProfiler.Configure(m_impl->eventConfig);

    // Descendant tracking for the in-process (LEGACY) probes, fed the
    // PIDs each probe lists. This process is the host, so it reaps the
    // adopted orphans itself (if adopt_orphans() was called).
    if (m_impl->SysLegacy() || m_impl->DiskLegacy()) {
        internal::AdoptionReaping reaping;
        reaping.mode    = internal::AdoptionReaping::Mode::InProcess;
        reaping.hostPid = static_cast<uint32_t>(::getpid());
        m_impl->discovery = std::make_unique<internal::ProcessDiscovery>(
            m_impl->discoverySettings,
            m_impl->SysLegacy()  ? &m_impl->systemProfiler : nullptr,
            m_impl->DiskLegacy() ? &m_impl->diskProfiler   : nullptr,
            reaping);
        using PD = internal::ProcessDiscovery;
        if (m_impl->SysLegacy())
            for (const auto& p : m_impl->sysConfig.Processes)
                m_impl->discovery->AddRoot(p.pid, p.alias, std::nullopt, PD::kSystemSink);
        if (m_impl->DiskLegacy())
            for (const auto& p : m_impl->diskConfig.Processes)
                m_impl->discovery->AddRoot(p.pid, p.alias, std::nullopt, PD::kDiskSink);
    }

    // Startup situation report: logged once, written to the manifest.
    if (m_impl->sysEnabled || m_impl->diskEnabled) {
        internal::SituationInputs in;
        in.sidecar = static_cast<bool>(m_impl->sidecar);
        if (m_impl->sidecar) {
            in.sidecarPid = m_impl->sidecar->child_pid();
            char buf[PATH_MAX] = {0};
            ssize_t n = ::readlink(("/proc/" + std::to_string(in.sidecarPid) + "/exe").c_str(),
                                   buf, sizeof(buf) - 1);
            in.observerBinary = n > 0 ? std::string(buf, static_cast<size_t>(n)) : "unknown";
        } else {
            char buf[PATH_MAX] = {0};
            ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
            in.observerBinary = n > 0 ? std::string(buf, static_cast<size_t>(n)) : "unknown";
        }
        in.discovery = m_impl->discoverySettings;
        auto lines = internal::ProbeSituation(in);
        internal::LogSituation("situation report:", lines);
        {
            std::lock_guard<std::mutex> lk(m_impl->situationMutex);
            m_impl->situation = std::move(lines);
        }
        std::vector<uint32_t> roots;
        for (const auto& p : m_impl->sysConfig.Processes)  roots.push_back(p.pid);
        for (const auto& p : m_impl->diskConfig.Processes) roots.push_back(p.pid);
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
        for (uint32_t pid : roots) m_impl->RecordRootSituation(pid, std::nullopt);
    }
    return ProfilerError::Ok;
}

void ProfilerSuite::Impl::RecordRootSituation(uint32_t pid,
                                              std::optional<bool> trackDescendants) {
    auto line = internal::ProbeRoot(pid, trackDescendants.value_or(discoverySettings.enabled),
                                    discoverySettings.intervalMs);
    internal::LogSituation("situation report:", {line});
    std::lock_guard<std::mutex> lk(situationMutex);
    for (auto& l : rootSituation) {
        if (l.check == line.check) { l = std::move(line); return; }
    }
    rootSituation.push_back(std::move(line));
}

void ProfilerSuite::Impl::WriteSessionManifest() {
    SessionMetadata meta;
    char hostbuf[256] = {0};
    gethostname(hostbuf, sizeof(hostbuf));
    meta.set_hostname(hostbuf);
    meta.mutable_producer()->set_name("cupti-profiler");
    meta.mutable_producer()->set_version(Version());
    meta.mutable_producer()->set_git_commit(GitCommit());
    meta.set_wall_clock_epoch_ns(startWallClockEpochNs);
    {
        std::time_t secs = startWallClockEpochNs / 1000000000ULL;
        uint64_t ns_part = startWallClockEpochNs % 1000000000ULL;
        std::tm tm_utc{};
        gmtime_r(&secs, &tm_utc);
        std::ostringstream iso;
        iso << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S")
            << "." << std::setw(9) << std::setfill('0') << ns_part << "Z";
        meta.set_start_iso8601(iso.str());
    }
    auto addProbe = [&](ProbeKind kind, const std::string& path, uint64_t hz) {
        auto* p = meta.add_probes();
        p->set_kind(kind);
        p->set_output_file(RelativeToManifest(path, sessionMetadataPath));
        p->set_sampling_frequency_hz(hz);
    };
    if (gpuEnabled)
        addProbe(PROBE_KIND_GPU,    gpuConfig.outputFile,
                 gpuConfig.samplingFrequencyHz);
    if (sysEnabled)
        addProbe(PROBE_KIND_SYSTEM, sysConfig.outputFile,
                 sysConfig.samplingFrequencyHz);
    if (diskEnabled)
        addProbe(PROBE_KIND_DISK,   diskConfig.outputFile,
                 diskConfig.samplingFrequencyHz);
    if (eventEnabled)
        addProbe(PROBE_KIND_EVENTS, eventConfig.outputFile, 0);

    // Inline the active MetricCatalog so the visualizer only needs
    // one file (session_metadata.pb) to bootstrap.
    if (catalog) {
        *meta.mutable_catalog() = catalog->Proto();
    }

    // Situation report. The subreaper line is re-probed: adopt_orphans()
    // may have been called after Configure().
    {
        std::lock_guard<std::mutex> lk(situationMutex);
        auto add = [&](const internal::SituationLine& l) {
            auto* c = meta.add_situation();
            c->set_check(l.check);
            c->set_observed(l.observed);
            c->set_consequence(l.consequence);
            c->set_degraded(l.degraded);
        };
        const auto subreaper = internal::ProbeSubreaper();
        for (const auto& l : situation) add(l.check == subreaper.check ? subreaper : l);
        for (const auto& l : rootSituation) add(l);
    }

    internal::WriteSessionMetadata(sessionMetadataPath, meta);
}

ProfilerError ProfilerSuite::Start() {
    ProfilerError result = ProfilerError::Ok;
    m_impl->startWallClockEpochNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Start();
    // Legacy sys/disk start in-process; Sidecar sys/disk are started
    // by the sidecar on receipt of MSG_START.
    if (m_impl->SysLegacy()) {
        m_impl->systemProfiler.Start();
        if (!m_impl->systemProfiler.IsRunning()) result = ProfilerError::ProbeStartFailed;
    }
    if (m_impl->DiskLegacy()) {
        m_impl->diskProfiler.Start();
        if (!m_impl->diskProfiler.IsRunning()) result = ProfilerError::ProbeStartFailed;
    }
    if (m_impl->eventEnabled) m_impl->eventProfiler.Start();
    if (m_impl->discovery) m_impl->discovery->Start();

    // Tell the sidecar to begin sampling. On failure it has exited (or
    // is exiting) without sampling anything: reap it and report the
    // error, after the in-process probes above have started, so the
    // caller can still decide to run without system/disk data.
    if (m_impl->sidecar) {
        if (auto e = m_impl->sidecar->SendStart(); e != ProfilerError::Ok) {
            std::cerr << "[ProfilerSuite] sidecar failed to start its probes ("
                      << ToString(e) << "); no system/disk data will be recorded\n";
            m_impl->sidecar.reset();
            if (result == ProfilerError::Ok) result = e;
        }
    }

    m_impl->started = true;
    m_impl->SyncHostReaper();

    // From here a catchable fatal signal stops the suite (final GPU
    // decode, every probe's flush, the sidecar's) before it takes its
    // course. See lifecycle.cpp.
    m_impl->running = true;
    internal::lifecycle::Register(m_impl.get(), internal::lifecycle::Order::Suite, "ProfilerSuite",
                                  [impl = m_impl.get()] { impl->Stop(); });
    if (m_impl->installSignalHandlers) {
        internal::lifecycle::InstallSignalHandlers();
        m_impl->signalHandlers = true;
    }

    // Emit the manifest now so live tailers (e.g. visualize_interactive.py
    // --live) have a starting point. Stop() re-emits the identical content
    // atomically.
    m_impl->WriteSessionManifest();
    return result;
}

void ProfilerSuite::Stop() { m_impl->Stop(); }

void ProfilerSuite::Impl::Stop() {
    std::lock_guard<std::mutex> stopLock(stopMutex);
    internal::lifecycle::StopScope stopping;
    if (stopped) return;
    stopped = true;
    internal::PassStopDelay();   // test-only; see <cupti_profiler/testing.h>
    auto* m_impl = this;
    // Fire off the shutdown signal to EVERY sample thread —
    // in-process AND the sidecar — as early as possible, so they all
    // wind down in parallel with our slow local teardown below. The
    // sidecar signal is fire-and-forget here (write MSG_STOP, don't
    // wait for the ack); we collect the ack at the very end via
    // JoinStopAck. Rationale: gpuProfiler.Stop() below does the GPU's
    // final decode, flush and CUPTI teardown; waiting for the sidecar's
    // ack first would serialize the two and let the sidecar sample past
    // the workload's real end for that long. (Every probe thread's wait
    // is interruptible, so no Stop() waits out an interval.)
    //
    // Discovery stops first (it only takes a condvar wake + join), so
    // nothing is registered during teardown and its final stats reach
    // the probes' last flush.
    if (m_impl->discovery) m_impl->discovery->Stop();
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.SignalStop();
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.SignalStop();
    if (m_impl->eventEnabled) m_impl->eventProfiler.SignalStop();
    bool sidecarStopSent = false;
    if (m_impl->sidecar) {
        if (auto e = m_impl->sidecar->SignalStop(); e != ProfilerError::Ok) {
            // It stopped on its own: SIGTERM/SIGINT (a terminal Ctrl-C
            // reaches it too) or an error. Say how, once.
            std::cerr << "[ProfilerSuite] sidecar had already stopped: "
                      << m_impl->sidecar->DescribeExit() << "\n";
        } else {
            sidecarStopSent = true;
        }
    }

    // Now join everything. Order doesn't matter much for timing
    // correctness (samples were bounded above by the SignalStop
    // fan-out); we keep the GPU/events → sidecar → sys/disk order
    // for consistency with the pre-sidecar codebase.
    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Stop();
    if (m_impl->eventEnabled) m_impl->eventProfiler.Stop();
    if (m_impl->sidecar) {
        if (sidecarStopSent) {
            if (auto e = m_impl->sidecar->JoinStopAck(); e != ProfilerError::Ok) {
                std::cerr << "[ProfilerSuite] sidecar did not acknowledge MSG_STOP ("
                          << ToString(e) << "): " << m_impl->sidecar->DescribeExit() << "\n";
            }
        }
        m_impl->sidecar.reset();
    }
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.Stop();
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.Stop();

    m_impl->WriteSessionManifest();
    running = false;
    if (signalHandlers) {
        internal::lifecycle::RemoveSignalHandlers();
        signalHandlers = false;
    }
    internal::lifecycle::Unregister(this);
}

void ProfilerSuite::AddTrackedProcess(uint32_t pid, std::string alias) {
    m_impl->AddTrackedProcess(pid, alias, std::nullopt);
}

void ProfilerSuite::AddTrackedProcess(uint32_t pid, std::string alias, bool trackDescendants) {
    m_impl->AddTrackedProcess(pid, alias, trackDescendants);
}

void ProfilerSuite::Impl::SyncHostReaper() {
    if (hostReaperSent || !started || !ChildSubreaperEnabled()) return;
    if (SysLegacy())  systemProfiler.SetHostReaper(static_cast<uint32_t>(::getpid()));
    if (DiskLegacy()) diskProfiler.SetHostReaper(static_cast<uint32_t>(::getpid()));
    if (sidecar && ((sysEnabled && !SysLegacy()) || (diskEnabled && !DiskLegacy()))) {
        if (auto e = sidecar->SendHostReaper(); e != ProfilerError::Ok) {
            std::cerr << "[ProfilerSuite] sidecar SendHostReaper: " << ToString(e) << "\n";
            return;
        }
    }
    hostReaperSent = true;
}

void ProfilerSuite::Impl::AddTrackedProcess(uint32_t pid, const std::string& alias,
                                            std::optional<bool> trackDescendants) {
    SyncHostReaper();
    // Fan out to every probe that supports per-PID sampling. Legacy
    // probes handle it in-process; Sidecar probes' add goes through
    // the sidecar over the pipe (single message serving both probes
    // when both are enabled — sidecar fans out on its side).
    if (SysLegacy())  systemProfiler.AddTrackedProcess(pid, alias);
    if (DiskLegacy()) diskProfiler.AddTrackedProcess(pid, alias);
    if (discovery) discovery->AddRoot(pid, alias, trackDescendants);
    if (sidecar) {
        if (auto e = sidecar->SendAddPid(pid, alias, trackDescendants);
            e != ProfilerError::Ok)
        {
            std::cerr << "[ProfilerSuite] sidecar SendAddPid(pid=" << pid
                      << "): " << ToString(e) << "\n";
        }
    }
    if (sysEnabled || diskEnabled) RecordRootSituation(pid, trackDescendants);
}

void ProfilerSuite::RemoveTrackedProcess(uint32_t pid) {
    const bool sys_legacy  = m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy;
    const bool disk_legacy = m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy;
    if (sys_legacy)  m_impl->systemProfiler.RemoveTrackedProcess(pid);
    if (disk_legacy) m_impl->diskProfiler.RemoveTrackedProcess(pid);
    if (m_impl->discovery) m_impl->discovery->RemoveRoot(pid);
    if (m_impl->sidecar) {
        if (auto e = m_impl->sidecar->SendRemovePid(pid);
            e != ProfilerError::Ok)
        {
            std::cerr << "[ProfilerSuite] sidecar SendRemovePid(pid=" << pid
                      << "): " << ToString(e) << "\n";
        }
    }
}

} // namespace cupti_profiler
