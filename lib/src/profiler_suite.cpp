#include <cupti_profiler/profiler_suite.h>

#include "metric_catalog.h"
#include "metric_catalog_builtins.h"
#include "profiler_config.pb.h"
#include "session_metadata.pb.h"
#include "session_metadata_writer.h"
#include "sidecar_process.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <sys/stat.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

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

    bool loaded = false;

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
ProfilerSuite::~ProfilerSuite() = default;
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

static void ResolvePIDZero(std::vector<TrackedProcess>& processes) {
    uint32_t myPID = static_cast<uint32_t>(getpid());
    for (auto& p : processes) {
        if (p.pid == 0) p.pid = myPID;
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
    // Serialize back to wire bytes for sidecar forwarding — cheaper
    // than re-serialising later, and keeps the sidecar's view exactly
    // as parsed (defaults populated where the .pbtxt omitted fields).
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
    m_impl->cachedConfigBytes = serializedProto;
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
        m_impl->gpuConfig.samplingFrequencyHz = g.sampling_frequency_hz() > 0 ? g.sampling_frequency_hz() : 10000;
        m_impl->gpuConfig.hwBufferSize = g.hw_buffer_size() > 0 ? g.hw_buffer_size() : 512 * 1024 * 1024;
        m_impl->gpuConfig.maxSamples = g.max_samples() > 0 ? g.max_samples() : 50000;
        m_impl->gpuConfig.flushIntervalMs = g.flush_interval_ms();
        m_impl->gpuConfig.outputFile = g.output_file();
        for (const auto& m : g.metrics()) {
            m_impl->gpuConfig.metrics.push_back(m);
        }
    }

    // System config
    if (proto.has_system() && proto.system().enabled()) {
        m_impl->sysEnabled = true;
        const auto& s = proto.system();
        m_impl->sysConfig.samplingFrequencyHz = s.sampling_frequency_hz() > 0 ? s.sampling_frequency_hz() : 100;
        m_impl->sysConfig.flushIntervalMs = s.flush_interval_ms() > 0 ? s.flush_interval_ms() : 5000;
        m_impl->sysConfig.outputFile = s.output_file();
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
        m_impl->diskConfig.samplingFrequencyHz = d.sampling_frequency_hz() > 0 ? d.sampling_frequency_hz() : 10;
        m_impl->diskConfig.flushIntervalMs = d.flush_interval_ms() > 0 ? d.flush_interval_ms() : 5000;
        m_impl->diskConfig.outputFile = d.output_file();
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
        m_impl->eventConfig.flushIntervalMs = e.flush_interval_ms() > 0 ? e.flush_interval_ms() : 5000;
        m_impl->eventConfig.outputFile = !e.output_file().empty() ? e.output_file() : "events.pb";
    }

    // Session metadata path
    m_impl->sessionMetadataPath = proto.session_metadata_file().empty()
        ? std::string("session_metadata.pb")
        : proto.session_metadata_file();

    // Metric catalog path (loaded at Configure()). Empty = default
    // location next to the binary.
    m_impl->metricCatalogPath = proto.metric_catalog_path();

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

    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Configure(m_impl->gpuConfig);
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
        // Steady_clock reference now, so sidecar samples share the
        // workload's t=0. wall_clock is deferred to Start().
        uint64_t steady_ref =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (auto e = m_impl->sidecar->SendSyncAnchor(steady_ref, /*wall=*/0);
            e != ProfilerError::Ok)
        {
            std::cerr << "[ProfilerSuite] sidecar SendSyncAnchor: " << ToString(e) << "\n";
            m_impl->sidecar.reset();
            return e;
        }
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
    return ProfilerError::Ok;
}

void ProfilerSuite::Impl::WriteSessionManifest() {
    SessionMetadata meta;
    char hostbuf[256] = {0};
    gethostname(hostbuf, sizeof(hostbuf));
    meta.set_hostname(hostbuf);
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
        p->set_output_file(path);
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

    internal::WriteSessionMetadata(sessionMetadataPath, meta);
}

void ProfilerSuite::Start() {
    m_impl->startWallClockEpochNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Start();
    // Legacy sys/disk start in-process; Sidecar sys/disk are started
    // by the sidecar on receipt of MSG_START.
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.Start();
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.Start();
    if (m_impl->eventEnabled) m_impl->eventProfiler.Start();

    // Nudge the sidecar to begin sampling (if one is running). Errors
    // from this handshake are logged; we don't fail Start() over them
    // because the in-process probes may still be usefully sampling.
    if (m_impl->sidecar) {
        if (auto e = m_impl->sidecar->SendStart(); e != ProfilerError::Ok) {
            std::cerr << "[ProfilerSuite] sidecar SendStart: "
                      << ToString(e) << "\n";
        }
    }

    // Emit the manifest now so live tailers (e.g. visualize_interactive.py
    // --live) have a starting point. Stop() re-emits the identical content
    // atomically.
    m_impl->WriteSessionManifest();
}

void ProfilerSuite::Stop() {
    // Signal in-process sample threads to stop immediately, before any
    // slow teardown (e.g., GPU CUPTI cleanup) blocks. Legacy sys/disk
    // signal here; Sidecar sys/disk signal via MSG_STOP over the pipe
    // below.
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.SignalStop();
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.SignalStop();
    if (m_impl->eventEnabled) m_impl->eventProfiler.SignalStop();

    // Tell the sidecar to flush + exit cleanly. The destructor's
    // SIGTERM path is the fallback if this handshake failed.
    if (m_impl->sidecar) {
        if (auto e = m_impl->sidecar->SendStop(); e != ProfilerError::Ok) {
            std::cerr << "[ProfilerSuite] sidecar SendStop: "
                      << ToString(e) << "\n";
        }
        m_impl->sidecar.reset();
    }

    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Stop();
    if (m_impl->sysEnabled  && m_impl->sysConfig.mode  == SystemProbeMode::Legacy)
        m_impl->systemProfiler.Stop();
    if (m_impl->diskEnabled && m_impl->diskConfig.mode == SystemProbeMode::Legacy)
        m_impl->diskProfiler.Stop();
    if (m_impl->eventEnabled) m_impl->eventProfiler.Stop();

    m_impl->WriteSessionManifest();
}

void ProfilerSuite::AddTrackedProcess(uint32_t pid, std::string alias) {
    // Fan out to every probe that supports per-PID sampling.
    if (m_impl->sysEnabled)  m_impl->systemProfiler.AddTrackedProcess(pid, alias);
    if (m_impl->diskEnabled) m_impl->diskProfiler.AddTrackedProcess(pid, std::move(alias));
}

void ProfilerSuite::RemoveTrackedProcess(uint32_t pid) {
    if (m_impl->sysEnabled)  m_impl->systemProfiler.RemoveTrackedProcess(pid);
    if (m_impl->diskEnabled) m_impl->diskProfiler.RemoveTrackedProcess(pid);
}

} // namespace cupti_profiler
