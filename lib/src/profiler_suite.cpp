#include <cupti_profiler/profiler_suite.h>

#include "metric_catalog.h"
#include "metric_catalog_builtins.h"
#include "profiler_config.pb.h"
#include "session_metadata.pb.h"
#include "session_metadata_writer.h"

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

void ProfilerSuite::Configure() {
    if (!m_impl->loaded) {
        std::cerr << "ProfilerSuite::Configure() called before LoadConfig()\n";
        return;
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
    if (m_impl->sysEnabled)   m_impl->systemProfiler.Configure(m_impl->sysConfig);
    if (m_impl->diskEnabled)  m_impl->diskProfiler.Configure(m_impl->diskConfig);
    if (m_impl->eventEnabled) m_impl->eventProfiler.Configure(m_impl->eventConfig);
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
    if (m_impl->sysEnabled)   m_impl->systemProfiler.Start();
    if (m_impl->diskEnabled)  m_impl->diskProfiler.Start();
    if (m_impl->eventEnabled) m_impl->eventProfiler.Start();

    // Emit the manifest now so live tailers (e.g. visualize_interactive.py
    // --live) have a starting point. Stop() re-emits the identical content
    // atomically.
    m_impl->WriteSessionManifest();
}

void ProfilerSuite::Stop() {
    // Signal all sample threads to stop immediately, before any
    // slow teardown (e.g., GPU CUPTI cleanup) blocks.
    if (m_impl->sysEnabled)   m_impl->systemProfiler.SignalStop();
    if (m_impl->diskEnabled)  m_impl->diskProfiler.SignalStop();
    if (m_impl->eventEnabled) m_impl->eventProfiler.SignalStop();

    if (m_impl->gpuEnabled)   m_impl->gpuProfiler.Stop();
    if (m_impl->sysEnabled)   m_impl->systemProfiler.Stop();
    if (m_impl->diskEnabled)  m_impl->diskProfiler.Stop();
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
