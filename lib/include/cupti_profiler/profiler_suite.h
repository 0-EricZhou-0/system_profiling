// ProfilerSuite — orchestrates GPU, System, and Disk profilers from a .pbtxt config.
#pragma once

#include <cupti_profiler/gpu_profiler.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/event_profiler.h>
#include <cupti_profiler/profiler_error.h>
#include <cupti_profiler/child_subreaper.h>

#include <memory>
#include <string>

#if defined(_WIN32)
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __declspec(dllexport)
  #else
    #define CUPTI_PROFILER_API __declspec(dllimport)
  #endif
#else
  #ifdef CUPTI_PROFILER_EXPORTS
    #define CUPTI_PROFILER_API __attribute__((visibility("default")))
  #else
    #define CUPTI_PROFILER_API
  #endif
#endif

namespace cupti_profiler {

class CUPTI_PROFILER_API ProfilerSuite {
public:
    ProfilerSuite();
    ~ProfilerSuite();
    ProfilerSuite(ProfilerSuite&&) noexcept;
    ProfilerSuite& operator=(ProfilerSuite&&) noexcept;

    /// Load configuration from a protobuf text format (.pbtxt) file.
    void LoadConfig(const std::string& pbtxtPath);

    /// Load configuration from a serialized ProfilerSuiteConfig protobuf
    /// (binary wire format). Used by language bindings that build the
    /// config in their own runtime and want to skip the .pbtxt round-trip.
    void LoadConfigFromBytes(const std::string& serializedProto);

    /// Access individual profilers (available after LoadConfig).
    GpuProfiler& GetGPUProfiler();
    SystemProfiler& GetSystemProfiler();
    DiskProfiler& GetDiskProfiler();
    EventProfiler& GetEventProfiler();

    /// Configure all enabled profilers.
    ///
    /// Also runs the startup situation report (which tracking
    /// guarantees hold here: /proc children support, pidfd, Yama,
    /// observer capabilities, secure-exec, the observer binary's
    /// filesystem, subreaper state, taskstats, and per listed root:
    /// same uid, spawned vs attached, io readable). It is logged once
    /// to stderr and written to session_metadata.pb.
    ///
    /// Under SystemProbeMode::Sidecar, this is where the sidecar
    /// process is spawned and sent its config. Failures surface here
    /// rather than as a silent no-sample run: SidecarNotFound,
    /// SidecarSpawnFailed, SidecarExited, SidecarBadHandshake (the
    /// sidecar rejected the config) or SidecarAffinityFailed
    /// (sidecar_cpus unusable). Returns InvalidConfig, with the reason
    /// on stderr, for an inconsistent GPU config (flush_interval_ms
    /// below decode_interval_ms); nothing is configured then.
    /// Returns ProfilerError::NotConfigured if LoadConfig has not
    /// been called yet.
    ProfilerError Configure();

    /// Start all enabled profilers.
    ///
    /// Returns ProfilerError::ProbeStartFailed if a System or Disk probe
    /// did not start (typically: its output file cannot be opened) —
    /// under SIDECAR the sidecar reports this and exits — or
    /// SidecarExited / SidecarBadHandshake if the sidecar did not answer.
    /// Every other enabled probe has still been started, so call Stop()
    /// as usual; the failed probe simply writes nothing.
    ///
    /// Unless ProfilerSuiteConfig.disable_signal_handlers is set, also
    /// installs handlers for the catchable signals whose default action
    /// ends the process: on one, the suite is stopped and flushed, then the
    /// previous handler (or the default action) runs. Stop() removes them.
    ProfilerError Start();

    /// Stop all enabled profilers: final GPU decode, every probe's last
    /// flush, the sidecar's; trace files closed. Runs once, whichever of
    /// the host or a signal comes first. Thread-safe.
    void Stop();

    /// Begin tracking a PID mid-run. Fans out to every enabled probe
    /// that supports per-PID sampling (currently System + Disk).
    /// Whether its descendants are tracked follows
    /// ProfilerSuiteConfig.process_discovery.enabled. Thread-safe.
    void AddTrackedProcess(uint32_t pid, std::string alias = {});

    /// Same, with an explicit per-root choice that overrides
    /// process_discovery.enabled for this root: trackDescendants=true
    /// follows its children (recursively unless direct_children_only),
    /// false traces this PID alone. Use it to follow a spawned server's
    /// tree without also following the host's own children.
    void AddTrackedProcess(uint32_t pid, std::string alias, bool trackDescendants);

    /// Stop tracking a PID mid-run. The PID appears one more time in
    /// the next flush of each affected probe (with TrackedProcessV2.
    /// removed=true) before being dropped. If it was a root whose
    /// descendants were tracked, its already-discovered descendants
    /// stay tracked until they exit. Thread-safe.
    void RemoveTrackedProcess(uint32_t pid);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cupti_profiler
