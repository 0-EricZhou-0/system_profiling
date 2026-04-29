// pybind11 wrapper exposing the cupti_profiler public C++ API to Python.
//
// Module name (as imported from Python): cupti_profiler._native
// The user-facing package is built up around this in cupti_profiler/__init__.py.
//
// Every .def() call here passes both py::arg("name") (so generated .pyi
// stubs carry real parameter names instead of arg0/arg1) and a short
// docstring (so the same stubs propagate API-level documentation to
// editors / type checkers). Keep doc strings terse — the canonical
// reference is the C++ header.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cupti_profiler/profiler_suite.h>
#include <cupti_profiler/event_profiler.h>
#include <cupti_profiler/gpu_profiler.h>
#include <cupti_profiler/system_profiler.h>
#include <cupti_profiler/disk_profiler.h>
#include <cupti_profiler/tracked_process.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace cupti_profiler;

PYBIND11_MODULE(_native, m) {
    m.doc() = "pybind11 bindings for the cupti_profiler suite "
              "(GPU + System + Disk + Events).";

    // -------------------------------------------------------------------
    // Shared TrackedProcess (PID + optional display alias) — used by both
    // SystemProfilerConfig and DiskProfilerConfig.
    // -------------------------------------------------------------------
    py::class_<TrackedProcess>(m, "TrackedProcess",
        "One tracked PID with an optional display alias. "
        "alias is empty by default; if set, visualizers render "
        "\"<alias> (PID xxx)\" instead of plain \"PID xxx\". "
        "pid=0 is resolved to the current process at config-load time.")
        .def(py::init<>())
        .def(py::init([](uint32_t pid, std::string alias) {
                 TrackedProcess p; p.pid = pid; p.alias = std::move(alias); return p;
             }), py::arg("pid"), py::arg("alias") = std::string{},
             "Construct from a PID and optional alias.")
        .def_readwrite("pid",   &TrackedProcess::pid,
            "PID to track. 0 = resolve to the current process at runtime.")
        .def_readwrite("alias", &TrackedProcess::alias,
            "Optional display name. Empty = no alias (visualizer falls back to \"PID xxx\").")
        .def("__repr__", [](const TrackedProcess& p) {
            if (p.alias.empty()) return "TrackedProcess(pid=" + std::to_string(p.pid) + ")";
            return "TrackedProcess(pid=" + std::to_string(p.pid) + ", alias='" + p.alias + "')";
        });

    // -------------------------------------------------------------------
    // Config structs (POD-style, public field assignment)
    // -------------------------------------------------------------------
    py::class_<ProfilerConfig>(m, "GpuProfilerConfig",
        "Configuration for GpuProfiler — CUPTI PM Sampling.")
        .def(py::init<>())
        .def_readwrite("device_index",          &ProfilerConfig::deviceIndex,
            "CUDA device ordinal to profile.")
        .def_readwrite("sampling_frequency_hz", &ProfilerConfig::samplingFrequencyHz,
            "PM sampling rate in Hz (default 10000).")
        .def_readwrite("hw_buffer_size",        &ProfilerConfig::hwBufferSize,
            "CUPTI hardware buffer size in bytes (default 512 MiB).")
        .def_readwrite("max_samples",           &ProfilerConfig::maxSamples,
            "Maximum samples retained in memory before flushing.")
        .def_readwrite("metrics",               &ProfilerConfig::metrics,
            "List of CUPTI metric names (e.g. 'sm__cycles_active.avg').")
        .def_readwrite("flush_interval_ms",     &ProfilerConfig::flushIntervalMs,
            "Periodic flush interval in ms; 0 = single write at Stop().")
        .def_readwrite("output_file",           &ProfilerConfig::outputFile,
            "Path to gpu_metrics.pb output file. Empty = no output.");

    py::class_<SystemProfilerConfig>(m, "SystemProfilerConfig",
        "Configuration for SystemProfiler — CPU + memory.")
        .def(py::init<>())
        .def_readwrite("sampling_frequency_hz", &SystemProfilerConfig::samplingFrequencyHz,
            "Tick rate in Hz (default 100).")
        .def_readwrite("processes",             &SystemProfilerConfig::Processes,
            "Processes to track per-process (list of TrackedProcess). "
            "Empty = system-wide only. PID 0 inside any entry is resolved "
            "to the current process at runtime.")
        .def_readwrite("flush_interval_ms",     &SystemProfilerConfig::flushIntervalMs,
            "Periodic flush interval in ms.")
        .def_readwrite("output_file",           &SystemProfilerConfig::outputFile,
            "Path to system_metrics.pb output file.");

    py::class_<DiskProfilerConfig>(m, "DiskProfilerConfig",
        "Configuration for DiskProfiler — per-device + per-process I/O.")
        .def(py::init<>())
        .def_readwrite("sampling_frequency_hz", &DiskProfilerConfig::samplingFrequencyHz,
            "Tick rate in Hz (default 10).")
        .def_readwrite("devices",               &DiskProfilerConfig::devices,
            "Block device names to track (e.g. ['nvme0n1', 'md0']).")
        .def_readwrite("processes",             &DiskProfilerConfig::Processes,
            "Processes to track per-process I/O (list of TrackedProcess). "
            "PID 0 inside any entry is resolved to the current process at runtime.")
        .def_readwrite("flush_interval_ms",     &DiskProfilerConfig::flushIntervalMs,
            "Periodic flush interval in ms.")
        .def_readwrite("output_file",           &DiskProfilerConfig::outputFile,
            "Path to disk_metrics.pb output file.");

    py::class_<EventProfilerConfig>(m, "EventProfilerConfig",
        "Configuration for EventProfiler — region + event annotations.")
        .def(py::init<>())
        .def_readwrite("flush_interval_ms", &EventProfilerConfig::flushIntervalMs,
            "Periodic flush interval in ms.")
        .def_readwrite("output_file",       &EventProfilerConfig::outputFile,
            "Path to events.pb output file.");

    // -------------------------------------------------------------------
    // EventTracker (with Domain enum)
    // -------------------------------------------------------------------
    py::class_<EventTracker> eventTracker(m, "EventTracker",
        "Records named time intervals (regions) and instantaneous events "
        "in a single time domain (Generic = host steady_clock, "
        "GPU = CUPTI clock). Constructed by EventProfiler — get an "
        "instance via EventProfiler.get_generic_tracker() / "
        "get_gpu_tracker(). Thread-safe across BeginRegion / EndRegion / "
        "MarkEvent.");
    py::enum_<EventTracker::Domain>(eventTracker, "Domain",
        "Time domain for an EventTracker.")
        .value("GENERIC", EventTracker::Domain::GENERIC,
               "Host steady_clock domain.")
        .value("GPU",     EventTracker::Domain::GPU,
               "CUPTI clock domain (cuptiGetTimestamp).")
        .export_values();
    eventTracker
        .def_property_readonly("domain", &EventTracker::GetDomain,
            "Time domain this tracker records in.")
        .def("begin_region", &EventTracker::BeginRegion, py::arg("name"),
            "Mark the start of a named region. Returns an opaque id to "
            "pass to end_region(). Safe to call from any thread.")
        .def("end_region",   &EventTracker::EndRegion, py::arg("region_id"),
            "Mark the end of a region previously started with "
            "begin_region(). Safe to call from a thread other than the "
            "one that called begin_region().")
        .def("mark_event",   &EventTracker::MarkEvent, py::arg("name"),
            "Record an instantaneous event with the given name.")
        .def("set_stream",
             [](EventTracker& self, std::uintptr_t handle) {
                 self.SetStream(reinterpret_cast<void*>(handle));
             },
             py::arg("stream_handle"),
            "GPU domain only — register the cudaStream_t (passed as an "
            "integer pointer) that subsequent begin_region/end_region/"
            "mark_event calls will record cudaEvents on. Must be called "
            "before any region/event call on a GPU-domain tracker.");

    // Re-export the enum at module scope so users can write
    // cupti_profiler.Domain.GPU.
    m.attr("Domain") = eventTracker.attr("Domain");

    // -------------------------------------------------------------------
    // Individual profilers
    // -------------------------------------------------------------------
    py::class_<GpuProfiler>(m, "GpuProfiler",
        "Records GPU PM samples via CUPTI. Constructed by ProfilerSuite.")
        .def("configure",  &GpuProfiler::Configure, py::arg("config"),
            "Initialize the profiler with the given GpuProfilerConfig. "
            "Caller must have an active CUDA context on config.device_index.",
            py::call_guard<py::gil_scoped_release>())
        .def("start",      &GpuProfiler::Start,
            "Start PM sampling, decode thread, and optional flush thread.",
            py::call_guard<py::gil_scoped_release>())
        .def("stop",       &GpuProfiler::Stop,
            "Stop PM sampling, join threads, write remaining data.",
            py::call_guard<py::gil_scoped_release>())
        .def("device_name",       &GpuProfiler::GetDeviceName,
            "Human-readable device name (available after configure()).")
        .def("chip_name",         &GpuProfiler::GetChipName,
            "GPU chip name (e.g. 'GH100').")
        .def("peak_dram_bw_gbps", &GpuProfiler::GetPeakDramBwGbps,
            "Theoretical peak DRAM bandwidth in GiB/s for this device.");

    py::class_<SystemProfiler>(m, "SystemProfiler",
        "Records CPU + memory ticks system-wide and per-PID.")
        .def("configure",   &SystemProfiler::Configure, py::arg("config"),
            "Initialize with the given SystemProfilerConfig.",
            py::call_guard<py::gil_scoped_release>())
        .def("start",       &SystemProfiler::Start,
            "Start the sampling and flush threads.",
            py::call_guard<py::gil_scoped_release>())
        .def("signal_stop", &SystemProfiler::SignalStop,
            "Non-blocking signal to stop sampling. Call stop() to join.")
        .def("stop",        &SystemProfiler::Stop,
            "Join threads, flush remaining data, close output file.",
            py::call_guard<py::gil_scoped_release>());

    py::class_<DiskProfiler>(m, "DiskProfiler",
        "Records per-device disk throughput / queue and per-PID I/O.")
        .def("configure",   &DiskProfiler::Configure, py::arg("config"),
            "Initialize with the given DiskProfilerConfig.",
            py::call_guard<py::gil_scoped_release>())
        .def("start",       &DiskProfiler::Start,
            "Start the sampling and flush threads.",
            py::call_guard<py::gil_scoped_release>())
        .def("signal_stop", &DiskProfiler::SignalStop,
            "Non-blocking signal to stop sampling.")
        .def("stop",        &DiskProfiler::Stop,
            "Join threads, flush remaining data, close output file.",
            py::call_guard<py::gil_scoped_release>());

    py::class_<EventProfiler>(m, "EventProfiler",
        "Owns one Generic-domain and one GPU-domain EventTracker, plus the "
        "background flush thread that drains them into events.pb.")
        .def("configure",   &EventProfiler::Configure, py::arg("config"),
            "Initialize with the given EventProfilerConfig.",
            py::call_guard<py::gil_scoped_release>())
        .def("start",       &EventProfiler::Start,
            "Start the periodic flush thread.",
            py::call_guard<py::gil_scoped_release>())
        .def("signal_stop", &EventProfiler::SignalStop,
            "Non-blocking signal to stop the flush thread.")
        .def("stop",        &EventProfiler::Stop,
            "Drain trackers (with forceResolve=true), close events.pb.",
            py::call_guard<py::gil_scoped_release>())
        .def("get_generic_tracker", &EventProfiler::GetGenericTracker,
             py::return_value_policy::reference_internal,
            "Returns the Generic-domain EventTracker (host steady_clock).")
        .def("get_gpu_tracker",     &EventProfiler::GetGpuTracker,
             py::return_value_policy::reference_internal,
            "Returns the GPU-domain EventTracker (CUPTI clock). Call "
            "set_stream() on it before recording any region/event.");

    // -------------------------------------------------------------------
    // ProfilerSuite (high-level orchestrator)
    // -------------------------------------------------------------------
    py::class_<ProfilerSuite>(m, "ProfilerSuite",
        "Orchestrates the GPU, System, Disk, and Events profilers from a "
        "single ProfilerSuiteConfig.")
        .def(py::init<>())
        .def("load_config",            &ProfilerSuite::LoadConfig,
             py::arg("pbtxt_path"),
            "Load configuration from a protobuf text-format (.pbtxt) file.")
        .def("load_config_from_bytes",
             [](ProfilerSuite& self, py::bytes serialized) {
                 self.LoadConfigFromBytes(std::string(serialized));
             },
             py::arg("serialized_proto"),
            "Load configuration from a serialized ProfilerSuiteConfig "
            "(binary wire format). Used by language bindings that build "
            "the config in-process; cupti_profiler.configure_suite() is "
            "the friendly wrapper around this.")
        .def("configure", &ProfilerSuite::Configure,
            "Configure all enabled sub-profilers. Must be called after "
            "load_config*.",
            py::call_guard<py::gil_scoped_release>())
        .def("start",     &ProfilerSuite::Start,
            "Start all enabled sub-profilers.",
            py::call_guard<py::gil_scoped_release>())
        .def("stop",      &ProfilerSuite::Stop,
            "Stop all sub-profilers and write session_metadata.pb.",
            py::call_guard<py::gil_scoped_release>())
        .def("get_gpu_profiler",    &ProfilerSuite::GetGPUProfiler,
             py::return_value_policy::reference_internal,
            "Returns the suite-owned GpuProfiler.")
        .def("get_system_profiler", &ProfilerSuite::GetSystemProfiler,
             py::return_value_policy::reference_internal,
            "Returns the suite-owned SystemProfiler.")
        .def("get_disk_profiler",   &ProfilerSuite::GetDiskProfiler,
             py::return_value_policy::reference_internal,
            "Returns the suite-owned DiskProfiler.")
        .def("get_event_profiler",  &ProfilerSuite::GetEventProfiler,
             py::return_value_policy::reference_internal,
            "Returns the suite-owned EventProfiler.");

    // -------------------------------------------------------------------
    // CUDA stream helpers — let Python tests get a real cudaStream_t
    // without depending on cupy / torch. Returns the pointer as an int.
    // -------------------------------------------------------------------
    m.def("create_cuda_stream", []() -> std::uintptr_t {
        cudaStream_t s = nullptr;
        cudaError_t err = cudaStreamCreate(&s);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamCreate failed: ")
                                     + cudaGetErrorString(err));
        }
        return reinterpret_cast<std::uintptr_t>(s);
    },
    "Allocate a fresh CUDA stream and return its pointer as an integer. "
    "Use destroy_cuda_stream() to release it.");

    m.def("destroy_cuda_stream",
        [](std::uintptr_t handle) {
            if (!handle) return;
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(handle));
        },
        py::arg("stream_handle"),
        "Destroy a CUDA stream previously returned by create_cuda_stream(). "
        "Safe to call with 0 (no-op).");
}
