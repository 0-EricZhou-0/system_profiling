---
title: "Profiler Suite — System Guide"
tags:
  - cupti
  - profiling
  - gpu
  - cpu
  - memory
  - disk
  - nvidia
  - documentation
---

# Profiler Suite

A shared library (`libcupti_profiler.so`) for profiling CUDA workloads end-to-end. It bundles four independent profilers under one `ProfilerSuite` driver:

| Profiler | What it samples | How |
| -------- | --------------- | --- |
| `GpuProfiler` | GPU hardware counters (SM, DRAM, PCIe, NVLink, …) | NVIDIA CUPTI PM Sampling — no kernel replay |
| `SystemProfiler` | CPU utilization + memory usage (system-wide and per-PID) | `/proc/stat`, `/proc/meminfo`, `/proc/<pid>/{stat,status}` |
| `DiskProfiler` | Per-device throughput + queue depth, per-PID I/O | `/proc/diskstats`, `/sys/block/<dev>/inflight`, `/proc/<pid>/io` |
| `EventProfiler` | Named regions and instantaneous markers across host + GPU clock domains | `cudaEvent` (GPU domain) + `steady_clock` (Generic domain) |

Each profiler can be used standalone or composed via `ProfilerSuite::LoadConfig(.pbtxt)`. Outputs are length-delimited protobuf streams plus a single `session_metadata.pb` manifest, all visualized as a unified plot by `tools/visualize_all.py`.

## Architecture

### Suite-level

```text ln:false
┌───────────────────────────────────────────────────────────────────────────┐
│  Your Application                                                         │
│   ProfilerSuite suite;                                                    │
│   suite.LoadConfig("config.pbtxt");                                       │
│   suite.Configure();                                                      │
│   suite.Start();                                                          │
│   // ... your CUDA + host workload ...                                    │
│   suite.Stop();                                                           │
└┬──────┬────────────────┬────────────────┬───────────────┬─────────────┬───┘
 │      ▼                ▼                ▼               ▼             ▼
 │ ┌──────────┐  ┌───────────────┐  ┌───────────┐  ┌──────────────┐  ┌─────┐
 │ │ GpuProf. │  │ SystemProf.   │  │ DiskProf. │  │ EventProf.   │  │     │
 │ │ CUPTI PM │  │ /proc/stat    │  │ diskstats │  │ regions +    │  │     │
 │ │ sampling │  │ /proc/meminfo │  │ /sys/blk/ │  │ events       │  │ ... │
 │ │          │  │ /proc/[pid]/* │  │ /proc/pid │  │ (GPU+host)   │  │     │
 │ └────┬─────┘  └───────┬───────┘  └─────┬─────┘  └──────┬───────┘  └──┬──┘
 │      ▼                ▼                ▼               ▼             ▼
 │gpu_metrics.pb system_metrics.pb disk_metrics.pb    events.pb      xxx.pb
 │      │                │                │               │             │
 │      ├────────────────┴────────────────┴───────────────┴─────────────┘
 │      │
 └▶ session_metadata.pb
        │
        └▶ tools/visualize_*.py ─▶ Human readable graph
```

Each profiler runs independently with its own sampling frequency, flush interval, and output file. They share only the wall-clock anchor written by `ProfilerSuite::Start()` so per-stream timestamps can be co-plotted.

### GPU profiler internals

```text ln:false
┌───────────────────────────────────────────────────┐
│  libcupti_profiler.so — GPU subsystem             │
│                                                   │
│ ┌────────────────┐  ┌───────────────────────────┐ │
│ │ GpuProfiler    │──│ CuptiPmSampling           │ │
│ │ (public API)   │  │ (target-side HW counters) │ │
│ └───┬────────────┘  └───────────────────────────┘ │
│     │                                             │
│ ┌───┴────────────┐  ┌───────────────────────────┐ │
│ │ RegionTracker  │  │ CuptiProfilerHost         │ │
│ │ (CUDA events)  │  │ (metric evaluation)       │ │
│ └────────────────┘  └───────────────────────────┘ │
│                                                   │
│ ┌────────────────┐  ┌───────────────────────────┐ │
│ │ Decode Thread  │  │ Flush Thread              │ │
│ │ (HW buf drain) │  │ (periodic .pb write)      │ │
│ └────────────────┘  └───────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

### System & disk profiler internals

```text ln:false
┌───────────────────────────────────────────────────┐
│  libcupti_profiler.so — System / Disk subsystems  │
│                                                   │
│ ┌────────────────┐    ┌───────────────────────┐   │
│ │ SystemProfiler │    │ DiskProfiler          │   │
│ │ (public API)   │    │ (public API)          │   │
│ └───┬────────────┘    └───────┬───────────────┘   │
│     │                         │                   │
│ ┌───┴────────────┐    ┌───────┴───────────────┐   │
│ │ proc_readers   │    │ disk_readers          │   │
│ │ (CPU + memory  │    │ (diskstats, inflight, │   │
│ │  delta calc)   │    │  per-PID io counters) │   │
│ └────────────────┘    └───────────────────────┘   │
│                                                   │
│ ┌──────────────────────┐  ┌──────────────────┐    │
│ │ system_flush_thread  │  │ disk_flush_thread│    │
│ │ (sample + serialize) │  │ (sample + write) │    │
│ └──────────────────────┘  └──────────────────┘    │
└───────────────────────────────────────────────────┘
```

Both subsystems run a single thread that samples and serializes inline — there's no separate decode stage because the source data (`/proc` text files) is already in host memory. CPU samples differentiate by computing per-tick deltas across consecutive reads of `/proc/stat` and `/proc/<pid>/stat`. Disk samples convert per-tick byte counters from `/proc/diskstats` into bytes-per-second using the wall time between reads.

### Event profiler internals

```text ln:false
┌────────────────────────────────────────────────────┐
│  libcupti_profiler.so — Events subsystem           │
│                                                    │
│ ┌────────────────┐                                 │
│ │ EventProfiler  │  owns two trackers:             │
│ │ (public API)   │                                 │
│ └───┬────────────┘                                 │
│     │                                              │
│ ┌───┴───────────────────┐  ┌────────────────────┐  │
│ │ GenericTracker        │  │ GpuTracker         │  │
│ │ (steady_clock)        │  │ (cudaEventRecord)  │  │
│ │ thread-safe begin/end │  │ on user stream     │  │
│ └───────────────────────┘  └────────────────────┘  │
│                                                    │
│ ┌────────────────────────────────────────────────┐ │
│ │ event_flush_thread — drains both trackers,     │ │
│ │ resolves cudaEvents into ns, serializes one    │ │
│ │ EventTrace per flush window                    │ │
│ └────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

### Data flow — GPU

```text ln:false
GPU Performance Monitor HW counters
        ↓  (sampled at configurable interval)
512 MB GPU ring buffer
        ↓  (drained every 5 ms by decode thread)
cuptiPmSamplingDecodeData() → raw counter samples
        ↓
cuptiProfilerHostEvaluateToGpuValues() → metric doubles
        ↓
SamplerRange vector (mutex-protected, host memory)
        ↓  (drained every flushIntervalMs by flush thread)
Length-delimited protobuf → gpu_metrics.pb
```

### Data flow — System

```text ln:false
/proc/stat, /proc/meminfo                            (system-wide)
/proc/<pid>/stat, /proc/<pid>/status                 (per-PID)
        ↓  (read at 1 / samplingFrequencyHz cadence)
proc_readers — compute deltas vs. previous sample
        ↓
CPUSystemSample / CPUProcessSample / Memory*Sample
        ↓  (accumulated in flush thread's vectors)
        ↓  (drained every flushIntervalMs)
Length-delimited protobuf → system_metrics.pb
```

### Data flow — Disk

```text ln:false
/proc/diskstats           (per-device byte counters)
/sys/block/<dev>/inflight (read/write queue depth)
/proc/<pid>/io            (per-PID rchar/wchar/read_bytes/write_bytes/cancelled_write_bytes)
        ↓  (read at 1 / samplingFrequencyHz cadence)
disk_readers — compute bytes/sec vs. previous sample
        ↓
DiskDeviceSample / DiskProcessSample
        ↓  (drained every flushIntervalMs)
Length-delimited protobuf → disk_metrics.pb
```

### Data flow — Events

```text ln:false
User code: tracker.BeginRegion("foo") / EndRegion(idx) / MarkEvent("bar")
        ↓
Generic domain: steady_clock::now() captured inline (mutex-protected map)
GPU domain:    cudaEventRecord on registered stream (resolved later)
        ↓  (drained every flushIntervalMs by event_flush_thread)
EventBuffer per active domain → EventTrace
        ↓
Length-delimited protobuf → events.pb
```

The `visualize_all.py` tool merges the four `.pb` files plus `session_metadata.pb` into one matplotlib figure with a shared x-axis.

---

## Project structure

```text ln:false
nvidia-profiling/
├── CMakeLists.txt                       Top-level CMake (project, find_package)
├── pyproject.toml                       scikit-build-core build backend (pip install)
├── configs/
│   └── example.pbtxt                    Reference suite config
├── proto/
│   ├── tracked_process.proto            Shared { pid, alias } message
│   ├── profiler_config.proto            ProfilerSuiteConfig + per-component configs
│   ├── gpu_metrics.proto                GPU counter trace
│   ├── system_metrics.proto             CPU + memory trace
│   ├── disk_metrics.proto               Disk device + per-PID I/O trace
│   ├── events.proto                     Region + event trace (multi-domain)
│   └── session_metadata.proto           Run manifest (probes, hostname, wall-clock)
├── lib/
│   ├── CMakeLists.txt                   Builds libcupti_profiler.so
│   ├── include/
│   │   └── cupti_profiler/
│   │       ├── profiler_suite.h         Suite orchestrator
│   │       ├── gpu_profiler.h           GPU profiler + RegionTracker
│   │       ├── system_profiler.h        CPU + memory profiler
│   │       ├── disk_profiler.h          Disk profiler
│   │       ├── event_profiler.h         Region/event tracker (Generic + GPU domains)
│   │       └── tracked_process.h        Shared TrackedProcess struct
│   └── src/
│       ├── profiler_suite.cpp                .pbtxt parsing + lifecycle fan-out
│       ├── session_metadata_writer.h/cpp     Writes session_metadata.pb
│       │
│       ├── gpu_profiler.cpp                  GpuProfiler + RegionTracker pimpl
│       ├── cupti_pm_sampling.h/cpp           Target-side PM sampling lifecycle
│       ├── profiler_host_internal.h/cpp      Host-side metric evaluation
│       ├── decode_thread.h/cpp               Background HW buffer drain
│       ├── flush_thread.h/cpp                GPU periodic protobuf serialization
│       │
│       ├── system_profiler.cpp               SystemProfiler pimpl
│       ├── proc_readers.h/cpp                /proc/stat, /proc/meminfo, /proc/<pid>/*
│       ├── system_flush_thread.h/cpp         System sample + serialize loop
│       │
│       ├── disk_profiler.cpp                 DiskProfiler pimpl
│       ├── disk_readers.h/cpp                /proc/diskstats, /sys/block, /proc/<pid>/io
│       ├── disk_flush_thread.h/cpp           Disk sample + serialize loop
│       │
│       ├── event_profiler.cpp                EventProfiler pimpl
│       ├── event_tracker.cpp                 EventTracker (Generic + GPU domains)
│       ├── event_tracker_internal.h          Internal ResolvedRegion / ResolvedEvent
│       ├── event_flush_thread.h/cpp          Resolve + serialize regions/events
│       │
│       └── helper_cupti.h                    Vendored NVIDIA error-check macros
├── examples/
│   ├── CMakeLists.txt
│   ├── gemm_profiling.cu                cuBLAS GEMM + vectorAdd, GPU-only profiling
│   ├── full_system_profiling.cu         ProfilerSuite + .pbtxt (C++)
│   └── full_system_profiling.py         ProfilerSuite + .pbtxt (Python via pybind11)
├── python/
│   ├── binding.cpp                      pybind11 bindings (full API surface)
│   └── cupti_profiler/                  Python package (proto pb2 + helpers)
├── tools/
│   ├── visualize_all.py                 Matplotlib full-suite visualizer (.png)
│   ├── visualize_interactive.py         Bokeh interactive visualizer (HTML)
│   ├── visualize_single.py              GPU-only matplotlib visualizer
│   └── environment.yml                  Conda environment
└── docs/
    ├── system-guide.md                  This file
    ├── full-system-overview.md          High-level overview + visuals
    ├── full-system-internals.md         Deep dive into per-component internals
    ├── cupti-overhead-analysis.md       PM Sampling overhead measurements
    ├── integration.md                   Downstream integration recipes
    └── examples/                        Per-example walkthroughs
```

---

## Building

### Prerequisites

- CUDA Toolkit 12.x with CUPTI
- GPU with compute capability >= 7.5 (Turing+)
- `GPU_TIME_INTERVAL` trigger mode requires Ampere GA10x+ / Hopper / Ada
- `libprotobuf-dev` (system package, must match `protoc` version)
- CMake >= 3.18

### Build commands

```bash title:"Build from source"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The build produces:
- `build/lib/libcupti_profiler.so` — shared library (all four profilers)
- `build/examples/gemm_profiling` — GPU-only example
- `build/examples/full_system_profiling` — full-suite example (GPU + system + disk + events)
- `build/python/cupti_profiler/_native*.so` — pybind11 bindings (when Python is available)

### CMake options

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `Protobuf_PROTOC_EXECUTABLE` | `/usr/bin/protoc` | Path to protoc (must match libprotobuf version) |

> [!IMPORTANT]
> Generated protobuf sources (`gpu_metrics.pb.h`, `gpu_metrics.pb.cc`) are placed in `build/lib/`, not in the source tree. They are never committed.

### Running the examples

```bash title:"Run with library in LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH

# GPU-only — writes gpu_metrics.pb in the cwd
./build/examples/gemm_profiling -d 0 -i 100000 -o gpu_metrics.pb

# Full suite — uses configs/example.pbtxt by default
./build/examples/full_system_profiling [-c your_config.pbtxt]

# Visualize all four streams together
python tools/visualize_all.py profiling_output/session_metadata.pb
```

---

## Public API reference

All public types live under `<cupti_profiler/...>` and the `cupti_profiler::` namespace. The most common entry point is `ProfilerSuite`, which loads a `.pbtxt` and orchestrates the four underlying profilers. Individual profilers are also usable on their own.

| Header | Types it exposes |
| ------ | ---------------- |
| `<cupti_profiler/profiler_suite.h>` | `ProfilerSuite` |
| `<cupti_profiler/gpu_profiler.h>` | `GpuProfiler`, `ProfilerConfig`, `RegionTracker`, `SamplerRange`, `Region` |
| `<cupti_profiler/system_profiler.h>` | `SystemProfiler`, `SystemProfilerConfig` |
| `<cupti_profiler/disk_profiler.h>` | `DiskProfiler`, `DiskProfilerConfig` |
| `<cupti_profiler/event_profiler.h>` | `EventProfiler`, `EventProfilerConfig`, `EventTracker` (with `Domain::{GENERIC,GPU}`) |
| `<cupti_profiler/tracked_process.h>` | `TrackedProcess` (shared by system + disk configs) |
| `<cupti_profiler/child_subreaper.h>` | `EnableChildSubreaper()`, `ChildSubreaperEnabled()` — see [Subreaper helper](#subreaper-helper-adopt_orphans) |

### `ProfilerSuite`

```cpp title:"<cupti_profiler/profiler_suite.h>"
class ProfilerSuite {
public:
    void LoadConfig(const std::string& pbtxtPath);
    void LoadConfigFromBytes(const std::string& serializedProto);

    GpuProfiler&    GetGPUProfiler();
    SystemProfiler& GetSystemProfiler();
    DiskProfiler&   GetDiskProfiler();
    EventProfiler&  GetEventProfiler();

    ProfilerError Configure();  // load catalog + configure all enabled probes
    ProfilerError Start();      // start all enabled probes + emit session_metadata.pb
    void Stop();                // stop, flush, re-emit session_metadata.pb

    // Mid-run PID tracking. Fans out to every probe that supports
    // per-PID sampling (currently System + Disk).
    void AddTrackedProcess(uint32_t pid, std::string alias = {});
    // ... with a per-root descendant-tracking override.
    void AddTrackedProcess(uint32_t pid, std::string alias, bool trackDescendants);
    void RemoveTrackedProcess(uint32_t pid);
};
```

| Method | Description |
| ------ | ----------- |
| `LoadConfig(path)` | Parse a protobuf text-format `.pbtxt` (`ProfilerSuiteConfig` schema) and apply it to all sub-profilers. Resolves `pid: 0` to the calling process. |
| `LoadConfigFromBytes(buf)` | Same as above but takes a serialized binary `ProfilerSuiteConfig`. Used by language bindings. |
| `Get*Profiler()` | Access individual sub-profilers — needed to grab `EventTracker` references for region annotation. |
| `Configure()` | Loads `MetricCatalog` (from `metric_catalog_path` in the config, or a default path next to the binary) and then calls `Configure()` on every sub-profiler whose `enabled = true`. With System or Disk enabled, also runs the [startup situation report](#startup-situation-report). |
| `Start()` / `Stop()` | Lifecycle fan-out. Both write `session_metadata.pb` (atomically — `.tmp` + `rename(2)`); the manifest carries the inlined `MetricCatalog` so visualizers don't need a separate catalog file. `Start()` returns `ProbeStartFailed` if a System/Disk probe did not start (e.g. its output file cannot be opened; under SIDECAR the sidecar reports it) or `SidecarExited` / `SidecarBadHandshake` if the sidecar did not answer; every other probe is running, so call `Stop()` as usual. Python's `start()` raises `RuntimeError`. |
| `AddTrackedProcess(pid, alias)` | Begin tracking a PID mid-run. First sample for the PID lands one sample-tick after `Add` returns (the first tick seeds the `/proc` baseline so the first delta isn't garbage); its CPU counts from then on (no head, unlike a discovered process). Its descendants are tracked if `process_discovery.enabled`. When it exits it is removed automatically ([exit detection](#process-table-and-exit-detection)). Thread-safe. |
| `AddTrackedProcess(pid, alias, trackDescendants)` | Same, overriding `process_discovery.enabled` for this root — see [Descendant tracking](#descendant-tracking). |
| `RemoveTrackedProcess(pid)` | Stop tracking a PID. The PID appears one more time in the next flush of each affected probe with `TrackedProcessV2.removed=true` (visualizer renders a removal marker), then is dropped. Descendants already discovered under it stay tracked until they exit. Thread-safe. |

### `ProfilerConfig` (GPU)

```cpp title:"lib/include/cupti_profiler/gpu_profiler.h"
struct ProfilerConfig {
    std::vector<int> deviceIndices;             // empty = {0}
    uint64_t samplingFrequencyHz = 10000;       // 10 kHz
    size_t hwBufferSize = 512 * 1024 * 1024;    // 512 MB
    uint64_t maxSamples = 50000;
    std::vector<std::string> metrics;

    uint64_t flushIntervalMs = 10000;           // 0 = no periodic flush
    std::string outputFile;                     // empty = no file output
};
```

| Field | Description |
| ----- | ----------- |
| `deviceIndices` | CUDA device ordinals to profile. One CUPTI PM-Sampling session is opened per index; samples from every device are funneled into a single trace and tagged with `gpu_index`. Empty defaults to `{0}` (device 0). |
| `samplingFrequencyHz` | HW counter sampling rate in Hz |
| `hwBufferSize` | GPU-side ring buffer size. 512 MB prevents overflow at 10 kHz |
| `maxSamples` | Decode buffer capacity (per decode cycle, not total) |
| `metrics` | CUPTI metric names to collect. Must fit in a single pass. The same set is applied to every device in `deviceIndices`. |
| `flushIntervalMs` | How often to write accumulated samples to disk. 0 disables periodic flush |
| `outputFile` | Path to the output `.pb` file. Empty disables file output |

> [!TIP]
> Query available metrics with `ncu --query-metrics --chip <chip_name>` or the CUPTI samples. On H100, there are ~962 base metrics with sub-metric rollup suffixes (`.avg`, `.max`, `.sum`, `.pct_of_peak_sustained_elapsed`, etc.).

### `GpuProfiler`

```cpp title:"Core lifecycle"
GpuProfiler profiler;
profiler.Configure(config);    // init CUPTI, validate device, build config image
profiler.Start();              // start HW sampling + decode thread + flush thread
// ... run your CUDA workload ...
profiler.Stop();               // stop sampling, join threads, write remaining data
```

| Method | Description |
| ------ | ----------- |
| `Configure(config)` | Initialize one CUPTI PM-Sampling session per device in `config.deviceIndices` |
| `Start()` | Begin PM sampling on every device + spawn one decode thread per device + one flush thread that merges across devices |
| `Stop()` | Stop sampling, join all decode threads + the flush thread, write final merged data |
| `DrainSamples()` | Atomically drain accumulated samples for the FIRST configured device. Multi-device readers should consume the on-disk trace. |
| `GetDeviceName()` | First configured device's name, e.g. `"NVIDIA H100 NVL"` |
| `GetChipName()` | First configured device's chip, e.g. `"GH100"` |
| `GetPeakDramBwGbps()` | First configured device's theoretical peak DRAM bandwidth in GB/s |

> [!WARNING]
> `Configure()` calls `cuInit(0)` internally. The caller must have already set the CUDA device (e.g., `cudaSetDevice()`). If you are using CUDA before calling `Configure()`, this is already satisfied.

### `RegionTracker`

Annotates named time regions within your workload. Backed by CUDA events for accurate GPU-side timestamps.

```cpp title:"Region annotation"
auto& regions = profiler.GetRegionTracker();
regions.SetStream(static_cast<void*>(myStream));  // call once

size_t idx = regions.Begin("forward pass");
// ... launch kernels on myStream ...
regions.End(idx);
```

| Method | Description |
| ------ | ----------- |
| `SetStream(void*)` | Attach to a CUDA stream. Pass your `cudaStream_t` cast to `void*`. Call once before `Begin`/`End`. |
| `Begin(name)` | Record a start event. Returns an index for `End()`. |
| `End(idx)` | Record an end event for a previously started region. |
| `Resolve()` | Convert CUDA events to absolute timestamps. Called automatically by `GpuProfiler::Stop()`. |
| `GetRegions()` | Access resolved regions. Available after `Stop()`. |

> [!NOTE]
> Regions are resolved using `cudaEventElapsedTime` which conflicts with active PM sampling. That's why `Resolve()` runs after `Stop()`, and regions only appear in the final protobuf message.

### `SamplerRange`

```cpp title:"Single PM sample"
struct SamplerRange {
    size_t rangeIndex;
    uint64_t startTimestamp;  // nanoseconds, CUPTI clock
    uint64_t endTimestamp;
    std::vector<double> metricValues;  // same order as config.metrics
};
```

### `Region`

```cpp title:"Resolved time region"
struct Region {
    std::string name;
    uint64_t startNs;
    uint64_t endNs;
};
```

### `TrackedProcess`

Used by both `SystemProfilerConfig` and `DiskProfilerConfig` to specify which PIDs to follow per-process and how to label them.

```cpp title:"<cupti_profiler/tracked_process.h>"
struct TrackedProcess {
    uint32_t pid = 0;       // 0 = resolved to the calling PID at config-load time
    std::string alias;      // optional display name; empty = no alias
};
```

| Field | Description |
| ----- | ----------- |
| `pid` | PID to track. `0` is a sentinel: `ProfilerSuite::LoadConfig*` rewrites it to `getpid()` so a config can self-attach without knowing its own PID. |
| `alias` | Display name for visualizers. Empty → label is `"PID 12345"`; non-empty → label is `"<alias> (PID 12345)"`. |

A profiler config with an empty `Processes` vector falls back to **system-wide samples only** — no per-PID rows.

### `SystemProfilerConfig` & `SystemProfiler`

```cpp title:"<cupti_profiler/system_profiler.h>"
struct SystemProfilerConfig {
    uint64_t samplingFrequencyHz = 100;          // 100 Hz default
    std::vector<TrackedProcess> Processes;       // empty = system-wide only
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;                      // e.g. "system_metrics.pb"
};

class SystemProfiler : public ProcessTrackingProbe {
public:
    void Configure(const SystemProfilerConfig& config);
    void Start();
    void SignalStop();   // non-blocking
    void Stop();         // join + flush + close

    // Inherited from ProcessTrackingProbe — call between Start() and
    // Stop() to adjust the tracked PID set mid-run. Thread-safe.
    void AddTrackedProcess(uint32_t pid, std::string alias);
    void RemoveTrackedProcess(uint32_t pid);
};
```

| Field | Description |
| ----- | ----------- |
| `samplingFrequencyHz` | How often `/proc/stat` and friends are polled. 100 Hz is a good default for second-scale workloads; 1000 Hz captures sub-second spikes. |
| `Processes` | Initial PIDs (with optional aliases) to sample per-process. `Add/RemoveTrackedProcess` may grow or shrink this set mid-run. See `TrackedProcess`. |
| `flushIntervalMs` | How often the in-memory sample buffer is serialized to `outputFile`. |
| `outputFile` | Path to the system trace `.pb`. Resolved against `output_dir` when driven by `ProfilerSuite`. |

`SystemProfiler` writes one `SystemMetricsTrace` per flush. CPU and memory readings at one tick are combined into a single `Sample` (system-wide) or `ProcessSample` (per-PID); `values[]` is ordered to match the per-scope FQN registry in `scope_metric_names[]` of the same trace. See the [Output format](#output-format) section for the schema.

### `DiskProfilerConfig` & `DiskProfiler`

```cpp title:"<cupti_profiler/disk_profiler.h>"
struct DiskProfilerConfig {
    uint64_t samplingFrequencyHz = 10;           // 10 Hz default
    std::vector<std::string> devices;            // e.g. {"nvme0n1", "md0"}
    std::vector<TrackedProcess> Processes;       // empty = device-only
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;                      // e.g. "disk_metrics.pb"
};

class DiskProfiler : public ProcessTrackingProbe {
public:
    void Configure(const DiskProfilerConfig& config);
    void Start();
    void SignalStop();
    void Stop();

    // Inherited from ProcessTrackingProbe.
    void AddTrackedProcess(uint32_t pid, std::string alias);
    void RemoveTrackedProcess(uint32_t pid);
};
```

| Field | Description |
| ----- | ----------- |
| `samplingFrequencyHz` | Polling rate for `/proc/diskstats` and `/sys/block/<dev>/inflight`. Disk counters update relatively slowly — 10 Hz is usually sufficient. |
| `devices` | Block devices to sample. Names match `/sys/block/<name>/`. Use `lsblk` or `cat /proc/diskstats` to enumerate. |
| `Processes` | PIDs to sample for `/proc/<pid>/io` (all five byte counters). Empty = device-only sampling. |
| `flushIntervalMs` / `outputFile` | Same semantics as `SystemProfilerConfig`. |

> [!NOTE]
> `/proc/<pid>/io` reports five cumulative byte counters: `rchar`/`wchar` (syscall layer, any fd, page-cache hits included, never `mmap`) and `read_bytes`/`write_bytes`/`cancelled_write_bytes` (storage layer: fetches from storage including `mmap` misses, pages dirtied, dirty pages discarded). The profiler emits each as its own bytes-per-second rate over the actual inter-sample time. What each one sees, with measured examples, is in [metric-model.md, "Per-PID I/O counters"](metric-model.md#per-pid-io-counters-who-records-what).

### `EventProfilerConfig`, `EventProfiler` & `EventTracker`

```cpp title:"<cupti_profiler/event_profiler.h>"
struct EventProfilerConfig {
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;             // e.g. "events.pb"
};

class EventProfiler {
public:
    void Configure(const EventProfilerConfig&);
    void Start();
    void SignalStop();
    void Stop();

    EventTracker& GetGenericTracker();  // host steady_clock domain
    EventTracker& GetGpuTracker();      // CUPTI clock (cudaEventRecord)
};

class EventTracker {
public:
    enum class Domain { GENERIC, GPU };
    Domain GetDomain() const;

    size_t BeginRegion(const std::string& name);
    void   EndRegion(size_t idx);
    void   MarkEvent(const std::string& name);

    // GPU-domain only — register the CUDA stream once before any Begin/End/Mark.
    void   SetStream(void* stream);
};
```

`EventProfiler` owns two `EventTracker`s, one per `Domain`. They share an output file (`events.pb`) but record into separate `EventBuffer`s within each `EventTrace` so the visualizer can colour-code them.

| Concept | Generic domain | GPU domain |
| ------- | -------------- | ---------- |
| Clock source | `std::chrono::steady_clock` | `cudaEvent` resolved via `cudaEventElapsedTime` |
| When timestamp is captured | At call-site, inline | At kernel launch (or wherever the event is recorded on the stream) |
| Setup | None | `SetStream(stream)` exactly once before any region/event call |
| Use it for | Host-side phases (data loading, allocation, validation) | Per-kernel or per-stream stages on the GPU |

> [!NOTE]
> `BeginRegion`/`EndRegion`/`MarkEvent` are thread-safe within a tracker. The opaque id returned by `BeginRegion` may be passed across threads — Begin on thread A and End on thread B is fully supported. `SetStream` is *not* thread-safe vs. begin/end; call it from setup code only.


---

## Integration guide

### Step 1 — Add to your CMake project

```cmake title:"Your CMakeLists.txt"
# Option A: subdirectory (if you vendor the repo)
add_subdirectory(third_party/nvidia-profiling)

# Option B: find the installed library
find_library(CUPTI_PROFILER cupti_profiler REQUIRED)

# Link your target
target_link_libraries(my_app PRIVATE cupti_profiler)
```

### Step 2 — Profile your workload

There are two ways to drive the library:

#### Option A — `ProfilerSuite` from a `.pbtxt` (recommended)

This is the path used by `examples/full_system_profiling.cu`. All four profilers are configured from a single text file; you add region annotations through the `EventTracker`s exposed by the suite.

```cpp title:"Suite-driven integration"
#include <cupti_profiler/profiler_suite.h>
#include <cuda_runtime.h>

int main() {
    cudaSetDevice(0);

    cupti_profiler::ProfilerSuite suite;
    suite.LoadConfig("config.pbtxt");   // see "Suite config (.pbtxt)" below
    suite.Configure();
    suite.Start();

    // Generic-domain regions for host work
    auto& host = suite.GetEventProfiler().GetGenericTracker();
    auto setup = host.BeginRegion("workload setup");

    // GPU-domain regions for device work
    auto& gpu = suite.GetEventProfiler().GetGpuTracker();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gpu.SetStream(static_cast<void*>(stream));

    host.EndRegion(setup);

    auto fwd = gpu.BeginRegion("forward");
    // ... launch kernels on stream ...
    gpu.EndRegion(fwd);

    cudaStreamSynchronize(stream);
    suite.Stop();   // writes gpu_metrics.pb, system_metrics.pb,
                    // disk_metrics.pb, events.pb, session_metadata.pb
}
```

#### Suite config (`.pbtxt`)

A reference config lives in `configs/example.pbtxt`. The minimal shape:

```protobuf title:"config.pbtxt"
output_dir: "profiling_output"
# Optional: path to a MetricCatalog pbtxt to OVERLAY onto the built-in
# registry (see lib/data/metric_catalog.pbtxt for a regenerated dump
# of what the runtime ships). Descriptors with FQNs already present are
# replaced; new FQNs are appended. Leave empty to use the built-ins
# only — no file is needed at runtime.
metric_catalog_path: ""

gpu {
    enabled: true
    # One CUPTI session is opened per index. Empty = [0].
    device_indices: 0
    sampling_frequency_hz: 10000
    metrics: "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
    metrics: "dram__read_throughput.avg.pct_of_peak_sustained_elapsed"
    output_file: "gpu_metrics.pb"
}

system {
    enabled: true
    sampling_frequency_hz: 100
    processes { pid: 0 alias: "self" }       # 0 → resolved at LoadConfig time
    output_file: "system_metrics.pb"
}

disk {
    enabled: true
    sampling_frequency_hz: 100
    devices: "nvme0n1"
    processes { pid: 0 alias: "self" }
    output_file: "disk_metrics.pb"
    # mode: SYSTEM_PROBE_MODE_SIDECAR  # see "Sidecar mode"
}

events {
    enabled: true
    output_file: "events.pb"
}

# Descendant tracking for system + disk (default: off).
process_discovery {
    enabled: false                 # true = also trace descendants of listed PIDs
    direct_children_only: false    # false = recursive
    scan_interval_ms: 100          # 0 = 100
}

# SIDECAR mode only: CPUs to pin the sidecar to. Empty = no pinning.
# sidecar_cpus: 7
```

Each block can be omitted or set `enabled: false` to skip that profiler. Each `processes { ... }` entry is independent — `pid` only, or `pid` + `alias`. Empty `processes` = system-wide samples only. The PID set may grow or shrink mid-run via `ProfilerSuite::AddTrackedProcess()` / `RemoveTrackedProcess()`, and a tracked process that exits is removed automatically ([exit detection](#process-table-and-exit-detection)). Every knob of sidecar mode, descendant tracking and the process table, with its default, is in the [configuration reference](#configuration-reference-sidecar-mode-descendant-tracking-process-table).

#### Option B — GPU-only with `GpuProfiler`

If you only need GPU counters and don't want a `.pbtxt`, the GPU profiler can be driven directly. This is the path used by `examples/gemm_profiling.cu`.

```cpp title:"GPU-only integration"
#include <cupti_profiler/gpu_profiler.h>
#include <cuda_runtime.h>

int main() {
    cudaSetDevice(0);

    cupti_profiler::ProfilerConfig config;
    config.deviceIndices = {0};            // multi-device: {0, 1, ...}
    config.samplingFrequencyHz = 10000;    // 10 kHz
    config.outputFile = "my_trace.pb";
    config.flushIntervalMs = 5000;
    config.metrics = {
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.per_cycle_active",
        "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
    };

    cupti_profiler::GpuProfiler profiler;
    profiler.Configure(config);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto& regions = profiler.GetRegionTracker();
    regions.SetStream(static_cast<void*>(stream));

    profiler.Start();

    size_t r = regions.Begin("inference");
    // ... launch kernels on stream ...
    regions.End(r);

    cudaStreamSynchronize(stream);
    profiler.Stop();
}
```

### Step 3 — Visualize

```bash title:"Generate plot"
conda activate gpu-profiling

# Suite-driven run → unified plot from session_metadata.pb
python tools/visualize_all.py profiling_output/session_metadata.pb -o run.png

# Or interactive Bokeh in the browser
python tools/visualize_interactive.py profiling_output/session_metadata.pb

# GPU-only trace → single-panel plot
python tools/visualize_single.py -i my_trace.pb -o my_trace.png
```

---

## Configuration reference: sidecar mode, descendant tracking, process table

Every knob these features added, where it lives, and its default. The
longer-standing fields (sampling rates, flush intervals, output files,
devices, GPU metrics) are described with their config structs under
[Public API reference](#public-api-reference).

| Knob | Where | Default | What it does |
|---|---|---|---|
| `system.mode`, `disk.mode` | `.pbtxt` / proto `SystemProfilerConfig.mode` (6), `DiskProfilerConfig.mode` (7); C++ `SystemProfilerConfig::mode`, `DiskProfilerConfig::mode` | `SYSTEM_PROBE_MODE_LEGACY` (unset = LEGACY) | `SYSTEM_PROBE_MODE_SIDECAR` runs that probe's sampler and flush threads in the `cupti-profiler-sidecar` process instead of this one, so their CPU is not charged to the workload. One sidecar serves both probes. See [Sidecar mode](#sidecar-mode) |
| `sidecar_cpus` | `.pbtxt` / proto `ProfilerSuiteConfig.sidecar_cpus` (9, repeated) | empty = no pinning | CPUs the sidecar and all its threads are pinned to. A CPU outside this process's allowed set fails `Configure()` with `SidecarAffinityFailed`. Ignored, with a note, when no probe is in SIDECAR mode |
| `CUPTI_PROFILER_SIDECAR` | environment variable, read at `Configure()` | unset | Path of the sidecar binary to run instead of the built-in one. Used only if it is an executable regular file; otherwise the built-in path is used silently |
| `CUPTI_PROFILER_SIDECAR_PATH` | CMake cache variable, build time | `<build dir>/tools/cupti-profiler-sidecar` | The built-in sidecar path. Without it and without `CUPTI_PROFILER_SIDECAR`, SIDECAR mode fails `Configure()` with `SidecarNotFound` |
| `process_discovery.enabled` | `.pbtxt` / proto `ProfilerSuiteConfig.process_discovery` (8) → `ProcessDiscoveryConfig.enabled` (1) | `false` | Also trace the descendants of listed PIDs, on both System and Disk. See [Descendant tracking](#descendant-tracking) |
| `process_discovery.direct_children_only` | `ProcessDiscoveryConfig.direct_children_only` (2) | `false` = recursive | `true`: only the listed roots' direct children |
| `process_discovery.scan_interval_ms` | `ProcessDiscoveryConfig.scan_interval_ms` (3) | `0` = 100 ms | How often every followed process's children are scanned; a child that lives less than one interval may be missed |
| `track_descendants` (per root) | Python `suite.add_tracked_process(pid, alias, track_descendants=None)`; C++ `ProfilerSuite::AddTrackedProcess(pid, alias, bool trackDescendants)` | `None` / the 2-argument overload = inherit `process_discovery.enabled` | Overrides `enabled` for that root only |
| `adopt_orphans()` | Python `cupti_profiler.adopt_orphans()`; C++ `EnableChildSubreaper()` (`<cupti_profiler/child_subreaper.h>`) | off; the library never sets it | Makes the calling process (the launcher) a child subreaper, so orphaned descendants are re-parented to it, and reaps those discovery saw adopted. See [what it changes](#what-changes-when-adopt_orphans-is-enabled) |
| `CUPTI_PROFILER_PROC_ROOT` | environment variable | unset = `/proc` | **Test-only, not supported.** Directory read instead of `/proc` for the process-table reads (`children`, `stat`, `comm`). See [the hook](#testing-hook-cupti_profiler_proc_root) |
| flush gate | C++ `<cupti_profiler/testing.h>`; Python `_native._testing_arm_flush_gate()`, `_testing_wait_flush_held()`, `_testing_release_flush_gate()` | disarmed | **Test-only, not supported.** Holds an in-process flush between writing and committing removals. See [the hook](#testing-hook-flush-gate) |

Fixed behaviour, not configurable: exit detection (a pidfd per tracked
process, polled every sample tick) is always on; `comm` is re-read every
100 ms; `comm_history` keeps at most 16 entries. See
[Process table and exit detection](#process-table-and-exit-detection).

## Sidecar mode

`mode: SYSTEM_PROBE_MODE_SIDECAR` on the System and/or Disk config moves
those probes' sampler and flush threads into a separate process,
`cupti-profiler-sidecar`, forked by `Configure()`, so their CPU is not
charged to the workload's PID. One sidecar serves both probes. GPU and
Events probes always stay in-process.

**Lifecycle.** `Configure()` spawns the sidecar and sends it the config;
`Start()` tells it to start its probes; `Stop()` sends `MSG_STOP` and
waits for its final flush. Without `MSG_STOP`, the sidecar still stops
the same way — probes stopped, final flush written, exit status 0 — when:

- **the host process exits**, however it dies (`kill -9` included). The
  sidecar holds a pidfd on the host *process*, so the thread that called
  `Configure()` may exit freely; and it works even when a process the
  host forked still holds the control pipe open;
- its **control pipe closes**;
- it receives **SIGTERM or SIGINT** (a terminal Ctrl-C reaches it with the
  host). The host's later `Stop()` then logs how the sidecar ended and
  returns normally; a C++ host that has not ignored SIGPIPE is not killed
  by writing to it.

**Which binary.** The sidecar is `cupti-profiler-sidecar`, found at the
path baked in at build time (CMake `CUPTI_PROFILER_SIDECAR_PATH`, default
`<build dir>/tools/cupti-profiler-sidecar`). The environment variable
**`CUPTI_PROFILER_SIDECAR`** overrides it at `Configure()` — e.g. for an
installed copy on a local filesystem, or one with file capabilities. It
is used only if it names an executable regular file; otherwise the
built-in path is used without a warning. If neither exists,
`Configure()` fails with `SidecarNotFound`. The startup situation report
records which binary ran (`observer`).

**Errors.** `Configure()` fails with `SidecarNotFound`,
`SidecarSpawnFailed`, `SidecarExited`, `SidecarBadHandshake` or
`SidecarAffinityFailed`; `Start()` with `ProbeStartFailed` when a probe
inside the sidecar could not start. Python raises `RuntimeError` naming
the code.

**CPU affinity (`sidecar_cpus`).** Empty by default: no pinning. List CPUs
to pin the sidecar and every thread it starts to them — e.g. a core the
workload does not use — so it interferes less with the workload, not
just shows up under a different PID:

```python
cp.configure_suite(suite, {..., "sidecar_cpus": [7]})
```

A CPU outside this process's allowed set fails `Configure()` with
`SidecarAffinityFailed`. The host's own affinity is never changed; with
no probe in SIDECAR mode the field is ignored with a note.

**Clocks.** Sidecar samples are stamped with `steady_clock`
(`CLOCK_MONOTONIC`), which is system-wide, so they line up with the
host's GPU and Events traces with no clock handshake. Measured: a marker
made at the same instant in the GPU trace (spin kernel) and the system
trace (RSS jump) lines up within one system tick, identically under
SIDECAR and LEGACY.

**Permissions.** None beyond LEGACY's: see
[Permissions for per-PID I/O](#permissions-for-per-pid-io).

## Process table and exit detection

Always on, for every tracked process — listed roots and discovered
processes alike, in LEGACY and SIDECAR, **with descendant tracking on or
off**.

**Exit detection.** Each probe (System, Disk) holds a pidfd on every
process it tracks, opened when the process is registered, and polls them
all once per sample tick (one `poll()` over all of them), right after
reading that tick's values. When a process exits — as soon as it is a
zombie, reaped or not — the probe:

1. stops sampling it: **no sample of it is later than its end time**;
2. emits it once more with `removed = true` and `end_time_ns`, in the
   next flush;
3. drops it from later flushes.

That entry is never sampled again. A process that later gets the same
PID number is a **different process**: it is not tracked unless it is
added again (`AddTrackedProcess`; a re-add made while the old entry's
removal is still pending is registered right after that flush) or
discovered as a child of a tracked process. Explicit
`RemoveTrackedProcess()` still works as before; such an entry has
`end_time_ns = 0`. A PID that does not exist when it is added is
recorded as exited straight away (logged, `removed = true` in the next
flush).

**The table.** Every flush of `system_metrics.pb` and `disk_metrics.pb`
carries one `TrackedProcessV2` per tracked process
(`proto/metric_sample.proto`):

| Field | Meaning |
|---|---|
| `pid` | the process ID |
| `parent_pid` | the parent when it was registered: for a discovered process, the tracked parent it was found under; for a root, its parent when it was listed. Not updated on reparenting |
| `discovered` | kind: `true` = found by descendant tracking, `false` = listed root |
| `label` | roots: the alias they were listed with; discovered: their root's alias |
| `alias` | display name: roots keep their listed alias; discovered processes are `<label>/<comm>`, following `comm` |
| `comm` | current `/proc/<pid>/comm` (≤ 15 characters), re-read every 100 ms |
| `comm_history` | every `comm` it has had, oldest first, each with the trace time it was first seen (the first at registration, later ones within 100 ms of the rename); at most 16 — the first and the 15 most recent |
| `start_time_ns` | when the process started: the kernel's `/proc/<pid>/stat` field 22 converted to the trace clock. **10 ms resolution** (`USER_HZ` ticks). 0 = unknown |
| `end_time_ns` | with `removed = true` after an exit: the first instant the probe saw it gone, on the trace clock. The exit happened **within one sampling tick** before it. 0 while alive and for a removal by request |
| `removed` | this is the entry's last flush |
| `cpu_before_discovery_ns` | System trace, discovered processes only: see [Head and tail CPU](#head-and-tail-cpu) |

The trace clock is the samples' `timestamp_ns` clock (`steady_clock` =
`CLOCK_MONOTONIC`, system-wide, so identical under SIDECAR). Field 22
counts on `CLOCK_BOOTTIME`, which also runs during suspend; the
conversion subtracts the current offset between the two.

Rebuilding the tree from a trace — keep each PID's latest row:

```python
table = {}
for frame in frames:                       # SystemMetricsTrace messages, in order
    for tp in frame.tracked_processes:
        table[tp.pid] = tp                 # latest alias, comm, end time
roots = [tp for tp in table.values() if not tp.discovered]
children = {}
for tp in table.values():
    if tp.discovered:
        children.setdefault(tp.parent_pid, []).append(tp)
```

Take the **latest** row, not the first: a discovered process's alias
changes when it renames itself (vLLM's `EngineCore` is found as
`vllm/python3.12` and becomes `vllm/VLLM::EngineCor` seconds later).

**On stderr**, discovery logs the tree as it grows and shrinks:

```text ln:false
[discovery] + 3858012 python3.12 (parent 3857844 vllm)
[discovery] - 3858336 ninja exited
```

### PID reuse

A PID names a process only while that process lives; once it has exited
and been reaped, the kernel may hand the number to a new process. Where
that could bite, and what guards it:

| Where | Guard |
|---|---|
| A listed PID, between the call and registration | the probe opens the pidfd first, reads `/proc/<pid>/stat`, then checks the pidfd is still alive: the recorded start time and parent belong to the process the pidfd pins |
| Samples: values are read from `/proc` **by number** every tick | **read-then-verify**: each tick reads every process's values first and polls the pidfds after; a reading of a process found gone is dropped. A kept reading was taken while the pinned process still existed (a zombie's number cannot be reused), so it is that process's |
| After a tracked process exits | its entry is removed and never sampled again. Per-PID baselines are keyed by the entry, not the number, so a new process never inherits an old one's |
| Registering a number whose old entry is still awaiting its removal flush | the new process gets its own entry once the old one is flushed and dropped, so a flush never carries one number twice |
| Discovery: a PID in a `children` file exits and is reused before `pidfd_open` | the parent check (the new owner's parent must be tracked), done after `pidfd_open` and checked against it; the probes duplicate that very pidfd |
| **Gap A: a discovered process exits and its number is reused before the next scan** (≤ one interval) | documented, not specifically guarded (user decision, 2026-09-25). Before the probes held their own pidfds, this could attribute up to one scan interval of the new process's samples to the old entry. **The probe-level pidfds remove that**: the probe marks the old entry exited at its next tick and never samples it again, whatever discovery does. What remains keyed by number is the **CPU tail** bookkeeping (`CpuTail`): it notices that an exited child has been reaped by its `/proc/<pid>` entry disappearing, and reads a reparented child's new parent by number. If the number is reused within one sampling tick of the reap, that child's tail can be missing or attributed to the wrong parent. `pid_max` is 4,194,304 on the compute nodes and allocation is sequential, so reuse needs the whole PID space to wrap within that window; never observed |

**Cost.** Per probe and sample tick: one `poll()` over all pidfds. Every
100 ms: one `/proc/<pid>/comm` read per tracked process. One pidfd per
tracked process per probe (plus discovery's own, with discovery on).

## Descendant tracking

Off by default: the System and Disk probes trace exactly the PIDs they
are given. Turn it on to also trace every descendant of a listed PID —
what a server that forks workers (vLLM's API server → `EngineCore`,
helpers, startup compile workers) needs.

### Configuration

```protobuf title:"proto/profiler_config.proto (excerpt)"
message ProcessDiscoveryConfig {
    bool   enabled              = 1;  // default false: trace listed PIDs only
    bool   direct_children_only = 2;  // default false => RECURSIVE
    uint64 scan_interval_ms     = 3;  // 0 => 100
}
message ProfilerSuiteConfig { ...; ProcessDiscoveryConfig process_discovery = 8; }
```

One setting drives **both** the System and Disk probes, which share the
tracked set. `direct_children_only` is inverted because proto3 bools
default to `false`: the default is recursive.

A per-root override wins over `enabled` for that root, so you can follow
a spawned server's tree without also following the host's own children:

```python
suite = cp.ProfilerSuite()
cp.configure_suite(suite, {..., "process_discovery": {"enabled": False, "scan_interval_ms": 100}})
suite.start()
server = subprocess.Popen(["vllm", "serve", ...])
suite.add_tracked_process(server.pid, "vllm", track_descendants=True)   # None = inherit
```

C++: `AddTrackedProcess(pid, alias, /*trackDescendants=*/true)`. PIDs
listed in the config's `processes` inherit `enabled`.

### Guarantee

**A process that is a child of a tracked process for at least one full
scan interval is discovered; once discovered, it is tracked until it
exits, regardless of reparenting.** A child that lives less than one
interval may be missed — lower `scan_interval_ms` to catch it (at the
cost below).

### How a scan works

One discovery thread per observer: in the workload process under
LEGACY, inside the sidecar under SIDECAR (the settings travel in the
serialized config, the per-root override in `MSG_ADD_PID`'s optional
trailing byte). It starts only once some root tracks its descendants.
Every `scan_interval_ms`:

1. For **every** followed process — listed roots with descendants on,
   and (recursive mode) every process already discovered under them,
   not just the tree currently reachable from a root — read
   `/proc/<pid>/task/*/children`, **every thread's** file: a child
   hangs off whichever thread forked it.
2. For each new PID: `pidfd_open` (raw syscall), then re-read its parent
   from `/proc/<pid>/stat` and **require that parent to be followed** —
   the PID-reuse guard (a PID listed in a `children` file can exit and be
   recycled by an unrelated process before `pidfd_open`). The pidfd is
   checked still-alive after the read, so the data describes the process
   the pidfd pins. The process is registered on both probes (which
   duplicate that pidfd) with `label` = the root's alias (its PID if it
   has none), `parent_pid` and `discovered = true`; its alias is
   `<label>/<comm>` and follows later `comm` changes (a forked child
   carries its parent's `comm` until it execs or renames itself, as
   vLLM's `EngineCore` does). It is logged:
   `[discovery] + <pid> <comm> (parent <ppid> <parent comm>)`.
3. Poll the held pidfds, to stop following processes that exited (and
   reap adopted ones, see below). Removing them from the trace is the
   probes' job: every tracked process — discovered or listed — is watched
   by the probes' own pidfds, see
   [Process table and exit detection](#process-table-and-exit-detection).
4. Record the scan's own wall time; every system/disk flush carries the
   cumulative `DiscoveryStats` (`scans`, `scan_p50_ns` / `p99` / `max`,
   `discovered`, `exited`, `rejected`).

Step 1 is what makes the guarantee hold across reparenting: when an
intermediate process exits, its discovered children leave the root's
tree but keep being scanned, so their own later children are found too.

### Cost

The scan is dominated by opening one `children` file per thread of every
followed process. Measured in C++ on a compute node (sprc01) against a
vLLM-shaped tree — root with 41 threads, children with 77 and 1 (119
threads, 3 processes), recursive:

| Interval | Per scan (library self-metric) | Observer CPU |
| -------- | ------------------------------ | ------------ |
| 100 ms (default) | p50 1.77–1.84 ms, p99 1.84–1.95 ms | **1.7–1.8% of one core** |
| 10 ms | p50 0.66–1.64 ms, p99 0.72–1.90 ms | 6.5–15% of one core (varies run to run: at this cadence the caches sometimes stay warm) |

A single `children` read costs ~4.5 µs when hot but ~14 µs after the
scan thread has slept for an interval (cold caches): a bare sweep of 120
threads measured 0.57 ms back-to-back and 1.65 ms after a 100 ms sleep.
Discovery's own bookkeeping adds ~0.15 ms. Under SIDECAR this CPU is the
sidecar's; under LEGACY it is charged to the host process.

### Limits

- **Short-lived children** (< 1 interval) may be missed; see the guarantee.
- **`CLONE_PARENT` from a root, and work delegated over IPC to processes
  outside the tree**, are the user's responsibility: track the process
  that actually spawns them.
- **Removing a root** (`RemoveTrackedProcess`) leaves its
  already-discovered descendants tracked until they exit. A root that
  exits is removed like any tracked process (see
  [exit detection](#process-table-and-exit-detection)); its descendants
  stay tracked.
- **PID reuse within one scan interval** ("gap A"): no longer able to
  misattribute samples; what remains is described under
  [PID reuse](#pid-reuse).
- **Aliases name the process as it is now**: a discovered process's alias
  follows its `comm`, so a reader should take the alias of the latest
  flush (or `comm_history`), not the first one seen.
- **The observer never tracks itself** (the sidecar is a child of the
  host, but is never discovered).
- With System and Disk in **different** modes (one LEGACY, one SIDECAR),
  each observer runs its own scan.
- CPU before a discovered process's first sample and after its last one
  is not in its samples; it is reported separately — see below.

### Head and tail CPU

A process's CPU clock counts from its fork, but a discovered process is
sampled only from its first sample (a baseline) to its last. The two
missing pieces are reported on the System trace, never folded into a
sample:

- **Head** — `TrackedProcessV2.cpu_before_discovery_ns`: its CPU clock at
  its first sample, i.e. what it used from fork until discovery found it.
  Recorded once.

> [!NOTE]
> **Roots and discovered processes are counted differently at the
> start (user decision, 2026-09-25).** A listed root gets **no head**:
> its CPU counts from when it was added (listed in the config: from
> `Start()`; `AddTrackedProcess` mid-run: from that call), and what it
> used before is not in the trace — `cpu_before_discovery_ns` is 0. A
> discovered process gets its CPU from fork to discovery as its head.
> So for a server attached some time after launch, the root's traced
> CPU is "since attach", while each child's `head + samples + tail` is
> its whole life. To get a root's whole life, add it right after
> spawning it, before it has done any work.
- **Tail** — a `CpuTail` in `SystemMetricsTrace.cpu_tails`, emitted once:
  CPU used after its last sample, up to its exit. Measured on its tracked
  parent: when the parent reaps it, the parent's `cutime + cstime` grow
  by the child's whole CPU (plus what the child itself reaped); the tail
  is that growth minus the child's clock at its last sample and minus
  its own reaped children's CPU.

For a discovered process tracked until it exits,
`head + Σ samples + tail` equals its total CPU (tests: within 10 ms of the
process's own clock at 2 Hz sampling, where tails were 0.27–0.42 s).

Limits of the tail:

- `cutime`/`cstime` are in `USER_HZ` ticks (10 ms), truncated per field:
  about ±20 ms, clamped at 0;
- it exists only once the parent **reaps** the child — a parent that never
  waits, or exits first, yields none, as does a child reparented to an
  untracked process;
- it also includes any short-lived child the same parent reaped in that
  interval that discovery never saw;
- when several tracked children of one parent are reaped within one
  sample interval, their shares cannot be told apart: **one `CpuTail`
  lists all of them in `pids`** with their combined tail, rather than a
  guessed split.

Cost: nothing with discovery off; otherwise one `/proc/<pid>/stat` read
per sample tick for each tracked process that has discovered children,
plus one per exited child until it is reaped.

### Subreaper helper: `adopt_orphans()`

When an intermediate process exits, its children are re-parented to the
nearest *subreaper* ancestor, or to init. Discovery keeps tracking the
ones it already found either way. **Opt in** to make your launcher that
subreaper — `cupti_profiler.adopt_orphans()` in Python,
`EnableChildSubreaper()` (`<cupti_profiler/child_subreaper.h>`) in C++;
the library never sets it by itself. Call it once, before spawning the
workload.

It buys: adopted orphans' **CPU and storage I/O fold into the
launcher's `getrusage`** (measured 0.001 s → 0.501 s for a 0.500 s
orphan), and orphans stay under the launcher instead of vanishing to
init. It does **not** buy: an orphan whose intermediate parent died
*before the first scan* shows up as the launcher's child,
indistinguishable from the launcher's other children, so it is not
auto-tracked.

**Reaping.** Adopted orphans become the launcher's zombies. A reaper that
calls `waitpid(-1)` would also steal the exit status of the launcher's
*own* children — `Popen.wait()` on the server would see `ECHILD`, which
Python reports as return code 0. So the helper reaps **only processes
discovery saw being adopted** — found with a parent other than the
launcher, and whose parent at exit is the launcher — one at a time,
through their pidfd (`waitid(P_PIDFD)`), after checking in `/proc` that
it is the launcher's zombie with the start time recorded at discovery.
Under LEGACY the discovery thread reaps directly; under SIDECAR the
sidecar sends the PID to the launcher over a dedicated pipe (sidecar
fd 5), and a thread in the launcher that otherwise sleeps in `read()`
reaps it.

#### What changes when `adopt_orphans()` is enabled

Every row was measured or verified on 2026-09-24 (kernel 5.15) unless
marked.

| Aspect | Without it | With it |
|---|---|---|
| **Who is marked** | — | the calling process (the launcher) only |
| **Inherited by its children?** | — | **no** — neither `fork` nor `Popen` children get it, so vLLM itself is never a subreaper (verified) |
| **Survives the launcher `execve`-ing?** | — | **yes** (verified) — if the launcher execs another program, that program is still a subreaper |
| **Where orphaned descendants go** | init, or the nearest existing subreaper such as `systemd --user` or `slurmstepd` | **the launcher** — their `PPid` becomes the launcher's PID |
| **Signals to the launcher** | none for orphans | **`SIGCHLD` for every adopted orphan that exits** — code that handles `SIGCHLD` or calls `waitpid(-1)` will see children it never started |
| **Zombies** | reaped by init | owned by the launcher until reaped. The helper reaps those discovery saw adopted; **orphans adopted before the first scan are not reaped by the helper** and stay zombies until the launcher exits. A zombie holds a PID and a process-table slot, no memory or CPU |
| **Accounting** | orphans' CPU and storage I/O are credited to init — invisible to you | credited to the launcher: `getrusage(RUSAGE_CHILDREN)` and `os.times()` child fields **increase** (measured 0.001 s → 0.501 s for a 0.500 s orphan; 32.0 MiB written → 32.0 MiB folded) |
| **Process group, session, signal delivery** | — | **unchanged** — Ctrl-C still reaches the orphans if they remain in the foreground process group |
| **Attached (not spawned) targets** | — | **no effect** — only descendants of the launcher are covered |
| **Cost** | — | zero at steady state; ~41 µs of launcher CPU per adopted orphan that exits, never on the target (measured 2026-09-25 on this implementation, SIDECAR mode: 41.4 µs, 95% CI ±3.0 µs, 5 runs × 600 orphans, all of it on the launcher's notice-reader thread; under LEGACY the cost is too small to separate from the in-process sampler's own CPU) |
| **Turning it off** | — | `prctl(PR_SET_CHILD_SUBREAPER, 0)`; already-adopted orphans stay adopted |

The startup situation report records whether it is set, so a trace
always says which guarantee it was collected under.

### Startup situation report

`ProfilerSuite::Configure()` (System or Disk enabled) checks, from the
observer's side, which of the guarantees above hold here, logs it once
to stderr, and writes it to `session_metadata.pb` as
`repeated SituationCheck situation` (`check`, `observed`, `consequence`,
`degraded`). Every line states what it means for the trace:

| Check | Why it matters |
| ----- | -------------- |
| `observer` | in-process (LEGACY) or the sidecar's PID and binary (SIDECAR) |
| `/proc/<pid>/task/<tid>/children` | absent (no `CONFIG_PROC_CHILDREN`) = no descendant tracking |
| `pidfd_open (syscall)` | probed with the raw syscall, not a language binding; absent = no descendant tracking |
| `descendant tracking` | the configured default, mode and interval, with the guarantee it implies |
| `yama ptrace_scope` | restricts ptrace *attach* only; `/proc/<pid>/io` reads are unaffected |
| `observer effective capabilities` | `CAP_DAC_READ_SEARCH` + `CAP_SYS_PTRACE` = other users' `io` readable |
| `secure-exec (AT_SECURE) of the observer` | setuid or file capabilities make the loader strip `LD_LIBRARY_PATH` — CUDA forward compat included |
| `CUDA forward-compat on LD_LIBRARY_PATH` | whether a newer CUDA runtime can run on the installed driver |
| `filesystem of <observer binary>` | `nosuid` mounts ignore file caps and setuid; **network filesystems (NFS, CIFS, Lustre, …) cannot hold file capabilities at all** — install the observer on a local filesystem to grant it any |
| `child subreaper (this process)` | re-probed on every manifest write, since `adopt_orphans()` may be called after `Configure()` |
| `taskstats per-PID query` | `EPERM` without `CAP_NET_ADMIN`: `/proc` backend only |
| `target <pid>` (one per root, added as roots are added) | uid (same / other user), **spawned** by this process vs **attached**, `/proc/<pid>/io` readable |

### Testing hook: `CUPTI_PROFILER_PROC_ROOT`

**Test-only, not a supported setting.** When set, discovery reads
`children`, `stat` and `comm` under this directory instead of `/proc`,
and so do the probes' process-table reads (`stat` and `comm` at
registration, the `comm` refresh), so the algorithm can be exercised on
a synthetic tree (`tests/python/test_discovery.py::test_pid_reuse_guard`).
pidfds and the probes' samples still use the real kernel, so every PID
in the synthetic tree must be a real, live process.

### Testing hook: flush gate

**Test-only, not a supported API** (`<cupti_profiler/testing.h>`,
Python `_native._testing_*`). Holds an in-process System/Disk flush
thread after it has written a flush and before it commits that flush's
removals, so a test can make a `RemoveTrackedProcess()` land exactly in
that window (`tests/python/test_removal_race.py`). Disarmed, it is one
atomic load per flush.

## Output format

A full-suite run produces five `.pb` files under `output_dir`:

| File | Schema | Contents |
| ---- | ------ | -------- |
| `gpu_metrics.pb` | `GPUMetricsTrace` (length-delimited) | GPU PM counter samples — combined across every `device_indices` entry |
| `system_metrics.pb` | `SystemMetricsTrace` (length-delimited) | CPU + memory samples (system + per-PID) |
| `disk_metrics.pb` | `DiskMetricsTrace` (length-delimited) | Disk device + per-PID I/O samples |
| `events.pb` | `EventTrace` (length-delimited) | Regions + events, Generic + GPU domains |
| `session_metadata.pb` | `SessionMetadata` (single message, **not** length-delimited) | Manifest of probes, hostname, wall-clock anchor, **inlined `MetricCatalog`** |

The three per-domain trace types share substructures (`TraceHeader`,
`ScopeMetricNames`, `Sample` / `ProcessSample` / `DeviceSample` /
`GPUSample`, `TrackedProcessV2`, `FlushStats`) defined in
`proto/metric_sample.proto`. Every sample is `(timestamp, [scope_key],
values[])` with `values[]` ordered to match the per-scope FQN list in
the same trace's `scope_metric_names[]`. See
[`docs/metric-model.md`](metric-model.md) for the full type system.

### GPU schema

```protobuf title:"proto/gpu_metrics.proto"
message GPUMetricsTrace {
    TraceHeader header                           = 1;
    // SCOPE_GPU FQNs — every GPUSample.values[i] aligns with this list.
    repeated ScopeMetricNames scope_metric_names = 2;
    // One entry per index in GPUProfilerConfig.device_indices.
    repeated GPUDeviceInfo tracked_gpus          = 3;
    repeated GPUSample samples                   = 4;
    repeated FlushStats flush_stats              = 5;
}
```

### System schema

```protobuf title:"proto/system_metrics.proto"
message SystemMetricsTrace {
    TraceHeader header                           = 1;
    // Two entries: SCOPE_SYSTEM (CPU + memory FQNs combined) and
    // SCOPE_PROCESS (per-PID CPU + memory FQNs combined).
    repeated ScopeMetricNames scope_metric_names = 2;
    // The process table. Grows mid-run via AddTrackedProcess() and
    // discovery. Entries with removed=true (removed by request, or
    // exited: then with end_time_ns) appear in exactly one flush as a
    // removal marker before being dropped.
    repeated TrackedProcessV2 tracked_processes  = 3;
    // One Sample per tick — values[] combines CPU% + mem bytes.
    repeated Sample        system_samples        = 4;
    // One ProcessSample per (tick × tracked PID).
    repeated ProcessSample process_samples       = 5;
    repeated FlushStats    flush_stats           = 6;
    // Present while descendant tracking runs (cumulative).
    DiscoveryStats   discovery_stats             = 7;
    // Exit tails of discovered processes, each emitted once.
    repeated CpuTail cpu_tails                   = 8;
}
message CpuTail { uint64 timestamp_ns; uint32 parent_pid;
                  repeated uint32 pids;               // several = not apportionable
                  uint64 cpu_after_last_sample_ns; }
```

### Disk schema

```protobuf title:"proto/disk_metrics.proto"
message DiskMetricsTrace {
    TraceHeader header                           = 1;
    // Two entries: SCOPE_DEVICE (BW + inflight) and SCOPE_PROCESS (the five per-PID /proc/<pid>/io counters).
    repeated ScopeMetricNames scope_metric_names = 2;
    repeated TrackedProcessV2 tracked_processes  = 3;
    repeated string  tracked_devices             = 4;
    repeated DeviceSample  device_samples        = 5;
    repeated ProcessSample process_samples       = 6;
    repeated FlushStats    flush_stats           = 7;
    DiscoveryStats   discovery_stats             = 8;
}
```

### Shared sample shape

```protobuf title:"proto/metric_sample.proto (excerpt)"
message ScopeMetricNames { Scope scope; repeated string fqns; }
message Sample           { uint64 timestamp_ns; repeated double values; }
message ProcessSample    { uint64 timestamp_ns; uint32 pid; repeated double values; }
message DeviceSample     { uint64 timestamp_ns; string device_name; repeated double values; }
message GPUSample        { uint64 timestamp_ns; uint32 gpu_index;   repeated double values; }

message TrackedProcessV2 { uint32 pid; string alias; bool removed;
                           uint32 parent_pid;   // parent when registered (roots too)
                           bool discovered;     // kind: true = found by descendant tracking
                           uint64 cpu_before_discovery_ns;  // system trace, discovered only
                           string label;        // root alias (discovered: alias = label/comm)
                           string comm;         // current comm, re-read every 100 ms
                           uint64 start_time_ns;   // kernel start time, trace clock, 10 ms res.
                           uint64 end_time_ns;     // exit seen (<= 1 tick late); 0 = alive/removed
                           repeated CommChange comm_history; }  // <= 16, oldest first
message CommChange       { uint64 timestamp_ns; string comm; }
message DiscoveryStats   { uint64 scan_interval_ns, scans, scan_p50_ns, scan_p99_ns,
                                  scan_max_ns, discovered, exited, rejected; }
message GPUDeviceInfo    { uint32 device_index; string device_name; string chip_name;
                           double peak_dram_bw_bytes_per_s, peak_pcie_bw_bytes_per_s,
                                  peak_nvlink_bw_bytes_per_s; }

message TraceHeader {
    string hostname; uint64 sampling_frequency_hz; uint32 host_cpu_count;
    ClockAnchors anchors;  // steady_clock + wall_clock_epoch + cupti_reference
}
message FlushStats { uint64 flush_byte_size; uint64 flush_interval_ns; }
```

### Events schema

Events split into two clock domains, written into the same trace and converted to `steady_clock` ns at view time using the metadata anchor.

```protobuf title:"proto/events.proto"
enum TimeDomain { TIME_DOMAIN_GENERIC = 1; TIME_DOMAIN_GPU = 2; }

message EventTrace {
    TraceMetadata metadata = 1;        // steady_clock + cupti_clock + wall-clock anchors
    repeated EventBuffer buffers = 2;  // one per active domain
    repeated EventFlushStats flush_stats = 3;
}

message EventBuffer { TimeDomain domain; repeated Region regions; repeated Event events; }
message Region { string name; uint64 start_timestamp_ns, end_timestamp_ns; }
message Event  { string name; uint64 timestamp_ns; }
```

### Session manifest

```protobuf title:"proto/session_metadata.proto"
enum ProbeKind { PROBE_KIND_GPU = 1; PROBE_KIND_SYSTEM = 2; PROBE_KIND_DISK = 3; PROBE_KIND_EVENTS = 4; }

message ActiveProbe { ProbeKind kind; string output_file; uint64 sampling_frequency_hz; }

message SessionMetadata {
    string hostname = 1;
    uint64 wall_clock_epoch_ns = 2;
    string start_iso8601 = 3;
    repeated ActiveProbe probes = 4;
    // Inlined active catalog (proto/metric_catalog.proto) so the
    // visualizer reads only ONE file to bootstrap.
    MetricCatalog catalog = 5;
    // Startup situation report (see "Startup situation report").
    repeated SituationCheck situation = 6;
}

message SituationCheck { string check; string observed; string consequence; bool degraded; }
```

`session_metadata.pb` is written atomically (`.tmp` + `rename(2)`) at
`ProfilerSuite::Start()` AND `Stop()` — tailers (live visualizer) never
observe a torn file. The Stop() copy may additionally carry situation
lines for roots added mid-run, and the current subreaper state.

### Length-delimited streaming format

Each per-probe `.pb` file contains one or more length-delimited messages of its corresponding trace type:

```text ln:false
[varint: message_size][serialized Trace]
[varint: message_size][serialized Trace]
...
[varint: message_size][serialized Trace]  ← final chunk
```

- Periodic flushes write incremental messages (a slice of samples for that window).
- Every message is **self-contained**: it carries the full `TraceHeader`, `scope_metric_names[]`, and (where applicable) `tracked_processes[]` / `tracked_gpus[]`. A live tailer joining mid-stream has everything it needs to plot.
- `flush_stats` carry the **previous** flush's byte size and interval (a flush can't include its own size).
- All visualization tools read and merge messages automatically.

### Reading in Python

```python title:"Reading the trace programmatically"
import gpu_metrics_pb2

def load_trace(path):
    with open(path, "rb") as f:
        data = f.read()

    traces = []
    offset = 0
    while offset < len(data):
        # Decode varint
        shift, msg_len, varint_bytes = 0, 0, 0
        while offset + varint_bytes < len(data):
            b = data[offset + varint_bytes]
            msg_len |= (b & 0x7F) << shift
            varint_bytes += 1
            shift += 7
            if (b & 0x80) == 0:
                break

        msg_start = offset + varint_bytes
        trace = gpu_metrics_pb2.GpuMetricsTrace()
        trace.ParseFromString(data[msg_start:msg_start + msg_len])
        traces.append(trace)
        offset = msg_start + msg_len

    # Merge all chunks
    merged = gpu_metrics_pb2.GpuMetricsTrace()
    merged.CopyFrom(traces[0])
    merged.ClearField("samples")
    merged.ClearField("regions")
    for t in traces:
        merged.samples.extend(t.samples)
        merged.regions.extend(t.regions)
    return merged
```

---

## GPU metrics reference

> [!TIP]
> The full type system (Counter / Ratio / Throughput, every legal rollup
> and submetric suffix, and the proposed extension that lets the same
> abstraction cover CPU/memory/disk for a generic post-processing scheme)
> is documented in [`metric-model.md`](metric-model.md). This section is
> the operational view — what to put in the `metrics:` list.

### Metric naming and types

CUPTI PM Sampling inherits the PerfWorks metric model. A fully-qualified metric name has the form:

```text ln:false
<entity>__<counter>[.<rollup>][.<submetric>]
```

- **`<entity>`** — the hardware unit being measured (`sm`, `smsp`, `dram`, `gpc`, `pcie`, `nvlrx`, `nvltx`, …).
- **`<counter>`** — the raw event being counted (`cycles_active`, `cycles_elapsed`, `warps_active`, `read_bytes`, …).
- **`<rollup>`** — aggregation across instances of the entity (e.g. across all SMs).
- **`<submetric>`** — post-rollup transformation (rate, percent of peak, etc.).

There are three base metric types, each accepting a different set of suffixes:

| Type | Description | Valid rollups | Valid submetrics | Example |
| ---- | ----------- | ------------- | ---------------- | ------- |
| **Counter** | Raw event count | `.sum`, `.avg`, `.min`, `.max` | `.per_second`, `.per_cycle_active`, `.per_cycle_elapsed`, `.pct_of_peak_sustained_{active,elapsed}`, `.pct_of_peak_burst_{active,elapsed}` | `dram__read_bytes.sum.per_second` |
| **Ratio** | Dimensionless quantity (already normalized) | `.ratio`, `.pct` | — | `smsp__average_warps_active_per_cycle_active.ratio` |
| **Throughput** | Pre-built utilization metric (% of peak) | `.avg`, `.max` | `.pct_of_peak_sustained_{active,elapsed}` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |

**Useful idioms:**
- Clock frequency in Hz: `<clock_domain>__cycles_elapsed.avg.per_second` — e.g. `gpc__cycles_elapsed.avg.per_second` (SM/GPC clock), `dram__cycles_elapsed.avg.per_second` (memory clock). All instances in a clock domain run synchronously, so `.min` / `.max` / `.avg` return the same number; only `.avg` (or `.sum` for `N × clock`) is useful.
- Per-window byte counters: `<bus>__{read,write}_bytes.sum` — cumulative-sum these post-hoc to get total bytes transferred.

The actual metric set is **architecture-specific** (Hopper exposes counters Ampere doesn't, etc.). To enumerate what's available on your device:

```bash ln:false
./cupti_pm_sampling --list-metrics
```

Each line is annotated with its type, so `grep Counter` / `grep Ratio` / `grep Throughput` buckets them. The Nsight Compute *Profiling Guide → Metrics Reference* has the canonical descriptions of what each entity and counter measures.

### Default metrics (used by example)

| Index | Metric | Category |
| ----- | ------ | -------- |
| 0 | `sm__cycles_active.avg` | SM utilization (average across SMs) |
| 1 | `sm__cycles_active.max` | SM utilization (busiest SM) |
| 2 | `sm__cycles_elapsed.avg` | Reference elapsed cycles (for normalization) |
| 3 | `sm__warps_active.avg` | Occupancy (average active warps/cycle) |
| 4 | `sm__warps_active.max` | Occupancy (busiest SM) |
| 5 | `dram__read_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM read BW % of peak |
| 6 | `dram__read_throughput.max.pct_of_peak_sustained_elapsed` | DRAM read BW max % |
| 7 | `dram__write_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM write BW % of peak |
| 8 | `dram__write_throughput.max.pct_of_peak_sustained_elapsed` | DRAM write BW max % |

> [!IMPORTANT]
> All metrics in a single `ProfilerConfig` must fit in one hardware pass. If you exceed the single-pass limit, `Configure()` will report the error. Reduce the metric count or choose metrics from the same counter group.

### Sampling interval guidance

| Interval | Frequency | Overhead | Use case |
| -------- | --------- | -------- | -------- |
| 1,000,000 ns | 1 kHz | Negligible | Long-running production monitoring |
| 100,000 ns | 10 kHz | Low | General profiling (default) |
| 10,000 ns | 100 kHz | Moderate | High-resolution analysis |
| 1,000 ns | 1 MHz | High | Short bursts only — risk of HW buffer overflow |

---

## System & disk metrics reference

Unlike GPU metrics — which are configurable via the `metrics:` list — system and disk metrics are **fixed**: every sample carries the full set listed below. The only knobs are sampling frequency and which PIDs/devices are tracked.

### CPU & memory (`SystemMetricsTrace`)

System-wide samples (always present):

| Field | Source | Units | Notes |
| ----- | ------ | ----- | ----- |
| `total_utilization_pct` | `/proc/stat` (cpu line) | % (0–100) | `1 − (idle + iowait) / total` across all CPUs |
| `user_pct` | `/proc/stat` user + nice | % | Time in userspace |
| `system_pct` | `/proc/stat` system + irq + softirq | % | Time in kernel |
| `iowait_pct` | `/proc/stat` iowait | % | CPU idle waiting for I/O |
| `total_bytes` | `/proc/meminfo` MemTotal | bytes | Static |
| `used_bytes` | derived from MemAvailable | bytes | `total − available` |
| `available_bytes` | `/proc/meminfo` MemAvailable | bytes | Kernel's "what's reclaimable for new allocations" |
| `buffers_bytes` | `/proc/meminfo` Buffers | bytes | Block-device cache |
| `cached_bytes` | `/proc/meminfo` Cached | bytes | Page cache |

Per-process samples (one row per tracked PID per sample tick):

| Field | Source | Units | Notes |
| ----- | ------ | ----- | ----- |
| `cpu_pct` | Process CPU clock (`clock_getcpuclockid` + `clock_gettime`, ns) | % of one CPU | Total on-CPU time across the whole thread group, including threads that exited between ticks, divided by actual wall-clock elapsed between ticks. No user/kernel/iowait split. No 10 ms `CLK_TCK` quantization, but a thread running without being descheduled is credited at each scheduler tick (`CONFIG_HZ`, 4 ms at 250 Hz), so a single sample can be off by up to one tick per running thread; the sum over samples is exact. |
| `rss_bytes` | `/proc/<pid>/status` VmRSS | bytes | Resident set size (physical pages) |
| `vms_bytes` | `/proc/<pid>/status` VmSize | bytes | Virtual memory size |
| `shared_bytes` | `/proc/<pid>/status` RssShmem | bytes | Resident shared memory |

> [!NOTE]
> Per-process CPU percentages can exceed 100% — they're normalized against one CPU, so a multi-threaded process pegging 4 cores reports ~400% combined.

### Disk (`DiskMetricsTrace`)

Per-device samples (one row per tracked device per sample tick):

| Field | Source | Units | Notes |
| ----- | ------ | ----- | ----- |
| `read_bytes_per_sec` | `/proc/diskstats` field 6 (sectors read) × 512 / Δt | B/s | Δt is wall time between consecutive samples |
| `write_bytes_per_sec` | `/proc/diskstats` field 10 (sectors written) × 512 / Δt | B/s | |
| `read_queue_depth` | `/sys/block/<dev>/inflight` | requests | Currently in-flight read requests |
| `write_queue_depth` | `/sys/block/<dev>/inflight` | requests | Currently in-flight write requests |

Per-process samples (one column per `/proc/<pid>/io` counter; rates over the actual Δt since the PID's previous sample):

| FQN | Source | Units | Sees |
| ----- | ------ | ----- | ----- |
| `proc__io_rchar.sum.per_second` | `rchar` delta / Δt | B/s | Bytes returned by `read`-family syscalls on any fd; page-cache hits included; never `mmap` |
| `proc__io_wchar.sum.per_second` | `wchar` delta / Δt | B/s | Bytes accepted by `write`-family syscalls on any fd (pipes and sockets too) |
| `proc__io_read_bytes.sum.per_second` | `read_bytes` delta / Δt | B/s | Bytes fetched from storage, including `mmap` faults that miss the cache; cache hits are 0 |
| `proc__io_write_bytes.sum.per_second` | `write_bytes` delta / Δt | B/s | File pages dirtied (counted at dirtying, not writeback) |
| `proc__io_cancelled_write_bytes.sum.per_second` | `cancelled_write_bytes` delta / Δt | B/s | Dirtied pages discarded before writeback (truncate/delete) |

> [!IMPORTANT]
> The two layers answer different questions; see [metric-model.md, "Per-PID I/O counters"](metric-model.md#per-pid-io-counters-who-records-what) for a measured table. In short: `rchar` − `read_bytes` is the page cache's contribution to `read()` I/O; warm `mmap` reads (e.g. model weights already in the page cache) are invisible to all of them; true disk writes ≈ `write_bytes` − `cancelled_write_bytes`. Traces written before 2026-09-25 carried `read_bytes`/`write_bytes` under the names `proc__io_rchar`/`proc__io_wchar`.

### Permissions for per-PID I/O

`/proc/<pid>/io` is the only `/proc` file the profiler reads that requires elevated permissions. On Linux it's mode `0400` (owner-only) and access additionally goes through `PTRACE_MODE_READ_FSCREDS`: the reader must have the target's uid **and** the target must be *dumpable*, unless the reader holds `CAP_SYS_PTRACE` (plus `CAP_DAC_READ_SEARCH` for the file mode). Yama's `kernel.yama.ptrace_scope` does **not** apply here — it restricts ptrace *attach* only — so no ancestor relationship and no `PR_SET_PTRACER` hint is needed (verified 2026-09-24 with `ptrace_scope = 1`). A process is not dumpable after it executes a setuid/setgid or file-capability binary, or after `prctl(PR_SET_DUMPABLE, 0)`.

```text ln:false
$ ls -la /proc/<pid>/io
-r--------  <user>  <user>  /proc/<pid>/io        ← owner-only, 0400
```

Symptom when permissions are missing: `disk_metrics.pb` is produced, per-device samples are populated normally, but **every per-PID `proc__io_*` rate reads as `0`** (or the profiler logs `EACCES` reading `/proc/<pid>/io`). Per-PID *CPU* and *memory* are unaffected because `/proc/<pid>/stat` and `/proc/<pid>/status` are world-readable.

**Fixes** (any one of):

1. **Run the profiler as the target PID's owner** — same uid, target dumpable. This covers the usual case: your own workload, launched by you or attached by PID, under LEGACY or SIDECAR alike. Changing `kernel.yama.ptrace_scope` makes no difference.

2. **Grant Linux file capabilities to the binary** — add `CAP_DAC_READ_SEARCH` (bypass file mode `0400`) and `CAP_SYS_PTRACE` (satisfy the `PTRACE_MODE_READ_FSCREDS` check):

   ```bash ln:false
   # Apply to the executable that links libcupti_profiler.so
   sudo setcap cap_dac_read_search,cap_sys_ptrace+eip ./build/examples/full_system_profiling

   # Verify
   getcap ./build/examples/full_system_profiling
   # → cap_dac_read_search,cap_sys_ptrace=eip
   ```

   For a Python script, you can't `setcap` the script — capabilities live on the executable, so apply them to the `python` interpreter you invoke (or to a copy of it dedicated to profiling, since this widens that interpreter's privileges):

   ```bash ln:false
   # Make a dedicated copy so you don't widen system Python
   cp $(which python3) ~/bin/python3-profiling
   sudo setcap cap_dac_read_search,cap_sys_ptrace+eip ~/bin/python3-profiling
   ~/bin/python3-profiling examples/full_system_profiling.py
   ```

3. **Run as root** — `sudo ./build/examples/full_system_profiling`. Simple but obviously broad; prefer (2) for production use.

> [!WARNING]
> Capabilities on a binary apply system-wide for everyone who can execute it. Don't `setcap` `/usr/bin/python3` directly — every Python invocation on the host inherits those capabilities. Either copy the interpreter or apply the caps to a single-purpose wrapper.

> [!NOTE]
> `LD_LIBRARY_PATH` is **stripped from the environment** when the kernel loads a binary that has file capabilities (a hardening measure). After `setcap`, you must either install `libcupti_profiler.so` to a system path (`/usr/lib`, `/usr/local/lib`, …) or set `RPATH`/`RUNPATH` on the binary at link time so it can find the library without `LD_LIBRARY_PATH`.

### Sampling frequency guidance

| Profile | Recommended Hz | Notes |
| ------- | -------------- | ----- |
| `system` (CPU + memory) | 100 Hz | Per-process CPU% needs at least ~50 Hz to resolve sub-second bursts; 100 Hz is the sweet spot. |
| `disk` (devices + per-PID I/O) | 10 – 100 Hz | `/proc/diskstats` updates relatively slowly; >100 Hz wastes cycles. |
| `events` | n/a | No periodic sampling — it's an inline log. Just choose `flush_interval_ms`. |

`/proc` parsing is the dominant cost. Empirically, `system` at 100 Hz with 4 tracked PIDs costs <0.1% of one CPU; `disk` at 100 Hz with 2 devices and 4 PIDs is similar.

---

## Design decisions

| Decision | Rationale |
| -------- | --------- |
| Pimpl on `GpuProfiler` and `RegionTracker` | Public header has zero CUDA/CUPTI/protobuf includes. Users compile with any C++17 compiler. |
| `void*` for `cudaStream_t` in public API | Avoids requiring `<cuda_runtime.h>` in the public header |
| Library is pure C++ (`.cpp`, no `.cu`) | No device code in the library. CUDA Runtime API calls work in `.cpp` linked against `cudart`. Only user code needs `nvcc`. |
| No `protobuf::ShutdownProtobufLibrary()` | Process-global operation, unsafe for a library to call. Left to the user if needed. |
| `cuInit(0)` in `Configure()` | Idempotent — safe to call multiple times. |
| Globals eliminated | `g_stopDecode`, `g_stopFlush`, `g_flushIntervalMs` are now `GpuProfiler::Impl` members. Enables multiple profiler instances (one per GPU). |
| Vendored `helper_cupti.h` | Avoids fragile dependency on CUPTI samples install path. Small file (just three error-check macros). |
