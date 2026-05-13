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
│ │  delta calc)   │    │  per-PID rchar/wchar) │   │
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
/proc/<pid>/io            (per-PID rchar/wchar)
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

    void Configure();    // configure all enabled profilers
    void Start();        // start all enabled profilers
    void Stop();         // stop, flush, and write session_metadata.pb
};
```

| Method | Description |
| ------ | ----------- |
| `LoadConfig(path)` | Parse a protobuf text-format `.pbtxt` (`ProfilerSuiteConfig` schema) and apply it to all sub-profilers. Resolves `pid: 0` to the calling process. |
| `LoadConfigFromBytes(buf)` | Same as above but takes a serialized binary `ProfilerSuiteConfig`. Used by language bindings. |
| `Get*Profiler()` | Access individual sub-profilers — needed to grab `EventTracker` references for region annotation. |
| `Configure()` | Calls `Configure()` on every sub-profiler whose `enabled = true`. Disabled profilers are no-ops. |
| `Start()` / `Stop()` | Lifecycle fan-out. `Stop()` also writes `session_metadata.pb` listing the active probes. |

### `ProfilerConfig` (GPU)

```cpp title:"lib/include/cupti_profiler/gpu_profiler.h"
struct ProfilerConfig {
    int deviceIndex = 0;
    uint64_t samplingIntervalNs = 100000;       // 0.1 ms = 10 kHz
    size_t hwBufferSize = 512 * 1024 * 1024;    // 512 MB
    uint64_t maxSamples = 50000;
    std::vector<std::string> metrics;

    uint64_t flushIntervalMs = 10000;           // 0 = no periodic flush
    std::string outputFile;                     // empty = no file output
};
```

| Field | Description |
| ----- | ----------- |
| `deviceIndex` | CUDA device to profile |
| `samplingIntervalNs` | HW counter sampling period in nanoseconds |
| `hwBufferSize` | GPU-side ring buffer size. 512 MB prevents overflow at 10 kHz |
| `maxSamples` | Decode buffer capacity (per decode cycle, not total) |
| `metrics` | CUPTI metric names to collect. Must fit in a single pass |
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
| `Configure(config)` | Initialize CUPTI subsystems. Requires active CUDA context on `config.deviceIndex` |
| `Start()` | Begin PM sampling and background threads |
| `Stop()` | Stop sampling, join threads, resolve regions, write final data to file |
| `DrainSamples()` | Atomically drain all accumulated samples. Safe to call during or after profiling |
| `GetRegionTracker()` | Access the `RegionTracker` for annotating workload phases |
| `GetDeviceName()` | e.g. `"NVIDIA H100 NVL"` |
| `GetChipName()` | e.g. `"GH100"` |
| `GetPeakDramBwGbps()` | Theoretical peak DRAM bandwidth in GB/s |

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

class SystemProfiler {
public:
    void Configure(const SystemProfilerConfig& config);
    void Start();
    void SignalStop();   // non-blocking
    void Stop();         // join + flush + close
};
```

| Field | Description |
| ----- | ----------- |
| `samplingFrequencyHz` | How often `/proc/stat` and friends are polled. 100 Hz is a good default for second-scale workloads; 1000 Hz captures sub-second spikes. |
| `Processes` | PIDs (with optional aliases) to sample per-process. See `TrackedProcess`. |
| `flushIntervalMs` | How often the in-memory sample buffer is serialized to `outputFile`. |
| `outputFile` | Path to the system trace `.pb`. Resolved against `output_dir` when driven by `ProfilerSuite`. |

`SystemProfiler` writes one `SystemMetricsTrace` per flush, each containing a slice of `cpu_system_samples`, `cpu_process_samples`, `memory_system_samples`, and `memory_process_samples`.

### `DiskProfilerConfig` & `DiskProfiler`

```cpp title:"<cupti_profiler/disk_profiler.h>"
struct DiskProfilerConfig {
    uint64_t samplingFrequencyHz = 10;           // 10 Hz default
    std::vector<std::string> devices;            // e.g. {"nvme0n1", "md0"}
    std::vector<TrackedProcess> Processes;       // empty = device-only
    uint64_t flushIntervalMs = 5000;
    std::string outputFile;                      // e.g. "disk_metrics.pb"
};

class DiskProfiler {
public:
    void Configure(const DiskProfilerConfig& config);
    void Start();
    void SignalStop();
    void Stop();
};
```

| Field | Description |
| ----- | ----------- |
| `samplingFrequencyHz` | Polling rate for `/proc/diskstats` and `/sys/block/<dev>/inflight`. Disk counters update relatively slowly — 10 Hz is usually sufficient. |
| `devices` | Block devices to sample. Names match `/sys/block/<name>/`. Use `lsblk` or `cat /proc/diskstats` to enumerate. |
| `Processes` | PIDs to sample for `/proc/<pid>/io` (rchar/wchar). Empty = device-only sampling. |
| `flushIntervalMs` / `outputFile` | Same semantics as `SystemProfilerConfig`. |

> [!NOTE]
> `/proc/<pid>/io` reports cumulative `rchar`/`wchar` (bytes read/written through the syscall layer, including page-cache hits). The profiler converts these into bytes-per-second using inter-sample wall time. This is **not** the same as physical disk traffic — for that, use the per-device samples.

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

gpu {
    enabled: true
    device_index: 0
    sampling_frequency_hz: 10000
    metrics: "sm__cycles_active.avg"
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
}

events {
    enabled: true
    output_file: "events.pb"
}
```

Each block can be omitted or set `enabled: false` to skip that profiler. Each `processes { ... }` entry is independent — `pid` only, or `pid` + `alias`. Empty `processes` = system-wide samples only.

#### Option B — GPU-only with `GpuProfiler`

If you only need GPU counters and don't want a `.pbtxt`, the GPU profiler can be driven directly. This is the path used by `examples/gemm_profiling.cu`.

```cpp title:"GPU-only integration"
#include <cupti_profiler/gpu_profiler.h>
#include <cuda_runtime.h>

int main() {
    cudaSetDevice(0);

    cupti_profiler::ProfilerConfig config;
    config.deviceIndex = 0;
    config.samplingIntervalNs = 100000;  // 10 kHz
    config.outputFile = "my_trace.pb";
    config.flushIntervalMs = 5000;
    config.metrics = {
        "sm__cycles_active.avg",
        "sm__cycles_elapsed.avg",
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

## Output format

A full-suite run produces five `.pb` files under `output_dir`:

| File | Schema | Contents |
| ---- | ------ | -------- |
| `gpu_metrics.pb` | `GpuMetricsTrace` (length-delimited) | GPU PM counter samples (regions live in `events.pb`) |
| `system_metrics.pb` | `SystemMetricsTrace` (length-delimited) | CPU + memory samples (system + per-PID) |
| `disk_metrics.pb` | `DiskMetricsTrace` (length-delimited) | Disk device + per-PID I/O samples |
| `events.pb` | `EventTrace` (length-delimited) | Regions + events, Generic + GPU domains |
| `session_metadata.pb` | `SessionMetadata` (single message, **not** length-delimited) | Manifest of probes, hostname, wall-clock anchor |

### GPU schema

```protobuf title:"proto/gpu_metrics.proto"
message GpuMetricsTrace {
    string device_name = 1;
    string chip_name = 2;
    uint64 sampling_frequency_hz = 3;
    repeated string metric_names = 4;
    repeated GpuMetricSample samples = 5;
    // Field 6 (regions) and field 8 (region_timing_method) were removed —
    // regions now live in events.pb (see proto/events.proto).
    reserved 6, 8;
    double peak_dram_bw_gbps = 7;
    double peak_pcie_bw_bytes_per_sec = 13;
    double peak_nvlink_bw_bytes_per_sec = 14;
    uint64 steady_clock_reference_ns = 9;
    uint64 cupti_reference_ns = 10;
    uint64 wall_clock_epoch_ns = 11;
    repeated GpuFlushStats flush_stats = 12;
}
```

### System schema

```protobuf title:"proto/system_metrics.proto"
message SystemMetricsTrace {
    string hostname = 1;
    uint64 sampling_frequency_hz = 2;
    repeated TrackedProcess tracked_processes = 3;   // PIDs + aliases
    repeated CPUSystemSample    cpu_system_samples    = 4;
    repeated CPUProcessSample   cpu_process_samples   = 5;
    repeated MemorySystemSample memory_system_samples = 6;
    repeated MemoryProcessSample memory_process_samples = 7;
    uint64 steady_clock_reference_ns = 8;
    uint64 wall_clock_epoch_ns       = 9;
    repeated SystemFlushStats flush_stats = 10;
}

message CPUSystemSample  { uint64 timestamp_ns; double total_utilization_pct, user_pct, system_pct, iowait_pct; }
message CPUProcessSample { uint64 timestamp_ns; uint32 pid; double user_pct, system_pct, iowait_pct; }
message MemorySystemSample { uint64 timestamp_ns; uint64 total_bytes, used_bytes, available_bytes, buffers_bytes, cached_bytes; }
message MemoryProcessSample{ uint64 timestamp_ns; uint32 pid; uint64 rss_bytes, vms_bytes, shared_bytes; }
```

### Disk schema

```protobuf title:"proto/disk_metrics.proto"
message DiskMetricsTrace {
    string hostname = 1;
    uint64 sampling_frequency_hz = 2;
    repeated string tracked_devices = 3;
    repeated TrackedProcess tracked_processes = 4;
    repeated DiskDeviceSample  device_samples  = 5;
    repeated DiskProcessSample process_samples = 6;
    uint64 steady_clock_reference_ns = 7;
    uint64 wall_clock_epoch_ns       = 8;
    repeated DiskFlushStats flush_stats = 9;
}

message DiskDeviceSample  { uint64 timestamp_ns; string device_name; double read_bytes_per_sec, write_bytes_per_sec; uint32 read_queue_depth, write_queue_depth; }
message DiskProcessSample { uint64 timestamp_ns; uint32 pid; double read_bytes_per_sec, write_bytes_per_sec; }
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
}
```

`session_metadata.pb` holds a single, **non**-length-delimited `SessionMetadata` message — `visualize_all.py` reads it first to discover which sibling `.pb` files to load.

### Length-delimited streaming format

Each per-probe `.pb` file contains one or more length-delimited messages of its corresponding trace type:

```text ln:false
[varint: message_size][serialized Trace]
[varint: message_size][serialized Trace]
...
[varint: message_size][serialized Trace]  ← final chunk
```

- Periodic flushes write incremental messages (a slice of samples for that window)
- For GPU traces: regions only appear in the final message (resolved at `Stop()`)
- For system/disk traces: every message contains complete `tracked_processes` so each is interpretable in isolation
- `flush_stats` carry the **previous** flush's byte size and interval (a flush can't include its own size)
- All visualization tools read and merge messages automatically

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
| `user_pct` | `/proc/<pid>/stat` utime delta | % of one CPU | Sum of utime across threads |
| `system_pct` | `/proc/<pid>/stat` stime delta | % of one CPU | Sum of stime across threads |
| `iowait_pct` | `/proc/<pid>/stat` delayacct_blkio_ticks delta | % of one CPU | Requires kernel `CONFIG_TASK_DELAY_ACCT`; 0 if unavailable |
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

Per-process samples:

| Field | Source | Units | Notes |
| ----- | ------ | ----- | ----- |
| `read_bytes_per_sec` | `/proc/<pid>/io` rchar delta / Δt | B/s | Includes page-cache hits — *not* physical disk reads |
| `write_bytes_per_sec` | `/proc/<pid>/io` wchar delta / Δt | B/s | Bytes the process *asked* to write, regardless of where they ended up |

> [!IMPORTANT]
> Per-PID I/O uses `rchar`/`wchar` (the syscall-layer counters), not `read_bytes`/`write_bytes` (the block-layer counters). The syscall counters always work; the block counters are 0 for buffered I/O. The visualizer plots both per-device and per-PID rows, so the gap between them is exactly the page-cache contribution.

### Permissions for per-PID I/O

`/proc/<pid>/io` is the only `/proc` file the profiler reads that requires elevated permissions. On Linux it's mode `0400` (owner-only) and access additionally goes through `PTRACE_MODE_READ_FSCREDS`, which on Ubuntu's default `kernel.yama.ptrace_scope = 1` restricts even same-user reads to direct ancestors of the target process.

```text ln:false
$ ls -la /proc/<pid>/io
-r--------  <user>  <user>  /proc/<pid>/io        ← owner-only, 0400
```

Symptom when permissions are missing: `disk_metrics.pb` is produced, per-device samples are populated normally, but **every per-PID `read_bytes_per_sec` and `write_bytes_per_sec` reads as `0`** (or the profiler logs `EACCES` reading `/proc/<pid>/io`). Per-PID *CPU* and *memory* are unaffected because `/proc/<pid>/stat` and `/proc/<pid>/status` are world-readable.

**Fixes** (any one of):

1. **Run the profiler as the target PID's owner with a parent-ancestor relationship** — e.g. spawn the workload from inside the profiler binary, or set `kernel.yama.ptrace_scope = 0` system-wide. This is the simplest path when you control how the workload is launched.

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
