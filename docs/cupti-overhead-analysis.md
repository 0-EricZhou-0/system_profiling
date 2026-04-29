---
title: "CUPTI Overhead Analysis: Performance Costs of GPU Profiling"
tags:
  - cupti
  - nvidia
  - profiling
  - gpu
  - performance
  - overhead
---

# CUPTI Overhead Analysis

CUPTI (CUDA Profiling Tools Interface) is NVIDIA's library for building GPU profiling and tracing tools. It underlies Nsight Systems, Nsight Compute, PyTorch Profiler (via Kineto), and TensorFlow Profiler. This document catalogs the overhead characteristics of each CUPTI subsystem, drawn from official NVIDIA documentation, community benchmarks, and the PM Sampling implementation in this repository.

## How this repository uses CUPTI

This project implements **PM (Performance Monitor) Sampling** — one of the lower-overhead CUPTI modes — to collect hardware counter metrics at a configurable frequency without kernel replay or serialization.

### Architecture overview

```text ln:false
GPU Workload (cuBLAS SGEMM + vectorAdd)
        ↓
CUPTI PM Sampling @ 10 kHz (GPU HW ring buffer, 512 MB)
        ↓
Background Decode Thread (drains every 5 ms)
        ↓
cuptiPmSamplingDecodeData() → raw counter samples
        ↓
cuptiProfilerHostEvaluateToGpuValues() → metric values
        ↓
SamplerRange collection (host memory, mutex-protected)
        ↓
Region timestamps resolved (CUDA events + CUPTI reference clock)
        ↓
WriteProtobuf() → gpu_metrics.pb
        ↓
visualize_single.py → gpu_metrics.png (matplotlib)
```

### Key API sequence

```cpp title:"cupti_pm_sampling.cu" fold:"CUPTI PM Sampling API flow"
// 1. Initialize profiler subsystem
cuptiProfilerInitialize();
cuptiProfilerHostInitialize();

// 2. Query device capabilities
cuptiDeviceGetChipName();
cuptiPmSamplingGetCounterAvailability();

// 3. Configure metrics (must fit single pass)
cuptiProfilerHostConfigAddMetrics();
cuptiProfilerHostGetConfigImage();
cuptiProfilerHostGetNumOfPasses();  // assert == 1

// 4. Enable and configure sampling
cuptiPmSamplingEnable();
cuptiPmSamplingSetConfig();         // interval_ns, GPU_TIME_INTERVAL trigger
cuptiPmSamplingCreateCounterDataImage();

// 5. Sample loop
cuptiPmSamplingStart();
// ... decode thread calls cuptiPmSamplingDecodeData() every 5 ms ...
cuptiPmSamplingStop();

// 6. Evaluate
cuptiProfilerHostEvaluateToGpuValues();  // raw counters → metric doubles
```

### Metrics collected

| Metric                                                        | Category        | Meaning                          |
| ------------------------------------------------------------- | --------------- | -------------------------------- |
| `sm__cycles_active.avg`                                       | SM Utilization  | Average SM cycles busy           |
| `sm__cycles_active.max`                                       | SM Utilization  | Busiest SM cycles                |
| `sm__cycles_elapsed.avg`                                      | SM Utilization  | Reference elapsed cycles         |
| `sm__warps_active.avg`                                        | Occupancy       | Average active warps per cycle   |
| `sm__warps_active.max`                                        | Occupancy       | Max active warps (busiest SM)    |
| `dram__read_throughput.avg.pct_of_peak_sustained_elapsed`     | Memory BW       | Read BW as % of peak             |
| `dram__read_throughput.max.pct_of_peak_sustained_elapsed`     | Memory BW       | Read BW max % across partitions  |
| `dram__write_throughput.avg.pct_of_peak_sustained_elapsed`    | Memory BW       | Write BW as % of peak            |
| `dram__write_throughput.max.pct_of_peak_sustained_elapsed`    | Memory BW       | Write BW max % across partitions |

### Configuration parameters

```text ln:false
CLI flags:
  -d, --device <idx>        GPU device index          (default: 0)
  -i, --interval <ns>       Sampling interval in ns   (default: 100,000 = 0.1 ms = 10 kHz)
  -o, --output <file>       Output protobuf file      (default: gpu_metrics.pb)

Hardcoded:
  Max samples:              50,000
  HW buffer size:           512 MB
  Decode poll interval:     5 ms
  Trigger mode:             GPU_TIME_INTERVAL (requires Ampere+)
  Min compute capability:   7.5 (Turing+)
```

> [!IMPORTANT]
> The single-pass constraint is enforced at startup. If the requested metrics cannot be collected in one pass, the program exits. This avoids kernel replay overhead entirely.

---

## CUPTI overhead by feature

Not all CUPTI modes are equal. Overhead spans three orders of magnitude depending on which subsystem is active.

### Activity tracing (lowest overhead)

Activity tracing collects timestamped records for CUDA API calls, kernel launches, memory copies, and other runtime events. Each event costs a few microseconds to log.

- **Typical overhead: 2–5%** of application runtime
- CUPTI creates a dedicated **worker thread** that handles buffer delivery and host-device synchronization, controllable via `cuptiActivityFlushPeriod()`
- The newer **Hardware Event System (HES)** on Blackwell GPUs provides more accurate timestamps with further reduced overhead compared to traditional software instrumentation

> [!TIP]
> Use `cuptiActivityEnableDriverApi()` and `cuptiActivityEnableRuntimeApi()` to trace only the APIs you care about, rather than enabling everything.

### Callback API (low–moderate overhead)

The callback API invokes user-registered functions on CUDA API entry/exit. The mechanism itself is lightweight — overhead depends on what the callback does.

> [!WARNING]
> NVIDIA documentation emphasizes: "the client should return as quickly as possible from these callbacks." A blocking callback stalls the application thread directly.

### PM Sampling (low–moderate overhead) — used in this repo

PM Sampling reads hardware performance counters at a fixed interval without replaying or serializing kernels. The overhead profile is:

- **No kernel serialization** — workload runs at full speed
- **No kernel replay** — metrics are sampled, not exhaustively measured
- **Background decode thread** adds CPU overhead proportional to sampling frequency
- **GPU-side cost** is the counter read itself, which is minimal at reasonable intervals

The main risk is **hardware buffer overflow** if the decode thread cannot drain samples fast enough. This repo mitigates that with a 512 MB buffer and 5 ms polling.

```text ln:false
Sampling interval vs. overhead tradeoff:

  Interval       Frequency    Overhead estimate
  ─────────────  ───────────  ─────────────────
  1,000,000 ns   1 kHz        Negligible
    100,000 ns   10 kHz       Low (this repo's default)
     10,000 ns   100 kHz      Moderate — watch for buffer pressure
      1,000 ns   1 MHz        High — likely buffer overflow
```

### PC Sampling (moderate–high overhead)

PC (Program Counter) sampling periodically samples the instruction pointer of running warps. Overhead scales with sampling frequency and GPU architecture.

| GPU generation | Driver      | Sampling rate | Measured overhead  |
| -------------- | ----------- | ------------- | ------------------ |
| Pascal (P4000) | CUDA 10.1   | Minimum       | **20×–50×**        |
| Volta+         | 418.67+     | Same workload | **2×–5×**          |
| Ampere+        | Modern      | Low (period 20–31) | **1–5%**      |
| Ampere+        | Modern      | Medium (period 10–19) | **5–15%**  |
| Ampere+        | Modern      | High (period 5–9) | **15–30%**    |

> [!WARNING]
> Earlier CUDA versions serialized execution when PC sampling was enabled, causing dramatically higher overhead. Always use recent drivers with Volta+ GPUs for PC sampling.

### Event/metric collection via Profiling API (very high overhead)

This is the most expensive mode, used by Nsight Compute for detailed per-kernel analysis.

- **Kernels are serialized** — each launch blocks until the kernel finishes and CUPTI post-processes the results
- **Kernel replay** — a single kernel can be replayed **up to 46 times** to collect all hardware counter sets
- **Software-patched metrics** incur the highest cost, as they modify the kernel binary and inject additional instructions
- **First-kernel overhead** — a "relatively high one-time overhead" for generating the metric configuration per context

```text ln:false
Effective slowdown for full metric collection:

  Scenario                          Slowdown
  ─────────────────────────────     ────────────────
  Single metric group, one pass     ~2×–5×
  Full metric set, many passes      ~10×–46×
  With software-patched metrics     Orders of magnitude
```

> [!CAUTION]
> Linaro Forge documentation warns that full metric collection "may have a significant impact on the target program, potentially resulting in orders of magnitude slowdown." Never enable this in production.

### Range Profiling API

Scoped version of event/metric collection — same overhead characteristics but limited to user-defined code regions. Still serializes and replays kernels within the range.

---

## Memory overhead

CUPTI imposes non-trivial memory costs independent of runtime overhead.

| Scenario                            | Official estimate | Observed (community) |
| ----------------------------------- | ----------------- | -------------------- |
| Per CUDA context (tracing)          | ~3 MB             | ~45 MB               |
| Multi-core subscription (8 cores)   | —                 | ~1.3 GB total        |
| Cause                               | CUPTI stores cubin images and module metadata per context |

> [!NOTE]
> NVIDIA acknowledged the memory overhead issue in September 2022 and indicated plans to reduce the footprint in future releases. The gap between official estimates and real-world measurements is significant.

---

## Persistent overhead after profiling ends

A critical finding for ML training workloads:

1. PyTorch Profiler (via Kineto) historically **did not call `cuptiFinalize()`** when profiling stopped
2. CUPTI hooks remained active, causing a **15% persistent training slowdown** even after the profiler UI showed "stopped"
3. Root cause: `cuptiFinalize()` was disabled because it caused crashes with CUDA Graphs (fixed in CUDA 12.6)
4. **Resolution**: PyTorch PR #146604 enabled teardown by default for CUDA 12.6+

```text ln:false
Workaround for older PyTorch versions:

  export TEARDOWN_CUPTI=1    # forces cuptiFinalize() on profiler stop
```

> [!WARNING]
> If you profile a training run and notice performance does not return to baseline after stopping the profiler, check whether `cuptiFinalize()` was called. This is a known issue across frameworks that use CUPTI.

---

## First-iteration / initialization overhead

- CUPTI instrumentation causes **high overhead on the first kernel launch** per context due to JIT compilation of instrumented code and module loading
- One user reported **~4 seconds** of `CUpti_ActivityOverhead` before actual kernel activities appeared
- Subsequent iterations have much lower overhead
- This repo mitigates this by including a "warmup" region in the workload phases

---

## Overhead self-tracking

CUPTI provides built-in overhead measurement via `CUpti_ActivityOverhead3` records:

```cpp title:"Overhead record fields"
struct CUpti_ActivityOverhead3 {
    CUpti_ActivityOverheadKind overheadKind;  // CUPTI, driver, or compiler
    uint64_t start;                           // Overhead start timestamp (ns)
    uint64_t end;                             // Overhead end timestamp (ns)
    // Also tracks: buffer requests, command buffer full events, UVM init
};
```

This lets profiling tools **subtract CUPTI's own overhead** from reported timelines — useful for validating that measurements are not dominated by the observer effect.

---

## Comparison with alternative profiling approaches

| Approach                          | Typical overhead | Best for                         |
| --------------------------------- | ---------------- | -------------------------------- |
| CUPTI Activity Tracing            | 2–5%             | Timeline / trace collection      |
| CUPTI PM Sampling (this repo)     | Low at 10 kHz    | Continuous HW counter monitoring |
| CUPTI PC Sampling (low rate)      | 1–5%             | Instruction-level hotspots       |
| CUPTI PC Sampling (high rate)     | 2×–5× (Volta+)   | Detailed warp analysis           |
| CUPTI Metric Collection (full)    | 10×–46×+          | Per-kernel deep analysis         |
| Nsight Systems (full trace)       | 2×–10×            | Comprehensive system analysis    |
| GPUprobe (eBPF + CUPTI)           | <4%              | Production continuous profiling  |
| USDT probes (no active trace)     | ~0% (NOP)        | Production monitoring            |
| Polar Signals (eBPF)              | "close to noise" | Production continuous profiling  |

---

## Best practices for minimizing overhead

### Buffer management

- Use **1–10 MB activity buffers** — smaller causes frequent allocation; larger delays delivery
- Enable `CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER` to reduce contention in multi-threaded apps
- Tune `cuptiActivityFlushPeriod()` to balance latency vs. overhead

### Selective collection

- Trace only the APIs you need via selective enable functions
- Prefer single-pass metric configurations (enforced in this repo)
- Use PM Sampling over full metric collection when trend data suffices

### Lifecycle management

- **Always call `cuptiFinalize()`** after profiling to remove persistent hooks
- For CUDA 12.6+, PyTorch handles this automatically
- For older versions, set `TEARDOWN_CUPTI=1`

### Sampling rate selection

```text ln:false
Rule of thumb for PM Sampling interval:

  Goal                              Recommended interval
  ──────────────────────────────    ────────────────────
  Low-overhead monitoring           1,000,000 ns (1 kHz)
  Balanced resolution/overhead      100,000 ns (10 kHz) ← this repo
  High-resolution analysis          10,000 ns (100 kHz)
  Maximum detail (short runs)       1,000 ns (1 MHz) — watch buffer overflow
```

---

## Design decisions in this repository

| Decision                                  | Rationale                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| PM Sampling over metric collection        | Avoids kernel serialization and replay                                       |
| Single-pass enforcement                   | Eliminates multi-pass overhead entirely                                      |
| `GPU_TIME_INTERVAL` trigger mode          | Stable frequency independent of workload, requires Ampere+                   |
| 512 MB hardware buffer                    | Prevents overflow at 10 kHz sampling with decode thread polling every 5 ms   |
| Background decode thread                  | Decouples buffer draining from workload execution                            |
| Event-based region timing                 | CUDA events + CUPTI reference clock avoids conflicts with active PM sampling |
| Protobuf serialization                    | Compact binary format, language-neutral, handles 50k samples efficiently     |
| 100 µs moving-average smoothing           | Reduces sampling noise while preserving workload dynamics in visualization   |

---

## References

- [CUPTI Official Documentation](https://docs.nvidia.com/cupti/main/main.html)
- [NVIDIA CUPTI Developer Page](https://developer.nvidia.com/cupti)
- [GPU Profiling Under the Hood — eunomia Survey (2025)](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)
- [CUDA GPU Profiling and Tracing Tutorial — eunomia](https://eunomia.dev/others/cuda-tutorial/08-profiling-tracing/)
- [Linaro Forge — CUPTI Performance Impact](https://docs.linaroforge.com/24.1/html/forge/map/cuda_profiling/performance_impact.html)
- [NVIDIA Forums — PC Sampling Large Slowdowns](https://forums.developer.nvidia.com/t/pc-sampling-leads-to-large-slow-downs-in-execution-time/80019)
- [NVIDIA Forums — CUPTI Instrumentation Overhead](https://forums.developer.nvidia.com/t/cupti-instrumentation-overhead/157045)
- [NVIDIA Forums — CUPTI Memory Overheads](https://forums.developer.nvidia.com/t/cupti-memory-overheads/221956)
- [PyTorch Issue #144455 — Close CUPTI After Profiling](https://github.com/pytorch/pytorch/issues/144455)
- [PyTorch PR #146604 — Enable CUPTI Teardown](https://github.com/pytorch/pytorch/pull/146604)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Continuous NVIDIA CUDA Profiling in Production — Polar Signals](https://www.polarsignals.com/blog/posts/2025/10/22/gpu-profiling)
- [LIBNVCD Multi-GPU Profiler — LLNL](https://www.osti.gov/servlets/purl/1874871)
- [`CUpti_ActivityOverhead3` Structure Reference](https://docs.nvidia.com/cupti/api/structCUpti__ActivityOverhead3.html)
