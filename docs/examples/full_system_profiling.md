# `full_system_profiling`

Full-machine example. Drives `ProfilerSuite` — the orchestrator over `GpuProfiler`,
`SystemProfiler`, and `DiskProfiler` — from a single `.pbtxt` config file.

Source: [`examples/full_system_profiling.cu`](../../examples/full_system_profiling.cu)

## What it does

Runs **the same GEMM + vecAdd workload** as [`gemm_profiling`](gemm_profiling.md), but
with three profilers collecting in parallel:

- **GPU** — CUPTI PM sampling at the rate set in the config (default 10 kHz).
  SM cycles, warp occupancy, DRAM throughput, PCIe read/write, NVLink rx/tx.
- **System** — `/proc`-based CPU + memory sampling (system-wide and per-PID)
  at a separate rate (default 100 Hz).
- **Disk** — `/proc/diskstats` + `/sys/block/*/inflight` + `/proc/[PID]/io` for
  per-device throughput and per-PID IO (default 100 Hz).

Each profiler runs two threads (sample + flush) and writes its own length-delimited
`.pb` file. Timestamps are anchored to `steady_clock` with a sync anchor so GPU,
CPU, and disk data line up on one timeline.

## Profiler setup (external `.pbtxt`)

The program takes only one flag — where to find the config:

```bash
./examples/full_system_profiling                       # uses configs/example.pbtxt
./examples/full_system_profiling -c my_config.pbtxt
```

The default path is resolved at runtime from `__FILE__` (via the
`SOURCE_FILE_DIR` / `DEFAULT_CONFIG_PATH` macros at the top of
[`examples/full_system_profiling.cu`](../../examples/full_system_profiling.cu)),
so running the binary from any working directory picks up the same
`configs/example.pbtxt` shipped with the source tree.

Everything else — device index, metrics, sampling rates, tracked PIDs, tracked
disks, output files, output directory — lives in the config file and is changeable
without a rebuild. See [`configs/example.pbtxt`](../../configs/example.pbtxt).

Sentinel: `pids: 0` is resolved to the current process PID at runtime, so you don't
need to know it in advance.

## Run

```bash
cd build
CUDA_VISIBLE_DEVICES=1 ./examples/full_system_profiling
```

Output: three `.pb` files (`gpu_metrics.pb`, `system_metrics.pb`, `disk_metrics.pb`)
in the current directory, or under `output_dir` if set in the config.

## Visualize

```bash
python tools/visualize_all.py \
    --gpu gpu_metrics.pb \
    --system system_metrics.pb \
    --disk disk_metrics.pb \
    -o full_profile.png
```

Panels (omitted when the corresponding trace is absent):
region timeline → SM utilization → active warps/cycle → DRAM GB/s →
PCIe GB/s → NVLink GB/s → CPU % → per-PID CPU % → system memory → per-PID RSS
→ disk throughput → per-PID disk IO → disk queue depth.

## Permission notes

- `/proc/[PID]/io` needs same-UID or `CAP_SYS_PTRACE`. The profiler warns once
  and skips per-process disk IO if unavailable. Fix with
  `sudo setcap cap_sys_ptrace=ep <binary>` or
  `sudo sysctl -w kernel.yama.ptrace_scope=0`.
- CUPTI PM sampling typically needs the NVIDIA driver's profiling restriction
  lifted (`nvidia-smi -pm 1` or `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`).

## When to use

- Correlating a GPU workload with the CPU, memory, and disk activity around it.
- Studying inference/serving pipelines where host-side overheads (tokenization,
  data loading, PCIe transfers) matter.
- Giving non-developers a knob (`.pbtxt`) to change what's collected without
  rebuilding.

For GPU-only runs with less machinery, use [`gemm_profiling`](gemm_profiling.md).
