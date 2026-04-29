# `gemm_profiling`

GPU-only example. Profiles a cuBLAS GEMM + memory-bound `vecAdd` workload using
`GpuProfiler` directly, no CPU/Disk collection, no config file.

Source: [`examples/gemm_profiling.cu`](../../examples/gemm_profiling.cu)

## What it does

Runs a scripted workload on a single CUDA stream while PM sampling is active:

1. **GEMM ramp** — `cublasSgemm` at sizes 512 → 1024 → 2048 → 4096 → 2048 → 1024 → 512.
   Iteration counts tuned so each phase takes roughly the same wall time.
2. **Idle dwell** — another 512×512 phase labeled `idle 512` to see the tail.
3. **`vecAdd` (mem-bound)** — 200 iterations on a 1 GB working set; pure memory-bound
   phase with no arithmetic intensity.

Each phase is bracketed with `RegionTracker::Begin(label) / End(idx)` so it shows up
as a colored band in the visualization.

## Profiler setup (hardcoded in `main()`)

| Field | Value |
|---|---|
| Device | `-d` flag, default 0 |
| Sampling frequency | `-f` flag, default 10 kHz |
| Output file | `-o` flag, default `gpu_metrics.pb` |
| Metrics | 9 — SM cycles active/elapsed, warps active, DRAM read/write throughput (avg + max) |

Change the metric list by editing the `config.metrics = { ... }` initializer in
`examples/gemm_profiling.cu` and rebuilding.

## Run

```bash
cd build
./examples/gemm_profiling                          # default 10kHz on device 0
./examples/gemm_profiling -d 1 -f 20000            # 20kHz on device 1
./examples/gemm_profiling -o runs/my_trace.pb
```

Output: one `.pb` file with the GPU trace.

## Visualize

```bash
python tools/visualize_single.py gpu_metrics.pb -o gpu_profile.png
```

(`visualize_single.py` — not `visualize_all.py` — since there's no system/disk data.)

## When to use

- Quick GPU-only studies (kernel micro-benchmarks, metric exploration).
- Regression runs where you don't need CPU/Disk context.
- Learning what `RegionTracker` does without the config-file indirection.

For a machine-wide correlated trace, use [`full_system_profiling`](full_system_profiling.md)
instead.
