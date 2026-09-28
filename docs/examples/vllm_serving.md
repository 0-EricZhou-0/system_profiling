# `vllm_serving_profiling.py` — a live vLLM server, process by process

Source: [`examples/vllm_serving_profiling.py`](../../examples/vllm_serving_profiling.py)
· panel layout: [`configs/vllm_serving_panels.pbtxt`](../../configs/vllm_serving_panels.pbtxt)

A `vllm serve` process is a tree: the API server you start, the
`EngineCore` it forks (which renames itself to `VLLM::EngineCore` after the
fork), a multiprocessing helper, and during startup a stream of short-lived
workers (compile workers, `ldconfig`, `ninja`, ...). This example traces
every one of them, from spawn to shutdown, from a small launcher.

![vLLM serving profile](../images/vllm_serving.png)

## What it does

1. Builds a `ProfilerSuite` with the **System and Disk probes in SIDECAR
   mode** (the samplers run in the `cupti-profiler-sidecar` child, not in
   the launcher), **descendant tracking on, recursive, 100 ms scan**, and an
   Events probe for the phase markers. With `--gpu`, also the GPU probe
   (see [GPU](#gpu-device-wide-counters-from-the-launcher)).
2. Starts `vllm serve <model>` and **immediately** calls
   `suite.add_tracked_process(pid, "vllm", track_descendants=True)`, before
   the server is ready, so the startup workers are found while they exist.
3. Waits for `/v1/models`, then drives six batches of completions, each for
   1/6 of `--load-seconds` (default 90 s), at concurrency 4, 16, 64, 64,
   16, 4. Every request has a fresh random prompt of 32–3000 words (so the
   prefix cache cannot short-cut it) and a fixed output of 16, 64 or 256
   tokens (`ignore_eos`). Each batch is a region in the trace.
4. Sends `SIGTERM` to the server, stops the suite, prints the trace's
   **process table**, and renders the figure with
   [`tools/visualize_all.py`](../tools/README.md) and the example's panel
   layout.

## Run

```bash
# the library importable (pip install . , or PYTHONPATH=build/python),
# and vLLM installed somewhere
python examples/vllm_serving_profiling.py --vllm /path/to/venv/bin/vllm --gpu
```

Every machine- or user-specific value is an argument:

| Argument | Default | What |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-0.8B` | model to serve |
| `--vllm` | `vllm` (on `PATH`) | the `vllm` executable |
| `--host`, `--port` | `127.0.0.1`, `8000` | where the server listens |
| `--output-dir` | `vllm_serving_profile` | trace directory; the figure is `<dir>/vllm_serving.png` unless `--png` |
| `--system-hz`, `--disk-hz` | 100, 100 | sampling rates (the library's defaults) |
| `--scan-interval-ms` | 100 | descendant tracking scan interval |
| `--flush-ms` | 5000 | flush interval of every probe |
| `--disk-device` | every whole block device in `/sys/block` | devices for the per-device panel (repeatable) |
| `--launcher-cpus` | not pinned | pin this launcher, which hosts the GPU probe, e.g. `8-15`; keep it off the L3 cache (CCD) of vLLM's cores, see [GPU probe cost and placement](../system-guide.md#gpu-probe-cost-and-placement) |
| `--gpu`, `--gpu-device`, `--gpu-hz` | off, 0, 1000 | GPU PM sampling from the launcher (the library's default is 100 Hz; 100–1000 Hz cost the same) |
| `--load-seconds` | 90 | duration of the request load |
| `--ready-timeout` | 900 | seconds to wait for `/v1/models` |
| after `--` | — | passed to `vllm serve` unchanged, e.g. `-- --max-model-len 4096` |

**What you set in the environment** (the example does not):

- **Where the weights are.** `vllm serve` downloads into the Hugging Face
  cache unless it finds them. For a shared, pre-populated cache, point the
  hub at it: `HF_HUB_CACHE=<dir that contains models--Qwen--Qwen3.5-0.8B>`
  (note: `HF_HOME=<dir>` looks in `<dir>/hub`, which is not the same
  directory), and `HF_HUB_OFFLINE=1` to forbid downloads.
- **CUDA.** Whatever vLLM needs, and for `--gpu` the CUPTI the library was
  built against on `LD_LIBRARY_PATH`. On a node whose driver is older than
  the CUDA toolkit, the toolkit's `compat` directory goes first on
  `LD_LIBRARY_PATH` (forward compatibility).
- The launcher's Python needs only the library (plus numpy and matplotlib
  for the figure); vLLM runs as its own executable, so its environment can
  be a different one.

## Reading the figure

Top to bottom: events (spawn, ready, shutdown) and the load batches as
regions; GPU (with `--gpu`); whole-host CPU; **per-process CPU and resident
memory**; per-device disk bandwidth; **per-process I/O** at the syscall layer
(`rchar` solid, `wchar` dashed) and the storage layer (`read_bytes` solid,
`write_bytes` dashed, `cancelled_write_bytes` dotted), each with its
cumulative companion.

Each process keeps one colour in every panel. Its label is its alias from the
process table: the root is `vllm`; processes found by descendant tracking are
`vllm/<comm>` **and say which tracked parent they were found under**
(`child of <pid>`). EngineCore appears under its renamed comm,
`vllm/VLLM::EngineCor` (comm is 15 characters), because the alias follows
renames; its rename history is in the table below.

### The run in the figure

`Qwen/Qwen3.5-0.8B` on one H100 NVL, vLLM 0.29 from PyPI, `--gpu`, the
defaults otherwise (System and Disk 50 Hz, GPU 500 Hz, 100 ms scan),
`-- --max-model-len 4096 --gpu-memory-utilization 0.80`, warm compile caches.
The server was ready 87 s after spawn and served 2412 requests in the 90 s of
load, none failed. The node was not otherwise idle: other jobs kept the
whole-host CPU panel at 5–20% busy (vLLM itself accounts for ~2 of the 128
CPUs). What the example printed at the end, the trace's process table (times
from the start of the trace; CPU = head + samples + exit tail):

| pid | parent | kind | comm (history) | start s | end s | CPU s |
|---|---|---|---|---|---|---|
| 1882216 | 1882197 | root | `vllm` | 0.0 | 183.8 | 64.36 |
| 1882384 | 1882216 | discovered | `python3.12` | 14.4 | 16.6 | 2.21 |
| 1882386 | 1882216 | discovered | `python3.12` | 14.5 | 16.7 | 2.40 |
| 1882478 | 1882216 | discovered | `python3.12` | 25.0 | 183.4 | 0.04 |
| 1882479 | 1882216 | discovered | `python3.12` → `VLLM::EngineCor` | 25.0 | 183.0 | 137.84 |
| 1882573 | 1882479 | discovered | `python3.12` | 33.9 | 36.6 | 2.66 |
| 1882575 | 1882479 | discovered | `python3.12` | 33.9 | 36.5 | 2.73 |

The **root** (1882216, `vllm`) is the API server; kind `root` because the
launcher listed it. **EngineCore** (1882479) was found as `python3.12` 25.0 s
in, before it renamed itself, and its alias followed the rename. The
**multiprocessing helper** (1882478) lives for the whole run at ~0 CPU. The
rest are **startup transients**: two pairs of short-lived Python workers
(~2.5 s of CPU each; one pair under the API server, one under EngineCore).
Processes that lived less than one scan interval can be missing: in this run
the oracle below saw 5 such (`file`, `ninja`, `tileiras` and two of
EngineCore's brief forks), each alive at most 80 ms.

The panels line up with vLLM's own log (times from the start of the trace):
EngineCore starts at 25.0 s and loads the weights at ~38 s (the step in its
cumulative `rchar`; vLLM: "Loading weights took 0.52 seconds"); its profiling
warmup and two CUDA graph captures (finished at 56 s and 61 s) are the GPU
activity during startup and its bursts of several cores; its init finishes at
65 s, and the API server, busy at ~100% of a core through most of startup,
spends ~20 s more before it answers `/v1/models` at 87 s. Under load both run
near one core each; resident memory reaches ~3.1 GiB (EngineCore) and
~2.3 GiB (API server). (`torch.compile` was a cache hit here, 0.3 s; with cold
caches startup takes minutes and spawns many more short-lived compiler
processes, each traced if it lives ≥ 100 ms.)

### Checked on seven runs

Six runs at the earlier defaults (System and Disk 100 Hz; two without
`--gpu`, then two pairs alternating) and the run in the figure, each with a
test-only oracle polling the run's cgroup every 10 ms and the launcher's
`getrusage(RUSAGE_CHILDREN)`:

| | result |
|---|---|
| Completeness: every process the oracle saw alive ≥ 100 ms is in the trace (both probes) | **7/7 runs** (7–8 such processes per run, 0 missed); the 3–9 missed per run were each seen alive ≤ 80.5 ms |
| Per-process CPU of the whole tree (Σ head + samples + tails) vs `getrusage(RUSAGE_CHILDREN)` once vLLM was reaped | trace **15–74 ms below** out of 196–221 s (0.008–0.035%; the figure's run: 74 ms of 212 s). What the trace cannot hold: the root's CPU after its last sample (a root has no exit tail: its parent, the launcher, is not tracked), and the CPU of the missed sub-100 ms processes |
| Descendant tracking scan (the trace's `DiscoveryStats`) | p50 0.72–1.25 ms, p99 1.7–2.8 ms, 6–10 processes discovered, all seen exiting (the figure's run: p50 0.92 ms, p99 2.5 ms, 6) |
| The sidecar's own CPU (exact, from `getrusage` across `stop()`) | At 100 Hz: **8.6–14.6% of one core** over the run, unpinned (8.6% on a quiet node without `--gpu`; 13.2–14.6% while the node was 40–70% busy or the launcher ran the GPU probe beside it). At the 50 Hz default, the figure's run: **8.3%** (8.8% under load, with `--gpu`). What each rate costs, measured: [Sampling frequency guidance](../system-guide.md#sampling-frequency-guidance). Pin it with `sidecar_cpus` to keep it off the server's cores |

## GPU: device-wide counters from the launcher

CUPTI PM Sampling runs in the launcher's own CUDA context, but the counters it
samples are the **device's**: SM activity, warps and DRAM throughput include
every context on the GPU, vLLM's among them. Measured in this example's runs:

| window | SM active (% of peak, mean) — run in the figure (500 Hz) | earlier run (1000 Hz) | earlier run (1000 Hz) |
|---|---|---|---|
| startup (spawn → ready) | 0.9 | 0.9 | 0.9 |
| load, concurrency 4 / 16 / 64 / 64 / 16 / 4 | 23.9 / 22.0 / 29.2 / 28.7 / 23.0 / 23.0 | 23.6 / 22.2 / 31.0 / 32.2 / 18.7 / 17.5 | 24.8 / 23.8 / 32.0 / 33.1 / 25.0 / 24.8 |
| idle, after the load | 0.0 | 0.0 | 0.0 |

SM activity follows vLLM's load, so **from a launcher the GPU panels are
meaningful, device-wide**. It did not conflict with vLLM: every run started
and served every request. What it costs, and where to put the launcher
(off the L3 cache of vLLM's cores, `--launcher-cpus`): [GPU probe cost and
placement](../system-guide.md#gpu-probe-cost-and-placement). Before
2026-09-28 the probe's collection loop re-initialized an 817 MB buffer
every ~60 ms, and at 500 Hz vLLM served 6.9% fewer requests per second;
that loop is gone. `--gpu` is off by default. Comparing runs with and
without `--gpu`: the probe's CUDA context itself makes a running vLLM
~5–8% faster per step (same page), so control for it.

What the launcher cannot give you: anything per context or per kernel
(the counters are whole-device, so another job on the same GPU would show up
too), and CUDA-event regions on vLLM's streams, which need the profiler
inside the server. The launcher's GPU probe costs a CUDA context on the
device (hence `--gpu-memory-utilization` below 0.9 in the runs here) and
one CPU thread in the launcher.

## I/O: what the per-process counters show for a model server

- **Weight loading shows in the syscall layer, not the storage layer.**
  vLLM 0.29's default safetensors loader reads the checkpoint with `read()`:
  EngineCore's `rchar` grows by the checkpoint's size (1.63 GiB for
  `Qwen/Qwen3.5-0.8B`; 1.66 GiB in the figure's 36–41 s window) in about
  0.5 s (vLLM logs "Loading weights took 0.52 seconds"), while `read_bytes`
  stays at 0
  because the file is already in the page cache. **A loader that `mmap`s
  from a warm cache would be invisible to every per-process counter**; see
  [per-PID I/O counters](../metric-model.md#per-pid-io-counters-who-records-what).
- **Reaped children's I/O is not counted twice.** When a process reaps a
  child, the kernel adds the child's lifetime I/O to the parent's
  `/proc/<pid>/io`: at shutdown, when the API server reaps EngineCore,
  EngineCore's ~2.4 GiB of `rchar` lands in the API server's counters
  (and, smaller, when EngineCore reaps its compile workers during
  startup). Since both are traced, the profiler subtracts each reaped
  traced child's last reading from its parent and records an
  `IoReapAdjustment`, so per-process I/O is each process's own and sums
  over the tree correctly; see
  [reaped children's I/O](../system-guide.md#reaped-childrens-io).
  In the figure's run, EngineCore's 2.43 GiB is subtracted from the API
  server at 183.1 s, so neither syscall-layer panel shows a jump at
  shutdown. A tracked process that reaps a child and then exits and is
  reaped itself within one sampling interval is handled too (the chain is
  subtracted at the first live tracked ancestor).
- `wchar` includes socket and pipe traffic (the server's responses, its
  log on stdout), so it is not a file-write rate.

## Limits

- A process that lives shorter than one scan interval (100 ms) can be
  missed; its CPU still shows up in its parent's exit tail if the parent is
  tracked and reaps it, and its I/O in the parent's (never subtracted,
  since it was never counted separately).
- Per-sample CPU of another process is quantized to scheduler ticks
  (4 ms at `CONFIG_HZ=250`): single samples of a busy process jitter
  around their true value; sums over time are exact.
- The per-device disk panel counts every process on the machine.
