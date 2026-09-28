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

1. Makes itself a **child subreaper** (`cp.adopt_orphans()`), before vLLM
   exists. Orphaned descendants then re-parent to the launcher instead of
   init, so they stay tracked, and the launcher's orphan reaper is what
   lets **reap chains** be resolved: a `sh -c` that reaps its compiler and
   exits within one sample interval (vLLM's kernel builds, with cold
   compile caches) folds the compiler's CPU and I/O into its own, and only
   with the reaper can the profiler tell that it did and take it out
   again. Without it those chains are flagged ambiguous in the trace
   (`CpuTail.ambiguous_pids` / `ambiguous_cpu_ns`, `IoReapAdjustment`
   `ambiguous`) and their CPU counted twice: a cold start measured 58% over
   the kernel's total, against −0.32% with it (see [cold compile caches](#the-run-in-the-figure)). What it changes for the
   launcher: [adopt_orphans()](../system-guide.md#what-changes-when-adopt_orphans-is-enabled).
2. Builds a `ProfilerSuite` with the **System and Disk probes in SIDECAR
   mode** (the samplers run in the `cupti-profiler-sidecar` child, not in
   the launcher), **descendant tracking on, recursive, 100 ms scan**, and an
   Events probe for the phase markers. With `--gpu`, also the GPU probe
   (see [GPU](#gpu-device-wide-counters-from-the-launcher)).
3. Starts `vllm serve <model>` and **immediately** calls
   `suite.add_tracked_process(pid, "vllm", track_descendants=True)`, before
   the server is ready, so the startup workers are found while they exist.
4. Waits for `/v1/models`, then drives six batches of completions, each for
   1/6 of `--load-seconds` (default 90 s), at concurrency 4, 16, 64, 64,
   16, 4. Every request has a fresh random prompt of 32–3000 words (so the
   prefix cache cannot short-cut it) and a fixed output of 16, 64 or 256
   tokens (`ignore_eos`). Each batch is a region in the trace.
5. Sends `SIGTERM` to the server, stops the suite, prints the trace's
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
defaults otherwise (System and Disk 100 Hz, GPU 1000 Hz, 100 ms scan, every
probe flushing every 5 s, the launcher a child subreaper),
`-- --max-model-len 4096 --gpu-memory-utilization 0.80`, warm compile caches,
launcher not pinned, on an otherwise idle node (the whole-host CPU panel
stays under 5%). The server was ready 76 s after spawn and served 3228
requests in the 90 s of load, none failed. What the example printed at the
end, the trace's process table (times from the start of the trace; CPU =
head + samples + exit tail):

| pid | parent | kind | comm (history) | start s | end s | CPU s |
|---|---|---|---|---|---|---|
| 4106078 | 4106058 | root | `vllm` | 0.0 | 172.6 | 66.95 |
| 4106199 | 4106078 | discovered | `python3.12` | 12.7 | 14.7 | 2.00 |
| 4106201 | 4106078 | discovered | `python3.12` | 12.7 | 14.6 | 2.08 |
| 4106206 | 4106078 | discovered | `ldconfig.real` | 12.9 | 12.9 | 0.01 |
| 4106305 | 4106078 | discovered | `python3.12` → `VLLM::EngineCor` | 21.4 | 171.5 | 130.62 |
| 4106304 | 4106078 | discovered | `python3.12` | 21.4 | 172.1 | 0.03 |
| 4106401 | 4106305 | discovered | `python3.12` | 29.1 | 31.2 | 2.24 |
| 4106403 | 4106305 | discovered | `python3.12` | 29.2 | 31.1 | 2.19 |
| 4106686 | 4106305 | discovered | `ninja` | 56.2 | 56.3 | 0.03 |

The **root** (4106078, `vllm`) is the API server; kind `root` because the
launcher listed it. **EngineCore** (4106305) was found as `python3.12` 21.4 s
in, before it renamed itself, and its alias followed the rename. The
**multiprocessing helper** (4106304) lives for the whole run at ~0 CPU. The
rest are **startup transients**: two pairs of short-lived Python workers
(~2.1 s of CPU each; one pair under the API server, one under EngineCore),
an `ldconfig` and a small `ninja` build. Processes that lived less than one
scan interval can be missing: in this run the oracle below saw 5 such.

The panels line up with vLLM's own log (times from the start of the trace):
EngineCore starts at 21.4 s and loads the weights at ~33 s (the step in its
cumulative `rchar`; vLLM: "Loading weights took 0.46 seconds"); its profiling
warmup and two CUDA graph captures (finished at 50 s and 53 s) are the GPU
activity during startup and its bursts of several cores; its init finishes at
57 s, and the API server, busy at ~100% of a core through most of startup,
answers `/v1/models` at 76 s. Under load both run near one core each;
resident memory reaches ~3.1 GiB (EngineCore) and ~2.3 GiB (API server).

**Cold compile caches.** With empty caches the same example takes 158 s to
be ready and runs ~90 short-lived compiler processes (`ninja`, `nvcc`, `sh`,
`cicc`, `cc1plus`, `ptxas`, ...), many of them `sh -c` shells that reap their
compiler and exit within one sample interval. Because the launcher adopts
orphans, those reap chains resolve: the trace's CPU for the whole tree was
**318.1 s against the kernel's 319.1 s (−0.32%)**; the difference is what
the ~65 processes shorter than a scan used. Without `adopt_orphans()` (the
same run from a harness) the chains are flagged ambiguous and the trace
reads **58% above** the kernel (`CpuTail.ambiguous_cpu_ns` carries the
excess). Such a run's figure is hard to read: ~160 processes overflow the
per-process panels' legends.

### Checked on nine runs

Six runs at earlier defaults (System and Disk 100 Hz, GPU 1000 Hz with the
old collection loop; two without `--gpu`, then two pairs alternating), one at
the 2026-09-25 defaults (System and Disk 50 Hz, GPU 500 Hz), and two at the
current ones (one before the launcher adopted orphans; the run in the
figure), each with a
test-only oracle polling the run's cgroup every 10 ms and the launcher's
`getrusage(RUSAGE_CHILDREN)`:

| | result |
|---|---|
| Completeness: every process the oracle saw alive ≥ 100 ms is in the trace (both probes) | **9/9 runs** (7–8 such processes per run, 0 missed); the 3–9 missed per run were each seen alive ≤ 80.5 ms (the figure's run: 7 such, 5 missed) |
| Per-process CPU of the whole tree (Σ head + samples + tails) vs `getrusage(RUSAGE_CHILDREN)` once vLLM was reaped | trace **15–74 ms below** out of 196–221 s (0.008–0.035%); the figure's run: **34 ms below**, of 206 s (0.017%). What the trace cannot hold: the root's CPU after its last sample (a root has no exit tail: its parent, the launcher, is not tracked), and the CPU of the missed sub-100 ms processes |
| Descendant tracking scan (the trace's `DiscoveryStats`) | p50 0.72–1.25 ms, p99 1.7–2.8 ms, 6–10 processes discovered, all seen exiting (the figure's run: p50 1.0 ms, p99 1.8 ms, 8) |
| The sidecar's own CPU (exact, from `getrusage` across `stop()`) | At 100 Hz: **8.6–14.6% of one core** over the run, unpinned (8.6% on a quiet node without `--gpu`; 13.2–14.6% while the node was 40–70% busy or the launcher ran the GPU probe beside it). The figure's run, at the 100 Hz default with `--gpu` on a quiet node: **10.9%** (12.2% under load). What each rate costs, measured: [Sampling frequency guidance](../system-guide.md#sampling-frequency-guidance). Pin it with `sidecar_cpus` to keep it off the server's cores |

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
