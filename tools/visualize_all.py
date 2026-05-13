#!/usr/bin/env python3
"""Unified visualization for GPU + CPU/Memory + Disk profiling traces.

Reads up to three protobuf trace files and produces a single multi-panel plot.
Each trace type uses its own clock domain — timestamps are normalized to
"time from first sample" independently.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add generated/proto/ to path for _pb2 modules
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "generated", "proto"))

YLIM_HEADROOM = 1.1
MAX_LABEL_X_OFFSET = 0.02
# Default smoothing window in seconds. 0.0 = no smoothing. Override via
# --smooth-window-s; the per-panel kernel size is derived as
# round(smooth_window_s / sample_interval_s) so the window holds the same
# wall time on every panel regardless of its sampling rate.
DEFAULT_SMOOTH_WINDOW_S = 0.0
REGION_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# Distinct palette for events so they don't visually mix with regions.
EVENT_COLORS = [
    "#000000", "#FF1493", "#00CED1", "#FFD700",
    "#7B68EE", "#228B22", "#A0522D", "#DC143C",
]

# === Layout constants (all in inches) =====================================
# Panel heights — tweak these to change the size of each subplot type.
PANEL_HEIGHT_METRIC = 1.7   # any metric panel (SM, DRAM, CPU, Disk, …)
PANEL_HEIGHT_EVENT  = 0.25  # event timeline strip (taller for the larger labels)
PANEL_HEIGHT_REGION = 0.10  # region timeline strip (just colored bars)
PANEL_HEIGHT_FOOTER = 0.7   # write-rate footer
# Spacings between panels.
SPACING_PANEL   = 0.55      # small — between adjacent panels in the same group
SPACING_SECTION = 0.90      # large — between component groups; dashed sep drawn here
# Vertical gap (inches) between the suptitle baseline and the top edge of
# the first strip (the Event panel). Larger = title floats higher above
# the plot. Tune this to taste; if you increase it past ~0.8 you'll
# probably want to also bump FIG_MARGIN_TOP so the title doesn't clip.
SPACING_TITLE_FORST_PANEL = 0.85
# Per-edge overrides for the annotation strips (no dashed sep on these).
SPACING_EVENT_REGION = 1.35 # between Event strip and Region strip
SPACING_AFTER_REGION = 1.45 # between Region strip and the panel just below it
# Page geometry.
FIG_WIDTH         = 21
FIG_MARGIN_TOP    = 1.2     # title + event labels above first strip
FIG_MARGIN_BOTTOM = 0.5
FIG_MARGIN_LEFT   = 1.0     # ylabel + tick labels
FIG_MARGIN_RIGHT  = 0.5
# ylabel x-offset, in axes-fraction units (negative = left of the axis).
# Same value applied to every panel → all ylabels line up vertically.
# Larger magnitude pushes labels farther from the axis. Default: -0.04.
YLABEL_X_AXES_FRAC = -0.025
YLABEL_FONTSIZE   = 11
# ==========================================================================

# ---------------------------------------------------------------------------
# Varint-delimited protobuf reader
# ---------------------------------------------------------------------------

def read_delimited_messages(data, proto_class):
    traces = []
    offset = 0
    while offset < len(data):
        shift, msg_len, varint_bytes = 0, 0, 0
        while offset + varint_bytes < len(data):
            b = data[offset + varint_bytes]
            msg_len |= (b & 0x7F) << shift
            varint_bytes += 1
            shift += 7
            if (b & 0x80) == 0:
                break
        else:
            break
        msg_start = offset + varint_bytes
        msg_end = msg_start + msg_len
        if msg_end > len(data) or msg_len == 0:
            break
        msg = proto_class()
        msg.ParseFromString(data[msg_start:msg_end])
        traces.append(msg)
        offset = msg_end
    if traces and offset == len(data):
        return traces
    msg = proto_class()
    msg.ParseFromString(data)
    return [msg]


def merge_gpu_traces(traces):
    import gpu_metrics_pb2
    if len(traces) == 1:
        return traces[0]
    merged = gpu_metrics_pb2.GpuMetricsTrace()
    merged.CopyFrom(traces[0])
    merged.ClearField("samples")
    for t in traces:
        merged.samples.extend(t.samples)
    return merged


def merge_event_traces(traces):
    """Merge multiple EventTrace messages into a single virtual trace by
    concatenating buffers domain-by-domain, plus a flat list of regions/events
    in CUPTI/steady_clock space already converted by domain.

    Returns dict: {
        "metadata": TraceMetadata (from first message that has one),
        "generic_regions": [Region],
        "generic_events":  [Event],
        "gpu_regions":     [Region],
        "gpu_events":      [Event],
    }
    """
    import events_pb2
    out = {"metadata": None,
           "generic_regions": [], "generic_events": [],
           "gpu_regions": [], "gpu_events": []}
    for t in traces:
        if t.HasField("metadata") and out["metadata"] is None:
            out["metadata"] = t.metadata
        for buf in t.buffers:
            if buf.domain == events_pb2.TIME_DOMAIN_GENERIC:
                out["generic_regions"].extend(buf.regions)
                out["generic_events"].extend(buf.events)
            elif buf.domain == events_pb2.TIME_DOMAIN_GPU:
                out["gpu_regions"].extend(buf.regions)
                out["gpu_events"].extend(buf.events)
    return out


def merge_system_traces(traces):
    import system_metrics_pb2
    if len(traces) == 1:
        return traces[0]
    merged = system_metrics_pb2.SystemMetricsTrace()
    merged.CopyFrom(traces[0])
    merged.ClearField("cpu_system_samples")
    merged.ClearField("cpu_process_samples")
    merged.ClearField("memory_system_samples")
    merged.ClearField("memory_process_samples")
    for t in traces:
        merged.cpu_system_samples.extend(t.cpu_system_samples)
        merged.cpu_process_samples.extend(t.cpu_process_samples)
        merged.memory_system_samples.extend(t.memory_system_samples)
        merged.memory_process_samples.extend(t.memory_process_samples)
    return merged


def merge_disk_traces(traces):
    import disk_metrics_pb2
    if len(traces) == 1:
        return traces[0]
    merged = disk_metrics_pb2.DiskMetricsTrace()
    merged.CopyFrom(traces[0])
    merged.ClearField("device_samples")
    merged.ClearField("process_samples")
    for t in traces:
        merged.device_samples.extend(t.device_samples)
        merged.process_samples.extend(t.process_samples)
    return merged


# Tunable: target samples processed per parallel task. Large enough that
# numpy's per-task setup overhead is amortized; small enough that the
# scheduler has lots of tasks to balance across cores. 1M samples ≈ a few
# tens of ms of work — a comfortable granule.
_SMOOTH_CHUNK_SAMPLES = 1_000_000

# Lazy module-level pool. Reused across every smooth() call so the threads
# stay warm and pool-construction cost (~1 ms) is paid only once per process.
_smooth_pool: ThreadPoolExecutor | None = None
_smooth_pool_workers: int = 0   # cached worker count for diagnostic prints


def _get_smooth_pool() -> ThreadPoolExecutor:
    global _smooth_pool, _smooth_pool_workers
    if _smooth_pool is None:
        _smooth_pool_workers = max(1, os.cpu_count() or 1)
        _smooth_pool = ThreadPoolExecutor(
            max_workers=_smooth_pool_workers, thread_name_prefix="smooth")
    return _smooth_pool


def _smooth_chunk(src, k: int, n: int, a: int, b: int, dst) -> None:
    """Boxcar-smooth output positions [a, b) of one 1-D series.

    `src` is the full series of length n (read-only). `dst` is the full
    output buffer; the worker writes only `dst[a:b]`. The input slice is
    expanded by half_l samples on the left and half_r on the right (a
    "halo"), clamped at the array boundaries, so every output position
    in [a, b) sees its full global window — chunk boundaries produce
    bit-identical results to a single-shot sequential pass.

    Each call is pure numpy (GIL is released throughout), so the pool
    runs N of these in parallel limited only by memory bandwidth.
    """
    half_l = (k - 1) // 2
    half_r = k - half_l
    in_start = max(0, a - half_l)
    in_end   = min(n,  b + half_r)

    sub = src[in_start:in_end]
    cs = np.empty(len(sub) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(sub, out=cs[1:])

    i = np.arange(a, b)
    g_lo = np.maximum(0, i - half_l)
    g_hi = np.minimum(n, i + half_r)
    l_lo = g_lo - in_start
    l_hi = g_hi - in_start
    dst[a:b] = (cs[l_hi] - cs[l_lo]) / (g_hi - g_lo)


# ---------------------------------------------------------------------------
# Progress-timing harness — module-level so panel builders can call it too.
# `_log_reset_anchor()` is called at the top of main() so deltas are reported
# from the start of "real" work (post-argparse). Each `_log("stage")` line
# prints "[<HH:MM:SS.mmm>] (+<delta>s) <stage>" to stdout.
# ---------------------------------------------------------------------------
_T_START_MONO: float | None = None


def _log_reset_anchor() -> None:
    global _T_START_MONO
    _T_START_MONO = time.monotonic()


def _log(stage: str) -> None:
    global _T_START_MONO
    if _T_START_MONO is None:
        _T_START_MONO = time.monotonic()
    now = time.time()
    secs = int(now)
    msecs = int((now - secs) * 1000)
    abs_str = time.strftime("%H:%M:%S", time.localtime(secs)) + f".{msecs:03d}"
    delta = time.monotonic() - _T_START_MONO
    print(f"[{abs_str}] (+{delta:7.3f}s) {stage}")


def smooth(data, k):
    """Boxcar moving average of width `k`, edge-aware, chunked-parallel.

    Splits the work into `_SMOOTH_CHUNK_SAMPLES`-wide tasks and dispatches
    them to a module-level `ThreadPoolExecutor`. Each task is the
    cumsum-based O(window) boxcar in `_smooth_chunk`, applied to a halo-
    padded input slice so chunk boundaries are seamless.

    Inputs below the chunk threshold run inline on the calling thread to
    avoid submit overhead; large inputs scale with `os.cpu_count()` up to
    the memory-bandwidth ceiling.

    Numerics: same algorithm as the single-shot cumsum boxcar — relative
    error stays at ~ε·N for series with large DC offsets, far below visual
    precision for the metrics we plot. See the smoothing discussion in
    docs/tools/README.md (TODO) if you ever need tighter bounds.
    """
    if k <= 1:
        return data

    if data.ndim == 1:
        n = len(data)
        if n == 0:
            return data.astype(np.float64, copy=True)
        out = np.empty(n, dtype=np.float64)
        if n >= _SMOOTH_CHUNK_SAMPLES:
            n_tasks = (n + _SMOOTH_CHUNK_SAMPLES - 1) // _SMOOTH_CHUNK_SAMPLES
            _get_smooth_pool()  # warm the pool so the worker count is known
            _log(f"  smooth shape=({n:,},) k={k:,} → "
                 f"{n_tasks} tasks across {_smooth_pool_workers} workers")
        else:
            _log(f"  smooth shape=({n:,},) k={k:,} → inline (below {_SMOOTH_CHUNK_SAMPLES:,})")
        _dispatch_1d(data, k, n, out)
        return out

    # 2-D: every column is an independent 1-D problem.
    n, m = data.shape
    if n == 0:
        return data.astype(np.float64, copy=True)
    out = np.empty_like(data, dtype=np.float64)

    if n * m < _SMOOTH_CHUNK_SAMPLES:
        # Whole matrix is below the task budget — skip the pool entirely.
        _log(f"  smooth shape=({n:,}, {m}) k={k:,} → inline "
             f"(n*m below {_SMOOTH_CHUNK_SAMPLES:,})")
        for c in range(m):
            _smooth_chunk(data[:, c], k, n, 0, n, out[:, c])
        return out

    pool = _get_smooth_pool()
    if n < _SMOOTH_CHUNK_SAMPLES:
        n_tasks = m
    else:
        chunks_per_col = (n + _SMOOTH_CHUNK_SAMPLES - 1) // _SMOOTH_CHUNK_SAMPLES
        n_tasks = m * chunks_per_col
    _log(f"  smooth shape=({n:,}, {m}) k={k:,} → "
         f"{n_tasks} tasks across {_smooth_pool_workers} workers")

    futures = []
    for c in range(m):
        col_in  = data[:, c]
        col_out = out[:, c]
        if n < _SMOOTH_CHUNK_SAMPLES:
            futures.append(pool.submit(
                _smooth_chunk, col_in, k, n, 0, n, col_out))
        else:
            for a in range(0, n, _SMOOTH_CHUNK_SAMPLES):
                b = min(n, a + _SMOOTH_CHUNK_SAMPLES)
                futures.append(pool.submit(
                    _smooth_chunk, col_in, k, n, a, b, col_out))
    for f in futures:
        f.result()
    return out


def _dispatch_1d(src, k, n, dst) -> None:
    """1-D dispatch: inline if below threshold, parallel chunks otherwise."""
    if n < _SMOOTH_CHUNK_SAMPLES:
        _smooth_chunk(src, k, n, 0, n, dst)
        return
    pool = _get_smooth_pool()
    futures = []
    for a in range(0, n, _SMOOTH_CHUNK_SAMPLES):
        b = min(n, a + _SMOOTH_CHUNK_SAMPLES)
        futures.append(pool.submit(_smooth_chunk, src, k, n, a, b, dst))
    for f in futures:
        f.result()


# ---------------------------------------------------------------------------
# GPU panels (matches quality of original visualize_single.py)
# ---------------------------------------------------------------------------

def pid_label(tracked_processes, pid):
    """Return "<alias> (PID xxx)" if `tracked_processes` carries an alias
    for this PID, else "PID xxx". Aliases come from the optional `alias`
    field on each TrackedProcess entry.

    Accepts either a `repeated TrackedProcess` proto field or any iterable
    of TrackedProcess-shaped objects (e.g. the list snapshot taken inside
    a prepare_* worker)."""
    pid = int(pid)
    for tp in tracked_processes or []:
        if int(tp.pid) == pid and tp.alias:
            return f"{tp.alias} (PID {pid})"
    return f"PID {pid}"


def gpu_to_steady(cupti_ts, cupti_ref, steady_ref):
    """Convert a CUPTI timestamp to steady_clock space using the sync anchor."""
    return cupti_ts - cupti_ref + steady_ref


def _smooth_kernel_size(smooth_window_s, sampling_freq_hz):
    """How many samples are in a `smooth_window_s` window for a series sampled
    at `sampling_freq_hz` Hz. Returns 1 (no-op) when smoothing is disabled
    or the window is shorter than one sample."""
    if smooth_window_s <= 0 or sampling_freq_hz <= 0:
        return 1
    return max(1, int(round(smooth_window_s * sampling_freq_hz)))


def prepare_gpu_data(gpu_traces, global_t0, smooth_window_s=0.0):
    """Extract proto samples to a flat numpy buffer, smoothed, returned
    as a dict that `build_gpu_panels` can plot from.

    `gpu_traces` is the list of GpuMetricsTrace chunks as parsed directly
    from gpu_metrics.pb (no `merge_gpu_traces` step needed). We walk
    samples across chunks via `itertools.chain.from_iterable` rather than
    extending one big protobuf `repeated` — saves the ~3 s the merge step
    costs on multi-chunk files. The "init artifact" (very first sample,
    which CUPTI emits with bogus values) is skipped from the very first
    chunk.

    No matplotlib here — safe to run in a worker thread alongside other
    prepare_*.
    """
    import itertools

    if not gpu_traces:
        return None
    total = sum(len(t.samples) for t in gpu_traces)
    if total <= 1:
        return None
    n = total - 1   # one global init artifact gets dropped

    head = gpu_traces[0]
    metric_names = list(head.metric_names)
    metric_idx = {name: i for i, name in enumerate(metric_names)}
    n_metrics = len(metric_names)

    _log(f"  gpu prep: extracting {n:,} timestamps across {len(gpu_traces)} chunks")
    sample_iter = itertools.chain.from_iterable(t.samples for t in gpu_traces)
    next(sample_iter, None)   # discard init artifact
    ts_cupti = np.fromiter((s.start_timestamp_ns for s in sample_iter),
                            dtype=np.float64, count=n)
    ts_steady = gpu_to_steady(ts_cupti,
                                head.cupti_reference_ns,
                                head.steady_clock_reference_ns)
    time_s = (ts_steady - global_t0) / 1e9

    _log(f"  gpu prep: extracting {n:,}×{n_metrics} metric values across {len(gpu_traces)} chunks")
    sample_iter = itertools.chain.from_iterable(t.samples for t in gpu_traces)
    next(sample_iter, None)   # discard init artifact
    vals = np.empty((n, n_metrics), dtype=np.float64)
    for i, s in enumerate(sample_iter):
        vals[i] = s.values

    k = _smooth_kernel_size(smooth_window_s, head.sampling_frequency_hz)
    _log(f"  gpu prep: smoothing (k={k:,})")
    vals = smooth(vals, k)
    _log("  gpu prep: done")

    return {
        "time_s": time_s,
        "vals": vals,
        "metric_idx": metric_idx,
        "metric_names": metric_names,
        "peak_dram_bw_gbps": head.peak_dram_bw_gbps,
        "peak_pcie_bw_bytes_per_sec": head.peak_pcie_bw_bytes_per_sec,
        "peak_nvlink_bw_bytes_per_sec": head.peak_nvlink_bw_bytes_per_sec,
        "device_name": head.device_name,
        "chip_name": head.chip_name,
    }


def build_gpu_panels(prepared, axes, panel_idx):
    """Render GPU panels from a `prepare_gpu_data` dict. Matplotlib calls
    only — must run on the main thread."""
    if prepared is None:
        return panel_idx
    _log("  gpu plot: panels")
    time_s = prepared["time_s"]
    vals = prepared["vals"]
    idx = prepared["metric_idx"]
    peak_dram_bw_gbps = prepared["peak_dram_bw_gbps"]
    peak_pcie_bw_bytes_per_sec = prepared["peak_pcie_bw_bytes_per_sec"]
    peak_nvlink_bw_bytes_per_sec = prepared["peak_nvlink_bw_bytes_per_sec"]

    # Panel: SM Utilization
    if "sm__cycles_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        ax = axes[panel_idx]
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        util_avg = np.where(elapsed > 0, vals[:, idx["sm__cycles_active.avg"]] / elapsed * 100, 0)
        if "sm__cycles_active.max" in idx:
            util_max = np.where(elapsed > 0, vals[:, idx["sm__cycles_active.max"]] / elapsed * 100, 0)
            ax.plot(time_s, util_max, lw=1.0, color="C0", label="max (busiest SM)")
        ax.plot(time_s, util_avg, lw=0.8, color="C0", alpha=0.45, label="avg (across all SMs)")
        ax.axhline(100, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, 100, "Max SM Util: 100%",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylabel("SM Utilization\n(%)")
        ax.set_ylim(0, 100 * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Panel: Active Warps / Cycle
    if "sm__warps_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        ax = axes[panel_idx]
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        occ_avg = np.where(elapsed > 0, vals[:, idx["sm__warps_active.avg"]] / elapsed, 0)
        if "sm__warps_active.max" in idx:
            occ_max = np.where(elapsed > 0, vals[:, idx["sm__warps_active.max"]] / elapsed, 0)
            ax.plot(time_s, occ_max, lw=1.0, color="C1", label="max (busiest SM)")
        ax.plot(time_s, occ_avg, lw=0.8, color="C1", alpha=0.45, label="avg (across all SMs)")
        max_warps = 64
        ax.axhline(max_warps, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, max_warps, f"Max Active Warps: {max_warps}",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylabel("Active Warps\n/ Cycle")
        ax.set_ylim(0, max_warps * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Panel: DRAM Bandwidth (binary GiB/s)
    if "dram__read_throughput.avg.pct_of_peak_sustained_elapsed" in idx:
        ax = axes[panel_idx]
        peak_gib = peak_dram_bw_gbps * 1e9 / (1024 ** 3)
        pct_to_gibps = peak_gib / 100.0

        rd_avg = vals[:, idx["dram__read_throughput.avg.pct_of_peak_sustained_elapsed"]] * pct_to_gibps
        wr_avg = vals[:, idx["dram__write_throughput.avg.pct_of_peak_sustained_elapsed"]] * pct_to_gibps

        if "dram__read_throughput.max.pct_of_peak_sustained_elapsed" in idx:
            rd_max = vals[:, idx["dram__read_throughput.max.pct_of_peak_sustained_elapsed"]] * pct_to_gibps
            wr_max = vals[:, idx["dram__write_throughput.max.pct_of_peak_sustained_elapsed"]] * pct_to_gibps
            ax.plot(time_s, rd_max, lw=1.0, color="C2", label="Read max")
            ax.plot(time_s, wr_max, lw=1.0, color="C3", label="Write max")

        ax.plot(time_s, rd_avg, lw=0.8, color="C2", alpha=0.45, label="Read avg")
        ax.plot(time_s, wr_avg, lw=0.8, color="C3", alpha=0.45, label="Write avg")

        ax.axhline(peak_gib, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, peak_gib, f"Max DRAM BW: {peak_gib:.0f} GiB/s",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylim(0, peak_gib * YLIM_HEADROOM)

        ax.set_ylabel("DRAM Bandwidth\n(GiB/s)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Panel: PCIe Bandwidth (bytes/sec → GiB/s)
    if "pcie__read_bytes.sum.per_second" in idx and "pcie__write_bytes.sum.per_second" in idx:
        ax = axes[panel_idx]
        rd_gibps = vals[:, idx["pcie__read_bytes.sum.per_second"]] / (1024 ** 3)
        wr_gibps = vals[:, idx["pcie__write_bytes.sum.per_second"]] / (1024 ** 3)
        ax.plot(time_s, rd_gibps, lw=1.0, color="C4", label="H→D (read)")
        ax.plot(time_s, wr_gibps, lw=1.0, color="C5", label="D→H (write)")
        ax.set_ylabel("PCIe Bandwidth\n(GiB/s)")

        # Stored value is per-direction; display as bi-directional (×2 for full-duplex sum).
        peak_pcie_gibps_bidi = peak_pcie_bw_bytes_per_sec * 2 / (1024 ** 3)
        ax.axhline(peak_pcie_gibps_bidi, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, peak_pcie_gibps_bidi,
                f"Max PCIe BW: {peak_pcie_gibps_bidi:.1f} GiB/s (Bi-directional)",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylim(0, peak_pcie_gibps_bidi * YLIM_HEADROOM)

        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Panel: Cumulative PCIe Bytes (each sample's pcie__*_bytes.sum is bytes
    # transferred during that sample's window; cumsum gives the running total).
    if "pcie__read_bytes.sum" in idx and "pcie__write_bytes.sum" in idx:
        ax = axes[panel_idx]
        rd_cum = np.cumsum(vals[:, idx["pcie__read_bytes.sum"]])
        wr_cum = np.cumsum(vals[:, idx["pcie__write_bytes.sum"]])
        max_bytes = float(max(rd_cum[-1], wr_cum[-1]))
        # Pick the largest binary unit such that max value is >= 1 in those units
        for div, unit in [(1024 ** 4, "TiB"), (1024 ** 3, "GiB"),
                           (1024 ** 2, "MiB"), (1024, "KiB"), (1, "B")]:
            if max_bytes >= div:
                break
        ax.plot(time_s, rd_cum / div, lw=1.0, color="C4", label="H→D cumulative")
        ax.plot(time_s, wr_cum / div, lw=1.0, color="C5", label="D→H cumulative")
        ax.text(0.995, rd_cum[-1] / div, f"  {rd_cum[-1] / div:.2f} {unit}",
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7, color="C4", fontweight="bold")
        ax.text(0.995, wr_cum[-1] / div, f"  {wr_cum[-1] / div:.2f} {unit}",
                transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7, color="C5", fontweight="bold")
        ax.set_ylabel(f"Cumulative PCIe Bytes\n({unit})")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Panel: NVLink Bandwidth (bytes/sec → GiB/s)
    if "nvlrx__bytes.sum.per_second" in idx and "nvltx__bytes.sum.per_second" in idx:
        ax = axes[panel_idx]
        rx_gibps = vals[:, idx["nvlrx__bytes.sum.per_second"]] / (1024 ** 3)
        tx_gibps = vals[:, idx["nvltx__bytes.sum.per_second"]] / (1024 ** 3)
        ax.plot(time_s, rx_gibps, lw=1.0, color="C6", label="RX")
        ax.plot(time_s, tx_gibps, lw=1.0, color="C7", label="TX")
        ax.set_ylabel("NVLink Bandwidth\n(GiB/s)")

        # Stored value is per-direction; display as bi-directional (×2).
        peak_nvl_gibps_bidi = peak_nvlink_bw_bytes_per_sec * 2 / (1024 ** 3)
        ax.axhline(peak_nvl_gibps_bidi, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, peak_nvl_gibps_bidi,
                f"Max NVLink BW: {peak_nvl_gibps_bidi:.1f} GiB/s (Bi-directional)",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylim(0, peak_nvl_gibps_bidi * YLIM_HEADROOM)

        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Convert regions to steady_clock space
    return panel_idx


# ---------------------------------------------------------------------------
# System panels (CPU + Memory)
# ---------------------------------------------------------------------------

def prepare_system_data(sys_trace, global_t0, smooth_window_s=0.0):
    """Extract + smooth CPU/memory samples into a dict that `build_system_panels`
    can plot from. Safe to run from a worker thread."""
    if not sys_trace:
        return None
    _log("  sys prep")
    k = _smooth_kernel_size(smooth_window_s, sys_trace.sampling_frequency_hz)
    out = {"tracked_processes": list(sys_trace.tracked_processes)}

    cpu_sys = list(sys_trace.cpu_system_samples)
    if cpu_sys:
        ts = np.array([s.timestamp_ns for s in cpu_sys], dtype=np.float64)
        out["cpu_sys"] = {
            "time_s": (ts - global_t0) / 1e9,
            "user":   smooth(np.array([s.user_pct for s in cpu_sys]), k),
            "sysp":   smooth(np.array([s.system_pct for s in cpu_sys]), k),
            "iowait": smooth(np.array([s.iowait_pct for s in cpu_sys]), k),
            "total":  smooth(np.array([s.total_utilization_pct for s in cpu_sys]), k),
        }

    cpu_proc = list(sys_trace.cpu_process_samples)
    if cpu_proc:
        pids = sorted(set(s.pid for s in cpu_proc))
        per_pid = {}
        for pid in pids:
            ps = [s for s in cpu_proc if s.pid == pid]
            ts = np.array([s.timestamp_ns for s in ps], dtype=np.float64)
            user = smooth(np.array([s.user_pct for s in ps]), k)
            sysp = smooth(np.array([s.system_pct for s in ps]), k)
            per_pid[pid] = {
                "time_s":  (ts - global_t0) / 1e9,
                "user":    user,
                "sysp":    sysp,
                "combined": user + sysp,
            }
        out["cpu_proc"] = {"pids": pids, "per_pid": per_pid}

    mem_sys = list(sys_trace.memory_system_samples)
    if mem_sys:
        ts = np.array([s.timestamp_ns for s in mem_sys], dtype=np.float64)
        GB = 1024 ** 3
        out["mem_sys"] = {
            "time_s":    (ts - global_t0) / 1e9,
            "used":      smooth(np.array([s.used_bytes    for s in mem_sys]) / GB, k),
            "buffers":   smooth(np.array([s.buffers_bytes for s in mem_sys]) / GB, k),
            "cached":    smooth(np.array([s.cached_bytes  for s in mem_sys]) / GB, k),
            "total_gib": mem_sys[0].total_bytes / GB if mem_sys[0].total_bytes else 0,
        }

    mem_proc = list(sys_trace.memory_process_samples)
    if mem_proc:
        pids = sorted(set(s.pid for s in mem_proc))
        per_pid = {}
        for pid in pids:
            ps = [s for s in mem_proc if s.pid == pid]
            ts = np.array([s.timestamp_ns for s in ps], dtype=np.float64)
            per_pid[pid] = {
                "time_s": (ts - global_t0) / 1e9,
                "rss":    smooth(np.array([s.rss_bytes for s in ps]) / (1024 ** 3), k),
            }
        out["mem_proc"] = {"pids": pids, "per_pid": per_pid}

    _log("  sys prep: done")
    return out


def build_system_panels(prepared, axes, panel_idx):
    """Render CPU + memory panels from a `prepare_system_data` dict. Main thread only."""
    if prepared is None:
        return panel_idx
    _log("  sys plot: panels")
    tracked = prepared.get("tracked_processes", [])

    # CPU system-wide
    if "cpu_sys" in prepared:
        d = prepared["cpu_sys"]
        ax = axes[panel_idx]
        time_s = d["time_s"]
        ax.stackplot(time_s, d["user"], d["sysp"], d["iowait"],
                     labels=["User", "System", "IOWait"],
                     colors=["#4c72b0", "#dd8452", "#c44e52"], alpha=0.7)
        ax.plot(time_s, d["total"], lw=1.0, color="black", label="Total")
        ax.axhline(100, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, 100, "Max CPU: 100%",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylabel("CPU Utilization\n(%)")
        ax.set_ylim(0, 100 * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if len(time_s) > 0:
            ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # CPU per-process
    if "cpu_proc" in prepared:
        c = prepared["cpu_proc"]
        ax = axes[panel_idx]
        max_val = 0
        last_time_s = None
        for i, pid in enumerate(c["pids"]):
            d = c["per_pid"][pid]
            time_s = d["time_s"]
            last_time_s = time_s
            max_val = max(max_val,
                           float(np.max(d["combined"])) if len(d["combined"]) > 0 else 0)
            label = pid_label(tracked, pid)
            ax.plot(time_s, d["combined"], lw=1.0, color=f"C{i}",
                    label=f"{label} (usr+sys)")
            ax.plot(time_s, d["user"], lw=0.8, color=f"C{i}",
                    alpha=0.5, ls="--", label=f"{label} (usr)")
        ax.set_ylabel("Process CPU\n(%)")
        ax.set_ylim(0, max(1, max_val) * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if last_time_s is not None and len(last_time_s) > 0:
            ax.set_xlim(left=0, right=last_time_s[-1] * 1.02)
        panel_idx += 1

    # Memory system-wide
    if "mem_sys" in prepared:
        d = prepared["mem_sys"]
        ax = axes[panel_idx]
        time_s = d["time_s"]
        ax.stackplot(time_s, d["used"], d["buffers"], d["cached"],
                     labels=["Used", "Buffers", "Cached"],
                     colors=["#c44e52", "#dd8452", "#4c72b0"], alpha=0.7)
        total_gib = d["total_gib"]
        if total_gib > 0:
            ax.axhline(total_gib, color="black", lw=1.2, ls=":", alpha=0.7)
            ax.text(MAX_LABEL_X_OFFSET, total_gib, f"Total: {total_gib:.0f} GiB",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
            ax.set_ylim(0, total_gib * YLIM_HEADROOM)
        ax.set_ylabel("System Memory\n(GiB)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if len(time_s) > 0:
            ax.set_xlim(left=0, right=time_s[-1] * 1.02)
        panel_idx += 1

    # Memory per-process
    if "mem_proc" in prepared:
        c = prepared["mem_proc"]
        ax = axes[panel_idx]
        max_val = 0
        last_time_s = None
        for i, pid in enumerate(c["pids"]):
            d = c["per_pid"][pid]
            time_s = d["time_s"]
            last_time_s = time_s
            max_val = max(max_val, float(np.max(d["rss"])) if len(d["rss"]) > 0 else 0)
            ax.plot(time_s, d["rss"], lw=1.0, color=f"C{i}",
                    label=f"{pid_label(tracked, pid)} RSS")
        ax.set_ylabel("Process Memory\n(GiB)")
        ax.set_ylim(0, max(0.01, max_val) * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if last_time_s is not None and len(last_time_s) > 0:
            ax.set_xlim(left=0, right=last_time_s[-1] * 1.02)
        panel_idx += 1

    return panel_idx


# ---------------------------------------------------------------------------
# Disk panels
# ---------------------------------------------------------------------------

def prepare_disk_data(disk_trace, global_t0, smooth_window_s=0.0):
    """Extract + smooth disk samples into a dict that `build_disk_panels` can plot from.
    Safe to run from a worker thread."""
    if not disk_trace:
        return None
    _log("  disk prep")
    k = _smooth_kernel_size(smooth_window_s, disk_trace.sampling_frequency_hz)
    dev_samples = list(disk_trace.device_samples)
    proc_samples = list(disk_trace.process_samples)
    if not dev_samples and not proc_samples:
        return None

    out = {"tracked_processes": list(disk_trace.tracked_processes)}
    devices = sorted(set(s.device_name for s in dev_samples)) if dev_samples else []
    out["devices"] = devices
    MB = 1024 ** 2

    if dev_samples:
        per_dev = {}
        for dev in devices:
            ds = [s for s in dev_samples if s.device_name == dev]
            ts = np.array([s.timestamp_ns for s in ds], dtype=np.float64)
            per_dev[dev] = {
                "time_s": (ts - global_t0) / 1e9,
                "rd":     smooth(np.array([s.read_bytes_per_sec  for s in ds]) / MB, k),
                "wr":     smooth(np.array([s.write_bytes_per_sec for s in ds]) / MB, k),
                "rdq":    smooth(np.array([s.read_queue_depth   for s in ds], dtype=np.float64), k),
                "wrq":    smooth(np.array([s.write_queue_depth  for s in ds], dtype=np.float64), k),
            }
        out["dev"] = per_dev

    if proc_samples:
        pids = sorted(set(s.pid for s in proc_samples))
        per_pid = {}
        for pid in pids:
            ps = [s for s in proc_samples if s.pid == pid]
            ts = np.array([s.timestamp_ns for s in ps], dtype=np.float64)
            rd = smooth(np.array([s.read_bytes_per_sec  for s in ps]) / MB, k)
            wr = smooth(np.array([s.write_bytes_per_sec for s in ps]) / MB, k)
            per_pid[pid] = {
                "time_s": (ts - global_t0) / 1e9,
                "rd":     rd,
                "wr":     wr,
            }
        out["proc"] = {"pids": pids, "per_pid": per_pid}

    _log("  disk prep: done")
    return out


def build_disk_panels(prepared, axes, panel_idx):
    """Render disk panels from a `prepare_disk_data` dict. Main thread only."""
    if prepared is None:
        return panel_idx
    _log("  disk plot: panels")
    tracked = prepared.get("tracked_processes", [])
    devices = prepared.get("devices", [])
    dev_data = prepared.get("dev")
    proc_data = prepared.get("proc")

    # Per-device bandwidth
    if dev_data:
        ax = axes[panel_idx]
        last_time_s = None
        for i, dev in enumerate(devices):
            d = dev_data[dev]
            last_time_s = d["time_s"]
            ax.plot(d["time_s"], d["rd"], lw=1.0, color=f"C{i*2}", label=f"{dev} read")
            ax.plot(d["time_s"], d["wr"], lw=1.0, color=f"C{i*2+1}", ls="--", label=f"{dev} write")
        ax.set_ylabel("Disk Bandwidth\n(MiB/s)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if last_time_s is not None and len(last_time_s) > 0:
            ax.set_xlim(left=0, right=last_time_s[-1] * 1.02)
        panel_idx += 1

    # Per-process disk IO
    if proc_data:
        ax = axes[panel_idx]
        max_val = 0
        last_time_s = None
        for i, pid in enumerate(proc_data["pids"]):
            d = proc_data["per_pid"][pid]
            last_time_s = d["time_s"]
            max_val = max(max_val,
                           float(np.max(d["rd"])) if len(d["rd"]) > 0 else 0,
                           float(np.max(d["wr"])) if len(d["wr"]) > 0 else 0)
            label = pid_label(tracked, pid)
            ax.plot(d["time_s"], d["rd"], lw=1.0, color=f"C{i*2}", label=f"{label} read")
            ax.plot(d["time_s"], d["wr"], lw=1.0, color=f"C{i*2+1}", ls="--", label=f"{label} write")
        ax.set_ylabel("Process Disk IO\n(MiB/s)")
        ax.set_ylim(0, max(0.01, max_val) * YLIM_HEADROOM)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if last_time_s is not None and len(last_time_s) > 0:
            ax.set_xlim(left=0, right=last_time_s[-1] * 1.02)
        panel_idx += 1

    # Queue depth (per-device)
    if dev_data:
        ax = axes[panel_idx]
        last_time_s = None
        for i, dev in enumerate(devices):
            d = dev_data[dev]
            last_time_s = d["time_s"]
            ax.plot(d["time_s"], d["rdq"], lw=1.0, color=f"C{i*2}", label=f"{dev} read Q")
            ax.plot(d["time_s"], d["wrq"], lw=1.0, color=f"C{i*2+1}", ls="--", label=f"{dev} write Q")
        ax.set_ylabel("Queue Depth")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if last_time_s is not None and len(last_time_s) > 0:
            ax.set_xlim(left=0, right=last_time_s[-1] * 1.02)
        panel_idx += 1

    return panel_idx


# ---------------------------------------------------------------------------
# Write-rate panels (FlushStats section)
# ---------------------------------------------------------------------------

def _flush_stats_series(trace, global_t0):
    """Extract (time_s, bytes_per_sec) arrays from a trace's flush_stats.
    Returns (None, None) if nothing usable is present."""
    if not trace or not hasattr(trace, "flush_stats"):
        return None, None
    stats = [s for s in trace.flush_stats if s.interval_ns > 0]
    if not stats:
        return None, None
    ts = np.array([s.timestamp_ns for s in stats], dtype=np.float64)
    bps = np.array([s.bytes_written * 1e9 / s.interval_ns for s in stats], dtype=np.float64)
    return (ts - global_t0) / 1e9, bps


def build_flush_panels(traces_with_labels, axes, panel_idx, global_t0):
    """Add write-rate panels — one per profiler source that has FlushStats.
    traces_with_labels: list of (trace, label_prefix, color) tuples.
    Returns new panel_idx."""
    for trace, label, color in traces_with_labels:
        time_s, bps = _flush_stats_series(trace, global_t0)
        if time_s is None or len(time_s) == 0:
            continue

        ax = axes[panel_idx]
        # Pick units adaptively: MiB/s if any point exceeds 1 MiB/s, else KiB/s
        max_bps = float(np.max(bps)) if len(bps) > 0 else 0.0
        if max_bps >= 1024 ** 2:
            y = bps / (1024 ** 2)
            unit = "MiB/s"
        else:
            y = bps / 1024
            unit = "KiB/s"
        ax.step(time_s, y, where="post", lw=1.1, color=color,
                label=f"{label} actual write rate")
        ax.fill_between(time_s, 0, y, step="post", alpha=0.15, color=color)
        mean_y = float(np.mean(y)) if len(y) > 0 else 0.0
        ax.axhline(mean_y, color=color, lw=0.8, ls=":", alpha=0.7)
        ax.text(0.995, mean_y, f"mean {mean_y:.2f} {unit}",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=7, color=color, fontweight="bold")
        ax.set_ylabel(f"{label} write rate\n({unit})")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        if len(time_s) > 0:
            ax.set_xlim(left=0, right=float(time_s[-1]) * 1.02)
        panel_idx += 1

    return panel_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified profiler visualization")
    parser.add_argument("metadata", help="session_metadata.pb (per-probe files are discovered from the manifest)")
    parser.add_argument("-o", "--output", default="full_profile.png", help="Output image")
    parser.add_argument("--smooth-window-s", type=float, default=DEFAULT_SMOOTH_WINDOW_S,
                        help="Boxcar-smooth every per-sample series within a window of "
                             "this many seconds (default: %(default)s = no smoothing). "
                             "Per-panel kernel sizes are derived from each probe's "
                             "sampling_frequency_hz so the wall-time width is uniform "
                             "across panels.")
    args = parser.parse_args()

    # Anchor the timing log at the start of "real" work (post-argparse).
    # `_log` is module-level so panel builders can use it too.
    _log_reset_anchor()
    _log("reading data")

    gpu_trace = sys_trace = disk_trace = None
    events = None

    import session_metadata_pb2
    session_meta = session_metadata_pb2.SessionMetadata()
    with open(args.metadata, "rb") as f:
        session_meta.ParseFromString(f.read())
    print(f"Session: {session_meta.hostname} @ {session_meta.start_iso8601} — "
          f"{len(session_meta.probes)} probes")

    # Probe paths are stored relative to the binary's launch directory.
    # Resolve them by trying the path verbatim, then relative to the
    # metadata's directory, then by basename inside the metadata's directory.
    meta_dir = os.path.dirname(os.path.abspath(args.metadata))
    def _resolve(rel):
        for cand in (rel,
                     os.path.join(meta_dir, rel),
                     os.path.join(meta_dir, os.path.basename(rel))):
            if os.path.exists(cand):
                return cand
        return None
    probe_paths = {}  # ProbeKind → resolved filesystem path
    for p in session_meta.probes:
        resolved = _resolve(p.output_file)
        if resolved is None:
            print(f"  warning: probe file not found: {p.output_file}")
            continue
        probe_paths[p.kind] = resolved

    gpu_path    = probe_paths.get(session_metadata_pb2.PROBE_KIND_GPU)
    system_path = probe_paths.get(session_metadata_pb2.PROBE_KIND_SYSTEM)
    disk_path   = probe_paths.get(session_metadata_pb2.PROBE_KIND_DISK)
    events_path = probe_paths.get(session_metadata_pb2.PROBE_KIND_EVENTS)

    gpu_traces = []           # list-of-chunks; used directly by prepare_gpu_data
    total_gpu_samples = 0     # cached sum across chunks (replaces len(gpu_trace.samples))
    gpu_duration_s = 0.0      # full-stream timestamp span; used by the footer's write-rate stat
    if gpu_path:
        import gpu_metrics_pb2
        with open(gpu_path, "rb") as f:
            data = f.read()
        gpu_traces = read_delimited_messages(data, gpu_metrics_pb2.GpuMetricsTrace)
        total_gpu_samples = sum(len(t.samples) for t in gpu_traces)
        # `gpu_trace` retained as a metadata handle for the title, footer,
        # flush-rate panel, and wall-clock anchor — all of which only need
        # the constant header fields (device_name, peak_*, sampling_freq_hz,
        # wall_clock_epoch_ns, …) that are identical across chunks. Sample
        # iteration goes through `gpu_traces` directly so we never have to
        # do the multi-second proto-level extend that `merge_gpu_traces`
        # used to perform.
        gpu_trace = gpu_traces[0] if gpu_traces else None
        if total_gpu_samples >= 2:
            t0_ns = gpu_traces[0].samples[0].start_timestamp_ns
            t1_ns = gpu_traces[-1].samples[-1].start_timestamp_ns
            if t1_ns > t0_ns:
                gpu_duration_s = (t1_ns - t0_ns) / 1e9
        print(f"GPU: {total_gpu_samples} samples ({len(gpu_traces)} chunks)")

    if events_path:
        import events_pb2
        with open(events_path, "rb") as f:
            data = f.read()
        events = merge_event_traces(read_delimited_messages(data, events_pb2.EventTrace))
        n_total = (len(events["generic_regions"]) + len(events["gpu_regions"])
                   + len(events["generic_events"]) + len(events["gpu_events"]))
        print(f"Events: {len(events['generic_regions'])} generic regions, "
              f"{len(events['gpu_regions'])} gpu regions, "
              f"{len(events['generic_events'])} generic events, "
              f"{len(events['gpu_events'])} gpu events ({n_total} total)")

    if system_path:
        import system_metrics_pb2
        with open(system_path, "rb") as f:
            data = f.read()
        sys_trace = merge_system_traces(read_delimited_messages(data, system_metrics_pb2.SystemMetricsTrace))
        print(f"System: {len(sys_trace.cpu_system_samples)} CPU samples, "
              f"{len(sys_trace.memory_system_samples)} mem samples")

    if disk_path:
        import disk_metrics_pb2
        with open(disk_path, "rb") as f:
            data = f.read()
        disk_trace = merge_disk_traces(read_delimited_messages(data, disk_metrics_pb2.DiskMetricsTrace))
        print(f"Disk: {len(disk_trace.device_samples)} device samples, "
              f"{len(disk_trace.process_samples)} process samples")

    _log("processing data")

    # Count panels: GPU gets a region timeline strip + a metric panel per metric group present
    n_gpu_panels = 0
    has_gpu_regions = False
    if gpu_trace and total_gpu_samples > 1:
        gpu_metric_set = set(gpu_trace.metric_names)
        if {"sm__cycles_active.avg", "sm__cycles_elapsed.avg"}.issubset(gpu_metric_set):
            n_gpu_panels += 1  # SM Utilization
        if {"sm__warps_active.avg", "sm__cycles_elapsed.avg"}.issubset(gpu_metric_set):
            n_gpu_panels += 1  # Active Warps / Cycle
        if "dram__read_throughput.avg.pct_of_peak_sustained_elapsed" in gpu_metric_set:
            n_gpu_panels += 1  # DRAM Bandwidth
        if {"pcie__read_bytes.sum.per_second", "pcie__write_bytes.sum.per_second"}.issubset(gpu_metric_set):
            n_gpu_panels += 1  # PCIe Bandwidth
        if {"pcie__read_bytes.sum", "pcie__write_bytes.sum"}.issubset(gpu_metric_set):
            n_gpu_panels += 1  # Cumulative PCIe Bytes
        if {"nvlrx__bytes.sum.per_second", "nvltx__bytes.sum.per_second"}.issubset(gpu_metric_set):
            n_gpu_panels += 1  # NVLink Bandwidth

    # Region timeline strip is shown when there are any regions to draw
    # (from either Generic or GPU domain in events.pb).
    has_gpu_regions = bool(events and (events["generic_regions"] or events["gpu_regions"]
                                        or events["generic_events"] or events["gpu_events"]))

    n_sys_panels = 0
    if sys_trace:
        if len(sys_trace.cpu_system_samples) > 0: n_sys_panels += 1
        if len(sys_trace.cpu_process_samples) > 0: n_sys_panels += 1
        if len(sys_trace.memory_system_samples) > 0: n_sys_panels += 1
        if len(sys_trace.memory_process_samples) > 0: n_sys_panels += 1

    n_disk_panels = 0
    if disk_trace:
        if len(disk_trace.device_samples) > 0: n_disk_panels += 2
        if len(disk_trace.process_samples) > 0: n_disk_panels += 1

    # Write-rate (FlushStats) panels — one per profiler source that has data
    n_flush_panels = 0
    flush_sources = []   # list of (trace, label, color)
    for trace, label, color in [(gpu_trace, "GPU", "C0"),
                                  (sys_trace, "System", "C1"),
                                  (disk_trace, "Disk", "C2")]:
        if trace and hasattr(trace, "flush_stats") and \
                any(s.interval_ns > 0 for s in trace.flush_stats):
            n_flush_panels += 1
            flush_sources.append((trace, label, color))

    # Events get their own strip below the region strip (so the two are
    # visually separated, since they're conceptually different — spans vs
    # instantaneous markers).
    has_events_strip = bool(events and (events["generic_events"] or events["gpu_events"]))

    # Write-rate footer gets its own panel at the bottom
    has_footer = bool(gpu_trace or sys_trace or disk_trace)

    n_panels = (1 if has_gpu_regions else 0) + (1 if has_events_strip else 0) \
               + n_gpu_panels + n_sys_panels + n_disk_panels \
               + n_flush_panels + (1 if has_footer else 0)

    if n_panels == 0:
        print("No data to plot")
        return

    # Build layout as a list of (group_kind, [(panel_kind, height_in), …]).
    # Events first, regions next, then GPU/Sys/Disk/Flush groups (each its own
    # section), footer last. Adjacent panels in the same group get SPACING_PANEL;
    # group boundaries get SPACING_SECTION (where the dashed separator goes).
    sections = []  # list of (group_kind, list of (panel_kind, height))
    # Event + Region share one section so no dashed separator falls between them.
    annotation_panels = []
    if has_events_strip:
        annotation_panels.append(("event",  PANEL_HEIGHT_EVENT))
    if has_gpu_regions:
        annotation_panels.append(("region", PANEL_HEIGHT_REGION))
    if annotation_panels:
        sections.append(("annot", annotation_panels))
    if n_gpu_panels > 0:
        sections.append(("gpu",    [("metric", PANEL_HEIGHT_METRIC)] * n_gpu_panels))
    if n_sys_panels > 0:
        sections.append(("sys",    [("metric", PANEL_HEIGHT_METRIC)] * n_sys_panels))
    if n_disk_panels > 0:
        sections.append(("disk",   [("metric", PANEL_HEIGHT_METRIC)] * n_disk_panels))
    if n_flush_panels > 0:
        sections.append(("flush",  [("metric", PANEL_HEIGHT_METRIC)] * n_flush_panels))
    if has_footer:
        sections.append(("footer", [("footer", PANEL_HEIGHT_FOOTER)]))

    total_metric_panels = n_gpu_panels + n_sys_panels + n_disk_panels + n_flush_panels

    # Resolve the gap to use between two adjacent panels inside the same group,
    # honoring the per-edge overrides for the annotation strips.
    def _within_group_gap(prev_kind, kind):
        if prev_kind == "event" and kind == "region":
            return SPACING_EVENT_REGION
        return SPACING_PANEL

    # Resolve the gap between the bottom of one section and the top of the next.
    # The dashed separator is always drawn — only the gap size changes when
    # the previous section ended with a region strip.
    def _between_section_gap(prev_section, next_section):
        prev_last_kind = prev_section[1][-1][0]
        if prev_last_kind == "region":
            return SPACING_AFTER_REGION, True
        return SPACING_SECTION, True

    panel_h_sum = sum(h for _, panels in sections for _, h in panels)
    small_sp_total = 0.0
    for _, panels in sections:
        for i in range(1, len(panels)):
            small_sp_total += _within_group_gap(panels[i-1][0], panels[i][0])
    section_break_total = 0.0
    for i in range(1, len(sections)):
        gap, _ = _between_section_gap(sections[i-1], sections[i])
        section_break_total += gap

    fig_h = (FIG_MARGIN_TOP + FIG_MARGIN_BOTTOM
             + panel_h_sum + small_sp_total + section_break_total)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))

    # Place axes top-to-bottom via fig.add_axes (fig-relative coords).
    # Each axis occupies [left, bottom, width, height] in fraction-of-figure.
    left_frac  = FIG_MARGIN_LEFT  / FIG_WIDTH
    width_frac = (FIG_WIDTH - FIG_MARGIN_LEFT - FIG_MARGIN_RIGHT) / FIG_WIDTH

    placed_axes = []          # flat list, top-to-bottom: (group_kind, panel_kind, ax)
    section_break_y_in = []   # y-positions (inches from bottom) where dashed sep belongs

    y_top_in = fig_h - FIG_MARGIN_TOP
    prev_kind = None
    for si, (group_kind, panels) in enumerate(sections):
        if si > 0:
            sec_gap, draw_dashed = _between_section_gap(sections[si-1], sections[si])
            if draw_dashed:
                # Mid-gap of the section break — that's where the dashed line sits.
                section_break_y_in.append(y_top_in - sec_gap / 2)
            y_top_in -= sec_gap
        for pi, (panel_kind, h) in enumerate(panels):
            if pi > 0:
                y_top_in -= _within_group_gap(prev_kind, panel_kind)
            y_top_in -= h
            ax = fig.add_axes([left_frac, y_top_in / fig_h,
                               width_frac, h / fig_h])
            placed_axes.append((group_kind, panel_kind, ax))
            prev_kind = panel_kind

    # Categorize axes by kind so the rest of the plotting code can address them.
    event_ax    = None
    timeline_ax = None
    footer_ax   = None
    metric_axes = []
    gpu_axes    = []
    sys_axes    = []
    disk_axes   = []
    flush_axes  = []
    for group_kind, panel_kind, ax in placed_axes:
        if   panel_kind == "event":  event_ax    = ax
        elif panel_kind == "region": timeline_ax = ax
        elif panel_kind == "footer": footer_ax   = ax
        elif panel_kind == "metric":
            metric_axes.append(ax)
            if   group_kind == "gpu":   gpu_axes.append(ax)
            elif group_kind == "sys":   sys_axes.append(ax)
            elif group_kind == "disk":  disk_axes.append(ax)
            elif group_kind == "flush": flush_axes.append(ax)

    all_axes = [ax for _, _, ax in placed_axes]

    # Compute global t0 = earliest PLOTTED data point in steady_clock ns.
    # GPU: sample[1] (sample[0] is skipped as init artifact)
    # CPU/Disk rates: sample[1] (sample[0] is delta baseline)
    # Memory: sample[0] (absolute values, no delta)
    first_data_ns = []
    if gpu_trace and total_gpu_samples > 1:
        cupti_ref = gpu_trace.cupti_reference_ns
        steady_ref = gpu_trace.steady_clock_reference_ns
        # The init artifact lives at sample[0] of the very first chunk;
        # the first *real* sample is whichever sample is at global index 1.
        # In practice the first chunk has thousands of samples, so this is
        # always gpu_traces[0].samples[1]; we add a guard for the corner
        # case where the first chunk had only the artifact in it.
        if len(gpu_traces[0].samples) > 1:
            first_real_ts = gpu_traces[0].samples[1].start_timestamp_ns
        else:
            first_real_ts = gpu_traces[1].samples[0].start_timestamp_ns
        first_data_ns.append(gpu_to_steady(first_real_ts, cupti_ref, steady_ref))
    if sys_trace:
        if len(sys_trace.cpu_system_samples) > 0:
            first_data_ns.append(sys_trace.cpu_system_samples[0].timestamp_ns)
        if len(sys_trace.memory_system_samples) > 0:
            first_data_ns.append(sys_trace.memory_system_samples[0].timestamp_ns)
    if disk_trace:
        if len(disk_trace.device_samples) > 0:
            first_data_ns.append(disk_trace.device_samples[0].timestamp_ns)
    global_t0 = min(first_data_ns) if first_data_ns else 0

    # ----- Preprocessing (parallel) ---------------------------------------
    # Each prepare_* task does proto → numpy + smoothing for one panel group
    # and returns a plain dict. They run concurrently in a small thread pool
    # (separate from _get_smooth_pool() to avoid nested-deadlock; the inner
    # smoothing uses the big smooth pool). Matplotlib calls happen
    # afterwards on the main thread.
    prep_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prep")
    prep_futures = {}
    if gpu_trace and n_gpu_panels > 0:
        prep_futures["gpu"] = prep_pool.submit(
            prepare_gpu_data, gpu_traces, global_t0, args.smooth_window_s)
    if sys_trace and n_sys_panels > 0:
        prep_futures["sys"] = prep_pool.submit(
            prepare_system_data, sys_trace, global_t0, args.smooth_window_s)
    if disk_trace and n_disk_panels > 0:
        prep_futures["disk"] = prep_pool.submit(
            prepare_disk_data, disk_trace, global_t0, args.smooth_window_s)
    n_prep_tasks = len(prep_futures)
    if n_prep_tasks:
        _log(f"  submitted {n_prep_tasks} preprocess tasks on a 4-worker pool")
    prepared = {k: f.result() for k, f in prep_futures.items()}
    prep_pool.shutdown(wait=True)
    if n_prep_tasks:
        _log("  preprocess: all tasks done")

    # ----- Plotting (sequential, main thread only) ------------------------
    panel_idx = 0
    if n_gpu_panels > 0:
        panel_idx = build_gpu_panels(prepared.get("gpu"), metric_axes, panel_idx)
    if n_sys_panels > 0:
        panel_idx = build_system_panels(prepared.get("sys"), metric_axes, panel_idx)
    if n_disk_panels > 0:
        panel_idx = build_disk_panels(prepared.get("disk"), metric_axes, panel_idx)
    if n_flush_panels > 0:
        panel_idx = build_flush_panels(flush_sources, metric_axes, panel_idx, global_t0)

    all_metric_axes = metric_axes

    # Compute global xmin (earliest plotted data point) and xmax
    global_xmin = float("inf")
    global_xmax = 0
    for ax in all_metric_axes:
        global_xmax = max(global_xmax, ax.get_xlim()[1])
        for line in ax.get_lines():
            xd = line.get_xdata()
            if len(xd) > 0:
                global_xmin = min(global_xmin, float(xd[0]))
        # Also check stacked areas (PolyCollections from stackplot)
        for coll in ax.collections:
            paths = coll.get_paths()
            if paths:
                verts = paths[0].vertices
                if len(verts) > 0:
                    global_xmin = min(global_xmin, float(verts[0, 0]))
    if global_xmin == float("inf"):
        global_xmin = 0
    for ax in all_metric_axes:
        ax.set_xlim(left=0, right=global_xmax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=30))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        yl = ax.get_ylim()
        ax.set_ylim(bottom=0, top=yl[1])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if timeline_ax is not None:
        timeline_ax.set_xlim(left=0, right=global_xmax)
    if event_ax is not None:
        event_ax.set_xlim(left=0, right=global_xmax)

    # Get wall-clock start time: find the trace whose steady_clock_reference_ns
    # is closest to global_t0, then compute wall_clock for the first data point.
    start_iso = None
    for trace in [gpu_trace, sys_trace, disk_trace]:
        if trace and hasattr(trace, 'wall_clock_epoch_ns') and trace.wall_clock_epoch_ns > 0:
            steady_ref = trace.steady_clock_reference_ns
            wall_ref = trace.wall_clock_epoch_ns
            # wall_clock at global_t0 = wall_ref + (global_t0 - steady_ref)
            start_epoch_ns = int(wall_ref + (global_t0 - steady_ref))
            start_dt = datetime.fromtimestamp(start_epoch_ns / 1e9, tz=timezone.utc)
            start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{start_epoch_ns % 1_000_000_000:09d}Z"
            break


    # --- Draw region + event annotations from events.pb across ALL panels ---
    # Each (region, start_ns_steady, end_ns_steady, name) for both domains.
    timeline_regions = []
    timeline_events = []
    if events:
        # Generic regions are already in steady_clock space.
        for r in events["generic_regions"]:
            timeline_regions.append((r.name, r.start_timestamp_ns, r.end_timestamp_ns))
        for e in events["generic_events"]:
            timeline_events.append((e.name, e.timestamp_ns))
        # GPU regions/events live in CUPTI clock; convert via the metadata anchors.
        meta = events["metadata"]
        if meta is not None:
            cupti_ref = meta.cupti_reference_ns
            steady_ref = meta.steady_clock_reference_ns
            for r in events["gpu_regions"]:
                timeline_regions.append((
                    r.name,
                    int(gpu_to_steady(r.start_timestamp_ns, cupti_ref, steady_ref)),
                    int(gpu_to_steady(r.end_timestamp_ns,   cupti_ref, steady_ref))))
            for e in events["gpu_events"]:
                timeline_events.append((
                    e.name,
                    int(gpu_to_steady(e.timestamp_ns, cupti_ref, steady_ref))))

    def _fmt_dur(s):
        """Adaptive duration formatter: ms for sub-second, seconds otherwise."""
        if s < 1.0:
            return f"{s * 1000:.1f} ms"
        return f"{s:.3f} s"

    if timeline_regions:
        for ri, (name, start_ns, end_ns) in enumerate(timeline_regions):
            r_start_s = (start_ns - global_t0) / 1e9
            r_end_s = (end_ns - global_t0) / 1e9
            dur_s = r_end_s - r_start_s
            color = REGION_COLORS[ri % len(REGION_COLORS)]

            # Shaded spans on ALL panels (kept very light so the line plots
            # remain dominant; full saturation lives on the timeline strip).
            for ax in all_metric_axes:
                ax.axvspan(r_start_s, r_end_s, alpha=0.08, color=color)
                ax.axvline(r_start_s, color=color, lw=0.6, ls="--", alpha=0.4)
                ax.axvline(r_end_s, color=color, lw=0.6, ls="--", alpha=0.4)

            # Timeline strip
            if timeline_ax is not None:
                timeline_ax.barh(0, dur_s, left=r_start_s, height=1.0,
                                 color=color, alpha=1, edgecolor=color, linewidth=0.8)

            # Vertical name above the timeline strip
            top_ax = timeline_ax if timeline_ax is not None else all_metric_axes[0]
            top_ax.annotate(name,
                            xy=(r_start_s, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, 4), textcoords="offset points",
                            fontsize=7, color=color, fontweight="bold",
                            rotation=90, rotation_mode="anchor",
                            ha="left", va="center",
                            annotation_clip=False)
            # Vertical duration below the timeline strip
            top_ax.annotate(_fmt_dur(dur_s),
                            xy=(r_start_s, 0.0), xycoords=("data", "axes fraction"),
                            xytext=(0, -4), textcoords="offset points",
                            fontsize=7, color=color, fontweight="bold",
                            rotation=90, rotation_mode="anchor",
                            ha="right", va="center",
                            annotation_clip=False)

        print(f"Regions: {len(timeline_regions)}")
        for name, start_ns, end_ns in timeline_regions:
            dur_s = (end_ns - start_ns) / 1e9
            print(f"  {name}: {_fmt_dur(dur_s)}")

    # Instantaneous events render on their own dedicated strip (event_ax).
    # Kept off the metric panels and the region strip so the two annotation
    # types stay visually distinct.
    if timeline_events and event_ax is not None:
        # Compute the steady_clock → wall_clock anchors so each event can show
        # an absolute timestamp below the axis.
        events_meta = events.get("metadata") if events else None
        wall_ref_ns = events_meta.wall_clock_epoch_ns if events_meta else 0
        steady_ref_ns = events_meta.steady_clock_reference_ns if events_meta else 0
        for ei, (name, ts) in enumerate(timeline_events):
            t_s = (ts - global_t0) / 1e9
            color = EVENT_COLORS[ei % len(EVENT_COLORS)]
            # Vertical line spanning the event strip
            event_ax.axvline(t_s, color=color, lw=1.2, ls="-", alpha=0.85)
            # Marker at center of strip
            event_ax.plot(t_s, 0, marker="v", markersize=8,
                          color=color, markeredgecolor="black", markeredgewidth=0.5)
            # Vertical name above the strip, left-aligned at the line so the
            # text reads upward from the strip top — same convention as the
            # region annotations.
            event_ax.annotate(name,
                              xy=(t_s, 1.0), xycoords=("data", "axes fraction"),
                              xytext=(0, 4), textcoords="offset points",
                              fontsize=7, color=color, fontweight="bold",
                              rotation=90, rotation_mode="anchor",
                              ha="left", va="center",
                              annotation_clip=False)
            # Vertical wall-clock + delta below the strip, right-aligned at
            # the line so the text reads upward into the strip bottom edge.
            if wall_ref_ns > 0:
                wall_ns = wall_ref_ns + (ts - steady_ref_ns)
                dt = datetime.fromtimestamp(wall_ns / 1e9, tz=timezone.utc)
                ms_part = (wall_ns % 1_000_000_000) // 1_000_000
                abs_str = dt.strftime("%H:%M:%S") + f".{ms_part:03d}"
                delta_s = (ts - global_t0) / 1e9
                delta_str = f"{delta_s:+.3f}s"
                event_ax.annotate(f"{abs_str}  ({delta_str})",
                                  xy=(t_s, 0.0), xycoords=("data", "axes fraction"),
                                  xytext=(0, -4), textcoords="offset points",
                                  fontsize=7, color=color, fontweight="bold",
                                  rotation=90, rotation_mode="anchor",
                                  ha="right", va="center",
                                  annotation_clip=False)
        print(f"Events: {len(timeline_events)}")

    # Configure timeline strip
    if timeline_ax is not None:
        timeline_ax.set_ylim(-0.5, 0.5)
        timeline_ax.set_yticks([])
        timeline_ax.spines["top"].set_visible(False)
        timeline_ax.spines["right"].set_visible(False)
        timeline_ax.spines["left"].set_visible(False)
        timeline_ax.tick_params(bottom=False, labelbottom=False)
        timeline_ax.set_ylabel("Region", fontsize=8)

    # Configure event strip
    if event_ax is not None:
        event_ax.set_ylim(-0.5, 0.5)
        event_ax.set_yticks([])
        event_ax.spines["top"].set_visible(False)
        event_ax.spines["right"].set_visible(False)
        event_ax.spines["left"].set_visible(False)
        event_ax.tick_params(bottom=False, labelbottom=False)
        event_ax.set_ylabel("Event", fontsize=8)
        event_ax.set_xlim(left=0, right=global_xmax)

    # X-axis label on every panel
    for ax in all_metric_axes:
        ax.set_xlabel("Time (s)")

    # Compute duration (s) from the plotted data span and derive the end
    # wall-clock by adding it to the manifest's start. global_xmax is now
    # in seconds (axis units), so this is just a passthrough.
    duration_s = global_xmax if global_xmax > 0 else 0.0

    def _fmt_duration_human(s):
        if s < 60:
            return f"{s:.2f}s"
        if s < 3600:
            m = int(s // 60)
            rem = s - 60 * m
            return f"{m}m {rem:.2f}s"
        h = int(s // 3600)
        s -= 3600 * h
        m = int(s // 60)
        rem = s - 60 * m
        return f"{h}h {m}m {rem:.2f}s"

    end_iso = None
    if session_meta.wall_clock_epoch_ns and duration_s > 0:
        end_epoch_ns = int(session_meta.wall_clock_epoch_ns + duration_s * 1e9)
        end_dt = datetime.fromtimestamp(end_epoch_ns / 1e9, tz=timezone.utc)
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{end_epoch_ns % 1_000_000_000:09d}Z"

    # Title with hostname / device + ISO start/end + duration + active probes.
    title_parts = ["Full-System Profile"]
    if session_meta.hostname:
        title_parts.append(session_meta.hostname)
    if gpu_trace:
        title_parts.append(f"{gpu_trace.device_name} ({gpu_trace.chip_name})")
    title_line1 = " — ".join(title_parts)
    if session_meta.start_iso8601:
        title_line1 += f"\nStart: {session_meta.start_iso8601}"
    if end_iso:
        title_line1 += f"   End: {end_iso}"
    if duration_s > 0:
        title_line1 += f"   Duration: {duration_s:.3f}s ({_fmt_duration_human(duration_s)})"
    if session_meta.probes:
        import session_metadata_pb2
        kind_to_name = {
            session_metadata_pb2.PROBE_KIND_GPU: "GPU",
            session_metadata_pb2.PROBE_KIND_SYSTEM: "System",
            session_metadata_pb2.PROBE_KIND_DISK: "Disk",
            session_metadata_pb2.PROBE_KIND_EVENTS: "Events",
        }
        probe_names = [kind_to_name.get(p.kind, "?") for p in session_meta.probes]
        title_line1 += f"\nProbes: {', '.join(probe_names)}"
    # Place the title SPACING_TITLE_FORST_PANEL inches above the top edge of
    # the first strip, expressed in figure-fraction units (suptitle's `y`).
    title_y_frac = (fig_h - FIG_MARGIN_TOP + SPACING_TITLE_FORST_PANEL) / fig_h
    fig.suptitle(title_line1, fontsize=12, y=title_y_frac)

    # --- Write-rate footer: estimated vs measured bytes/sec per source ---
    def _fmt_rate(bps):
        if bps >= 1024 ** 3:
            return f"{bps / (1024 ** 3):7.2f} GiB/s"
        if bps >= 1024 ** 2:
            return f"{bps / (1024 ** 2):7.2f} MiB/s"
        if bps >= 1024:
            return f"{bps / 1024:7.2f} KiB/s"
        return f"{bps:7.0f}   B/s"

    def _trace_duration_s(trace):
        # pick whichever sample vector is available and non-empty
        for attr in ("samples", "cpu_system_samples", "memory_system_samples",
                      "device_samples", "process_samples"):
            v = getattr(trace, attr, None)
            if v is not None and len(v) >= 2:
                ts_field = "start_timestamp_ns" if attr == "samples" else "timestamp_ns"
                t0 = getattr(v[0], ts_field)
                t1 = getattr(v[-1], ts_field)
                if t1 > t0:
                    return (t1 - t0) / 1e9
        return 0.0

    footer_rows = []  # (label, est_bps, meas_bps, n_samples)
    total_est = 0.0
    total_meas = 0.0
    total_samples = 0

    # GPU estimate: 2 varint timestamps (~10 B each) + N doubles (9 B w/ tag) + ~3 B tag/len
    if gpu_trace:
        n_metrics = len(list(gpu_trace.metric_names))
        bytes_per_sample = 2 * 10 + n_metrics * 9 + 3
        est = gpu_trace.sampling_frequency_hz * bytes_per_sample
        meas = 0.0
        # GPU duration spans all chunks; precomputed at read time so we
        # don't have to fall back to `_trace_duration_s` (which would look
        # only at gpu_trace.samples = chunk 0, since merge was dropped).
        if gpu_path and gpu_duration_s > 0:
            meas = os.path.getsize(gpu_path) / gpu_duration_s
        n_samp = max(0, total_gpu_samples - 1)
        footer_rows.append(("GPU", est, meas, n_samp))
        total_est += est
        total_meas += meas
        total_samples += n_samp

    # System estimate: CPUSys (~40 B) + CPUProc × N_pid (~40 B) + MemSys (~50 B) + MemProc × N_pid (~40 B)
    if sys_trace:
        n_pids = len(list(sys_trace.tracked_processes)) if sys_trace.tracked_processes else 1
        bytes_per_tick = 40 + 40 * n_pids + 50 + 40 * n_pids
        est = sys_trace.sampling_frequency_hz * bytes_per_tick
        meas = 0.0
        if system_path:
            dur = _trace_duration_s(sys_trace)
            if dur > 0:
                meas = os.path.getsize(system_path) / dur
        # System samples tick at the configured frequency; the per-tick row
        # count is whichever sample stream is populated (cpu_system_samples).
        n_samp = max(len(sys_trace.cpu_system_samples), len(sys_trace.memory_system_samples))
        footer_rows.append(("System", est, meas, n_samp))
        total_est += est
        total_meas += meas
        total_samples += n_samp

    # Disk estimate: DeviceSample × N_dev (~50 B) + ProcessSample × N_pid (~35 B)
    if disk_trace:
        n_dev = len(list(disk_trace.tracked_devices)) if disk_trace.tracked_devices else 1
        n_pids = len(list(disk_trace.tracked_processes)) if disk_trace.tracked_processes else 0
        bytes_per_tick = 50 * n_dev + 35 * n_pids
        est = disk_trace.sampling_frequency_hz * bytes_per_tick
        meas = 0.0
        if disk_path:
            dur = _trace_duration_s(disk_trace)
            if dur > 0:
                meas = os.path.getsize(disk_path) / dur
        # device_samples is one row per device per tick; divide to get tick count.
        n_samp = (len(disk_trace.device_samples) // n_dev) if n_dev > 0 else len(disk_trace.device_samples)
        footer_rows.append(("Disk", est, meas, n_samp))
        total_est += est
        total_meas += meas
        total_samples += n_samp

    # Events: not periodic, count region+event records as the "sample" count.
    if events:
        n_samp = (len(events["generic_regions"]) + len(events["gpu_regions"])
                  + len(events["generic_events"]) + len(events["gpu_events"]))
        meas = 0.0
        if events_path:
            # Events flush is driven by user activity, not a fixed rate, so
            # estimate is left at 0 — only the measured rate is meaningful.
            dur = duration_s
            if dur > 0:
                meas = os.path.getsize(events_path) / dur
        footer_rows.append(("Events", 0.0, meas, n_samp))
        total_meas += meas
        total_samples += n_samp

    if footer_rows:
        lines = ["Write rate — estimated vs measured (file_size / trace_duration):"]
        for label, est, meas, n_samp in footer_rows:
            est_str = _fmt_rate(est) if est > 0 else "      —      "
            lines.append(f"  {label:<7} est {est_str}   |   measured {_fmt_rate(meas)}   |   samples {n_samp:>8}")
        lines.append(f"  {'Total':<7} est {_fmt_rate(total_est)}   |   measured {_fmt_rate(total_meas)}   |   samples {total_samples:>8}")
        footer_text = "\n".join(lines)
    else:
        footer_text = None

    # fig.align_ylabels skips axes that lack a subplotspec (which is everything
    # we created via fig.add_axes), so do it manually. axes-fraction units mean
    # all panels (which share the same width) end up with their labels at the
    # same horizontal figure position.
    for ax in all_axes:
        if ax.get_ylabel():
            ax.yaxis.set_label_coords(YLABEL_X_AXES_FRAC, 0.5)
            ax.yaxis.label.set_fontsize(YLABEL_FONTSIZE)

    if footer_text and footer_ax is not None:
        footer_ax.axis("off")
        footer_ax.text(0.01, 0.95, footer_text,
                       transform=footer_ax.transAxes,
                       fontsize=10, family="monospace",
                       va="top", ha="left")

    # Dashed separator lines at the precomputed section-break y-positions
    # (in inches from figure bottom; convert to figure fraction for transFigure).
    for y_in in section_break_y_in:
        line_y = y_in / fig_h
        fig.add_artist(plt.Line2D(
            [0.01, 0.99], [line_y, line_y],
            transform=fig.transFigure, color="#888888", lw=1.5,
            linestyle=(0, (5, 3)), clip_on=False))

    _log("plotting")
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
    _log("done")


if __name__ == "__main__":
    main()
