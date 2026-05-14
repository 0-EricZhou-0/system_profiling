#!/usr/bin/env python3
"""Full-system profile renderer.

Reads the per-probe `.pb` files referenced by `session_metadata.pb`
and produces a tall, multi-panel PNG. Layout (top to bottom):

  - Event strip   : instantaneous markers from events.pb
  - Region strip  : named time intervals from events.pb, with shaded
                    overlays mirrored on every metric panel below
  - GPU panels    : every layout entry whose FQNs were emitted by the
                    GPU probe
  - System panels : CPU + memory + per-PID CPU/RSS
  - Disk panels   : per-device bandwidth, per-PID I/O
  - Flush panels  : write-rate per probe over time (one per probe)
  - Footer        : write-rate table (estimated vs measured)

Panel content is driven by `MetricCatalog` (inlined into
`session_metadata.pb` or supplied via `--catalog`) and `PanelLayout`
(`configs/visualizer_panels.pbtxt` by default). Section grouping is
derived from each FQN's source probe (recorded by the projector at
ingest time), so the layout pbtxt stays free of group hints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Sibling tools package + generated proto package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "generated" / "proto"))

from google.protobuf.internal.decoder import _DecodeVarint32  # noqa: E402

import metric_catalog_pb2 as mc_pb  # noqa: E402
import gpu_metrics_pb2  # noqa: E402
import system_metrics_pb2  # noqa: E402
import disk_metrics_pb2  # noqa: E402
import session_metadata_pb2  # noqa: E402
import events_pb2  # noqa: E402

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

PANEL_HEIGHT_METRIC = 1.7   # any metric panel
PANEL_HEIGHT_EVENT  = 0.25  # event timeline strip
PANEL_HEIGHT_REGION = 0.10  # region timeline strip
PANEL_HEIGHT_FOOTER = 0.7   # write-rate text footer

SPACING_PANEL        = 0.55   # within a section
SPACING_SECTION      = 0.90   # between sections (dashed sep at midpoint)
SPACING_EVENT_REGION = 2.50   # event strip -> region strip
SPACING_AFTER_REGION = 1.45   # region strip -> first metric panel
SPACING_TITLE        = 0.85   # title baseline above first strip

FIG_WIDTH         = 21.0
FIG_MARGIN_TOP    = 1.2
FIG_MARGIN_BOTTOM = 0.5
FIG_MARGIN_LEFT   = 1.0
FIG_MARGIN_RIGHT  = 0.5

YLABEL_X_AXES_FRAC = -0.025
YLABEL_FONTSIZE    = 11

# When a panel has a known peak, set ymax = peak * YLIM_HEADROOM
# unconditionally so every panel with a peak uses the same headroom
# fraction (panel-level `y_max` overrides are ignored for these).
YLIM_HEADROOM      = 1.10
# Bold "Max: …" label rendered at the peak height. X is in axes
# fraction (just inside the y-axis); Y is in data coordinates.
MAX_LABEL_X_AXES_FRAC = 0.005

REGION_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
EVENT_COLORS = [
    "#000000", "#FF1493", "#00CED1", "#FFD700",
    "#7B68EE", "#228B22", "#A0522D", "#DC143C",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()

def _log(msg: str) -> None:
    now = time.localtime()
    ms = int((time.time() - int(time.time())) * 1000)
    abs_ts = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}.{ms:03d}"
    delta = time.perf_counter() - _T0
    print(f"[{abs_ts}] (+{delta:6.3f}s) {msg}", flush=True)


# ---------------------------------------------------------------------------
# Wire reading
# ---------------------------------------------------------------------------

def _read_delimited(path: str | Path, msg_cls) -> list:
    with open(path, "rb") as f:
        buf = f.read()
    out = []
    pos = 0
    while pos < len(buf):
        size, new_pos = _DecodeVarint32(buf, pos)
        pos = new_pos
        m = msg_cls()
        m.ParseFromString(buf[pos:pos + size])
        out.append(m)
        pos += size
    return out


def _load_session_metadata(path: str | Path) -> session_metadata_pb2.SessionMetadata:
    with open(path, "rb") as f:
        meta = session_metadata_pb2.SessionMetadata()
        meta.ParseFromString(f.read())
    return meta


def _resolve_probe_path(metadata_path: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    cands = [Path.cwd() / pp, metadata_path.parent / pp.name, metadata_path.parent / pp]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _gpu_to_steady(ts_ns: int, cupti_ref_ns: int, steady_ref_ns: int) -> int:
    if cupti_ref_ns == 0:
        return ts_ns
    return int(ts_ns) + int(steady_ref_ns) - int(cupti_ref_ns)


def _load_events(path: Path):
    """Load events.pb and return (regions, events, metadata) all in
    steady_clock space. GPU-domain entries are converted via the
    metadata anchor pair."""
    traces = _read_delimited(path, events_pb2.EventTrace)
    if not traces:
        return [], [], None
    # Use the first non-empty metadata for the anchor.
    meta = None
    for t in traces:
        if t.HasField("metadata") and t.metadata.steady_clock_reference_ns:
            meta = t.metadata
            break
    regions: list[tuple[str, int, int]] = []
    events:  list[tuple[str, int]]      = []
    cupti_ref  = meta.cupti_reference_ns        if meta else 0
    steady_ref = meta.steady_clock_reference_ns if meta else 0
    for t in traces:
        for buf in t.buffers:
            is_gpu = buf.domain == events_pb2.TIME_DOMAIN_GPU
            for r in buf.regions:
                s = _gpu_to_steady(r.start_timestamp_ns, cupti_ref, steady_ref) if is_gpu \
                    else r.start_timestamp_ns
                e = _gpu_to_steady(r.end_timestamp_ns,   cupti_ref, steady_ref) if is_gpu \
                    else r.end_timestamp_ns
                regions.append((r.name, int(s), int(e)))
            for ev in buf.events:
                ts = _gpu_to_steady(ev.timestamp_ns, cupti_ref, steady_ref) if is_gpu \
                    else ev.timestamp_ns
                events.append((ev.name, int(ts)))
    return regions, events, meta


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth(vals: np.ndarray, k: int) -> np.ndarray:
    n = vals.size
    if k <= 1 or n == 0:
        return vals
    k = min(k, n)
    half_l = k // 2
    half_r = k - half_l - 1
    csum = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    idx = np.arange(n, dtype=np.int64)
    lo = np.maximum(idx - half_l, 0)
    hi = np.minimum(idx + half_r + 1, n)
    win = (hi - lo).astype(np.float64)
    return ((csum[hi] - csum[lo]) / win).astype(vals.dtype, copy=False)


def _kernel_size(sample_freq_hz: float, window_s: float) -> int:
    return max(1, int(round(sample_freq_hz * window_s)))


# ---------------------------------------------------------------------------
# Unit formatting
# ---------------------------------------------------------------------------

def _format_unit_axis(unit: int, peak_hint: float | None):
    """Return (scale_fn, ylabel)."""
    if unit in (mc_pb.UNIT_PCT, mc_pb.UNIT_PCT_OF_CORE):
        return (lambda v: v, "%")
    if unit == mc_pb.UNIT_RATIO:
        return (lambda v: v, "ratio")
    if unit == mc_pb.UNIT_REQUESTS:
        return (lambda v: v, "requests in-flight")
    if unit == mc_pb.UNIT_HZ:
        return (lambda v: v / 1e6, "MHz")
    if unit == mc_pb.UNIT_BYTES:
        ref = peak_hint if (peak_hint and peak_hint > 0) else 1024.0 ** 3
        if ref >= 1024.0 ** 3:
            return (lambda v: v / (1024.0 ** 3), "GiB")
        if ref >= 1024.0 ** 2:
            return (lambda v: v / (1024.0 ** 2), "MiB")
        return (lambda v: v / 1024.0, "KiB")
    if unit == mc_pb.UNIT_BYTES_PER_SEC:
        if peak_hint is not None and peak_hint >= 1024.0 ** 3:
            return (lambda v: v / (1024.0 ** 3), "GiB/s")
        return (lambda v: v / (1024.0 ** 2), "MiB/s")
    return (lambda v: v, "")


# ---------------------------------------------------------------------------
# Peak resolution
# ---------------------------------------------------------------------------

def _resolve_panel_peak(panel, descriptor, projector: TraceProjector) -> float | None:
    which = panel.WhichOneof("peak")
    if which == "peak_constant":
        return float(panel.peak_constant)
    if which == "peak_from_descriptor_ref":
        return projector.lookup_first_value(panel.peak_from_descriptor_ref)
    if which == "peak_from_expr":
        fn = metric_catalog.PEAK_EXPRS.get(panel.peak_from_expr)
        return float(fn(projector.host)) if fn else None
    if which == "peak_from_gpu_info":
        if not projector.gpu_info:
            return None
        first = next(iter(projector.gpu_info.values()))
        v = float(getattr(first, panel.peak_from_gpu_info, 0.0))
        return v if v > 0 else None
    return metric_catalog.resolve_peak(descriptor, projector.host,
                                       projector.lookup_first_value)


# ---------------------------------------------------------------------------
# Series labels
# ---------------------------------------------------------------------------

def _series_label(series: metric_layout.ResolvedSeries,
                  projector: TraceProjector) -> str:
    fqn = series.fqn
    key = series.scope_key
    if series.scope == mc_pb.SCOPE_SYSTEM:
        return fqn
    if series.scope == mc_pb.SCOPE_PROCESS:
        tp = projector.tracked_processes.get(int(key))
        if tp and tp.alias:
            return f"{fqn}  [{tp.alias} (PID {key})]"
        return f"{fqn}  [PID {key}]"
    if series.scope == mc_pb.SCOPE_DEVICE:
        return f"{fqn}  [{key}]"
    if series.scope == mc_pb.SCOPE_GPU:
        info = projector.gpu_info.get(int(key))
        if info and info.device_name:
            return f"{fqn}  [GPU {key}: {info.device_name}]"
        return f"{fqn}  [GPU {key}]"
    return fqn


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _render_metric_panel(
    ax,
    panel,
    series_list: list[metric_layout.ResolvedSeries],
    projector: TraceProjector,
    projection: dict,
    sample_freq_hz: float,
    smooth_window_s: float,
    t0_ns: int,
    pid_color_map: dict[int, str],
) -> None:
    ax.set_title(panel.title, fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)

    unit = panel.unit_override if panel.unit_override != mc_pb.UNIT_UNSPECIFIED \
        else series_list[0].descriptor.unit
    peak_hint = _resolve_panel_peak(panel, series_list[0].descriptor, projector)
    scale_fn, ylabel = _format_unit_axis(unit, peak_hint)
    ax.set_ylabel(ylabel)

    color_idx = 0
    for series in series_list:
        if (series.scope == mc_pb.SCOPE_PROCESS
                and int(series.scope_key) in pid_color_map):
            color = pid_color_map[int(series.scope_key)]
        else:
            color = _COLOR_CYCLE[color_idx % len(_COLOR_CYCLE)]
            color_idx += 1

        ts_ns, vals = projection[(series.fqn, series.scope_key)]
        if ts_ns.size == 0:
            continue

        if series.descriptor.smoothable and smooth_window_s > 0 and sample_freq_hz > 0:
            k = _kernel_size(sample_freq_hz, smooth_window_s)
            vals = _smooth(vals.astype(np.float64), k)

        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        ax.plot(time_s, scale_fn(vals),
                color=color, linewidth=0.9,
                label=_series_label(series, projector))

    peak_scaled = None
    if peak_hint is not None and peak_hint > 0:
        peak_scaled = scale_fn(peak_hint)
        # Dotted black horizontal reference line at the peak.
        ax.axhline(peak_scaled, color="black", linestyle=":",
                   linewidth=1.4, alpha=0.7)
        # Bold black "Max: …" label near the y-axis at peak height
        # (data y, axes-fraction x). Plain numeric formatter so very
        # large peaks (e.g. ncpus_x_100 = 12800, peak DRAM in MiB/s)
        # don't end up as `1.28e+04`.
        unit_str = f" {ylabel}" if ylabel else ""
        ax.text(MAX_LABEL_X_AXES_FRAC, peak_scaled,
                f"Max: {_fmt_plain(peak_scaled)}{unit_str}",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left",
                fontsize=8, color="black", fontweight="bold",
                zorder=5)

    # Y-axis: always pinned to 0 at the bottom. If a peak is known,
    # cap the top at peak * YLIM_HEADROOM (same headroom for every
    # peak-bearing panel). No peak -> let matplotlib pick the top.
    if peak_scaled is not None:
        ax.set_ylim(0.0, peak_scaled * YLIM_HEADROOM)
    else:
        ax.set_ylim(bottom=0.0)

    # Plain (non-scientific) numeric tick labels on both axes.
    for axis in (ax.xaxis, ax.yaxis):
        fmt = ticker.ScalarFormatter(useOffset=False, useMathText=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))


# ---------------------------------------------------------------------------
# Region + event strips
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 1.0 else f"{s:.3f} s"


def _fmt_plain(v: float) -> str:
    """Plain (non-scientific) numeric formatter — used for the bold
    `Max: …` labels and any other axis-anchored numeric text. Avoids
    `1.2e+04` style output regardless of magnitude."""
    if v == int(v):
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _render_region_strip(ax, regions, t0_ns: int, xmax_s: float) -> None:
    ax.set_xlim(0, xmax_s)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Region", fontsize=8)
    for ri, (name, start_ns, end_ns) in enumerate(regions):
        r_start_s = (start_ns - t0_ns) / 1e9
        dur_s = (end_ns - start_ns) / 1e9
        color = REGION_COLORS[ri % len(REGION_COLORS)]
        ax.barh(0, dur_s, left=r_start_s, height=1.0,
                color=color, alpha=1.0, edgecolor=color, linewidth=0.8)
        ax.annotate(name,
                    xy=(r_start_s, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold",
                    rotation=90, rotation_mode="anchor",
                    ha="left", va="center", annotation_clip=False)
        ax.annotate(_fmt_dur(dur_s),
                    xy=(r_start_s, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -4), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold",
                    rotation=90, rotation_mode="anchor",
                    ha="right", va="center", annotation_clip=False)


def _render_event_strip(ax, events, t0_ns: int, xmax_s: float,
                        wall_ref_ns: int, steady_ref_ns: int) -> None:
    ax.set_xlim(0, xmax_s)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Event", fontsize=8)
    for ei, (name, ts) in enumerate(events):
        t_s = (ts - t0_ns) / 1e9
        color = EVENT_COLORS[ei % len(EVENT_COLORS)]
        ax.axvline(t_s, color=color, lw=1.2, ls="-", alpha=0.85)
        ax.plot(t_s, 0, marker="v", markersize=8,
                color=color, markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(name,
                    xy=(t_s, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold",
                    rotation=90, rotation_mode="anchor",
                    ha="left", va="center", annotation_clip=False)
        if wall_ref_ns > 0:
            wall_ns = wall_ref_ns + (ts - steady_ref_ns)
            dt = datetime.fromtimestamp(wall_ns / 1e9, tz=timezone.utc)
            ms_part = (wall_ns % 1_000_000_000) // 1_000_000
            abs_str = dt.strftime("%H:%M:%S") + f".{ms_part:03d}"
            delta_s = (ts - t0_ns) / 1e9
            # Two-line rotated label: wall clock + delta on parallel
            # rotated lines. Shorter than one concatenated string, so
            # the strip's perpendicular footprint is smaller.
            label = f"{abs_str}\n{delta_s:+.3f}s"
            ax.annotate(label,
                        xy=(t_s, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(0, -4), textcoords="offset points",
                        fontsize=7, color=color, fontweight="bold",
                        rotation=90, rotation_mode="anchor",
                        ha="right", va="center", annotation_clip=False)


def _overlay_regions(metric_axes, regions, t0_ns: int) -> None:
    for ri, (_name, start_ns, end_ns) in enumerate(regions):
        r_start_s = (start_ns - t0_ns) / 1e9
        r_end_s   = (end_ns   - t0_ns) / 1e9
        color = REGION_COLORS[ri % len(REGION_COLORS)]
        for ax in metric_axes:
            ax.axvspan(r_start_s, r_end_s, alpha=0.08, color=color)
            ax.axvline(r_start_s, color=color, lw=0.6, ls="--", alpha=0.4)
            ax.axvline(r_end_s,   color=color, lw=0.6, ls="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Flush stats panel
# ---------------------------------------------------------------------------

def _build_flush_series(traces, first_sample_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Walk traces in order and build (ts_s, MiBps) arrays. Each
    flush_stats entry contributes one point at the running
    cumulative-end of `flush_interval_ns`."""
    times_ns = []
    rates = []
    cur_ns = first_sample_ns
    for t in traces:
        for fs in t.flush_stats:
            if fs.flush_interval_ns == 0:
                continue
            cur_ns += int(fs.flush_interval_ns)
            bps = float(fs.flush_byte_size) / (fs.flush_interval_ns / 1e9)
            times_ns.append(cur_ns)
            rates.append(bps)
    return np.asarray(times_ns, dtype=np.int64), \
           np.asarray(rates,    dtype=np.float64)


def _render_flush_panel(ax, sources, t0_ns: int, xmax_s: float) -> None:
    """sources: list of (label, color, ts_ns_array, rate_Bps_array)."""
    ax.set_title("Per-probe Write Rate (file emission)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("MiB/s")
    for label, color, ts_ns, rates in sources:
        if ts_ns.size == 0:
            continue
        time_s = (ts_ns - t0_ns) / 1e9
        ax.plot(time_s, rates / (1024.0 ** 2), color=color, linewidth=1.0,
                marker="o", markersize=2.5, label=label)
    ax.set_xlim(0, xmax_s)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _fmt_rate(bps: float) -> str:
    if bps >= 1024 ** 3: return f"{bps / 1024**3:7.2f} GiB/s"
    if bps >= 1024 ** 2: return f"{bps / 1024**2:7.2f} MiB/s"
    if bps >= 1024:      return f"{bps / 1024:7.2f} KiB/s"
    return f"{bps:7.0f}   B/s"


def _build_footer_text(rows: list[tuple[str, float, float, int]]) -> str:
    lines = ["Write rate — estimated vs measured (file_size / trace_duration):"]
    total_est = total_meas = 0.0
    total_samp = 0
    for label, est, meas, n_samp in rows:
        est_str = _fmt_rate(est) if est > 0 else "      —      "
        lines.append(f"  {label:<7} est {est_str}   |   measured {_fmt_rate(meas)}   "
                     f"|   samples {n_samp:>8}")
        total_est += est
        total_meas += meas
        total_samp += n_samp
    lines.append(f"  {'Total':<7} est {_fmt_rate(total_est)}   |   measured "
                 f"{_fmt_rate(total_meas)}   |   samples {total_samp:>8}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _ingest_probes(
    projector: TraceProjector,
    meta: session_metadata_pb2.SessionMetadata,
    metadata_path: Path,
):
    """Returns dict with per-probe traces, paths, sample frequencies, and
    cumulative flush-rate series."""
    out = {
        "gpu":    {"traces": [], "path": None, "freq_hz": 0, "n_samples": 0,
                   "duration_s": 0.0},
        "system": {"traces": [], "path": None, "freq_hz": 0, "n_samples": 0,
                   "duration_s": 0.0},
        "disk":   {"traces": [], "path": None, "freq_hz": 0, "n_samples": 0,
                   "duration_s": 0.0},
        "events": {"regions": [], "events": [], "meta": None, "path": None},
    }

    for probe in meta.probes:
        out_path = _resolve_probe_path(metadata_path, probe.output_file)
        if not out_path.exists():
            _log(f"  skip {out_path} (not found)")
            continue

        if probe.kind == session_metadata_pb2.PROBE_KIND_GPU:
            traces = _read_delimited(out_path, gpu_metrics_pb2.GPUMetricsTrace)
            for t in traces:
                projector.ingest_gpu(t)
            n = sum(len(t.samples) for t in traces)
            out["gpu"].update({"traces": traces, "path": out_path,
                                "freq_hz": probe.sampling_frequency_hz,
                                "n_samples": n})
            _log(f"  gpu:    {len(traces)} flushes, {n} samples")
        elif probe.kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
            traces = _read_delimited(out_path, system_metrics_pb2.SystemMetricsTrace)
            for t in traces:
                projector.ingest_system(t)
            n_sys = sum(len(t.system_samples)  for t in traces)
            n_proc = sum(len(t.process_samples) for t in traces)
            out["system"].update({"traces": traces, "path": out_path,
                                   "freq_hz": probe.sampling_frequency_hz,
                                   "n_samples": max(n_sys, n_proc)})
            _log(f"  system: {len(traces)} flushes, {n_sys} sys + {n_proc} proc samples")
        elif probe.kind == session_metadata_pb2.PROBE_KIND_DISK:
            traces = _read_delimited(out_path, disk_metrics_pb2.DiskMetricsTrace)
            for t in traces:
                projector.ingest_disk(t)
            n_dev  = sum(len(t.device_samples)  for t in traces)
            n_proc = sum(len(t.process_samples) for t in traces)
            out["disk"].update({"traces": traces, "path": out_path,
                                 "freq_hz": probe.sampling_frequency_hz,
                                 "n_samples": max(n_dev, n_proc)})
            _log(f"  disk:   {len(traces)} flushes, {n_dev} dev + {n_proc} proc samples")
        elif probe.kind == session_metadata_pb2.PROBE_KIND_EVENTS:
            regions, events, ev_meta = _load_events(out_path)
            out["events"].update({"regions": regions, "events": events,
                                   "meta": ev_meta, "path": out_path})
            _log(f"  events: {len(regions)} regions, {len(events)} events")

    return out


def _first_last_ns(projector: TraceProjector,
                   projection) -> tuple[int | None, int | None]:
    """min/max timestamp across all projected series (already in
    steady_clock space — GPU samples were converted at ingest)."""
    first = None
    last = None
    for ts, _vals in projection.values():
        if ts.size == 0:
            continue
        f = int(ts[0]); l = int(ts[-1])
        first = f if first is None or f < first else first
        last  = l if last  is None or l > last  else last
    return first, last


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------

def _panel_group(series_list: list[metric_layout.ResolvedSeries],
                 fqn_to_probe: dict[str, str]) -> str:
    """Decide which probe a panel belongs to. Group by the first
    matched FQN's source probe; defaults to "gpu" for unmapped FQNs
    (synthesized GPU descriptors)."""
    if not series_list:
        return "gpu"
    return fqn_to_probe.get(series_list[0].fqn, "gpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", help="Path to session_metadata.pb")
    parser.add_argument("-o", "--output", default="full_profile.png",
                        help="Output PNG path (default: full_profile.png)")
    parser.add_argument("--catalog", default=None,
                        help="Override MetricCatalog pbtxt (default: inlined "
                             "in session_metadata.pb)")
    parser.add_argument("--panel-layout", default=None,
                        help="Override PanelLayout pbtxt (default: "
                             "configs/visualizer_panels.pbtxt)")
    parser.add_argument("--smooth-window-s", type=float, default=0.0,
                        help="Boxcar smoothing window in seconds. 0 = none.")
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    _log(f"loading session metadata from {metadata_path}")
    meta = _load_session_metadata(metadata_path)

    if args.catalog:
        catalog = metric_catalog.load_catalog(args.catalog)
        _log(f"catalog: {len(catalog.metrics)} descriptors (from {args.catalog})")
    else:
        catalog = metric_catalog.load_catalog_from_session_metadata(meta)
        _log(f"catalog: {len(catalog.metrics)} descriptors (inlined)")

    layout_path = Path(args.panel_layout) if args.panel_layout \
        else _HERE.parent / "configs" / "visualizer_panels.pbtxt"
    layout = metric_layout.load_panel_layout(layout_path)
    _log(f"layout: {len(layout.panels)} panels (from {layout_path})")

    projector = TraceProjector(catalog)
    _log("ingesting probe data")
    probes = _ingest_probes(projector, meta, metadata_path)

    _log("projecting")
    proj = projector.project()
    if not proj:
        _log("no samples — nothing to plot")
        return 1

    t0_ns, t_end_ns = _first_last_ns(projector, proj)
    # Include events in the time range so the strips don't overflow.
    for _name, ts in probes["events"]["events"]:
        if t0_ns is None or ts < t0_ns: t0_ns = ts
        if t_end_ns is None or ts > t_end_ns: t_end_ns = ts
    for _name, s, e in probes["events"]["regions"]:
        if t0_ns is None or s < t0_ns: t0_ns = s
        if t_end_ns is None or e > t_end_ns: t_end_ns = e
    if t0_ns is None or t_end_ns is None or t_end_ns <= t0_ns:
        _log("no plottable time range")
        return 1
    xmax_s = (t_end_ns - t0_ns) / 1e9

    # Resolve catalog index (with synthesized fallback for GPU FQNs)
    catalog_index = metric_catalog.build_index(catalog)
    for (fqn, _key) in proj.keys():
        if fqn not in catalog_index:
            catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

    series_keys = list(proj.keys())
    resolved = []
    for panel in layout.panels:
        series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
        if not series:
            _log(f"  skip panel {panel.title!r}  (no matching series)")
            continue
        resolved.append((panel, series))
    if not resolved:
        _log("no panels resolved any series")
        return 1

    # Group resolved panels by source probe.
    groups: dict[str, list] = {"gpu": [], "system": [], "disk": []}
    for panel, series in resolved:
        g = _panel_group(series, projector.fqn_to_probe)
        groups.setdefault(g, []).append((panel, series))

    # Stable PID color map: shared across every per-PID panel so the
    # same PID renders in the same color everywhere.
    pid_color_map: dict[int, str] = {}
    seen_pids: list[int] = []
    for _g, panels in groups.items():
        for _p, series_list in panels:
            for s in series_list:
                if s.scope == mc_pb.SCOPE_PROCESS:
                    pid = int(s.scope_key)
                    if pid not in pid_color_map:
                        pid_color_map[pid] = _COLOR_CYCLE[len(seen_pids) % len(_COLOR_CYCLE)]
                        seen_pids.append(pid)

    # Flush sources
    flush_sources = []
    for key, color in (("gpu", "C0"), ("system", "C1"), ("disk", "C2")):
        p = probes[key]
        if not p["traces"]:
            continue
        # Anchor at the trace's first sample timestamp so the rate
        # series sits on the same x axis as the metric panels.
        first_ns = None
        for t in p["traces"]:
            for attr in ("samples", "system_samples", "device_samples"):
                arr = getattr(t, attr, None)
                if arr is not None and len(arr) > 0:
                    if attr == "samples":
                        first_ns = int(t.samples[0].timestamp_ns)
                        # Convert GPU to steady_clock if needed
                        if key == "gpu":
                            anchors = t.header.anchors
                            first_ns = _gpu_to_steady(
                                first_ns,
                                anchors.cupti_reference_ns,
                                anchors.steady_clock_reference_ns)
                    else:
                        first_ns = int(arr[0].timestamp_ns)
                    break
            if first_ns is not None:
                break
        if first_ns is None:
            continue
        ts_ns, rates = _build_flush_series(p["traces"], first_ns)
        # Convert GPU cumulative ts to steady too (the cumulative
        # offset is already in steady space because we converted the
        # anchor; flush_interval_ns is wall delta, domain-independent).
        flush_sources.append((key.upper(), color, ts_ns, rates))

    # ---------------- Build inches-based layout ----------------
    sections: list[tuple[str, list[tuple[str, float]]]] = []

    annot_panels: list[tuple[str, float]] = []
    has_events  = bool(probes["events"]["events"])
    has_regions = bool(probes["events"]["regions"])
    if has_events:  annot_panels.append(("event",  PANEL_HEIGHT_EVENT))
    if has_regions: annot_panels.append(("region", PANEL_HEIGHT_REGION))
    if annot_panels:
        sections.append(("annot", annot_panels))

    for group_key in ("gpu", "system", "disk"):
        if groups.get(group_key):
            sections.append((group_key,
                             [("metric", PANEL_HEIGHT_METRIC)] * len(groups[group_key])))

    has_flush = bool(flush_sources)
    if has_flush:
        sections.append(("flush", [("metric", PANEL_HEIGHT_METRIC)]))

    has_footer = bool(probes["gpu"]["traces"] or probes["system"]["traces"]
                       or probes["disk"]["traces"])
    if has_footer:
        sections.append(("footer", [("footer", PANEL_HEIGHT_FOOTER)]))

    if not sections:
        _log("nothing to render")
        return 1

    def _within_group_gap(prev_kind, kind):
        if prev_kind == "event" and kind == "region":
            return SPACING_EVENT_REGION
        return SPACING_PANEL

    def _between_section_gap(prev_section, next_section):
        return (SPACING_AFTER_REGION if prev_section[1][-1][0] == "region"
                else SPACING_SECTION)

    panel_h_sum = sum(h for _g, panels in sections for _k, h in panels)
    small_sp_total = sum(_within_group_gap(panels[i-1][0], panels[i][0])
                          for _g, panels in sections for i in range(1, len(panels)))
    section_break_total = sum(_between_section_gap(sections[i-1], sections[i])
                               for i in range(1, len(sections)))
    fig_h = (FIG_MARGIN_TOP + FIG_MARGIN_BOTTOM
             + panel_h_sum + small_sp_total + section_break_total)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    left_frac  = FIG_MARGIN_LEFT / FIG_WIDTH
    width_frac = (FIG_WIDTH - FIG_MARGIN_LEFT - FIG_MARGIN_RIGHT) / FIG_WIDTH

    placed: list[tuple[str, str, object]] = []
    section_break_y_in: list[float] = []
    y_top_in = fig_h - FIG_MARGIN_TOP
    prev_kind = None
    for si, (group_key, panels) in enumerate(sections):
        if si > 0:
            gap = _between_section_gap(sections[si-1], sections[si])
            section_break_y_in.append(y_top_in - gap / 2)
            y_top_in -= gap
        for pi, (panel_kind, h) in enumerate(panels):
            if pi > 0:
                y_top_in -= _within_group_gap(prev_kind, panel_kind)
            y_top_in -= h
            ax = fig.add_axes([left_frac, y_top_in / fig_h,
                                width_frac, h / fig_h])
            placed.append((group_key, panel_kind, ax))
            prev_kind = panel_kind

    # Sort axes by role
    event_ax = region_ax = footer_ax = None
    metric_axes_by_group: dict[str, list] = {"gpu": [], "system": [], "disk": []}
    flush_ax = None
    for group_key, panel_kind, ax in placed:
        if panel_kind == "event":  event_ax = ax
        elif panel_kind == "region": region_ax = ax
        elif panel_kind == "footer": footer_ax = ax
        elif panel_kind == "metric":
            if group_key == "flush":
                flush_ax = ax
            else:
                metric_axes_by_group[group_key].append(ax)

    all_metric_axes = []
    for g in ("gpu", "system", "disk"):
        all_metric_axes.extend(metric_axes_by_group[g])
    if flush_ax is not None:
        all_metric_axes.append(flush_ax)

    # ---------------- Render metric panels ----------------
    _log(f"rendering {sum(len(v) for v in metric_axes_by_group.values())} metric panels"
         f"{' + 1 flush panel' if flush_ax else ''}")
    sample_freq_for = {"gpu": probes["gpu"]["freq_hz"],
                       "system": probes["system"]["freq_hz"],
                       "disk": probes["disk"]["freq_hz"]}
    for group_key in ("gpu", "system", "disk"):
        for ax, (panel, series_list) in zip(
                metric_axes_by_group[group_key], groups[group_key]):
            _render_metric_panel(
                ax, panel, series_list, projector, proj,
                sample_freq_hz=sample_freq_for[group_key],
                smooth_window_s=args.smooth_window_s,
                t0_ns=t0_ns,
                pid_color_map=pid_color_map,
            )

    if flush_ax is not None:
        _render_flush_panel(flush_ax, flush_sources, t0_ns, xmax_s)

    # ---------------- Strips + overlays ----------------
    if event_ax is not None:
        ev_meta = probes["events"]["meta"]
        _render_event_strip(event_ax, probes["events"]["events"], t0_ns, xmax_s,
                            wall_ref_ns=ev_meta.wall_clock_epoch_ns if ev_meta else 0,
                            steady_ref_ns=ev_meta.steady_clock_reference_ns if ev_meta else 0)
    if region_ax is not None:
        _render_region_strip(region_ax, probes["events"]["regions"], t0_ns, xmax_s)

    if probes["events"]["regions"]:
        _overlay_regions(all_metric_axes, probes["events"]["regions"], t0_ns)

    # Common xlim + xlabel
    for ax in all_metric_axes:
        ax.set_xlim(0, xmax_s)
        ax.set_xlabel("Time (s)")

    # Align ylabels
    all_axes = [ax for _g, _k, ax in placed]
    for ax in all_axes:
        if ax.get_ylabel():
            ax.yaxis.set_label_coords(YLABEL_X_AXES_FRAC, 0.5)
            ax.yaxis.label.set_fontsize(YLABEL_FONTSIZE)

    # Dashed section separators
    for y_in in section_break_y_in:
        fig.add_artist(plt.Line2D(
            [0.01, 0.99], [y_in / fig_h, y_in / fig_h],
            transform=fig.transFigure, color="#888888", lw=1.5,
            linestyle=(0, (5, 3)), clip_on=False))

    # ---------------- Footer ----------------
    if footer_ax is not None:
        rows = []

        def _est_bytes_per_sample(n_metrics: int, extra_tag: int = 0) -> int:
            # rough: 2 varint timestamps (~10 B) + N doubles (9 B) + ~3 B tag
            return 2 * 10 + n_metrics * 9 + 3 + extra_tag

        # Per-probe duration in seconds, from the projected series.
        def _probe_duration_s(probe_key: str) -> float:
            ts_min = ts_max = None
            for (fqn, _k), (ts, _v) in proj.items():
                src = projector.fqn_to_probe.get(fqn)
                if src != probe_key or ts.size == 0:
                    continue
                a, b = int(ts[0]), int(ts[-1])
                if ts_min is None or a < ts_min: ts_min = a
                if ts_max is None or b > ts_max: ts_max = b
            if ts_min is None or ts_max is None or ts_max <= ts_min:
                return 0.0
            return (ts_max - ts_min) / 1e9

        # GPU row
        if probes["gpu"]["traces"]:
            n_metrics = sum(len(smn.fqns)
                            for smn in probes["gpu"]["traces"][0].scope_metric_names)
            est = probes["gpu"]["freq_hz"] * _est_bytes_per_sample(n_metrics)
            dur = _probe_duration_s("gpu")
            meas = os.path.getsize(probes["gpu"]["path"]) / dur if dur > 0 else 0.0
            rows.append(("GPU", est, meas, probes["gpu"]["n_samples"]))

        # System row
        if probes["system"]["traces"]:
            traces = probes["system"]["traces"]
            n_sys_fqns = n_proc_fqns = 0
            for smn in traces[0].scope_metric_names:
                if smn.scope == mc_pb.SCOPE_SYSTEM:  n_sys_fqns  = len(smn.fqns)
                if smn.scope == mc_pb.SCOPE_PROCESS: n_proc_fqns = len(smn.fqns)
            n_pids = len(traces[0].tracked_processes) or 1
            bytes_per_tick = (_est_bytes_per_sample(n_sys_fqns)
                              + n_pids * _est_bytes_per_sample(n_proc_fqns, extra_tag=4))
            est = probes["system"]["freq_hz"] * bytes_per_tick
            dur = _probe_duration_s("system")
            meas = os.path.getsize(probes["system"]["path"]) / dur if dur > 0 else 0.0
            rows.append(("System", est, meas, probes["system"]["n_samples"]))

        # Disk row
        if probes["disk"]["traces"]:
            traces = probes["disk"]["traces"]
            n_dev_fqns = n_proc_fqns = 0
            for smn in traces[0].scope_metric_names:
                if smn.scope == mc_pb.SCOPE_DEVICE:  n_dev_fqns  = len(smn.fqns)
                if smn.scope == mc_pb.SCOPE_PROCESS: n_proc_fqns = len(smn.fqns)
            n_devs = len(traces[0].tracked_devices)  or 1
            n_pids = len(traces[0].tracked_processes) or 0
            bytes_per_tick = (n_devs * _est_bytes_per_sample(n_dev_fqns, extra_tag=8)
                              + n_pids * _est_bytes_per_sample(n_proc_fqns, extra_tag=4))
            est = probes["disk"]["freq_hz"] * bytes_per_tick
            dur = _probe_duration_s("disk")
            meas = os.path.getsize(probes["disk"]["path"]) / dur if dur > 0 else 0.0
            rows.append(("Disk", est, meas, probes["disk"]["n_samples"]))

        # Events row — emission rate is user-driven, only measured is meaningful.
        if probes["events"]["path"]:
            n_samp = len(probes["events"]["regions"]) + len(probes["events"]["events"])
            dur = xmax_s
            meas = os.path.getsize(probes["events"]["path"]) / dur if dur > 0 else 0.0
            rows.append(("Events", 0.0, meas, n_samp))

        footer_text = _build_footer_text(rows) if rows else None
        footer_ax.axis("off")
        if footer_text:
            footer_ax.text(0.01, 0.95, footer_text,
                           transform=footer_ax.transAxes,
                           fontsize=10, family="monospace",
                           va="top", ha="left")

    # ---------------- Title ----------------
    title_parts = ["Full-System Profile"]
    if meta.hostname:
        title_parts.append(meta.hostname)
    if projector.gpu_info:
        first_gpu = next(iter(projector.gpu_info.values()))
        if first_gpu.device_name:
            title_parts.append(f"{first_gpu.device_name} ({first_gpu.chip_name})")
    title_line = " — ".join(title_parts)
    if meta.start_iso8601:
        title_line += f"\nStart: {meta.start_iso8601}"
    # End time: t0 wall + duration
    end_iso = ""
    if meta.wall_clock_epoch_ns and xmax_s > 0:
        end_ns = int(meta.wall_clock_epoch_ns + xmax_s * 1e9)
        dt = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
        end_iso = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{end_ns % 1_000_000_000:09d}Z"
        title_line += f"   End: {end_iso}   Duration: {xmax_s:.3f}s"
    if meta.probes:
        kind_to_name = {
            session_metadata_pb2.PROBE_KIND_GPU: "GPU",
            session_metadata_pb2.PROBE_KIND_SYSTEM: "System",
            session_metadata_pb2.PROBE_KIND_DISK: "Disk",
            session_metadata_pb2.PROBE_KIND_EVENTS: "Events",
        }
        probe_names = [kind_to_name.get(p.kind, "?") for p in meta.probes]
        title_line += f"\nProbes: {', '.join(probe_names)}"

    title_y_frac = (fig_h - FIG_MARGIN_TOP + SPACING_TITLE) / fig_h
    fig.suptitle(title_line, fontsize=12, y=title_y_frac)

    out_path = Path(args.output).resolve()
    _log(f"saving to {out_path}")
    fig.savefig(out_path, dpi=150)
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
