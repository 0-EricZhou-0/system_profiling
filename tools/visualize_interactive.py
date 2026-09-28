#!/usr/bin/env python3
"""Interactive Bokeh visualizer for a cupti_profiler run.

Static mode (default):
    python tools/visualize_interactive.py profiling_output/session_metadata.pb
    # → opens http://localhost:8000 with the rendered page.

Live mode (--live):
    python tools/visualize_interactive.py --live \\
        profiling_output/session_metadata.pb
    # → opens a Bokeh server that tails the .pb files as the suite
    #   writes them, streaming new samples into the running document.

Catalog + panel layout are loaded the same way as visualize_all.py:
the catalog is inlined into session_metadata.pb; the layout defaults
to configs/visualizer_panels.pbtxt.

Live-mode dynamic series allocation (mid-run PID join, removal
markers) lands in the follow-up commit.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

# Sibling tools + generated proto packages.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "generated" / "proto"))

from google.protobuf.internal.decoder import _DecodeVarint32  # noqa: E402

import metric_catalog_pb2 as mc_pb  # noqa: E402
import panels_pb2 as panels_pb  # noqa: E402
import gpu_metrics_pb2  # noqa: E402
import system_metrics_pb2  # noqa: E402
import disk_metrics_pb2  # noqa: E402
import events_pb2  # noqa: E402
import session_metadata_pb2  # noqa: E402

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
import metric_suffix  # noqa: E402
import panel_legend  # noqa: E402
import units  # noqa: E402
import process_timeline  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402

from bokeh.application import Application  # noqa: E402
from bokeh.application.handlers.function import FunctionHandler  # noqa: E402
from bokeh.embed import file_html  # noqa: E402
from bokeh.themes import built_in_themes  # noqa: E402
from bokeh.layouts import column, row  # noqa: E402
from bokeh.models import (BoxAnnotation, BoxZoomTool, ColumnDataSource,  # noqa: E402
                          Button, CustomJS, HoverTool, InlineStyleSheet, Label, Legend,
                          LegendItem, PanTool,
                          Range1d, ResetTool, SaveTool, Span, WheelZoomTool)
from bokeh.palettes import Category10  # noqa: E402
from bokeh.plotting import figure  # noqa: E402
from bokeh.resources import INLINE  # noqa: E402
from bokeh.server.server import Server  # noqa: E402


_T0 = time.perf_counter()

def _log(msg: str) -> None:
    now = time.localtime()
    ms = int((time.time() - int(time.time())) * 1000)
    ts = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}.{ms:03d}"
    print(f"[{ts}] (+{time.perf_counter() - _T0:6.3f}s) {msg}", flush=True)


# ---------------------------------------------------------------------------
# Wire reading (same helpers as visualize_all.py)
# ---------------------------------------------------------------------------

def _read_delimited(path: str | Path, msg_cls) -> list:
    with open(path, "rb") as f:
        buf = f.read()
    out, pos = [], 0
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


def _resolve_path(metadata_path: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    for c in (Path.cwd() / pp, metadata_path.parent / pp.name, metadata_path.parent / pp):
        if c.exists():
            return c
    return Path.cwd() / pp


def _ingest_probes(
    projector: TraceProjector,
    meta: session_metadata_pb2.SessionMetadata,
    metadata_path: Path,
) -> tuple[dict[str, int], dict[str, dict]]:
    """Ingest each probe's .pb file into the projector.

    Returns (sample_freqs, probes_info) where probes_info holds the
    per-probe path + first-trace scope info + sample count needed to
    render the "Write rate" footer. The first trace suffices because
    ScopeMetricNames is emitted identically on every flush.
    """
    sample_freqs: dict[str, int] = {}
    probes_info: dict[str, dict] = {}
    for probe in meta.probes:
        out = _resolve_path(metadata_path, probe.output_file)
        if not out.exists():
            _log(f"  skip {out} (not found)")
            continue
        if probe.kind == session_metadata_pb2.PROBE_KIND_GPU:
            traces = _read_delimited(out, gpu_metrics_pb2.GPUMetricsTrace)
            for t in traces:
                projector.ingest_gpu(t)
            sample_freqs["gpu"] = probe.sampling_frequency_hz
            probes_info["gpu"] = {
                "path": out,
                "traces_head": traces[0] if traces else None,
                "n_samples": sum(len(t.samples) for t in traces),
            }
        elif probe.kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
            traces = _read_delimited(out, system_metrics_pb2.SystemMetricsTrace)
            for t in traces:
                projector.ingest_system(t)
            sample_freqs["system"] = probe.sampling_frequency_hz
            n_sys  = sum(len(t.system_samples)  for t in traces)
            n_proc = sum(len(t.process_samples) for t in traces)
            probes_info["system"] = {
                "path": out,
                "traces_head": traces[0] if traces else None,
                "n_samples": max(n_sys, n_proc),
            }
        elif probe.kind == session_metadata_pb2.PROBE_KIND_DISK:
            traces = _read_delimited(out, disk_metrics_pb2.DiskMetricsTrace)
            for t in traces:
                projector.ingest_disk(t)
            sample_freqs["disk"] = probe.sampling_frequency_hz
            n_dev  = sum(len(t.device_samples)  for t in traces)
            n_proc = sum(len(t.process_samples) for t in traces)
            probes_info["disk"] = {
                "path": out,
                "traces_head": traces[0] if traces else None,
                "n_samples": max(n_dev, n_proc),
            }
        elif probe.kind == session_metadata_pb2.PROBE_KIND_EVENTS:
            probes_info["events"] = {
                "path": out,
                "traces_head": None,
                "n_samples": 0,   # regions + events counted at render time
            }
    return sample_freqs, probes_info


# ---------------------------------------------------------------------------
# Write-rate footer (mirrors visualize_all.py's table)
# ---------------------------------------------------------------------------

def _fmt_rate(bps: float) -> str:
    return f"{units.fmt_bytes(bps, rate=True):>13}"


def _est_bytes_per_sample(n_metrics: int, extra_tag: int = 0) -> int:
    # rough: 2 varint timestamps (~10 B) + N doubles (9 B) + ~3 B tag
    return 2 * 10 + n_metrics * 9 + 3 + extra_tag


def _probe_duration_s(projector: TraceProjector, proj, probe_key: str) -> float:
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


def _build_write_rate_rows(
    projector: TraceProjector,
    proj,
    sample_freqs: dict[str, int],
    probes_info: dict[str, dict],
    events_regions: int,
    events_ns: int,
    xmax_s: float,
) -> list[tuple[str, float, float, int]]:
    """Same rows as visualize_all.py's footer: (label, est_bps, meas_bps, n_samples)."""
    rows: list[tuple[str, float, float, int]] = []

    # GPU
    info = probes_info.get("gpu")
    if info and info["traces_head"] is not None:
        head = info["traces_head"]
        n_metrics = sum(len(smn.fqns) for smn in head.scope_metric_names)
        est = sample_freqs.get("gpu", 0) * _est_bytes_per_sample(n_metrics)
        dur = _probe_duration_s(projector, proj, "gpu")
        meas = os.path.getsize(info["path"]) / dur if dur > 0 else 0.0
        rows.append(("GPU", est, meas, info["n_samples"]))

    # System
    info = probes_info.get("system")
    if info and info["traces_head"] is not None:
        head = info["traces_head"]
        n_sys_fqns = n_proc_fqns = 0
        for smn in head.scope_metric_names:
            if smn.scope == mc_pb.SCOPE_SYSTEM:  n_sys_fqns  = len(smn.fqns)
            if smn.scope == mc_pb.SCOPE_PROCESS: n_proc_fqns = len(smn.fqns)
        n_pids = len(head.tracked_processes) or 1
        bytes_per_tick = (_est_bytes_per_sample(n_sys_fqns)
                          + n_pids * _est_bytes_per_sample(n_proc_fqns, extra_tag=4))
        est = sample_freqs.get("system", 0) * bytes_per_tick
        dur = _probe_duration_s(projector, proj, "system")
        meas = os.path.getsize(info["path"]) / dur if dur > 0 else 0.0
        rows.append(("System", est, meas, info["n_samples"]))

    # Disk
    info = probes_info.get("disk")
    if info and info["traces_head"] is not None:
        head = info["traces_head"]
        n_dev_fqns = n_proc_fqns = 0
        for smn in head.scope_metric_names:
            if smn.scope == mc_pb.SCOPE_DEVICE:  n_dev_fqns  = len(smn.fqns)
            if smn.scope == mc_pb.SCOPE_PROCESS: n_proc_fqns = len(smn.fqns)
        n_devs = len(head.tracked_devices)   or 1
        n_pids = len(head.tracked_processes) or 0
        bytes_per_tick = (n_devs * _est_bytes_per_sample(n_dev_fqns, extra_tag=8)
                          + n_pids * _est_bytes_per_sample(n_proc_fqns, extra_tag=4))
        est = sample_freqs.get("disk", 0) * bytes_per_tick
        dur = _probe_duration_s(projector, proj, "disk")
        meas = os.path.getsize(info["path"]) / dur if dur > 0 else 0.0
        rows.append(("Disk", est, meas, info["n_samples"]))

    # Events — no estimated rate (user-driven emission)
    info = probes_info.get("events")
    if info and info["path"] and info["path"].exists():
        n_samp = events_regions + events_ns
        dur = xmax_s
        meas = os.path.getsize(info["path"]) / dur if dur > 0 else 0.0
        rows.append(("Events", 0.0, meas, n_samp))

    return rows


def _format_write_rate_footer(rows: list[tuple[str, float, float, int]]) -> str:
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
# Unit scaling — same logic as visualize_all.py
# ---------------------------------------------------------------------------

def _format_unit_axis(unit: int, data_max: float | None):
    """Return (scale_fn, ylabel). Bytes and bytes/s: the unit from the
    largest value plotted on the axis (units.byte_unit: 2x thresholds)."""
    if unit in (mc_pb.UNIT_PCT, mc_pb.UNIT_PCT_OF_CORE):
        return (lambda v: v, "%")
    if unit == mc_pb.UNIT_RATIO:
        return (lambda v: v, "ratio")
    if unit == mc_pb.UNIT_REQUESTS:
        return (lambda v: v, "requests in-flight")
    if unit == mc_pb.UNIT_HZ:
        return (lambda v: v / 1e6, "MHz")
    if unit in (mc_pb.UNIT_BYTES, mc_pb.UNIT_BYTES_PER_SEC):
        div, label = units.byte_unit(data_max, rate=unit == mc_pb.UNIT_BYTES_PER_SEC)
        return (lambda v: v / div, label)
    return (lambda v: v, "")


def _smooth(vals: np.ndarray, k: int) -> np.ndarray:
    """Boxcar smoothing with a window of `k` samples — mirrors
    visualize_all._smooth. Used to suppress per-sample noise on
    smoothable metrics when the user opts in via --smooth-window-s."""
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


def _make_plot_tools(include_save: bool = True) -> tuple[list, BoxZoomTool, WheelZoomTool]:
    """Build the per-figure tool set:

      - PanTool         (toolbar only — not the active drag)
      - WheelZoomTool   ctrl-modifier gated, so plain scroll falls
                        through to the browser (page scroll) and
                        ctrl+scroll triggers x-axis zoom; maintain_focus
                        off so cursor-anchored zoom near an edge that
                        already touches a bound absorbs the remaining
                        zoom on the other side instead of being dropped
      - BoxZoomTool     click-drag rectangle on the plot zooms the
                        x-axis to the selected region (rubber-band)
      - ResetTool
      - SaveTool        (optional — skipped on strips)

    Returns (tools, box_zoom, wheel_zoom) so callers can pin
    active_drag and active_scroll to the right instances."""
    pan   = PanTool(dimensions="width")
    wheel = WheelZoomTool(dimensions="width",
                          modifiers={"ctrl": True},
                          maintain_focus=False)
    boxz  = BoxZoomTool(dimensions="width")
    tools: list = [pan, wheel, boxz, ResetTool()]
    if include_save:
        tools.append(SaveTool())
    return tools, boxz, wheel


def _decimate_to_hz(ts_ns: np.ndarray, vals: np.ndarray,
                    source_hz: float, target_hz: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Stride-based downsampling: keep every Nth sample where
    N = source_hz / target_hz. No-op when target_hz <= 0, source_hz
    <= 0, or source_hz <= target_hz."""
    if target_hz <= 0 or source_hz <= 0 or source_hz <= target_hz:
        return ts_ns, vals
    step = max(1, int(round(source_hz / target_hz)))
    return ts_ns[::step], vals[::step]


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


# Legend labels: shared with visualize_all.py.
_series_label = panel_legend.series_label


# ---------------------------------------------------------------------------
# Panel building (Bokeh)
# ---------------------------------------------------------------------------

# Bokeh color palette — Category10 has 10 distinct colors; cycle past that.
_PALETTE = list(Category10[10])

# Fixed plot-area borders so every panel ends at the same right edge
# regardless of how wide its legend is. Without these the legends sit
# in a variable-width right column, which jiggles the plot frames
# left and right across panels and makes legend labels start at
# different x. _FRAME_WIDTH locks the plot frame itself.
#
# Strips don't actually render a toolbar (Bokeh suppresses it on
# figures with height < ~100 px), so their left border is the bare
# _BORDER_LEFT_PX. Metric panels DO render a 30 px left-side toolbar,
# and Bokeh expands their effective left border to ~98 px (toolbar +
# y-axis padding) regardless of min_border_left. _STRIP_LEFT_PX shims
# the strips' left border so their plot frames line up with the
# metric panels' frames despite the missing toolbar reservation.
_BORDER_LEFT_PX  = 80
_STRIP_LEFT_PX   = 98
_FRAME_WIDTH     = 860
# Plot-frame height of a metric panel; the title, the legend above and
# the axis below add to it (so a long legend never squeezes the plot).
_FRAME_HEIGHT    = 150


# Bokeh `output_backend` applied to every figure. main() overrides via
# --render-backend. canvas is the default because for our trace volume
# (~6k pts × ~15 panels) it's roughly 4-5× faster to first paint than
# webgl (measured: canvas 585 ms vs webgl 2687 ms in headless
# Chromium; some real-world GPU drivers stall on webgl ReadPixels and
# blow up by 20-50×). webgl wins on pan/zoom repaints, so it's still
# available as an opt-in.
_RENDER_BACKEND = "canvas"

# Headroom above a known peak when sizing y_range, so the dashed
# peak-reference line isn't drawn right at the plot edge.
_YLIM_HEADROOM = 1.10

# --fit-axis-to-data (see units.OFFSCALE_FACTOR). Set by main() / build_static().
_FIT_AXIS = False


# Per-theme colors. The plot internals are handled by Bokeh's
# `dark_minimal` theme; this table covers the bits Bokeh doesn't
# touch (page background, loading overlay, sticky-strip fills,
# dashed separator). Mutated by main() from --theme.
_THEMES = {
    "light": {
        "bokeh_theme":   None,
        "page_bg":       "#ffffff",
        "page_fg":       "#333333",
        "strip_bg":      "#ffffff",
        "strip_border":  "#888888",
        "overlay_track": "#e0e0e0",
        "overlay_accent": "#3870c4",
    },
    "dark": {
        "bokeh_theme":    "dark_minimal",
        # Page bg = dark_minimal's `border_fill_color`; strip bg =
        # dark_minimal's `background_fill_color`. Keeping them in sync
        # so the sticky strips and metric panels share one continuous
        # dark surface with no visual seam.
        "page_bg":        "#15191C",
        "page_fg":        "#E0E0E0",
        "strip_bg":       "#20262B",
        "strip_border":   "#555555",
        "overlay_track":  "#2a3036",
        "overlay_accent": "#5b9bd5",
    },
}
_THEME = "light"


# Legends sit above their panel (outside the plot frame), in as many
# columns as the widest label allows across the frame. Which entries:
# panel_legend (at most LEGEND_MAX_ENTRIES, the rest drawn in
# OTHER_COLOR and toggled together by one "+k more" entry).
_LEGEND_FONT_PX   = 8 * 96 / 72   # 8pt
_LEGEND_CHAR_EM   = 0.45          # average glyph width of the UI font, in em
_LEGEND_ENTRY_PAD = 20 + 5 + 3 + 10   # glyph_width + label_standoff + spacing + slack


def _legend_ncols(labels: list[str]) -> int:
    if not labels:
        return 1
    widest = max(len(lab) for lab in labels) * _LEGEND_CHAR_EM * _LEGEND_FONT_PX
    return max(1, min(len(labels), int(_FRAME_WIDTH // (widest + _LEGEND_ENTRY_PAD))))


# Series line widths (70% of the earlier 1.2 / 0.8; the static PNG's
# are scaled the same way).
_SERIES_LINE_WIDTH = 0.84
_OTHER_LINE_WIDTH  = 0.56

# panel_legend's line style index -> Bokeh line_dash.
_METRIC_DASHES = ["solid", "dashed", "dotted", "dashdot"]


def _legend_above(fig, items: list[tuple[str, list]], key: list | None = None) -> None:
    """Legend above the plot: items = [(label, renderers)]; an entry's
    swatch draws every renderer's glyph, the last on top. Click an entry
    to hide its lines. `key` (the line-style entries, same form) goes on
    a row of its own, above the rest."""
    def legend(entries, ncols):
        return Legend(items=[LegendItem(label=lab, renderers=list(rs)) for lab, rs in entries],
                      ncols=ncols, location="top_left", click_policy="hide",
                      label_text_font_size="8pt", padding=4, margin=2,
                      spacing=3, border_line_color="#cccccc", border_line_alpha=1.0)
    if items:
        fig.add_layout(legend(items, _legend_ncols([lab for lab, _rs in items])), "above")
    if key:
        # Added after, so Bokeh stacks it farther from the plot: on top.
        fig.add_layout(legend(key, len(key)), "above")


def _draw_plan(fig, p: panel_legend.Plan, sources: dict) -> tuple[list, list]:
    """Draw every series of a panel_legend.Plan (sources: (fqn,
    scope_key) -> ColumnDataSource) and return its legend items: listed
    series in their colour and line style, the rest light grey. Every
    entry but "+k more" gets a swatch of its own, 1.5 px wide as the PNG's
    legend handles (the series lines are thinner) (a one-point
    line at negative time, outside the x range's bounds) so it shows the
    process's colour, or the style in black; clicking it hides all its
    lines."""
    lines = {}
    for key in p.order:
        color, st, listed = p.styles[key]
        lines[key] = fig.line("x", "y", source=sources[key],
                              color=color if listed else panel_legend.OTHER_COLOR,
                              line_dash=_METRIC_DASHES[st],
                              line_width=_SERIES_LINE_WIDTH if listed else _OTHER_LINE_WIDTH)
    items, key = [], []
    for label, color, st, keys in p.entries:
        rs = [lines[k] for k in keys if k in lines]
        if color != panel_legend.OTHER_COLOR:
            # Last: a legend item draws all its renderers' swatches, in order.
            rs.append(fig.line(x=[-2.0, -1.0], y=[0.0, 0.0], color=color,
                               line_dash=_METRIC_DASHES[st], line_width=1.5))
        (key if color == panel_legend.METRIC_COLOR else items).append((label, rs))
    return items, key


def _attach_unified_hover(
    fig,
    series_list: list[metric_layout.ResolvedSeries],
    projection: dict,
    scale_fn,
    t0_ns: int,
    label_bases: dict[tuple, str],
    projector: TraceProjector,
    value_fmt: str = "0.000",
    unit_suffix: str = "",
) -> None:
    """Wire up a HoverTool that shows every series in this panel in a
    single popup, regardless of which series the cursor sits over (or
    whether some have been legend-hidden).

    Implementation: build an invisible 'anchor' line glyph whose CDS
    has x = union of every series's timestamps and one y_i column per
    series (np.interp aligns series with different sampling). Bind the
    HoverTool to *only* that anchor, with one tooltip row per series."""
    aligned: list[tuple[metric_layout.ResolvedSeries, np.ndarray, np.ndarray]] = []
    for s in series_list:
        ts_ns, vals = projection[(s.fqn, s.scope_key)]
        if ts_ns.size == 0:
            continue
        aligned.append((s, ts_ns, vals))
    if not aligned:
        return
    union_ts_ns = np.unique(np.concatenate([ts for _, ts, _ in aligned]))
    union_x_s   = (union_ts_ns.astype(np.int64) - t0_ns) / 1e9
    data: dict = {"x": union_x_s, "_anchor_y": np.zeros_like(union_x_s)}
    tooltips: list[tuple[str, str]] = [("t", "@x{0.000}s")]
    for i, (s, ts_ns, vals) in enumerate(aligned):
        ts_s   = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        scaled = scale_fn(vals.astype(np.float64))
        if union_ts_ns.size == ts_ns.size and np.array_equal(union_ts_ns, ts_ns):
            y_aligned = scaled
        else:
            y_aligned = np.interp(union_x_s, ts_s, scaled,
                                  left=np.nan, right=np.nan)
        col = f"y_{i}"
        data[col] = y_aligned
        label = _series_label(s, projector,
                              base=label_bases[(s.fqn, s.scope_key)])
        unit_part = f" {unit_suffix}" if unit_suffix else ""
        tooltips.append((label, f"@{col}{{{value_fmt}}}{unit_part}"))
    anchor_cds = ColumnDataSource(data=data)
    anchor = fig.line("x", "_anchor_y", source=anchor_cds,
                      line_alpha=0, line_width=0)
    fig.add_tools(HoverTool(renderers=[anchor], mode="vline",
                            attachment="below", tooltips=tooltips))


def _panel_title(panel, series_list: list[metric_layout.ResolvedSeries]) -> str:
    """The layout's title if it gives one; otherwise what the panel's
    series share (metric_layout.shared_title)."""
    if panel.title:
        return panel.title
    if not series_list:
        return panel.series_glob
    return metric_layout.shared_title(series_list)


def _build_panel(
    panel,
    series_list: list[metric_layout.ResolvedSeries],
    projector: TraceProjector,
    projection: dict,
    t0_ns: int,
    x_range=None,
    pid_colors: dict | None = None,
    metric_colors: dict | None = None,
) -> tuple:
    """Build one Bokeh figure for one panel. Returns
    (figure, dict[(fqn, scope_key) -> ColumnDataSource]) so live mode
    can stream new rows in. pid_colors: panel_legend.pid_color_map."""
    unit = panel.unit_override if panel.unit_override != mc_pb.UNIT_UNSPECIFIED \
        else series_list[0].descriptor.unit
    peak_hint = _resolve_panel_peak(panel, series_list[0].descriptor, projector)
    # The unit from what is plotted (not the ceiling).
    scale_fn, ylabel = _format_unit_axis(
        unit, units.largest(projection[(s.fqn, s.scope_key)][1] for s in series_list))

    tools, box_zoom, wheel_zoom = _make_plot_tools()
    fig_kwargs = dict(
        title=_panel_title(panel, series_list),
        x_axis_label="time (s)",
        y_axis_label=ylabel,
        width=1200, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools=tools,
        toolbar_location="left",
        active_drag=box_zoom,
        active_scroll=wheel_zoom,
        output_backend=_RENDER_BACKEND,
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    fig = figure(**fig_kwargs)

    label_bases = metric_layout.disambiguate_short_labels(series_list)
    plan = panel_legend.plan(series_list, projector, projection, pid_colors or {},
                             metric_colors=metric_colors)

    cds_by_key: dict[tuple, ColumnDataSource] = {}
    for key in plan.styles:
        ts_ns, vals = projection[key]
        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        cds_by_key[key] = ColumnDataSource(data=dict(
            x=time_s, y=scale_fn(vals.astype(np.float64)),
        ))
    legend_items, legend_key = _draw_plan(fig, plan, cds_by_key)

    # Peak reference line + y-range. When the panel has a known peak,
    # pin the view to [0, peak*headroom] with hard bounds so pan/zoom
    # can't drift past the meaningful range. When no peak is known,
    # only clamp the floor at 0 — every metric this profiler emits is
    # non-negative, and a drifting negative axis just wastes space.
    data_max = units.largest(projection[(s.fqn, s.scope_key)][1] for s in series_list)
    if _FIT_AXIS and units.off_scale(peak_hint, data_max):
        # The axis fits the data; the ceiling, far above it, is written.
        unit_str = f" {ylabel}" if ylabel else ""
        fig.y_range.bounds = (0.0, None)
        fig.add_layout(Label(x=4, x_units="screen", y=_FRAME_HEIGHT - 4, y_units="screen",
                             text=f"Peak: {_fmt_total(scale_fn(peak_hint))}{unit_str} (off-scale)",
                             text_font_size="8pt", text_font_style="bold",
                             text_baseline="top"))
    elif peak_hint is not None and peak_hint > 0:
        scaled_peak = scale_fn(peak_hint)
        upper = scaled_peak * _YLIM_HEADROOM
        fig.y_range = Range1d(start=0.0, end=upper, bounds=(0.0, upper))
        fig.add_layout(Span(location=scaled_peak, dimension="width",
                            line_color="red", line_dash="dashed",
                            line_alpha=0.6, line_width=1.5))
        # The line is the panel's ceiling (catalog / layout peak), not
        # the data's maximum: say so, as visualize_all.py does.
        unit_str = f" {ylabel}" if ylabel else ""
        fig.add_layout(Label(x=4, x_units="screen", y=scaled_peak, y_units="data",
                             text=f"Peak: {_fmt_total(scaled_peak)}{unit_str}",
                             text_font_size="8pt", text_font_style="bold",
                             text_baseline="bottom"))
    else:
        fig.y_range.bounds = (0.0, None)

    # Per-panel y_min/y_max overrides (pbtxt).
    if panel.y_min != 0.0:
        fig.y_range.start = panel.y_min
    if panel.y_max != 0.0:
        fig.y_range.end = panel.y_max

    # Single-popup hover: one tooltip listing every series' value at
    # the cursor x (interpolated where sampling rates differ).
    _attach_unified_hover(
        fig, series_list, projection, scale_fn, t0_ns,
        label_bases, projector,
        unit_suffix=ylabel,
    )

    _legend_above(fig, legend_items, legend_key)
    return fig, cds_by_key


# ---------------------------------------------------------------------------
# Aggregation companion panels
# ---------------------------------------------------------------------------

# Source unit → integrated unit. Mirrors the table in visualize_all.py.
_INTEGRATED_UNIT = {
    mc_pb.UNIT_BYTES_PER_SEC: mc_pb.UNIT_BYTES,
}


def _trapz_cumulative(ts_ns: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Trapezoidal cumulative integral of (timestamps_ns, values)."""
    if ts_ns.size < 2:
        return np.zeros_like(vals, dtype=np.float64)
    dt_s = np.diff(ts_ns.astype(np.int64)) / 1e9
    avg  = 0.5 * (vals[:-1].astype(np.float64) + vals[1:].astype(np.float64))
    inc  = avg * dt_s
    out  = np.empty(vals.size, dtype=np.float64)
    out[0]  = 0.0
    out[1:] = np.cumsum(inc)
    return out


def _fmt_total(v: float) -> str:
    """Plain (non-scientific) number for a run total."""
    if v == int(v):
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _panel_has_cumulative_companion(panel,
                                     series_list: list[metric_layout.ResolvedSeries]
                                     ) -> bool:
    """Whether `panel` opted into PANEL_AGGREGATION_INTEGRATE AND the
    source unit has a mapped integrated unit. Returns False (with no
    warning) for non-integrable units — visualize_all.py logs the
    skip; here we keep the live-render path quiet since this gets
    called every tick."""
    if panel.aggregation not in (panels_pb.PANEL_AGGREGATION_INTEGRATE,
                                 panels_pb.PANEL_AGGREGATION_INTEGRATE_SUM):
        return False
    if not series_list:
        return False
    src_unit = (panel.unit_override
                if panel.unit_override != mc_pb.UNIT_UNSPECIFIED
                else series_list[0].descriptor.unit)
    return src_unit in _INTEGRATED_UNIT


def _build_cumulative_panel(
    panel,
    series_list: list[metric_layout.ResolvedSeries],
    projector: TraceProjector,
    projection: dict,
    t0_ns: int,
    x_range=None,
    display_hz: float = 0.0,
    source_hzs: dict[tuple[str, object], float] | None = None,
    pid_colors: dict | None = None,
    metric_colors: dict | None = None,
    aggregated: bool = False,
) -> tuple:
    """Bokeh equivalent of visualize_all._render_integrated_panel
    (aggregated: the series are panel_legend.aggregate() sums).

    Returns (figure, cds_by_key) — same shape as _build_panel so the
    static / live machinery treats it like a regular panel for layout
    purposes. The CDS values are pre-cumulated; in live mode the
    coordinator doesn't yet stream into these (cumulative requires
    per-series running-total state — tracked as a follow-up).

    Cumulation runs on the full-resolution series so the displayed
    total reflects the actual integrated value; `display_hz` only
    affects how many points are sent to the browser for plotting.
    """
    src_unit = (panel.unit_override
                if panel.unit_override != mc_pb.UNIT_UNSPECIFIED
                else series_list[0].descriptor.unit)
    integrated_unit = _INTEGRATED_UNIT.get(src_unit, mc_pb.UNIT_UNSPECIFIED)

    # First pass: cumulate every series so we can size the byte-axis.
    # Cumulate on the full-resolution arrays for an accurate total,
    # then optionally decimate the (ts, cum) pair for display only.
    cumulatives: list[tuple[metric_layout.ResolvedSeries, np.ndarray, np.ndarray]] = []
    full_totals: list[tuple[metric_layout.ResolvedSeries, np.ndarray]] = []
    max_total = 0.0
    for series in series_list:
        ts_ns, vals = projection[(series.fqn, series.scope_key)]
        if ts_ns.size == 0:
            continue
        cum = _trapz_cumulative(ts_ns, vals.astype(np.float64))
        full_totals.append((series, cum))
        if cum.size and cum[-1] > max_total:
            max_total = float(cum[-1])
        if display_hz > 0 and source_hzs is not None:
            src_hz = source_hzs.get((series.fqn, series.scope_key), 0.0)
            ts_ns, cum = _decimate_to_hz(ts_ns, cum, src_hz, display_hz)
        cumulatives.append((series, ts_ns, cum))

    scale_fn, ylabel = _format_unit_axis(integrated_unit, max_total)
    base_title = _panel_title(panel, series_list)

    tools, box_zoom, wheel_zoom = _make_plot_tools()
    fig_kwargs = dict(
        title=(f"{base_title}  (cumulative, {series_list[0].scope_key})" if aggregated
               else f"{base_title}  (cumulative)"),
        x_axis_label="time (s)",
        y_axis_label=ylabel,
        width=1200, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools=tools,
        toolbar_location="left",
        active_drag=box_zoom,
        active_scroll=wheel_zoom,
        output_backend=_RENDER_BACKEND,
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    fig = figure(**fig_kwargs)

    label_bases = metric_layout.disambiguate_short_labels(series_list)

    cds_by_key: dict[tuple, ColumnDataSource] = {}
    # Stash pre-cumulated values so the unified hover sees the
    # cumulative curve rather than the source rate.
    cumulative_projection: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}
    # Legend: the largest run totals (full resolution), each in its label.
    plan = panel_legend.plan(
        series_list, projector, projection, pid_colors or {},
        totals={(s.fqn, s.scope_key): float(c[-1]) if c.size else 0.0 for s, c in full_totals},
        fmt_total=units.fmt_bytes,          # each total in its own unit
        metric_colors=metric_colors, aggregated=aggregated)
    for series, ts_ns, cum in cumulatives:
        key = (series.fqn, series.scope_key)
        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        cds_by_key[key] = ColumnDataSource(data=dict(x=time_s, y=scale_fn(cum)))
        cumulative_projection[key] = (ts_ns, cum)
    legend_items, legend_key = _draw_plan(fig, plan, cds_by_key)

    # Cumulative curves are non-negative monotonic — clamp the floor
    # at 0 and let the upper auto-fit.
    fig.y_range.start = 0.0
    fig.y_range.bounds = (0.0, None)

    _attach_unified_hover(
        fig, [s for s, _ts, _cum in cumulatives],
        cumulative_projection, scale_fn, t0_ns,
        label_bases, projector,
        unit_suffix=ylabel,
    )

    _legend_above(fig, legend_items, legend_key)
    return fig, cds_by_key


# ---------------------------------------------------------------------------
# Event / region strips
# ---------------------------------------------------------------------------

def _gpu_to_steady(ts_ns: int, cupti_ref_ns: int, steady_ref_ns: int) -> int:
    if cupti_ref_ns == 0:
        return ts_ns
    return ts_ns - cupti_ref_ns + steady_ref_ns


def _load_events(path: Path):
    """Load events.pb and return (regions, events). Regions are
    (name, start_ns, end_ns); events are (name, ts_ns). GPU-domain
    entries are converted to steady_clock via the per-trace anchor."""
    traces = _read_delimited(path, events_pb2.EventTrace)
    if not traces:
        return [], []
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
    return regions, events


def _load_events_for_session(meta: session_metadata_pb2.SessionMetadata,
                              metadata_path: Path):
    """Find the events probe in the session metadata, load it, and
    return (regions, events). Returns ([], []) if no events probe was
    configured or its .pb is missing."""
    for probe in meta.probes:
        if probe.kind != session_metadata_pb2.PROBE_KIND_EVENTS:
            continue
        out = _resolve_path(metadata_path, probe.output_file)
        if not out.exists():
            return [], []
        return _load_events(out)
    return [], []


def _build_region_strip(regions, t0_ns: int, x_range) -> "figure":
    """Thin strip figure with one colored bar per region. Hover shows
    name + start/end."""
    tools, box_zoom, wheel_zoom = _make_plot_tools(include_save=False)
    fig = figure(
        title="Regions",
        width=1200, height=70, frame_width=_FRAME_WIDTH,
        min_border_left=_STRIP_LEFT_PX,
        x_range=x_range, y_range=(0.0, 1.0),
        tools=tools,
        toolbar_location="left",
        active_drag=box_zoom,
        active_scroll=wheel_zoom,
        output_backend=_RENDER_BACKEND,
    )
    # Solid fills on both the plot area and the surrounding frame
    # so metric panels can't bleed through during scroll. The two
    # strip figures are wrapped in a single sticky Column at the
    # call site (gap-free), so the sticky positioning lives there
    # rather than on each strip.
    fig.background_fill_color = _THEMES[_THEME]["strip_bg"]
    fig.border_fill_color     = _THEMES[_THEME]["strip_bg"]
    fig.yaxis.visible = False
    fig.ygrid.visible = False
    fig.xaxis.visible = False
    if not regions:
        return fig
    names, lefts, rights, colors = [], [], [], []
    for i, (name, s, e) in enumerate(regions):
        names.append(name)
        lefts.append((s - t0_ns) / 1e9)
        rights.append((e - t0_ns) / 1e9)
        colors.append(_PALETTE[i % len(_PALETTE)])
    cds = ColumnDataSource(data=dict(
        name=names, left=lefts, right=rights,
        top=[0.85] * len(regions), bottom=[0.15] * len(regions),
        color=colors,
    ))
    g = fig.quad(left="left", right="right", top="top", bottom="bottom",
                 source=cds, fill_color="color", fill_alpha=0.6,
                 line_color="color", line_width=1.0)
    fig.add_tools(HoverTool(renderers=[g],
        tooltips=[("region", "@name"),
                  ("start", "@left{0.000}s"),
                  ("end",   "@right{0.000}s")]))
    return fig


def _build_event_strip(events, t0_ns: int, x_range) -> "figure":
    """Thin strip figure with one inverted-triangle marker per event
    plus a vertical hairline. Hover shows name + timestamp."""
    tools, box_zoom, wheel_zoom = _make_plot_tools(include_save=False)
    fig = figure(
        title="Events",
        width=1200, height=70, frame_width=_FRAME_WIDTH,
        min_border_left=_STRIP_LEFT_PX,
        x_range=x_range, y_range=(0.0, 1.0),
        tools=tools,
        toolbar_location="left",
        active_drag=box_zoom,
        active_scroll=wheel_zoom,
        output_backend=_RENDER_BACKEND,
    )
    fig.background_fill_color = _THEMES[_THEME]["strip_bg"]
    fig.border_fill_color     = _THEMES[_THEME]["strip_bg"]
    fig.yaxis.visible = False
    fig.ygrid.visible = False
    fig.xaxis.visible = False
    if not events:
        return fig
    names, xs, colors = [], [], []
    for i, (name, ts) in enumerate(events):
        names.append(name)
        xs.append((ts - t0_ns) / 1e9)
        colors.append(_PALETTE[i % len(_PALETTE)])
    cds = ColumnDataSource(data=dict(
        name=names, x=xs, y=[0.5] * len(events), color=colors,
    ))
    g = fig.scatter("x", "y", source=cds, marker="inverted_triangle",
                    size=12, color="color")
    for x, c in zip(xs, colors):
        fig.add_layout(Span(location=x, dimension="height",
                            line_color=c, line_width=1.0, line_alpha=0.5))
    fig.add_tools(HoverTool(renderers=[g],
        tooltips=[("event", "@name"), ("t", "@x{0.000}s")]))
    return fig


_LANE_PX = 16   # process timeline: pixels per lane
_TIMELINE_CHAR_PX = 7 * 96 / 72 * 0.55   # 7pt label: an upper estimate of a character's width


def _timeline_data(projector: TraceProjector, proj: dict, t0_ns: int, t_end_ns: int):
    """(processes, lanes, n_lanes, links) of the process timeline."""
    first, last = {}, {}
    for (fqn, key), (ts, _v) in proj.items():
        d = projector.descriptors.get(fqn)
        if ts.size and d is not None and d.scope == mc_pb.SCOPE_PROCESS:
            first[key] = min(first.get(key, int(ts[0])), int(ts[0]))
            last[key] = max(last.get(key, int(ts[-1])), int(ts[-1]))
    procs = process_timeline.build(projector.process_table, t0_ns, t_end_ns,
                                   first_sample_ns=first, last_sample_ns=last)
    lanes, n_lanes = process_timeline.pack_lanes(procs)
    return procs, lanes, n_lanes, process_timeline.fork_links(procs, lanes)


def _build_process_timeline(procs, lanes, n_lanes, links, t0_ns: int, t_end_ns: int,
                            x_range, pid_colors: dict) -> "figure":
    """Bokeh equivalent of visualize_all._render_process_timeline: one bar
    per process (lifetime, lane packed), listed roots outlined solid,
    orphans dashed, fork links from parent to child; `comm (pid)` on the
    bars wide enough at the full view, everything in the hover."""
    # Every process labelled (process_timeline.place_labels): inside its
    # bar where the label fits at the full view, else in rows under the
    # lanes with a leader, spread so none overlaps another.
    text_px = lambda t: len(t) * _TIMELINE_CHAR_PX
    placed, n_rows = process_timeline.place_labels(procs, lanes, t0_ns, t_end_ns,
                                                   _FRAME_WIDTH, text_px, pad_units=10.0)
    height = process_timeline.height_in_lanes(n_lanes, n_rows)
    fig = figure(
        title="Processes (bars: lifetime, packed into the fewest lanes; lines: fork links)",
        width=1200, frame_height=max(1, round(height * _LANE_PX)), frame_width=_FRAME_WIDTH,
        min_border_left=_STRIP_LEFT_PX, x_range=x_range, y_range=Range1d(height, 0),
        tools=[], toolbar_location=None, output_backend=_RENDER_BACKEND,
    )
    fig.yaxis.visible = False
    fig.ygrid.visible = False
    fig.xaxis.visible = False
    span_s = max((t_end_ns - t0_ns) / 1e9, 1e-9)
    by_kind: dict = {k: dict(left=[], right=[], top=[], bottom=[], color=[], name=[], pid=[],
                             ppid=[], kind=[], start=[], end=[])
                     for k in (process_timeline.ROOT, process_timeline.DISCOVERED,
                               process_timeline.ORPHAN)}
    for p in procs:
        lane = lanes[p.key]
        left, right = (p.start_ns - t0_ns) / 1e9, (p.end_ns - t0_ns) / 1e9
        color = pid_colors.get(p.pid, "#9e9e9e")
        d = by_kind[p.kind]
        d["left"].append(left)
        d["right"].append(max(right, left + span_s * 1e-4))
        d["top"].append(lane + 0.11)
        d["bottom"].append(lane + 0.89)
        d["color"].append(color)
        d["name"].append(process_timeline.name_history(p))
        d["pid"].append(p.pid)
        d["ppid"].append(p.ppid)
        d["kind"].append(p.kind + (", started before the trace" if p.started_before else "")
                         + (", alive at the end" if p.alive else ""))
        d["start"].append(left)
        d["end"].append(right)
    style = {process_timeline.ROOT: dict(line_color="black", line_width=1.5),
             process_timeline.DISCOVERED: dict(line_color="color", line_width=0.5),
             process_timeline.ORPHAN: dict(line_color="black", line_width=1.0,
                                           line_dash="dashed")}
    shown = []
    names = {process_timeline.ROOT: "listed root", process_timeline.DISCOVERED: "discovered",
             process_timeline.ORPHAN: "orphan (parent not tracked)"}
    bars = []
    for kind, d in by_kind.items():
        bars.append(fig.quad(left="left", right="right", top="top", bottom="bottom",
                             source=ColumnDataSource(d), fill_color="color", fill_alpha=0.85,
                             **style[kind]))
        # Legend swatch: the kind's outline on neutral grey (a bar's own
        # colour is its process's, not the kind's), drawn at negative time,
        # outside the x range's bounds.
        swatch = dict(style[kind], line_color=("#bbbbbb" if kind == process_timeline.DISCOVERED
                                               else style[kind]["line_color"]))
        shown.append((names[kind], fig.quad(left=[-2.0], right=[-1.0], top=[0.1], bottom=[0.9],
                                            fill_color="#bbbbbb", **swatch)))
    link = dict(x=[(lk.t_ns - t0_ns) / 1e9 for lk in links],
                y0=[lk.parent_lane + 0.5 for lk in links],
                y1=[lk.child_lane + 0.5 for lk in links])
    seg = fig.segment(x0="x", y0="y0", x1="x", y1="y1", source=ColumnDataSource(link),
                      line_color="#444444", line_width=1.0)
    fig.scatter("x", "y0", source=ColumnDataSource(link), size=3, color="#444444")
    shown.append(("fork link (parent -> child)", seg))
    inside = dict(x=[], y=[], text=[])
    outside = dict(x=[], y=[], text=[])
    leaders = dict(x0=[], y0=[], x1=[], y1=[])
    for lab in placed:
        if lab.row < 0:
            inside["x"].append(lab.x_s)
            inside["y"].append(lab.lane + 0.5)
            inside["text"].append(lab.text)
        else:
            y = process_timeline.label_row_y(n_lanes, lab.row)
            outside["x"].append(lab.x_s)
            outside["y"].append(y)
            outside["text"].append(lab.text)
            leaders["x0"].append(lab.anchor_s)
            leaders["y0"].append(lab.lane + 0.89)
            leaders["x1"].append(lab.x_s)
            leaders["y1"].append(y - process_timeline.LABEL_ROW * 0.42)
    lead = fig.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=ColumnDataSource(leaders),
                       line_color="#aaaaaa", line_width=0.5)
    fig.renderers.remove(lead)
    fig.renderers.insert(0, lead)                      # under the bars
    fig.text(x="x", y="y", text="text", source=ColumnDataSource(inside),
             text_align="center", text_baseline="middle", text_font_size="7pt",
             text_color="white")
    fig.text(x="x", y="y", text="text", source=ColumnDataSource(outside),
             text_align="center", text_baseline="middle", text_font_size="7pt",
             text_color="#333333")
    fig.add_tools(HoverTool(renderers=bars, tooltips=[
        ("process", "@name (@pid)"), ("parent", "@ppid"), ("kind", "@kind"),
        ("start", "@start{0.000}s"), ("end", "@end{0.000}s")]))
    _legend_above(fig, [(lab, [r]) for lab, r in shown])
    fig.legend[0].click_policy = "none"
    return fig


def _overlay_regions(figs: list, regions, t0_ns: int) -> None:
    """Add a translucent BoxAnnotation per region to each metric panel."""
    if not regions or not figs:
        return
    for fig in figs:
        for i, (_name, s, e) in enumerate(regions):
            fig.add_layout(BoxAnnotation(
                left=(s - t0_ns) / 1e9, right=(e - t0_ns) / 1e9,
                fill_color=_PALETTE[i % len(_PALETTE)],
                fill_alpha=0.08, line_alpha=0.0,
            ))


# ---------------------------------------------------------------------------
# Loading overlay (static HTML)
# ---------------------------------------------------------------------------

# CSS + DOM for a full-page spinner that covers the page until Bokeh
# finishes hydrating every document on the page. Without this, a multi-MB
# HTML can look frozen for several seconds while the browser parses JSON
# and lays out canvases.
def _loading_overlay_head(theme: dict) -> str:
    """CSS for the loading overlay + page body background. Pulls
    colors from the active theme (light or dark)."""
    return f"""\
<style>
  html, body {{ background: {theme["page_bg"]}; color: {theme["page_fg"]}; }}
  #cupti-loading-overlay {{
    position: fixed; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px;
    background: {theme["page_bg"]};
    z-index: 999999;
    transition: opacity 0.25s ease;
    font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: {theme["page_fg"]};
  }}
  #cupti-loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
  #cupti-loading-overlay .spinner {{
    width: 44px; height: 44px;
    border: 4px solid {theme["overlay_track"]};
    border-top-color: {theme["overlay_accent"]};
    border-radius: 50%;
    animation: cupti-spin 0.9s linear infinite;
  }}
  #cupti-loading-overlay .label  {{ font-weight: 500; }}
  #cupti-loading-overlay .meta   {{ opacity: 0.7; font-size: 12px; font-variant-numeric: tabular-nums; }}
  /* Indeterminate bar — slides L→R via pure CSS so it animates even
     while the JS thread is blocked in Bokeh hydration. JS-driven
     progress can't update during that window. */
  #cupti-loading-overlay .bar {{
    width: 280px; height: 6px; background: {theme["overlay_track"]}; border-radius: 3px;
    overflow: hidden; position: relative;
  }}
  #cupti-loading-overlay .bar::after {{
    content: ""; position: absolute; top: 0; left: 0;
    width: 40%; height: 100%; background: {theme["overlay_accent"]}; border-radius: 3px;
    animation: cupti-slide 1.4s ease-in-out infinite;
  }}
  @keyframes cupti-spin  {{ to {{ transform: rotate(360deg); }} }}
  @keyframes cupti-slide {{
    0%   {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(350%); }}
  }}
</style>
"""

_LOADING_OVERLAY_BODY_TEMPLATE = """\
<div id="cupti-loading-overlay">
  <div class="spinner"></div>
  <div class="label">Rendering profile…</div>
  <div class="bar"></div>
  <div class="meta" id="cupti-loading-meta">0.0s elapsed</div>
</div>
<script>
(function () {
  var overlay = document.getElementById("cupti-loading-overlay");
  var meta    = document.getElementById("cupti-loading-meta");
  if (!overlay) return;
  var nFigs   = __N_FIGS__;
  var t0      = performance.now();

  // The elapsed-time text can't update while the JS thread is blocked
  // by Bokeh hydration (which is the slow part). Once the thread is
  // free, this tick resumes and shows the final elapsed value before
  // we fade out. Once canvases start appearing we also report a panel
  // count so users can see real progress in the final paint phase.
  function tick() {
    var dt = ((performance.now() - t0) / 1000).toFixed(1);
    var nCanvas = Math.min(document.querySelectorAll("canvas").length, nFigs);
    if (nCanvas > 0) {
      meta.textContent = dt + "s elapsed  ·  " + nCanvas + " / " + nFigs + " panels";
    } else {
      meta.textContent = dt + "s elapsed";
    }
    setTimeout(tick, 100);
  }
  tick();

  // Readiness heuristic: bokeh emits `<div data-root-id="...">` at
  // page-template time (empty) and populates it with the layout
  // subtree once embed_items has run. So `children.length > 0` on
  // that div is the most reliable "rendered" signal across browsers.
  // We avoid Bokeh.documents[0].is_ready (undefined on this version,
  // would keep the overlay up forever) and the canvas count (zero in
  // headless test environments, but works in real browsers).
  function rendered() {
    var root = document.querySelector("[data-root-id]");
    return !!root && root.children.length > 0;
  }
  function check() {
    if (rendered()) {
      requestAnimationFrame(function () {
        overlay.classList.add("hidden");
        setTimeout(function () { overlay.remove(); }, 350);
      });
      return;
    }
    setTimeout(check, 80);
  }
  check();
})();
</script>
"""


def _loading_overlay_body(n_figures: int) -> str:
    return _LOADING_OVERLAY_BODY_TEMPLATE.replace("__N_FIGS__", str(n_figures))


def _inject_write_rate_footer(html: str, text: str, theme: dict) -> str:
    """Append the write-rate table as a raw <pre> block right before
    the last </body>. Sidesteps Bokeh's widget layout (which serialises
    Div/PreText into docs_json but doesn't materialise a DOM node under
    this Bokeh version) and its Div HTML-escaping.
    """
    import html as _html
    color = theme.get("axis_label", "#888888")
    footer = (
        f"<pre style=\"margin:12px 0 12px 12px;font-family:monospace;"
        f"font-size:11px;line-height:1.35;color:{color};\">"
        f"{_html.escape(text)}"
        f"</pre>"
    )
    i = html.rfind("</body>")
    if i < 0:
        return html
    return html[:i] + footer + html[i:]


def _inject_loading_overlay(html: str, n_figures: int) -> str:
    """Inject a determinate progress overlay into the bokeh-generated
    HTML. Hides once every Bokeh document reports is_ready *and* the
    number of rendered <canvas> elements has caught up with
    `n_figures` — which lets users see real progress (X / N panels +
    elapsed seconds) instead of an indeterminate spinner.

    Anchors on the LAST `</head>` / `</body>` in the source, not the
    first. Bokeh's embedded minified JS contains string literals
    like `"<body></body></html>"` (DOMParser fallback templates in
    the sanitizer/DOMPurify code path), and replacing the first
    occurrence splices the overlay's `</script></body></html>` into
    the middle of a JS string — corrupting every subsequent JS
    statement and producing a blank page with 0 canvases. The last
    occurrence is guaranteed to be the real closing tag because
    Bokeh emits the doc's `<script>` blocks BEFORE the closing
    `</body>`. The overlay is `position: fixed`, so DOM insertion
    point doesn't affect its visual placement.
    """
    def _replace_last(hay: str, needle: str, replacement: str) -> str:
        i = hay.rfind(needle)
        if i < 0:
            return hay
        return hay[:i] + replacement + hay[i + len(needle):]

    theme = _THEMES[_THEME]
    html = _replace_last(html, "</head>",
                         _loading_overlay_head(theme) + "</head>")
    html = _replace_last(html, "</body>",
                         _loading_overlay_body(n_figures) + "</body>")
    return html


# ---------------------------------------------------------------------------
# Foldable panels, pinned timeline, hotkeys (static page)
# ---------------------------------------------------------------------------

# The pinned process timeline scrolls inside the sticky band beyond this
# share of the window height: with the event and region strips (~140 px)
# it keeps the band under about half of a laptop-height window, so at
# least half is left for the panels scrolling under it.
_TIMELINE_MAX_VH = 35

_FOLD_CSS = InlineStyleSheet(css=".bk-btn { text-align: left; font-weight: bold; "
                                 "border: none; background: transparent; padding: 2px 4px; }")


def _fold(fig, title: str):
    """Header button of a foldable panel: '▾ title' expanded, '▸ title'
    collapsed (the figure hidden, the header kept). The button carries the
    panel's title; the figure's own is dropped so it is not shown twice."""
    fig.title.text = ""
    btn = Button(label="\u25be " + title, sizing_mode="stretch_width", height=24,
                 stylesheets=[_FOLD_CSS], name="fold")
    btn.js_on_click(CustomJS(args=dict(fig=fig, btn=btn, title=title), code="""
        fig.visible = !fig.visible;
        btn.label = (fig.visible ? "\u25be " : "\u25b8 ") + title;
    """))
    return btn, column([btn, fig], spacing=0, sizing_mode="stretch_width")


_FOLD_ALL_JS = """
    for (let i = 0; i < figs.length; i++) {
        figs[i].visible = !collapse;
        btns[i].label = (collapse ? "\u25b8 " : "\u25be ") + titles[i];
    }
"""


def _fold_all_buttons(folds: list):
    """'Collapse all' / 'Expand all' for folds = [(fig, button, title)]."""
    args = dict(figs=[f for f, _b, _t in folds], btns=[b for _f, b, _t in folds],
                titles=[t for _f, _b, t in folds])
    out = []
    for label, collapse in (("Collapse all", True), ("Expand all", False)):
        b = Button(label=label, width=110, height=26, name="fold-all")
        b.js_on_click(CustomJS(args=dict(args, collapse=collapse), code=_FOLD_ALL_JS))
        out.append(b)
    return out


_KEYS = [("r / 0", "reset zoom"), ("= / +", "zoom in (2x, around the centre)"),
         ("-", "zoom out"), ("\u2190 / \u2192", "pan 10% of the view (Shift: 50%)"),
         ("c", "collapse / expand all panels"), ("?", "this help")]


def _hotkeys_script(xr_id: str, t_end_s: float, folds: list) -> str:
    """Keyboard shortcuts on the shared x-range of every panel and the
    fold state, plus a help overlay ('?'). Plain JS on the document,
    ignored while typing in an input / textarea."""
    import json
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in _KEYS)
    fold_ids = json.dumps([[f.id, b.id, t] for f, b, t in folds])
    return f"""
<div id="cupti-keys-help" style="display:none;position:fixed;right:16px;bottom:16px;z-index:1000;
     background:#fff;border:1px solid #bbb;border-radius:6px;padding:8px 12px;
     font:12px sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.2)">
  <div style="font-weight:bold;margin-bottom:4px">Keys</div><table>{rows}</table></div>
<script>
(function () {{
  const XR = {json.dumps(xr_id)}, T_END = {t_end_s!r}, FOLDS = {fold_ids};
  function model(id) {{
    const docs = (window.Bokeh && Bokeh.documents) || [];
    for (const d of docs) {{ const m = d.get_model_by_id(id); if (m) return m; }}
    return null;
  }}
  function setRange(xr, a, b) {{
    const hi = T_END * 1.4, w = b - a;
    if (a < 0) {{ a = 0; b = w; }}
    if (b > hi) {{ b = hi; a = Math.max(0, hi - w); }}
    xr.start = a; xr.end = b;
  }}
  function foldAll() {{
    const any = FOLDS.some(([f]) => model(f) && model(f).visible);
    for (const [f, b, t] of FOLDS) {{
      const fig = model(f), btn = model(b);
      if (!fig || !btn) continue;
      fig.visible = !any;
      btn.label = (any ? "\u25b8 " : "\u25be ") + t;
    }}
  }}
  window.cuptiHotkey = function (key, shift) {{
    const xr = model(XR);
    if (!xr) return false;
    const a = xr.start, b = xr.end, w = b - a, c = (a + b) / 2;
    switch (key) {{
      case "r": case "0": setRange(xr, 0, T_END); break;
      case "=": case "+": setRange(xr, c - w / 4, c + w / 4); break;
      case "-": case "_": setRange(xr, c - w, c + w); break;
      case "ArrowLeft":  setRange(xr, a - w * (shift ? 0.5 : 0.1), b - w * (shift ? 0.5 : 0.1)); break;
      case "ArrowRight": setRange(xr, a + w * (shift ? 0.5 : 0.1), b + w * (shift ? 0.5 : 0.1)); break;
      case "c": foldAll(); break;
      case "?": {{ const h = document.getElementById("cupti-keys-help");
                  h.style.display = h.style.display === "none" ? "block" : "none"; break; }}
      default: return false;
    }}
    return true;
  }};
  // The shared x-range as [start, end] (for checking the keys).
  window.cuptiHotkey.range = function () {{
    const xr = model(XR);
    return xr ? [xr.start, xr.end] : null;
  }};
  document.addEventListener("keydown", function (e) {{
    const t = e.target;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (window.cuptiHotkey(e.key, e.shiftKey)) e.preventDefault();
  }});
}})();
</script>
"""


# ---------------------------------------------------------------------------
# Static rendering path
# ---------------------------------------------------------------------------

class StaticDocument:
    """_build_static_document's result: the page's root layout, the
    sticky strips (event, region) and the panels as (panel, kind, figure)
    top to bottom, kind "metric" or "cumulative"."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _render_static(
    projector: TraceProjector,
    layout: metric_layout.PanelLayout,
    catalog_index: dict[str, mc_pb.MetricDescriptor],
    out_path: Path,
    title: str,
    meta: session_metadata_pb2.SessionMetadata,
    metadata_path: Path,
    sample_freqs: dict[str, int],
    probes_info: dict[str, dict],
    smooth_window_s: float,
    display_hz: float,
) -> None:
    doc = _build_static_document(projector, layout, catalog_index, meta, metadata_path,
                                 sample_freqs, probes_info, smooth_window_s, display_hz)
    if doc is None:
        return
    theme = _THEMES[_THEME]
    # file_html silently ignores theme string names — resolve to a
    # Theme object via built_in_themes. None falls through to stock.
    bokeh_theme = (built_in_themes[theme["bokeh_theme"]]
                   if theme["bokeh_theme"] else None)
    html = _static_html(doc, title, bokeh_theme, theme)
    out_path.write_text(html)
    _log(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KiB; "
         f"{doc.n_figures} panels)")


def _static_html(doc, title: str, bokeh_theme, theme: dict) -> str:
    html = file_html(doc.root, INLINE, title=title, theme=bokeh_theme)
    html = _inject_loading_overlay(html, doc.n_figures)
    if doc.footer_text:
        html = _inject_write_rate_footer(html, doc.footer_text, theme)
    if doc.x_range is not None:
        i = html.rfind("</body>")
        html = html[:i] + _hotkeys_script(doc.x_range.id, doc.t_end_s, doc.folds) + html[i:]
    return html


def static_page(metadata, **kw) -> str:
    """build_static() rendered to the page's HTML (not written)."""
    doc = build_static(metadata, **kw)
    return _static_html(doc, "Profile", None, _THEMES[_THEME])


def build_static(metadata, *, catalog=None, panel_layout=None,
                 smooth_window_s: float = 0.0, display_hz: float = 0.0,
                 unit_scale_factor: float = units.DEFAULT_SCALE_FACTOR,
                 fit_axis_to_data: bool = False):
    """Load the trace whose session_metadata.pb is `metadata` and build
    the static page's Bokeh document (not written). Returns a
    StaticDocument, or None when there is nothing to render.
    unit_scale_factor: see units.py (ValueError < 1); fit_axis_to_data:
    see --fit-axis-to-data."""
    global _FIT_AXIS
    units.set_scale_factor(unit_scale_factor)
    _FIT_AXIS = fit_axis_to_data
    metadata_path = Path(metadata).resolve()
    meta = _load_session_metadata(metadata_path)
    cat = (metric_catalog.load_catalog(catalog) if catalog
           else metric_catalog.load_catalog_from_session_metadata(meta))
    layout_path = Path(panel_layout) if panel_layout \
        else _HERE.parent / "configs" / "visualizer_panels.pbtxt"
    layout = metric_layout.load_panel_layout(layout_path)
    projector = TraceProjector(cat)
    sample_freqs, probes_info = _ingest_probes(projector, meta, metadata_path)
    return _build_static_document(projector, layout, metric_catalog.build_index(cat),
                                  meta, metadata_path, sample_freqs, probes_info,
                                  smooth_window_s, display_hz)


def _build_static_document(
    projector: TraceProjector,
    layout: metric_layout.PanelLayout,
    catalog_index: dict[str, mc_pb.MetricDescriptor],
    meta: session_metadata_pb2.SessionMetadata,
    metadata_path: Path,
    sample_freqs: dict[str, int],
    probes_info: dict[str, dict],
    smooth_window_s: float,
    display_hz: float,
):
    proj = projector.project()
    if not proj:
        _log("no samples — nothing to render")
        return None
    t0_ns = min(int(ts[0]) for ts, _ in proj.values() if ts.size > 0)
    t_end_ns = max(int(ts[-1]) for ts, _ in proj.values() if ts.size > 0)
    series_keys = list(proj.keys())

    # Promote synthesized GPU descriptors.
    for (fqn, _) in series_keys:
        if fqn not in catalog_index:
            catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

    # Optional boxcar smoothing on smoothable metrics, then optional
    # stride-decimation to a target display rate. Smoothing acts as
    # the anti-aliasing filter so decimation doesn't fold high-
    # frequency content back into the visible band. The cumulative
    # companions stay at raw resolution so the integrated total
    # matches reality.
    needs_pass = smooth_window_s > 0 or display_hz > 0
    if needs_pass:
        smoothed_proj = {}
        for (fqn, scope_key), (ts_ns, vals) in proj.items():
            d = catalog_index.get(fqn)
            probe = projector.fqn_to_probe.get(fqn, "")
            freq = sample_freqs.get(probe, 0)
            if (d is not None and d.smoothable
                    and smooth_window_s > 0 and freq > 0 and ts_ns.size > 0):
                k = _kernel_size(freq, smooth_window_s)
                vals = _smooth(vals.astype(np.float64), k)
            if display_hz > 0:
                ts_ns, vals = _decimate_to_hz(ts_ns, vals, freq, display_hz)
            smoothed_proj[(fqn, scope_key)] = (ts_ns, vals)
        if smooth_window_s > 0:
            _log(f"smoothing window: {smooth_window_s*1000:.0f} ms boxcar")
        if display_hz > 0:
            _log(f"display rate:     {display_hz:g} Hz (stride decimation)")
    else:
        smoothed_proj = proj

    regions, events = _load_events_for_session(meta, metadata_path)
    _log(f"events: {len(regions)} regions, {len(events)} events")
    # Extend the time window so events/regions outside the metric
    # window aren't clipped by the x-axis bounds.
    for _n, ts in events:
        if ts < t0_ns:    t0_ns    = ts
        if ts > t_end_ns: t_end_ns = ts
    for _n, s, e in regions:
        if s < t0_ns:    t0_ns    = s
        if e > t_end_ns: t_end_ns = e

    resolved = [(panel, metric_layout.resolve_panel_series(panel, catalog_index, series_keys))
                for panel in layout.panels]
    resolved = [(panel, series) for panel, series in resolved if series]
    # One colour per process on the whole page (panels and timeline).
    pid_colors = panel_legend.pid_color_map([(p, s, "metric") for p, s in resolved], proj)
    # One hue per base metric across the page (statistics share it).
    metric_colors = panel_legend.metric_color_map([(p, s, "metric") for p, s in resolved])

    metric_figs: list = []
    figs: list = []
    panel_figs: list = []
    shared_x = None
    for panel, series in resolved:
        fig, _ = _build_panel(panel, series, projector, smoothed_proj, t0_ns,
                              x_range=shared_x, pid_colors=pid_colors,
                              metric_colors=metric_colors)
        if shared_x is None:
            shared_x = fig.x_range
        figs.append(fig)
        metric_figs.append(fig)
        panel_figs.append((panel, "metric", fig))
        # Companion cumulative panel, directly below — opt-in per
        # panel via aggregation: PANEL_AGGREGATION_INTEGRATE. Uses the
        # unsmoothed `proj` so the integrated total is faithful; the
        # display-only decimation happens inside the builder.
        if _panel_has_cumulative_companion(panel, series):
            cum_series, cum_proj = series, proj
            summed = panel.aggregation == panels_pb.PANEL_AGGREGATION_INTEGRATE_SUM
            if summed:           # each metric summed over its instances first
                cum_series, sums = panel_legend.aggregate(series, proj)
                cum_proj = {**proj, **sums}
            source_hzs = {
                (s.fqn, s.scope_key):
                    sample_freqs.get(projector.fqn_to_probe.get(s.fqn, ""), 0)
                for s in cum_series
            }
            cum_fig, _ = _build_cumulative_panel(
                panel, cum_series, projector, cum_proj, t0_ns, x_range=shared_x,
                display_hz=display_hz, source_hzs=source_hzs, pid_colors=pid_colors,
                metric_colors=metric_colors, aggregated=summed)
            figs.append(cum_fig)
            metric_figs.append(cum_fig)
            panel_figs.append((panel, "cumulative", cum_fig))
        elif panel.aggregation == panels_pb.PANEL_AGGREGATION_INTEGRATE:
            _log(f"  panel {panel.title!r}: aggregation INTEGRATE set but "
                 "source unit has no integrated mapping — skipping companion")

    # Pan/zoom guardrails on the shared x range.
    #   - left bound at 0 keeps the no-negative-time invariant
    #   - right bound is dynamic: t_end + 0.4 * current_window_length
    #     (so when zoomed all the way out, the trace fills at least
    #     ~60% of the view; tighter views get a tighter cap, so users
    #     can't pan/zoom into a sea of empty space)
    # The "current window length" piece can't be expressed as a
    # static bound, so a CustomJS callback recomputes it on every
    # x_range change.
    if shared_x is not None:
        x_end_s = (t_end_ns - t0_ns) / 1e9
        shared_x.bounds = (0.0, x_end_s + x_end_s * 0.4)
        dyn_bounds_cb = CustomJS(args=dict(rng=shared_x, t_end=x_end_s),
                                 code="""
            const wl = rng.end - rng.start;
            if (wl <= 0) return;
            rng.bounds = [0.0, t_end + wl * 0.4];
        """)
        shared_x.js_on_change("start", dyn_bounds_cb)
        shared_x.js_on_change("end",   dyn_bounds_cb)

    # Region-shaded overlays on every metric panel (translucent so
    # they don't drown the traces).
    _overlay_regions(metric_figs, regions, t0_ns)

    # Strips above the panels: events first (so its hairlines line up
    # vertically with the panels below), then regions.
    strips: list = []
    if shared_x is not None and events:
        strips.append(_build_event_strip(events, t0_ns, shared_x))
    if shared_x is not None and regions:
        strips.append(_build_region_strip(regions, t0_ns, shared_x))

    # The process timeline, right under the (sticky) strips, scrolling.
    timeline_fig = None
    if shared_x is not None:
        procs, lanes, n_lanes, links = _timeline_data(projector, proj, t0_ns, t_end_ns)
        if procs:
            timeline_fig = _build_process_timeline(
                procs, lanes, n_lanes, links, t0_ns, t_end_ns, shared_x, pid_colors)

    # Drop the Bokeh logo from every toolbar (15 logos across the
    # page is visual noise; the framework is implicit from the file
    # extension). The wheel-zoom's maintain_focus=False is set at
    # construction time inside _make_plot_tools.
    for f in strips + figs:
        f.toolbar.logo = None

    # Wrap both strips in their own Column with spacing=0 so there's
    # no transparent gap between them during scroll, and make THAT
    # the sticky element. The dashed bottom border separates the
    # sticky band from the scrolling metric panels.
    # Every panel (and the timeline) foldable under a header that keeps
    # its title; collapse / expand all at the top.
    folds: list = []                                   # (fig, button, title)
    folded_panels: list = []
    for panel, kind, fig in panel_figs:
        title = fig.title.text                      # the builders' titles
        btn, wrapped = _fold(fig, title)
        folds.append((fig, btn, title))
        folded_panels.append(wrapped)
    timeline_block = None
    if timeline_fig is not None:
        tl_title = timeline_fig.title.text
        tl_btn, timeline_block = _fold(timeline_fig, tl_title)
        folds.insert(0, (timeline_fig, tl_btn, tl_title))
        # Pinned with the strips; beyond _TIMELINE_MAX_VH of the window it
        # scrolls inside the band.
        # The figure keeps its full height (Bokeh would otherwise shrink it
        # to fit the cap); the block scrolls over it.
        timeline_fig.height_policy = "fixed"
        timeline_block.height_policy = "fit"
        timeline_block.styles = {"max-height": f"{_TIMELINE_MAX_VH}vh", "overflow-y": "auto"}

    theme = _THEMES[_THEME]
    layout_children: list = []
    # The fold-all buttons ride in the sticky band too, always at hand.
    band = ([row(_fold_all_buttons(folds))] if folds else []) + strips \
        + ([timeline_block] if timeline_block is not None else [])
    if band:
        strip_col = column(band, spacing=0, sizing_mode="stretch_width")
        strip_col.styles = {
            "position":      "sticky",
            "top":           "0",
            "z-index":       "50",
            "background":    theme["strip_bg"],
            "border-bottom": f"1px dashed {theme['strip_border']}",
        }
        layout_children.append(strip_col)
    layout_children.extend(folded_panels)

    # Write-rate footer (mirrors visualize_all.py's static-PNG table).
    # Kept at the very bottom of the scrolling column so it doesn't
    # interfere with the sticky strip band or the metric panels above.
    footer_rows = _build_write_rate_rows(
        projector=projector,
        proj=proj,
        sample_freqs=sample_freqs,
        probes_info=probes_info,
        events_regions=len(regions),
        events_ns=len(events),
        xmax_s=(t_end_ns - t0_ns) / 1e9,
    )
    # Footer text is stashed and injected as raw HTML in _inject_footer
    # below (after file_html produces the Bokeh document). We tried both
    # Div (which HTML-escapes) and PreText (which serializes to
    # docs_json but doesn't materialise into the DOM under this
    # Bokeh version's widget layout), so we sidestep the widget layer.
    footer_text = _format_write_rate_footer(footer_rows) if footer_rows else None

    layout_root = column(layout_children, sizing_mode="stretch_width")
    return StaticDocument(root=layout_root, strips=strips, panel_figs=panel_figs,
                          timeline=timeline_fig, footer_text=footer_text,
                          n_figures=len(strips) + len(figs) + (timeline_fig is not None),
                          band=strip_col if band else None, timeline_block=timeline_block,
                          folds=folds, x_range=shared_x,
                          t_end_s=(t_end_ns - t0_ns) / 1e9)


# ---------------------------------------------------------------------------
# Minimal HTTP server (static mode)
# ---------------------------------------------------------------------------

def _serve(html_path: Path, host: str, port: int, open_browser: bool) -> None:
    serve_dir = html_path.parent
    fname = html_path.name

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, fmt, *args):
            pass

    # SO_REUSEADDR so restarts during a TIME_WAIT window succeed.
    class _ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = _ReusableTCPServer((host, port), _Handler)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/{fname}"
    _log(f"serving at {url} (Ctrl-C to quit)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("shutting down server")
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _unit_scale_factor(text: str) -> float:
    """argparse type for --unit-scale-factor (a clear error below 1)."""
    try:
        f = float(text)
        units.set_scale_factor(f)          # validates
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return f


def main() -> int:
    global _RENDER_BACKEND, _THEME, _FIT_AXIS
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", help="Path to session_metadata.pb")
    parser.add_argument("-o", "--output", default="profile.html",
                        help="Output HTML path (default: profile.html)")
    parser.add_argument("--catalog", default=None,
                        help="Override MetricCatalog pbtxt")
    parser.add_argument("--panel-layout", default=None,
                        help="Override PanelLayout pbtxt")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-serve", action="store_true",
                        help="Render HTML and exit (don't start a server).")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open a browser tab.")
    parser.add_argument("--live", action="store_true",
                        help="Stream new samples as the suite writes them.")
    parser.add_argument("--poll-interval-ms", type=int, default=1000,
                        help="Live mode: tail-poll interval in ms (default: 1000)")
    parser.add_argument("--live-bootstrap-timeout-s", type=float, default=30.0,
                        help="Live mode: how long to wait for session_metadata.pb "
                             "to appear (default: 30.0)")
    parser.add_argument("--allow-websocket-origin", action="append", default=None,
                        help="Live mode: extra origin allowed for the Bokeh "
                             "WebSocket. Repeatable. Default: '*' (any).")
    parser.add_argument("--fit-axis-to-data", action="store_true",
                        help="Size a panel's y-axis to its data when its "
                             "ceiling (the Peak line) is more than "
                             f"{units.OFFSCALE_FACTOR:g}x the largest plotted "
                             "value; the Peak is then written as 'Peak: <value> "
                             "(off-scale)' instead of drawn. Default: off (the "
                             "axis reaches the Peak).")
    parser.add_argument("--unit-scale-factor", type=_unit_scale_factor,
                        default=units.DEFAULT_SCALE_FACTOR, metavar="F",
                        help="Byte-unit threshold: an axis (and every run total "
                             "and footer rate) uses the largest prefix P with "
                             "max value >= F x P (default: %(default)s; 1 = "
                             "switch as soon as a value reaches the prefix; "
                             "must be >= 1).")
    parser.add_argument("--smooth-window-s", type=float, default=0.0,
                        help="Boxcar smoothing window in seconds applied to "
                             "every smoothable metric. 0 = none. Kernel size "
                             "is per-probe (uses each probe's sampling rate). "
                             "The cumulative companion panels stay unsmoothed "
                             "so integrated totals remain faithful.")
    parser.add_argument("--display-hz", type=float, default=0.0,
                        help="Downsample every series to this rate (in Hz) "
                             "before rendering, via stride decimation. 0 "
                             "(the default) keeps the raw sampling rate. "
                             "Applied after smoothing (so smoothing acts as "
                             "the anti-aliasing filter). Cumulative panels "
                             "still compute their total from the raw series; "
                             "decimation only affects how many points are "
                             "sent to the browser.")
    parser.add_argument("--theme", default=_THEME, choices=tuple(_THEMES),
                        help="Color theme. 'light' (default) keeps the "
                             "stock white-on-grey Bokeh styling; 'dark' "
                             "applies Bokeh's dark_minimal theme to every "
                             "plot and flips the page background, loading "
                             "overlay, and sticky-strip fills to match.")
    parser.add_argument("--render-backend", default=_RENDER_BACKEND,
                        choices=("webgl", "canvas", "svg"),
                        help="Bokeh output backend per panel. canvas (the "
                             "default) is ~4-5× faster to first paint with "
                             "our trace volume; webgl wins on pan/zoom "
                             "repaint smoothness but pays a steep init cost "
                             "per plot (and on some GPU drivers stalls "
                             "catastrophically on ReadPixels).")
    args = parser.parse_args()
    _RENDER_BACKEND = args.render_backend
    _THEME = args.theme
    units.set_scale_factor(args.unit_scale_factor)   # static and live mode alike
    _FIT_AXIS = args.fit_axis_to_data

    metadata_path = Path(args.metadata).resolve()
    if args.live:
        return _run_live(args, metadata_path)


    _log(f"loading session metadata from {metadata_path}")
    meta = _load_session_metadata(metadata_path)

    if args.catalog:
        catalog = metric_catalog.load_catalog(args.catalog)
    else:
        catalog = metric_catalog.load_catalog_from_session_metadata(meta)
    _log(f"catalog: {len(catalog.metrics)} descriptors")

    layout_path = Path(args.panel_layout) if args.panel_layout \
        else _HERE.parent / "configs" / "visualizer_panels.pbtxt"
    layout = metric_layout.load_panel_layout(layout_path)
    _log(f"layout: {len(layout.panels)} panels")

    projector = TraceProjector(catalog)
    _log("ingesting probes")
    sample_freqs, probes_info = _ingest_probes(projector, meta, metadata_path)

    catalog_index = metric_catalog.build_index(catalog)
    out_path = Path(args.output).resolve()
    title = f"Profile — {meta.hostname}  {meta.start_iso8601}"
    _log("rendering")
    _render_static(projector, layout, catalog_index, out_path, title,
                   meta, metadata_path,
                   sample_freqs=sample_freqs,
                   probes_info=probes_info,
                   smooth_window_s=args.smooth_window_s,
                   display_hz=args.display_hz)

    if not args.no_serve:
        _serve(out_path, args.host, args.port, not args.no_browser)
    return 0


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def _run_live(args, metadata_path: Path) -> int:
    """Spin up a Bokeh server that tails the .pb files and streams
    new samples into the document on every periodic callback."""
    import live_tail  # local import — only needed in --live mode

    _log(f"--live  waiting for {metadata_path} (timeout {args.live_bootstrap_timeout_s}s)")
    try:
        meta = live_tail.wait_for_metadata(
            metadata_path, args.live_bootstrap_timeout_s, _log)
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.catalog:
        catalog = metric_catalog.load_catalog(args.catalog)
    else:
        catalog = metric_catalog.load_catalog_from_session_metadata(meta)
    _log(f"catalog: {len(catalog.metrics)} descriptors")

    layout_path = Path(args.panel_layout) if args.panel_layout \
        else _HERE.parent / "configs" / "visualizer_panels.pbtxt"
    layout = metric_layout.load_panel_layout(layout_path)
    _log(f"layout: {len(layout.panels)} panels")

    # Cumulative companion panels need per-(panel, series) running-
    # total state that the current LiveCoordinator doesn't carry — see
    # _build_cumulative_panel docstring. Refuse to start live mode if
    # the active layout asks for any integrate-aggregation panel,
    # rather than silently freezing the cumulative at first paint.
    # The companion lives in the static path and as a planned follow-
    # up for live mode.
    for panel in layout.panels:
        if panel.aggregation == panels_pb.PANEL_AGGREGATION_INTEGRATE:
            raise NotImplementedError(
                f"panel {panel.title or panel.series_glob!r} has "
                "aggregation: PANEL_AGGREGATION_INTEGRATE, but cumulative "
                "companion panels are not supported in --live mode yet. "
                "Either drop --live and render statically, or set "
                "aggregation: PANEL_AGGREGATION_UNSPECIFIED on the panel "
                "in the layout pbtxt."
            )

    def make_doc(doc):
        """One coordinator + one document per browser session."""
        # Each browser session gets its own coordinator + projector
        # state. Multiple browser tabs see independent live updates
        # but each tails the same files.
        coordinator = live_tail.LiveCoordinator(
            catalog=catalog, layout=layout,
            metadata_path=metadata_path, meta=meta,
            log=_log,
            series_factory=_live_series_factory,
            panel_factory=None,  # built inline below
            t0_ns=None,
        )

        # Initial paint — read everything currently on disk.
        coordinator.tick()

        # If we still have no samples, that's OK; an empty layout will
        # populate as the workload runs. Build panels eagerly.
        catalog_index = coordinator.catalog_index
        proj = coordinator.projector.project()
        for (fqn, _) in proj.keys():
            if fqn not in catalog_index:
                catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

        series_keys = list(proj.keys())
        figs = []
        shared_x = None
        for panel in layout.panels:
            series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
            if not series and not _panel_should_eagerly_exist(panel, layout):
                continue
            fig, cds_by_key, scale_fn = _build_live_panel(
                panel, series, coordinator.projector, proj,
                coordinator.t0_ns or 0, x_range=shared_x,
            )
            if shared_x is None:
                shared_x = fig.x_range
            entry = live_tail._PanelEntry(
                panel=panel, figure=fig, scale_fn=scale_fn,
                palette_idx=len(series),
            )
            coordinator.register_panel(entry)
            for series_key, cds in cds_by_key.items():
                coordinator.register_series(series_key, cds)
            figs.append(fig)

        doc.add_root(column(figs, sizing_mode="stretch_width"))
        doc.title = f"Live profile — {meta.hostname}"
        doc.add_periodic_callback(coordinator.tick, args.poll_interval_ms)

    origins = args.allow_websocket_origin or ["*"]
    app = Application(FunctionHandler(make_doc))
    server = Server({"/": app},
                    address=args.host, port=args.port,
                    allow_websocket_origin=origins)
    server.start()
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/"
    _log(f"live server at {url}  (poll every {args.poll_interval_ms} ms)")
    _log("Ctrl-C to stop")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        _log("stopping")
    return 0


def _panel_should_eagerly_exist(panel, layout) -> bool:
    """Whether to build a panel that currently has no matching series.
    Eagerly build SYSTEM/DEVICE/GPU panels (those scopes don't grow
    mid-run), but defer PROCESS panels until at least one PID joins."""
    return panel.scope != mc_pb.SCOPE_PROCESS


def _build_live_panel(panel, series_list, projector, projection, t0_ns,
                       x_range=None):
    """Like _build_panel but also returns the scale_fn so the
    coordinator can apply it consistently to streamed deltas."""
    unit = panel.unit_override if panel.unit_override != mc_pb.UNIT_UNSPECIFIED \
        else (series_list[0].descriptor.unit if series_list else mc_pb.UNIT_COUNT)
    peak_hint = _resolve_panel_peak(
        panel,
        series_list[0].descriptor if series_list else metric_catalog.MetricDescriptor(),
        projector,
    )
    scale_fn, ylabel = _format_unit_axis(unit, peak_hint)

    tools, box_zoom, wheel_zoom = _make_plot_tools()
    fig_kwargs = dict(
        title=_panel_title(panel, series_list),
        x_axis_label="time (s)", y_axis_label=ylabel,
        width=1200, height=240, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools=tools,
        toolbar_location="left",
        active_drag=box_zoom,
        active_scroll=wheel_zoom,
        output_backend=_RENDER_BACKEND,
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    fig = figure(**fig_kwargs)

    cds_by_key: dict[tuple, ColumnDataSource] = {}
    for i, series in enumerate(series_list):
        color = _PALETTE[i % len(_PALETTE)]
        ts_ns, vals = projection[(series.fqn, series.scope_key)]
        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        cds = ColumnDataSource(data=dict(
            x=list(time_s), y=list(scale_fn(vals.astype(np.float64))),
        ))
        cds_by_key[(series.fqn, series.scope_key)] = cds
        fig.line("x", "y", source=cds, color=color, line_width=1.2,
                 legend_label=_series_label(series, projector))

    if peak_hint is not None and peak_hint > 0:
        fig.add_layout(Span(location=scale_fn(peak_hint),
                            dimension="width", line_color="red",
                            line_dash="dashed", line_alpha=0.6,
                            line_width=1.5))

    if panel.y_min != 0.0:
        fig.y_range.start = panel.y_min
    if panel.y_max != 0.0:
        fig.y_range.end = panel.y_max

    fig.add_tools(HoverTool(
        tooltips=[("t", "@x{0.000}s"), ("y", "@y{0.000}")],
        mode="vline",
    ))
    if fig.legend:
        legend = fig.legend[0]
        legend.click_policy = "hide"
        legend.label_text_font_size = "8pt"
        legend.location = "top_left"
        fig.add_layout(legend, "right")

    return fig, cds_by_key, scale_fn


def _live_series_factory(entry, fqn, scope_key, color, ts_s, vals,
                          descriptor, projector) -> ColumnDataSource:
    """Called by LiveCoordinator when a new (Scope, key) joins
    mid-run. Allocates a CDS, attaches it to the panel's figure, and
    returns the CDS so the coordinator can stream subsequent rows in."""
    cds = ColumnDataSource(data=dict(x=list(ts_s), y=list(vals)))
    # Short legend label — just the pretty counter name (the panel
    # title already conveys the entity + suffix).
    base = metric_suffix.pretty_counter(descriptor.counter, descriptor.entity) or fqn
    if descriptor.scope == mc_pb.SCOPE_PROCESS:
        tp = projector.tracked_processes.get(int(scope_key))
        if tp and tp.alias:
            label = f"{base}  [{tp.alias} (PID {scope_key})]"
        else:
            label = f"{base}  [PID {scope_key}]"
    elif descriptor.scope == mc_pb.SCOPE_GPU:
        if len(projector.gpu_info) <= 1:
            label = base
        else:
            label = f"{base}  [GPU {scope_key}]"
    elif descriptor.scope == mc_pb.SCOPE_DEVICE:
        label = f"{base}  [{scope_key}]"
    elif descriptor.scope == mc_pb.SCOPE_SYSTEM:
        label = base
    else:
        label = f"{base}  [scope_key={scope_key}]"
    entry.figure.line("x", "y", source=cds, color=color, line_width=1.2,
                       legend_label=label)
    return cds


if __name__ == "__main__":
    sys.exit(main())
