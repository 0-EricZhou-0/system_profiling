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
from metric_projector import TraceProjector  # noqa: E402

from bokeh.application import Application  # noqa: E402
from bokeh.application.handlers.function import FunctionHandler  # noqa: E402
from bokeh.embed import file_html  # noqa: E402
from bokeh.layouts import column  # noqa: E402
from bokeh.models import (BoxAnnotation, ColumnDataSource, CustomJS,  # noqa: E402
                          HoverTool, Range1d, Span, WheelZoomTool)
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
) -> dict[str, int]:
    sample_freqs: dict[str, int] = {}
    for probe in meta.probes:
        out = _resolve_path(metadata_path, probe.output_file)
        if not out.exists():
            _log(f"  skip {out} (not found)")
            continue
        if probe.kind == session_metadata_pb2.PROBE_KIND_GPU:
            for t in _read_delimited(out, gpu_metrics_pb2.GPUMetricsTrace):
                projector.ingest_gpu(t)
            sample_freqs["gpu"] = probe.sampling_frequency_hz
        elif probe.kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
            for t in _read_delimited(out, system_metrics_pb2.SystemMetricsTrace):
                projector.ingest_system(t)
            sample_freqs["system"] = probe.sampling_frequency_hz
        elif probe.kind == session_metadata_pb2.PROBE_KIND_DISK:
            for t in _read_delimited(out, disk_metrics_pb2.DiskMetricsTrace):
                projector.ingest_disk(t)
            sample_freqs["disk"] = probe.sampling_frequency_hz
    return sample_freqs


# ---------------------------------------------------------------------------
# Unit scaling — same logic as visualize_all.py
# ---------------------------------------------------------------------------

def _format_unit_axis(unit: int, peak_hint: float | None):
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


def _relax_wheel_zoom(fig) -> None:
    """Flip WheelZoomTool.maintain_focus = False on every wheel-zoom
    tool attached to the figure.

    Bokeh's default behavior anchors the wheel zoom at the cursor and
    refuses to apply a zoom step that would push *either* side of the
    range past its bounds. Near the left edge with bounds=(0, …), the
    zoom-out attempt extends below 0 and the whole gesture is rejected
    — so the user gets a 'stuck' feeling. Setting maintain_focus=False
    tells Bokeh: 'if one side hits a bound, absorb the remaining zoom
    on the other side instead of cancelling.'"""
    for t in fig.tools:
        if isinstance(t, WheelZoomTool):
            t.maintain_focus = False


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


def _series_label(series: metric_layout.ResolvedSeries,
                  projector: TraceProjector,
                  base: str | None = None) -> str:
    """Compact legend label. Hover tooltips can carry the long form.

    `base` overrides `series.label_short`; pass the disambiguated
    label from `metric_layout.disambiguate_short_labels` when rendering
    multiple series in one panel so colliding entries (e.g. avg/max
    rollups) stay distinguishable."""
    if base is None:
        base = series.label_short
    key = series.scope_key
    if series.scope == mc_pb.SCOPE_SYSTEM:
        return base
    if series.scope == mc_pb.SCOPE_PROCESS:
        tp = projector.tracked_processes.get(int(key))
        if tp and tp.alias:
            return f"{base}  [{tp.alias} (PID {key})]"
        return f"{base}  [PID {key}]"
    if series.scope == mc_pb.SCOPE_DEVICE:
        return f"{base}  [{key}]"
    if series.scope == mc_pb.SCOPE_GPU:
        # Single-GPU runs need no scope-key suffix; with multiple GPUs,
        # label by index only — the device-name string is redundant
        # with the panel title and just inflates the legend.
        if len(projector.gpu_info) <= 1:
            return base
        return f"{base}  [GPU {key}]"
    return base


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
        "page_bg":        "#15191c",
        "page_fg":        "#cdd0d4",
        "strip_bg":       "#20262d",
        "strip_border":   "#555555",
        "overlay_track":  "#333a40",
        "overlay_accent": "#5b9bd5",
    },
}
_THEME = "light"


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
    """Pbtxt `title:` wins; otherwise fall back to the resolved
    descriptor's `description` (catalog) or `metric_suffix.label_for`
    (FQN suffix table)."""
    if panel.title:
        return panel.title
    if not series_list:
        return panel.series_glob
    d = series_list[0].descriptor
    if d.description:
        return d.description
    return metric_suffix.label_for(d.entity, d.counter, d.rollup, d.submetric)


def _build_panel(
    panel,
    series_list: list[metric_layout.ResolvedSeries],
    projector: TraceProjector,
    projection: dict,
    t0_ns: int,
    x_range=None,
) -> tuple:
    """Build one Bokeh figure for one panel. Returns
    (figure, dict[(fqn, scope_key) -> ColumnDataSource]) so live mode
    can stream new rows in."""
    unit = panel.unit_override if panel.unit_override != mc_pb.UNIT_UNSPECIFIED \
        else series_list[0].descriptor.unit
    peak_hint = _resolve_panel_peak(panel, series_list[0].descriptor, projector)
    scale_fn, ylabel = _format_unit_axis(unit, peak_hint)

    fig_kwargs = dict(
        title=_panel_title(panel, series_list),
        x_axis_label="time (s)",
        y_axis_label=ylabel,
        width=1200, height=240, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        toolbar_location="left",
        active_drag="xpan", active_scroll="xwheel_zoom",
        output_backend=_RENDER_BACKEND,
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    fig = figure(**fig_kwargs)

    label_bases = metric_layout.disambiguate_short_labels(series_list)

    cds_by_key: dict[tuple, ColumnDataSource] = {}
    for i, series in enumerate(series_list):
        color = _PALETTE[i % len(_PALETTE)]
        ts_ns, vals = projection[(series.fqn, series.scope_key)]
        if ts_ns.size == 0:
            continue
        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        cds = ColumnDataSource(data=dict(
            x=time_s, y=scale_fn(vals.astype(np.float64)),
        ))
        cds_by_key[(series.fqn, series.scope_key)] = cds
        fig.line("x", "y", source=cds, color=color, line_width=1.2,
                 legend_label=_series_label(series, projector,
                                            base=label_bases[(series.fqn, series.scope_key)]))

    # Peak reference line + y-range. When the panel has a known peak,
    # pin the view to [0, peak*headroom] with hard bounds so pan/zoom
    # can't drift past the meaningful range. When no peak is known,
    # only clamp the floor at 0 — every metric this profiler emits is
    # non-negative, and a drifting negative axis just wastes space.
    if peak_hint is not None and peak_hint > 0:
        scaled_peak = scale_fn(peak_hint)
        upper = scaled_peak * _YLIM_HEADROOM
        fig.y_range = Range1d(start=0.0, end=upper, bounds=(0.0, upper))
        fig.add_layout(Span(location=scaled_peak, dimension="width",
                            line_color="red", line_dash="dashed",
                            line_alpha=0.6, line_width=1.5))
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

    # Click-to-hide legend entries (only present if a glyph added one).
    # Move the legend out of the plot area to the right so it never
    # occludes the trace.
    if fig.legend:
        legend = fig.legend[0]
        legend.click_policy = "hide"
        legend.label_text_font_size = "8pt"
        legend.location = "top_left"
        fig.add_layout(legend, "right")

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


def _panel_has_cumulative_companion(panel,
                                     series_list: list[metric_layout.ResolvedSeries]
                                     ) -> bool:
    """Whether `panel` opted into PANEL_AGGREGATION_INTEGRATE AND the
    source unit has a mapped integrated unit. Returns False (with no
    warning) for non-integrable units — visualize_all.py logs the
    skip; here we keep the live-render path quiet since this gets
    called every tick."""
    if panel.aggregation != panels_pb.PANEL_AGGREGATION_INTEGRATE:
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
) -> tuple:
    """Bokeh equivalent of visualize_all._render_integrated_panel.

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
    max_total = 0.0
    for series in series_list:
        ts_ns, vals = projection[(series.fqn, series.scope_key)]
        if ts_ns.size == 0:
            continue
        cum = _trapz_cumulative(ts_ns, vals.astype(np.float64))
        if cum.size and cum[-1] > max_total:
            max_total = float(cum[-1])
        if display_hz > 0 and source_hzs is not None:
            src_hz = source_hzs.get((series.fqn, series.scope_key), 0.0)
            ts_ns, cum = _decimate_to_hz(ts_ns, cum, src_hz, display_hz)
        cumulatives.append((series, ts_ns, cum))

    scale_fn, ylabel = _format_unit_axis(integrated_unit,
                                          peak_hint=max_total if max_total > 0 else None)
    base_title = _panel_title(panel, series_list)

    fig_kwargs = dict(
        title=f"{base_title}  (cumulative)",
        x_axis_label="time (s)",
        y_axis_label=ylabel,
        width=1200, height=240, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        toolbar_location="left",
        active_drag="xpan", active_scroll="xwheel_zoom",
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
    for i, (series, ts_ns, cum) in enumerate(cumulatives):
        color = _PALETTE[i % len(_PALETTE)]
        time_s = (ts_ns.astype(np.int64) - t0_ns) / 1e9
        cds = ColumnDataSource(data=dict(x=time_s, y=scale_fn(cum)))
        cds_by_key[(series.fqn, series.scope_key)] = cds
        cumulative_projection[(series.fqn, series.scope_key)] = (ts_ns, cum)
        fig.line("x", "y", source=cds, color=color, line_width=1.2,
                 legend_label=_series_label(series, projector,
                                            base=label_bases[(series.fqn, series.scope_key)]))

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

    if fig.legend:
        legend = fig.legend[0]
        legend.click_policy = "hide"
        legend.label_text_font_size = "8pt"
        legend.location = "top_left"
        fig.add_layout(legend, "right")

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
    fig = figure(
        title="Regions",
        width=1200, height=70, frame_width=_FRAME_WIDTH,
        min_border_left=_STRIP_LEFT_PX,
        x_range=x_range, y_range=(0.0, 1.0),
        tools="xpan,xwheel_zoom,reset",
        toolbar_location="left",
        active_drag="xpan", active_scroll="xwheel_zoom",
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
    fig = figure(
        title="Events",
        width=1200, height=70, frame_width=_FRAME_WIDTH,
        min_border_left=_STRIP_LEFT_PX,
        x_range=x_range, y_range=(0.0, 1.0),
        tools="xpan,xwheel_zoom,reset",
        toolbar_location="left",
        active_drag="xpan", active_scroll="xwheel_zoom",
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


def _inject_loading_overlay(html: str, n_figures: int) -> str:
    """Inject a determinate progress overlay into the bokeh-generated
    HTML. Hides once every Bokeh document reports is_ready *and* the
    number of rendered <canvas> elements has caught up with
    `n_figures` — which lets users see real progress (X / N panels +
    elapsed seconds) instead of an indeterminate spinner.

    Anchors on the closing tags `</head>` / `</body>` because the
    opening `<body>` collides with literal string occurrences inside
    bokeh's embedded minified JS source (e.g. `must be under <body>`
    in error messages). The overlay is `position: fixed`, so DOM
    insertion point doesn't affect its visual placement."""
    theme = _THEMES[_THEME]
    if "</head>" in html:
        html = html.replace("</head>",
                            _loading_overlay_head(theme) + "</head>", 1)
    if "</body>" in html:
        html = html.replace("</body>",
                            _loading_overlay_body(n_figures) + "</body>", 1)
    return html


# ---------------------------------------------------------------------------
# Static rendering path
# ---------------------------------------------------------------------------

def _render_static(
    projector: TraceProjector,
    layout: metric_layout.PanelLayout,
    catalog_index: dict[str, mc_pb.MetricDescriptor],
    out_path: Path,
    title: str,
    meta: session_metadata_pb2.SessionMetadata,
    metadata_path: Path,
    sample_freqs: dict[str, int],
    smooth_window_s: float,
    display_hz: float,
) -> None:
    proj = projector.project()
    if not proj:
        _log("no samples — nothing to render")
        return
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

    metric_figs: list = []
    figs: list = []
    shared_x = None
    for panel in layout.panels:
        series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
        if not series:
            continue
        fig, _ = _build_panel(panel, series, projector, smoothed_proj, t0_ns,
                              x_range=shared_x)
        if shared_x is None:
            shared_x = fig.x_range
        figs.append(fig)
        metric_figs.append(fig)
        # Companion cumulative panel, directly below — opt-in per
        # panel via aggregation: PANEL_AGGREGATION_INTEGRATE. Uses the
        # unsmoothed `proj` so the integrated total is faithful; the
        # display-only decimation happens inside the builder.
        if _panel_has_cumulative_companion(panel, series):
            source_hzs = {
                (s.fqn, s.scope_key):
                    sample_freqs.get(projector.fqn_to_probe.get(s.fqn, ""), 0)
                for s in series
            }
            cum_fig, _ = _build_cumulative_panel(
                panel, series, projector, proj, t0_ns, x_range=shared_x,
                display_hz=display_hz, source_hzs=source_hzs)
            figs.append(cum_fig)
            metric_figs.append(cum_fig)
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

    # Free up wheel-zoom near the bounds so the gesture isn't rejected
    # when the cursor sits next to an edge that already touches a bound.
    # Also drop the Bokeh logo from every toolbar (15 logos across the
    # page is visual noise; the framework is implicit from the file
    # extension).
    for f in strips + figs:
        _relax_wheel_zoom(f)
        f.toolbar.logo = None

    # Wrap both strips in their own Column with spacing=0 so there's
    # no transparent gap between them during scroll, and make THAT
    # the sticky element. The dashed bottom border separates the
    # sticky band from the scrolling metric panels.
    theme = _THEMES[_THEME]
    layout_children: list = []
    if strips:
        strip_col = column(strips, spacing=0, sizing_mode="stretch_width")
        strip_col.styles = {
            "position":      "sticky",
            "top":           "0",
            "z-index":       "50",
            "background":    theme["strip_bg"],
            "border-bottom": f"1px dashed {theme['strip_border']}",
        }
        layout_children.append(strip_col)
    layout_children.extend(figs)
    all_figs = strips + figs  # for the loading-overlay panel count
    layout_root = column(layout_children, sizing_mode="stretch_width")
    html = file_html(layout_root, INLINE, title=title,
                     theme=theme["bokeh_theme"])
    html = _inject_loading_overlay(html, len(all_figs))
    out_path.write_text(html)
    _log(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KiB; "
         f"{len(all_figs)} panels)")


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

def main() -> int:
    global _RENDER_BACKEND, _THEME
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
    sample_freqs = _ingest_probes(projector, meta, metadata_path)

    catalog_index = metric_catalog.build_index(catalog)
    out_path = Path(args.output).resolve()
    title = f"Profile — {meta.hostname}  {meta.start_iso8601}"
    _log("rendering")
    _render_static(projector, layout, catalog_index, out_path, title,
                   meta, metadata_path,
                   sample_freqs=sample_freqs,
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

    fig_kwargs = dict(
        title=_panel_title(panel, series_list),
        x_axis_label="time (s)", y_axis_label=ylabel,
        width=1200, height=240, frame_width=_FRAME_WIDTH,
        min_border_left=_BORDER_LEFT_PX,
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        toolbar_location="left",
        active_drag="xpan", active_scroll="xwheel_zoom",
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
    base = metric_suffix.pretty_counter(descriptor.counter) or fqn
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
