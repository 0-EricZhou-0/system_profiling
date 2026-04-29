"""Interactive Bokeh visualizer for a cupti_profiler run.

Mirrors the panel set of tools/visualize_all.py — events + region
timeline strip on top, GPU / System / Disk metric panels stacked below
with a shared X axis. Output is a self-contained .html file with synced
pan/zoom, per-panel hover tooltips, and a vertical dashed crosshair
that follows the cursor across every panel. By default the same HTML
is also served over a small built-in HTTP server.

Run:
    python tools/visualize_interactive.py \
        profiling_output/session_metadata.pb -o profile.html
"""

import argparse
import http.server
import os
import socketserver
import sys
from datetime import datetime, timezone

# Make tools/ importable so we can reuse the .pb loaders that
# tools/visualize_all.py already has, and add generated/proto/ for the
# *_pb2 modules.
_TOOLS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated", "proto"))

import numpy as np
from bokeh.embed import file_html
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import (BoxAnnotation, ColumnDataSource, CrosshairTool,
                          HoverTool, Label, Range1d, Span)
from bokeh.plotting import figure
from bokeh.resources import CDN, INLINE

import visualize_all as va  # reuse the existing loaders


# ----- Layout / style constants -------------------------------------------

PLOT_WIDTH        = 1500
PANEL_HEIGHT      = 220
ANNOT_HEIGHT      = 110
TITLE_FONT_SIZE   = "13pt"
PANEL_FONT_SIZE   = "11pt"
# Padding above the highest expected sample value (and the dashed
# reference line) before pan/zoom hits the upper bound. Matches
# YLIM_HEADROOM in tools/visualize_all.py.
YLIM_HEADROOM     = 1.1

REGION_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e",
                 "#9467bd", "#8c564b", "#e377c2", "#17becf"]
EVENT_COLORS  = ["#000000", "#FF1493", "#00CED1", "#FFD700",
                 "#7B68EE", "#228B22", "#A0522D", "#DC143C"]


# ----- Data loading helpers ------------------------------------------------

def _pid_label(trace, pid):
    """"<alias> (PID xxx)" if the trace carries an alias for this PID,
    else "PID xxx". Aliases come from the optional `alias` field on each
    TrackedProcess entry in `tracked_processes`."""
    pid = int(pid)
    for tp in getattr(trace, "tracked_processes", []) or []:
        if int(tp.pid) == pid and tp.alias:
            return f"{tp.alias} (PID {pid})"
    return f"PID {pid}"


def _resolve_probe_paths(meta, meta_path):
    """Same path-resolution logic as visualize_all.py: absolute → meta-dir
    → basename-in-meta-dir."""
    import session_metadata_pb2

    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    out = {}
    for p in meta.probes:
        for cand in (p.output_file,
                     os.path.join(meta_dir, p.output_file),
                     os.path.join(meta_dir, os.path.basename(p.output_file))):
            if os.path.exists(cand):
                out[p.kind] = cand
                break
    return out


def _load_all(metadata_path):
    """Returns (gpu_trace, sys_trace, disk_trace, events, session_meta)."""
    import session_metadata_pb2

    session_meta = session_metadata_pb2.SessionMetadata()
    with open(metadata_path, "rb") as f:
        session_meta.ParseFromString(f.read())

    paths = _resolve_probe_paths(session_meta, metadata_path)

    gpu_trace = sys_trace = disk_trace = events = None

    if session_metadata_pb2.PROBE_KIND_GPU in paths:
        import gpu_metrics_pb2
        with open(paths[session_metadata_pb2.PROBE_KIND_GPU], "rb") as f:
            gpu_trace = va.merge_gpu_traces(
                va.read_delimited_messages(f.read(), gpu_metrics_pb2.GpuMetricsTrace))

    if session_metadata_pb2.PROBE_KIND_SYSTEM in paths:
        import system_metrics_pb2
        with open(paths[session_metadata_pb2.PROBE_KIND_SYSTEM], "rb") as f:
            sys_trace = va.merge_system_traces(
                va.read_delimited_messages(f.read(), system_metrics_pb2.SystemMetricsTrace))

    if session_metadata_pb2.PROBE_KIND_DISK in paths:
        import disk_metrics_pb2
        with open(paths[session_metadata_pb2.PROBE_KIND_DISK], "rb") as f:
            disk_trace = va.merge_disk_traces(
                va.read_delimited_messages(f.read(), disk_metrics_pb2.DiskMetricsTrace))

    if session_metadata_pb2.PROBE_KIND_EVENTS in paths:
        import events_pb2
        with open(paths[session_metadata_pb2.PROBE_KIND_EVENTS], "rb") as f:
            events = va.merge_event_traces(
                va.read_delimited_messages(f.read(), events_pb2.EventTrace))

    return gpu_trace, sys_trace, disk_trace, events, session_meta


def _global_t0(gpu_trace, sys_trace, disk_trace):
    """Earliest data point across the active probes, in steady_clock ns."""
    cands = []
    if gpu_trace and len(gpu_trace.samples) > 1:
        cands.append(va.gpu_to_steady(gpu_trace.samples[1].start_timestamp_ns,
                                       gpu_trace.cupti_reference_ns,
                                       gpu_trace.steady_clock_reference_ns))
    if sys_trace and len(sys_trace.cpu_system_samples) > 0:
        cands.append(sys_trace.cpu_system_samples[0].timestamp_ns)
    if disk_trace and len(disk_trace.device_samples) > 0:
        cands.append(disk_trace.device_samples[0].timestamp_ns)
    return min(cands) if cands else 0


def _global_xmax_ms(gpu_trace, sys_trace, disk_trace, events, global_t0):
    """Latest data point across the active probes, expressed in ms from
    global_t0. Used to set the shared X axis bounds."""
    cands = []
    if gpu_trace and len(gpu_trace.samples) > 1:
        last = gpu_trace.samples[-1]
        cands.append(va.gpu_to_steady(last.start_timestamp_ns,
                                       gpu_trace.cupti_reference_ns,
                                       gpu_trace.steady_clock_reference_ns))
    if sys_trace and len(sys_trace.cpu_system_samples) > 0:
        cands.append(sys_trace.cpu_system_samples[-1].timestamp_ns)
    if disk_trace and len(disk_trace.device_samples) > 0:
        cands.append(disk_trace.device_samples[-1].timestamp_ns)
    if events:
        for r in events["generic_regions"]:
            cands.append(r.end_timestamp_ns)
        for e in events["generic_events"]:
            cands.append(e.timestamp_ns)
        meta = events["metadata"]
        if meta is not None:
            cref, sref = meta.cupti_reference_ns, meta.steady_clock_reference_ns
            for r in events["gpu_regions"]:
                cands.append(int(va.gpu_to_steady(r.end_timestamp_ns, cref, sref)))
            for e in events["gpu_events"]:
                cands.append(int(va.gpu_to_steady(e.timestamp_ns, cref, sref)))
    if not cands:
        return 0.0
    return (max(cands) - global_t0) / 1e6


# ----- Per-panel builders --------------------------------------------------

def _empty_figure(title, x_range, height=PANEL_HEIGHT, y_label=""):
    kwargs = dict(width=PLOT_WIDTH, height=height, title=title,
                  tools="pan,box_zoom,xwheel_zoom,reset,save",
                  active_scroll="xwheel_zoom",
                  output_backend="canvas")
    if x_range is not None:
        kwargs["x_range"] = x_range
    fig = figure(**kwargs)
    fig.xaxis.axis_label = "Time (ms)"
    fig.yaxis.axis_label = y_label
    fig.title.text_font_size = PANEL_FONT_SIZE
    # Clamp y-axis to >= 0 — every metric panel here is non-negative
    # (rates, percentages, byte counts), so don't let pan/zoom drift the
    # view below the x-axis. (None upper bound = still auto-fits up.)
    fig.y_range.bounds = (0, None)
    return fig


# Reuse a single Label import — Bokeh's Label is one per annotation.
from bokeh.models import Label as _BokehLabel  # noqa: E402


def _move_legend_outside(fig, location="right"):
    """Detach the auto-built legend from inside the plot frame and dock
    it on the figure's right edge so glyphs aren't covered."""
    if not fig.legend:
        return
    legend = fig.legend[0]
    legend.click_policy = "hide"
    legend.label_text_font_size = "8pt"
    # Take it off the central layout (where it's drawn inside the plot
    # frame by default) and add it as a side panel.
    fig.add_layout(legend, location)


def _attach_anchored_hover(fig, src, tooltips, time_col="t"):
    """Wire up a HoverTool that fires for any cursor x position in the
    figure (independent of which visible glyphs are showing) and renders
    the tooltip at the bottom edge of the plot.

    Implementation: add an invisible "anchor" line at y=0 that spans every
    timestamp in the source, then bind the hover to *that* renderer.
    Because the anchor is never hidden by legend clicks and always covers
    the full x extent, the tooltip stays usable even when individual
    series have been toggled off. attachment='below' positions the popup
    just under the plot frame so timestamps land at the bottom of every
    panel.
    """
    if "_anchor_y" not in src.data:
        n = len(src.data[time_col])
        src.data["_anchor_y"] = np.zeros(n, dtype=np.float64)
    anchor = fig.line(time_col, "_anchor_y", source=src,
                       line_alpha=0, line_width=0)
    fig.add_tools(HoverTool(renderers=[anchor], mode="vline",
                             attachment="below", tooltips=tooltips))
    return anchor


def _clamp_and_mark_peak(fig, peak, *, label=None):
    """Pin the y-range to [0, peak * YLIM_HEADROOM], draw a dashed
    horizontal reference line at y=peak, and (optionally) annotate it.

    Replaces the auto-fitting DataRange1d with a fixed Range1d so the
    initial view matches the clamp; pan/zoom outside [0, peak*HEADROOM]
    is rejected by `bounds`.
    """
    if peak is None or peak <= 0:
        return
    upper = peak * YLIM_HEADROOM
    fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))

    fig.add_layout(Span(location=peak, dimension="width",
                         line_color="black", line_dash="dotted",
                         line_width=1.2, line_alpha=0.7))

    if label:
        # Anchor the label inside the plot area (8 px from the left edge,
        # just above the peak line) so it's stable under pan/zoom.
        fig.add_layout(_BokehLabel(
            x=8, y=peak, x_units="screen", y_units="data",
            text=label, text_font_size="8pt",
            text_color="black", text_baseline="bottom"))


def build_annot_panel(events, x_range, gpu_trace, global_t0):
    """Single strip showing regions (colored bars at y=0) + events (vertical
    lines + triangle markers at y=0.5). Hover shows the name and duration
    or absolute timestamp.
    """
    kwargs = dict(width=PLOT_WIDTH, height=ANNOT_HEIGHT, title="Regions / Events",
                  # Pin the y-range and refuse pan/zoom outside [-1, 1] so the
                  # strip can't slide off-screen when the user scrolls. The
                  # strip is purely categorical (no real units), so the same
                  # initial view doubles as the bounds.
                  y_range=Range1d(start=-1, end=1, bounds=(-1, 1)),
                  tools="pan,box_zoom,xwheel_zoom,reset,save",
                  active_scroll="xwheel_zoom")
    if x_range is not None:
        kwargs["x_range"] = x_range
    fig = figure(**kwargs)
    fig.yaxis.visible = False
    fig.ygrid.grid_line_color = None
    fig.title.text_font_size = PANEL_FONT_SIZE
    fig.xaxis.axis_label = "Time (ms)"

    # Build the merged region list (generic first, then GPU converted via metadata).
    regions = []
    instant_events = []
    if events:
        for r in events["generic_regions"]:
            regions.append((r.name, r.start_timestamp_ns, r.end_timestamp_ns))
        for e in events["generic_events"]:
            instant_events.append((e.name, e.timestamp_ns))
        meta = events["metadata"]
        if meta is not None:
            cref, sref = meta.cupti_reference_ns, meta.steady_clock_reference_ns
            for r in events["gpu_regions"]:
                regions.append((r.name,
                                int(va.gpu_to_steady(r.start_timestamp_ns, cref, sref)),
                                int(va.gpu_to_steady(r.end_timestamp_ns,   cref, sref))))
            for e in events["gpu_events"]:
                instant_events.append((e.name,
                                        int(va.gpu_to_steady(e.timestamp_ns, cref, sref))))

    # Regions: rectangle (left, right, top, bottom) per region.
    if regions:
        names, lefts, rights, dur_ms, colors = [], [], [], [], []
        for i, (name, s_ns, e_ns) in enumerate(regions):
            names.append(name)
            lefts.append((s_ns - global_t0) / 1e6)
            rights.append((e_ns - global_t0) / 1e6)
            dur_ms.append((e_ns - s_ns) / 1e6)
            colors.append(REGION_COLORS[i % len(REGION_COLORS)])
        src = ColumnDataSource(dict(name=names, left=lefts, right=rights,
                                     dur_ms=dur_ms, color=colors,
                                     bottom=[-0.4] * len(names),
                                     top=[0.4] * len(names)))
        region_renderer = fig.quad(
            left="left", right="right", bottom="bottom", top="top",
            color="color", alpha=0.7, line_color="color", source=src,
            legend_label="region")
        # Scope to the quad renderer so hovering an event marker won't
        # also fire this hover and surface bogus "@left = ???" cells.
        fig.add_tools(HoverTool(renderers=[region_renderer],
                                 tooltips=[("region", "@name"),
                                           ("start (ms)", "@left{0.00}"),
                                           ("end (ms)", "@right{0.00}"),
                                           ("duration (ms)", "@dur_ms{0.00}")]))

    # Events: vertical line + triangle marker.
    if instant_events:
        en, ex, ec = [], [], []
        for i, (name, ts) in enumerate(instant_events):
            en.append(name)
            ex.append((ts - global_t0) / 1e6)
            ec.append(EVENT_COLORS[i % len(EVENT_COLORS)])
        src = ColumnDataSource(dict(name=en, x=ex, color=ec,
                                     y=[0.7] * len(en),
                                     y0=[-1.0] * len(en),
                                     y1=[1.0] * len(en)))
        event_marker = fig.scatter(
            x="x", y="y", marker="inverted_triangle", size=12,
            color="color", line_color="black", source=src,
            legend_label="event")
        # Vertical lines so the event is visible across the strip height.
        fig.segment(x0="x", y0="y0", x1="x", y1="y1",
                    line_color="color", line_width=1.5, line_alpha=0.6,
                    source=src)
        # Hover only fires on the marker glyph — not on the region quads
        # (which have no @x column and would render as "???").
        fig.add_tools(HoverTool(renderers=[event_marker],
                                 tooltips=[("event", "@name"),
                                           ("t (ms)", "@x{0.00}")]))

    _move_legend_outside(fig)
    return fig


def build_gpu_panels(gpu_trace, x_range, global_t0):
    """SM Utilization, DRAM throughput, PCIe throughput — only the three the
    matplotlib version emits most prominently. Each is its own figure.
    """
    panels = []
    if not gpu_trace or len(gpu_trace.samples) <= 1:
        return panels

    samples = list(gpu_trace.samples)[1:]  # skip init artifact
    cref, sref = gpu_trace.cupti_reference_ns, gpu_trace.steady_clock_reference_ns
    ts_steady = np.array([s.start_timestamp_ns for s in samples], dtype=np.float64)
    time_ms = (va.gpu_to_steady(ts_steady, cref, sref) - global_t0) / 1e6

    metric_names = list(gpu_trace.metric_names)
    idx = {name: i for i, name in enumerate(metric_names)}
    vals = np.zeros((len(samples), len(metric_names)))
    for i, s in enumerate(samples):
        for j, v in enumerate(s.values):
            vals[i, j] = v

    if "sm__cycles_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        # One ColumnDataSource per panel + a single HoverTool scoped to one
        # renderer = single combined tooltip box (no stacked popups when
        # multiple series share the figure).
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        cds_data = {
            "t":   time_ms,
            "avg": np.where(elapsed > 0,
                             vals[:, idx["sm__cycles_active.avg"]] / elapsed * 100, 0),
        }
        if "sm__cycles_active.max" in idx:
            cds_data["max"] = np.where(elapsed > 0,
                                        vals[:, idx["sm__cycles_active.max"]] / elapsed * 100, 0)
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("GPU SM Utilization", x_range, y_label="%")
        avg_line = fig.line("t", "avg", source=src, line_width=1,
                             color="#1f77b4", legend_label="avg")
        last = avg_line
        if "max" in cds_data:
            last = fig.line("t", "max", source=src, line_width=1.4,
                             color="#d62728", legend_label="max")
        tt = [("t (ms)", "@t{0.00}"), ("avg", "@avg{0.0}%")]
        if "max" in cds_data:
            tt.append(("max", "@max{0.0}%"))
        _attach_anchored_hover(fig, src, tt)
        _clamp_and_mark_peak(fig, 100, label="Max SM Util: 100%")
        _move_legend_outside(fig)
        panels.append(fig)

    if "sm__warps_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        cds_data = {
            "t":   time_ms,
            "avg": np.where(elapsed > 0,
                             vals[:, idx["sm__warps_active.avg"]] / elapsed, 0),
        }
        if "sm__warps_active.max" in idx:
            cds_data["max"] = np.where(elapsed > 0,
                                        vals[:, idx["sm__warps_active.max"]] / elapsed, 0)
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("Active Warps / Cycle", x_range,
                             y_label="warps / cycle")
        fig.line("t", "avg", source=src, line_width=0.8, alpha=0.45,
                 color="#ff7f0e", legend_label="avg (across all SMs)")
        last = fig.renderers[-1]
        if "max" in cds_data:
            last = fig.line("t", "max", source=src, line_width=1.0,
                             color="#ff7f0e", legend_label="max (busiest SM)")
        tt = [("t (ms)", "@t{0.00}"), ("avg", "@avg{0.00}")]
        if "max" in cds_data:
            tt.append(("max", "@max{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        # H100 / GH100: 64 warps/SM is the architectural cap.
        _clamp_and_mark_peak(fig, 64, label="Max Active Warps: 64")
        _move_legend_outside(fig)
        panels.append(fig)

    drum_avg_key = "dram__read_throughput.avg.pct_of_peak_sustained_elapsed"
    if drum_avg_key in idx:
        # Convert "% of peak" → GiB/s for direct readability, matching
        # the matplotlib visualizer.
        peak_dram_gibps = (gpu_trace.peak_dram_bw_gbps * 1e9 / (1024 ** 3)
                           if gpu_trace.peak_dram_bw_gbps > 0 else None)
        scale = (peak_dram_gibps / 100.0) if peak_dram_gibps else 1.0
        unit = "GiB/s" if peak_dram_gibps else "% of peak"

        cds_data = {"t": time_ms, "rd": vals[:, idx[drum_avg_key]] * scale}
        if "dram__write_throughput.avg.pct_of_peak_sustained_elapsed" in idx:
            cds_data["wr"] = vals[:, idx["dram__write_throughput.avg.pct_of_peak_sustained_elapsed"]] * scale
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("GPU DRAM Bandwidth", x_range, y_label=unit)
        fig.line("t", "rd", source=src, line_width=1, color="#2ca02c",
                 legend_label="read avg")
        last = fig.renderers[-1]
        if "wr" in cds_data:
            last = fig.line("t", "wr", source=src, line_width=1, color="#ff7f0e",
                             legend_label="write avg")
        tt = [("t (ms)", "@t{0.00}"), (f"read ({unit})", "@rd{0.00}")]
        if "wr" in cds_data:
            tt.append((f"write ({unit})", "@wr{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        if peak_dram_gibps:
            _clamp_and_mark_peak(fig, peak_dram_gibps,
                                  label=f"Max DRAM BW: {peak_dram_gibps:.0f} GiB/s")
        _move_legend_outside(fig)
        panels.append(fig)

    if "pcie__read_bytes.sum.per_second" in idx:
        cds_data = {"t": time_ms,
                    "rd": vals[:, idx["pcie__read_bytes.sum.per_second"]] / (1024 ** 3)}
        if "pcie__write_bytes.sum.per_second" in idx:
            cds_data["wr"] = vals[:, idx["pcie__write_bytes.sum.per_second"]] / (1024 ** 3)
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("GPU PCIe Bandwidth", x_range, y_label="GiB/s")
        fig.line("t", "rd", source=src, line_width=1, color="#9467bd",
                 legend_label="H→D")
        last = fig.renderers[-1]
        if "wr" in cds_data:
            last = fig.line("t", "wr", source=src, line_width=1, color="#8c564b",
                             legend_label="D→H")
        tt = [("t (ms)", "@t{0.00}"), ("H→D (GiB/s)", "@rd{0.00}")]
        if "wr" in cds_data:
            tt.append(("D→H (GiB/s)", "@wr{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        # PCIe trace metadata stores per-direction peak; matplotlib panel
        # plots both directions on the same axis and uses the bidirectional
        # peak (×2) as the reference. Mirror that here.
        peak_pcie_bidi = (gpu_trace.peak_pcie_bw_bytes_per_sec * 2 / (1024 ** 3)
                          if gpu_trace.peak_pcie_bw_bytes_per_sec > 0 else None)
        if peak_pcie_bidi:
            _clamp_and_mark_peak(fig, peak_pcie_bidi,
                                  label=f"Max PCIe BW: {peak_pcie_bidi:.1f} GiB/s (Bi-directional)")
        _move_legend_outside(fig)
        panels.append(fig)

    if "pcie__read_bytes.sum" in idx and "pcie__write_bytes.sum" in idx:
        # Per-sample raw bytes; cumsum gives the running total.
        rd_cum = np.cumsum(vals[:, idx["pcie__read_bytes.sum"]])
        wr_cum = np.cumsum(vals[:, idx["pcie__write_bytes.sum"]])
        max_bytes = float(max(rd_cum[-1] if rd_cum.size else 0,
                               wr_cum[-1] if wr_cum.size else 0))
        # Pick the largest binary unit such that the max value is ≥ 1 in it.
        for div, unit in [(1024 ** 4, "TiB"), (1024 ** 3, "GiB"),
                          (1024 ** 2, "MiB"), (1024, "KiB"), (1, "B")]:
            if max_bytes >= div:
                break
        src = ColumnDataSource({"t": time_ms,
                                 "rd": rd_cum / div, "wr": wr_cum / div})
        fig = _empty_figure("Cumulative PCIe Bytes", x_range, y_label=unit)
        fig.line("t", "rd", source=src, line_width=1, color="#9467bd",
                 legend_label="H→D cumulative")
        last = fig.line("t", "wr", source=src, line_width=1, color="#8c564b",
                         legend_label="D→H cumulative")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             (f"H→D ({unit})", "@rd{0.00}"),
             (f"D→H ({unit})", "@wr{0.00}")])
        # Cumulative — auto-fit to data, just clamp >= 0.
        upper = max(0.01, max_bytes / div) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if "nvlrx__bytes.sum.per_second" in idx and "nvltx__bytes.sum.per_second" in idx:
        rx_gibps = vals[:, idx["nvlrx__bytes.sum.per_second"]] / (1024 ** 3)
        tx_gibps = vals[:, idx["nvltx__bytes.sum.per_second"]] / (1024 ** 3)
        src = ColumnDataSource({"t": time_ms, "rx": rx_gibps, "tx": tx_gibps})
        fig = _empty_figure("GPU NVLink Bandwidth", x_range, y_label="GiB/s")
        fig.line("t", "rx", source=src, line_width=1, color="#e377c2",
                 legend_label="RX")
        last = fig.line("t", "tx", source=src, line_width=1, color="#17becf",
                         legend_label="TX")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             ("RX (GiB/s)", "@rx{0.00}"),
             ("TX (GiB/s)", "@tx{0.00}")])
        peak_nvl_bidi = (gpu_trace.peak_nvlink_bw_bytes_per_sec * 2 / (1024 ** 3)
                         if gpu_trace.peak_nvlink_bw_bytes_per_sec > 0 else None)
        if peak_nvl_bidi:
            _clamp_and_mark_peak(fig, peak_nvl_bidi,
                                  label=f"Max NVLink BW: {peak_nvl_bidi:.1f} GiB/s (Bi-directional)")
        _move_legend_outside(fig)
        panels.append(fig)

    return panels


def build_system_panels(sys_trace, x_range, global_t0):
    panels = []
    if not sys_trace:
        return panels

    if len(sys_trace.cpu_system_samples) > 0:
        s = sys_trace.cpu_system_samples
        time_ms = (np.array([x.timestamp_ns for x in s]) - global_t0) / 1e6
        user = np.array([x.user_pct for x in s])
        sysp = np.array([x.system_pct for x in s])
        iow  = np.array([x.iowait_pct for x in s])
        # Use the trace's authoritative total_utilization_pct rather than
        # summing components; matches visualize_all.py's "Total" line.
        total = np.array([x.total_utilization_pct for x in s])
        # Stacked vareas need cumulative tops as columns in the source.
        src = ColumnDataSource({
            "t": time_ms,
            "user": user, "sys": sysp, "iow": iow, "total": total,
            "y_user_top": user,
            "y_sys_top":  user + sysp,
            "y_iow_top":  user + sysp + iow,
        })
        fig = _empty_figure("CPU Utilization (system-wide)", x_range, y_label="%")
        fig.varea(x="t", y1=0,            y2="y_user_top", source=src,
                  color="#4c72b0", alpha=0.7, legend_label="User")
        fig.varea(x="t", y1="y_user_top", y2="y_sys_top",  source=src,
                  color="#dd8452", alpha=0.7, legend_label="System")
        fig.varea(x="t", y1="y_sys_top",  y2="y_iow_top",  source=src,
                  color="#c44e52", alpha=0.7, legend_label="IOWait")
        total_line = fig.line("t", "total", source=src, color="black",
                               line_width=1, legend_label="Total")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             ("User", "@user{0.0}%"),
             ("System", "@sys{0.0}%"),
             ("IOWait", "@iow{0.0}%"),
             ("Total", "@total{0.0}%")])
        _clamp_and_mark_peak(fig, 100, label="Max CPU: 100%")
        _move_legend_outside(fig)
        panels.append(fig)

    if len(sys_trace.cpu_process_samples) > 0:
        proc = list(sys_trace.cpu_process_samples)
        pids = sorted(set(s.pid for s in proc))
        # All PIDs share the per-tick timestamp set; take the first PID's
        # timestamps as the canonical x axis.
        per_pid = {pid: [s for s in proc if s.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms = (np.array([x.timestamp_ns for x in base]) - global_t0) / 1e6
        cds_data = {"t": time_ms}
        max_val = 0.0
        for pid in pids:
            samps = per_pid[pid]
            user = np.array([x.user_pct for x in samps])
            sysp = np.array([x.system_pct for x in samps])
            n = len(time_ms)
            if user.size != n: user = np.resize(user, n)
            if sysp.size != n: sysp = np.resize(sysp, n)
            cds_data[f"pid_{pid}_user"] = user
            cds_data[f"pid_{pid}_sum"]  = user + sysp
            max_val = max(max_val, float((user + sysp).max() if user.size else 0))
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("CPU Utilization (per process)", x_range, y_label="%")
        last = None
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            color = REGION_COLORS[i % 8]
            label = _pid_label(sys_trace, pid)
            fig.line("t", f"pid_{pid}_sum", source=src, line_width=1, color=color,
                     legend_label=f"{label} (usr+sys)")
            last = fig.line("t", f"pid_{pid}_user", source=src, line_width=0.8,
                             alpha=0.5, line_dash="dashed", color=color,
                             legend_label=f"{label} (usr)")
            tt.append((f"{label} usr+sys", f"@pid_{pid}_sum{{0.0}}%"))
            tt.append((f"{label} usr",     f"@pid_{pid}_user{{0.0}}%"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(1.0, max_val) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if len(sys_trace.memory_system_samples) > 0:
        s = sys_trace.memory_system_samples
        time_ms = (np.array([x.timestamp_ns for x in s]) - global_t0) / 1e6
        used    = np.array([x.used_bytes    for x in s]) / (1024 ** 3)
        buffers = np.array([x.buffers_bytes for x in s]) / (1024 ** 3)
        cached  = np.array([x.cached_bytes  for x in s]) / (1024 ** 3)
        # Stack: Used → Buffers → Cached, matching visualize_all.py.
        src = ColumnDataSource({
            "t": time_ms,
            "used": used, "buffers": buffers, "cached": cached,
            "y_used_top":    used,
            "y_buffers_top": used + buffers,
            "y_cached_top":  used + buffers + cached,
        })
        fig = _empty_figure("System Memory", x_range, y_label="GiB")
        fig.varea(x="t", y1=0,                y2="y_used_top",    source=src,
                  color="#c44e52", alpha=0.7, legend_label="Used")
        fig.varea(x="t", y1="y_used_top",     y2="y_buffers_top", source=src,
                  color="#dd8452", alpha=0.7, legend_label="Buffers")
        fig.varea(x="t", y1="y_buffers_top",  y2="y_cached_top",  source=src,
                  color="#4c72b0", alpha=0.7, legend_label="Cached")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             ("Used (GiB)",    "@used{0.00}"),
             ("Buffers (GiB)", "@buffers{0.00}"),
             ("Cached (GiB)",  "@cached{0.00}")])
        # First sample's total_bytes is the install RAM size (constant).
        total_gib = s[0].total_bytes / (1024 ** 3) if s[0].total_bytes else 0
        if total_gib > 0:
            _clamp_and_mark_peak(fig, total_gib,
                                  label=f"Total: {total_gib:.0f} GiB")
        _move_legend_outside(fig)
        panels.append(fig)

    if len(sys_trace.memory_process_samples) > 0:
        proc = list(sys_trace.memory_process_samples)
        pids = sorted(set(s.pid for s in proc))
        per_pid = {pid: [s for s in proc if s.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms = (np.array([x.timestamp_ns for x in base]) - global_t0) / 1e6
        cds_data = {"t": time_ms}
        max_val = 0.0
        for pid in pids:
            samps = per_pid[pid]
            rss = np.array([x.rss_bytes for x in samps]) / (1024 ** 3)
            n = len(time_ms)
            if rss.size != n: rss = np.resize(rss, n)
            cds_data[f"pid_{pid}_rss"] = rss
            max_val = max(max_val, float(rss.max() if rss.size else 0))
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("Process Memory (RSS)", x_range, y_label="GiB")
        last = None
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            last = fig.line("t", f"pid_{pid}_rss", source=src, line_width=1,
                             color=REGION_COLORS[i % 8],
                             legend_label=f"{_pid_label(sys_trace, pid)} RSS")
            tt.append((f"{_pid_label(sys_trace, pid)} RSS", f"@pid_{pid}_rss{{0.00}} GiB"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(0.01, max_val) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    return panels


def build_disk_panels(disk_trace, x_range, global_t0):
    panels = []
    if not disk_trace or len(disk_trace.device_samples) == 0:
        return panels

    # Group device samples by device name.
    by_dev = {}
    for s in disk_trace.device_samples:
        by_dev.setdefault(s.device_name, []).append(s)

    fig = _empty_figure("Disk Bandwidth (per device)", x_range, y_label="MiB/s")

    # Build one combined ColumnDataSource keyed on the union of device
    # timestamps. All devices tick together in the C++ flush thread, so
    # the timestamp arrays match across devices in practice.
    devs = list(by_dev.keys())
    base_samps = by_dev[devs[0]]
    time_ms = (np.array([x.timestamp_ns for x in base_samps]) - global_t0) / 1e6
    cds_data = {"t": time_ms}
    data_max = 0.0
    for dev in devs:
        samps = by_dev[dev]
        rd = np.array([x.read_bytes_per_sec for x in samps]) / (1024 ** 2)
        wr = np.array([x.write_bytes_per_sec for x in samps]) / (1024 ** 2)
        # Pad / truncate if a device's sample count drifts from the base.
        n = len(time_ms)
        if rd.size != n:
            rd = np.resize(rd, n)
            wr = np.resize(wr, n)
        cds_data[f"{dev}_rd"] = rd
        cds_data[f"{dev}_wr"] = wr
        data_max = max(data_max, float(rd.max() if rd.size else 0),
                                  float(wr.max() if wr.size else 0))
    src = ColumnDataSource(cds_data)

    last = None
    for palette_idx, dev in enumerate(devs):
        color = REGION_COLORS[palette_idx % 8]
        fig.line("t", f"{dev}_rd", source=src, line_width=1.2, color=color,
                 legend_label=f"{dev} read")
        last = fig.line("t", f"{dev}_wr", source=src, line_width=1.0,
                         line_dash="dashed", color=color,
                         legend_label=f"{dev} write")
    tt = [("t (ms)", "@t{0.00}")]
    for dev in devs:
        tt.append((f"{dev} read",  f"@{dev}_rd{{0.00}} MiB/s"))
        tt.append((f"{dev} write", f"@{dev}_wr{{0.00}} MiB/s"))
    _attach_anchored_hover(fig, src, tt)
    # No theoretical peak for disk — clamp to observed-data max with the
    # standard headroom, mirroring tools/visualize_all.py.
    if data_max > 0:
        upper = max(0.01, data_max) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
    _move_legend_outside(fig)
    panels.append(fig)

    # Per-process disk IO
    if len(disk_trace.process_samples) > 0:
        proc = list(disk_trace.process_samples)
        pids = sorted(set(s.pid for s in proc))
        per_pid = {pid: [s for s in proc if s.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms = (np.array([x.timestamp_ns for x in base]) - global_t0) / 1e6
        cds_data = {"t": time_ms}
        max_val = 0.0
        for pid in pids:
            samps = per_pid[pid]
            rd = np.array([x.read_bytes_per_sec for x in samps]) / (1024 ** 2)
            wr = np.array([x.write_bytes_per_sec for x in samps]) / (1024 ** 2)
            n = len(time_ms)
            if rd.size != n: rd = np.resize(rd, n)
            if wr.size != n: wr = np.resize(wr, n)
            cds_data[f"pid_{pid}_rd"] = rd
            cds_data[f"pid_{pid}_wr"] = wr
            max_val = max(max_val, float(rd.max() if rd.size else 0),
                                    float(wr.max() if wr.size else 0))
        src = ColumnDataSource(cds_data)
        fig = _empty_figure("Process Disk IO", x_range, y_label="MiB/s")
        last = None
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            color = REGION_COLORS[i % 8]
            label = _pid_label(disk_trace, pid)
            fig.line("t", f"pid_{pid}_rd", source=src, line_width=1, color=color,
                     legend_label=f"{label} read")
            last = fig.line("t", f"pid_{pid}_wr", source=src, line_width=1,
                             line_dash="dashed", color=color,
                             legend_label=f"{label} write")
            tt.append((f"{label} read",  f"@pid_{pid}_rd{{0.00}} MiB/s"))
            tt.append((f"{label} write", f"@pid_{pid}_wr{{0.00}} MiB/s"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(0.01, max_val) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    # Per-device queue depth
    cds_data = {"t": (np.array([x.timestamp_ns for x in by_dev[devs[0]]])
                       - global_t0) / 1e6}
    q_max = 0.0
    for dev in devs:
        samps = by_dev[dev]
        rdq = np.array([x.read_queue_depth  for x in samps], dtype=np.float64)
        wrq = np.array([x.write_queue_depth for x in samps], dtype=np.float64)
        n = len(cds_data["t"])
        if rdq.size != n: rdq = np.resize(rdq, n)
        if wrq.size != n: wrq = np.resize(wrq, n)
        cds_data[f"{dev}_rdq"] = rdq
        cds_data[f"{dev}_wrq"] = wrq
        q_max = max(q_max, float(rdq.max() if rdq.size else 0),
                            float(wrq.max() if wrq.size else 0))
    src = ColumnDataSource(cds_data)
    fig = _empty_figure("Disk Queue Depth (per device)", x_range, y_label="depth")
    last = None
    tt = [("t (ms)", "@t{0.00}")]
    for i, dev in enumerate(devs):
        color = REGION_COLORS[i % 8]
        fig.line("t", f"{dev}_rdq", source=src, line_width=1, color=color,
                 legend_label=f"{dev} read Q")
        last = fig.line("t", f"{dev}_wrq", source=src, line_width=1,
                         line_dash="dashed", color=color,
                         legend_label=f"{dev} write Q")
        tt.append((f"{dev} read Q",  f"@{dev}_rdq{{0}}"))
        tt.append((f"{dev} write Q", f"@{dev}_wrq{{0}}"))
    _attach_anchored_hover(fig, src, tt)
    upper = max(1.0, q_max) * YLIM_HEADROOM
    fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
    _move_legend_outside(fig)
    panels.append(fig)

    return panels


# ----- Main ----------------------------------------------------------------

def _serve_html(html_bytes, port, host="0.0.0.0"):
    """Tiny static HTTP server that returns the same HTML document for any
    GET request. Blocks until Ctrl-C."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, format, *args):
            # Quieter than the default access-log dump on stderr.
            sys.stderr.write(f"[viz-bokeh] {self.address_string()} - "
                             f"{format % args}\n")

    # Allow the port to be reused immediately on script restart.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), Handler) as srv:
        url = f"http://localhost:{port}/"
        print(f"\nServing on {url}  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata",
                        help="session_metadata.pb (per-probe files are discovered from it)")
    parser.add_argument("-o", "--output", default="profile.html",
                        help="Path to also write the HTML to (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port to serve on (default: %(default)s)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: %(default)s — all interfaces)")
    parser.add_argument("--no-serve", action="store_true",
                        help="Skip the HTTP server; just write the HTML file")
    args = parser.parse_args()

    gpu_trace, sys_trace, disk_trace, events, session_meta = _load_all(args.metadata)
    print(f"Session: {session_meta.hostname} @ {session_meta.start_iso8601}")
    print(f"Probes:  {[p.output_file for p in session_meta.probes]}")

    global_t0 = _global_t0(gpu_trace, sys_trace, disk_trace)

    # Shared x-range — pan/zoom on one panel syncs the rest. Bounds are
    # set so the user can't drag below 0 (no "before the run started"
    # blank space) or past xmax + 80% of the data span (room to scroll
    # right if labels are clipped, but not infinite).
    xmax_ms = _global_xmax_ms(gpu_trace, sys_trace, disk_trace, events, global_t0)
    if xmax_ms <= 0:
        xmax_ms = 1.0  # nothing to plot, but keep the range valid
    x_upper_bound = xmax_ms * 1.8  # xmax + 0.8 * (xmax - 0)
    shared_x = Range1d(start=0, end=xmax_ms,
                        bounds=(0, x_upper_bound))

    annot = build_annot_panel(events, shared_x, gpu_trace, global_t0)
    panels = [annot]
    panels += build_gpu_panels(gpu_trace, shared_x, global_t0)
    panels += build_system_panels(sys_trace, shared_x, global_t0)
    panels += build_disk_panels(disk_trace, shared_x, global_t0)

    # Shared synced crosshair: a single CrosshairTool with one vertical
    # Span overlay, attached to every panel. Bokeh 3.x propagates the
    # cursor's x across all figures sharing the tool, so moving the mouse
    # in any panel draws the dashed line at the same x on all of them.
    crosshair = CrosshairTool(overlay=Span(
        dimension="height", line_dash="dashed",
        line_color="#666666", line_width=1, line_alpha=0.6))
    for fig in panels:
        fig.add_tools(crosshair)

    # Title block as a Div above the column.
    from bokeh.models import Div
    title_lines = [f"<b>Full-System Profile</b> — {session_meta.hostname}"]
    if gpu_trace:
        title_lines[0] += f" — {gpu_trace.device_name} ({gpu_trace.chip_name})"
    if session_meta.start_iso8601:
        title_lines.append(f"Start: {session_meta.start_iso8601}")
    if session_meta.probes:
        import session_metadata_pb2
        kind_to_name = {
            session_metadata_pb2.PROBE_KIND_GPU:    "GPU",
            session_metadata_pb2.PROBE_KIND_SYSTEM: "System",
            session_metadata_pb2.PROBE_KIND_DISK:   "Disk",
            session_metadata_pb2.PROBE_KIND_EVENTS: "Events",
        }
        title_lines.append("Probes: " + ", ".join(kind_to_name.get(p.kind, "?")
                                                   for p in session_meta.probes))
    title_div = Div(text="<br>".join(title_lines),
                    styles={"font-size": TITLE_FONT_SIZE,
                             "padding-bottom": "10px"})

    # Per-figure widths are fixed (PLOT_WIDTH); the column itself just stacks.
    layout = column(title_div, *panels)

    # Render the layout to an HTML document. INLINE bundles BokehJS into
    # the page so the served copy doesn't need outbound network.
    page_title = "cupti_profiler — Bokeh visualization"
    html = file_html(layout, INLINE, title=page_title)
    html_bytes = html.encode("utf-8")

    # Always write the file too, so the user can copy / share / re-open
    # without keeping the server up.
    with open(args.output, "wb") as f:
        f.write(html_bytes)
    print(f"Saved interactive plot to {args.output}")

    if not args.no_serve:
        _serve_html(html_bytes, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
