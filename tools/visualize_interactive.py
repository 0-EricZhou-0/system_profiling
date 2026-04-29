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
import live_tail as lt      # projection + tail support shared with --live


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

def _pid_label(tracked_processes, pid):
    """"<alias> (PID xxx)" if the run carries an alias for this PID,
    else "PID xxx". Aliases come from the optional `alias` field on each
    TrackedProcess entry. Accepts either a `repeated TrackedProcess`
    proto field or a plain list of `TrackedProcess` messages.
    """
    pid = int(pid)
    for tp in tracked_processes or []:
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


# Note: `live_tail.global_t0_from_traces` and `live_tail.global_xmax_ms`
# are the canonical implementations of the time-range helpers; both the
# static and live entrypoints route through them so the shared x-axis is
# computed identically in either mode.


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


def build_annot_panel(annot_data, x_range):
    """Single strip showing regions (colored bars at y=0) + events
    (vertical lines + triangle markers at y=0.5). Hover shows the name
    and duration or absolute timestamp.

    `annot_data` is the projection-output dict:
        {"annot_regions": {col: list, ...} | None,
         "annot_events":  {col: list, ...} | None}

    Returns (figure, cds_by_key).
    """
    kwargs = dict(width=PLOT_WIDTH, height=ANNOT_HEIGHT, title="Regions / Events",
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

    cds_by_key = {}

    region_data = annot_data.get("annot_regions")
    # Always create the CDS — even if currently empty — so live tick can
    # populate it later when the first regions arrive.
    region_cds = ColumnDataSource(region_data or dict(
        name=[], left=[], right=[], dur_ms=[], color=[], bottom=[], top=[]))
    cds_by_key["annot_regions"] = region_cds
    region_renderer = fig.quad(
        left="left", right="right", bottom="bottom", top="top",
        color="color", alpha=0.7, line_color="color", source=region_cds,
        legend_label="region")
    fig.add_tools(HoverTool(renderers=[region_renderer],
                             tooltips=[("region", "@name"),
                                       ("start (ms)", "@left{0.00}"),
                                       ("end (ms)", "@right{0.00}"),
                                       ("duration (ms)", "@dur_ms{0.00}")]))

    event_data = annot_data.get("annot_events")
    event_cds = ColumnDataSource(event_data or dict(
        name=[], x=[], color=[], y=[], y0=[], y1=[]))
    cds_by_key["annot_events"] = event_cds
    event_marker = fig.scatter(
        x="x", y="y", marker="inverted_triangle", size=12,
        color="color", line_color="black", source=event_cds,
        legend_label="event")
    fig.segment(x0="x", y0="y0", x1="x", y1="y1",
                line_color="color", line_width=1.5, line_alpha=0.6,
                source=event_cds)
    fig.add_tools(HoverTool(renderers=[event_marker],
                             tooltips=[("event", "@name"),
                                       ("t (ms)", "@x{0.00}")]))

    _move_legend_outside(fig)
    return fig, cds_by_key


def build_gpu_panels(gpu_frame, x_range):
    """Build SM/DRAM/PCIe/NVLink panels from a projected gpu frame.

    `gpu_frame` is `{"data": {key: cds-shaped dict | None}, "meta": {...}}`
    produced by `live_tail.project_gpu`. Returns `(panels, cds_by_key)`.
    """
    panels = []
    cds_by_key = {}
    data = gpu_frame["data"]
    meta = gpu_frame["meta"]

    if data["gpu_sm_util"] is not None:
        src = ColumnDataSource(data["gpu_sm_util"])
        cds_by_key["gpu_sm_util"] = src
        fig = _empty_figure("GPU SM Utilization", x_range, y_label="%")
        fig.line("t", "avg", source=src, line_width=1,
                 color="#1f77b4", legend_label="avg")
        if "max" in data["gpu_sm_util"]:
            fig.line("t", "max", source=src, line_width=1.4,
                     color="#d62728", legend_label="max")
        tt = [("t (ms)", "@t{0.00}"), ("avg", "@avg{0.0}%")]
        if "max" in data["gpu_sm_util"]:
            tt.append(("max", "@max{0.0}%"))
        _attach_anchored_hover(fig, src, tt)
        _clamp_and_mark_peak(fig, 100, label="Max SM Util: 100%")
        _move_legend_outside(fig)
        panels.append(fig)

    if data["gpu_warps"] is not None:
        src = ColumnDataSource(data["gpu_warps"])
        cds_by_key["gpu_warps"] = src
        fig = _empty_figure("Active Warps / Cycle", x_range,
                             y_label="warps / cycle")
        fig.line("t", "avg", source=src, line_width=0.8, alpha=0.45,
                 color="#ff7f0e", legend_label="avg (across all SMs)")
        if "max" in data["gpu_warps"]:
            fig.line("t", "max", source=src, line_width=1.0,
                     color="#ff7f0e", legend_label="max (busiest SM)")
        tt = [("t (ms)", "@t{0.00}"), ("avg", "@avg{0.00}")]
        if "max" in data["gpu_warps"]:
            tt.append(("max", "@max{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        # H100 / GH100: 64 warps/SM is the architectural cap.
        _clamp_and_mark_peak(fig, 64, label="Max Active Warps: 64")
        _move_legend_outside(fig)
        panels.append(fig)

    if data["gpu_dram"] is not None:
        peak = meta.get("peak_dram_gibps")
        unit = "GiB/s" if peak else "% of peak"
        src = ColumnDataSource(data["gpu_dram"])
        cds_by_key["gpu_dram"] = src
        fig = _empty_figure("GPU DRAM Bandwidth", x_range, y_label=unit)
        fig.line("t", "rd", source=src, line_width=1, color="#2ca02c",
                 legend_label="read avg")
        if "wr" in data["gpu_dram"]:
            fig.line("t", "wr", source=src, line_width=1, color="#ff7f0e",
                     legend_label="write avg")
        tt = [("t (ms)", "@t{0.00}"), (f"read ({unit})", "@rd{0.00}")]
        if "wr" in data["gpu_dram"]:
            tt.append((f"write ({unit})", "@wr{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        if peak:
            _clamp_and_mark_peak(fig, peak,
                                  label=f"Max DRAM BW: {peak:.0f} GiB/s")
        _move_legend_outside(fig)
        panels.append(fig)

    if data["gpu_pcie"] is not None:
        src = ColumnDataSource(data["gpu_pcie"])
        cds_by_key["gpu_pcie"] = src
        fig = _empty_figure("GPU PCIe Bandwidth", x_range, y_label="GiB/s")
        fig.line("t", "rd", source=src, line_width=1, color="#9467bd",
                 legend_label="H→D")
        if "wr" in data["gpu_pcie"]:
            fig.line("t", "wr", source=src, line_width=1, color="#8c564b",
                     legend_label="D→H")
        tt = [("t (ms)", "@t{0.00}"), ("H→D (GiB/s)", "@rd{0.00}")]
        if "wr" in data["gpu_pcie"]:
            tt.append(("D→H (GiB/s)", "@wr{0.00}"))
        _attach_anchored_hover(fig, src, tt)
        peak_pcie_bidi = meta.get("peak_pcie_bidi_gibps")
        if peak_pcie_bidi:
            _clamp_and_mark_peak(fig, peak_pcie_bidi,
                                  label=f"Max PCIe BW: {peak_pcie_bidi:.1f} GiB/s (Bi-directional)")
        _move_legend_outside(fig)
        panels.append(fig)

    if data["gpu_pcie_cum"] is not None:
        src = ColumnDataSource(data["gpu_pcie_cum"])
        cds_by_key["gpu_pcie_cum"] = src
        unit = meta.get("pcie_cum_unit", "B")
        fig = _empty_figure("Cumulative PCIe Bytes", x_range, y_label=unit)
        fig.line("t", "rd", source=src, line_width=1, color="#9467bd",
                 legend_label="H→D cumulative")
        fig.line("t", "wr", source=src, line_width=1, color="#8c564b",
                 legend_label="D→H cumulative")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             (f"H→D ({unit})", "@rd{0.00}"),
             (f"D→H ({unit})", "@wr{0.00}")])
        # Cumulative — auto-fit, just keep >= 0. Cap from the initial frame;
        # in live mode, axis bound stays put as new samples scroll past.
        max_val = max(float(np.max(data["gpu_pcie_cum"]["rd"])
                              if len(data["gpu_pcie_cum"]["rd"]) else 0),
                       float(np.max(data["gpu_pcie_cum"]["wr"])
                              if len(data["gpu_pcie_cum"]["wr"]) else 0))
        upper = max(0.01, max_val) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if data["gpu_nvlink"] is not None:
        src = ColumnDataSource(data["gpu_nvlink"])
        cds_by_key["gpu_nvlink"] = src
        fig = _empty_figure("GPU NVLink Bandwidth", x_range, y_label="GiB/s")
        fig.line("t", "rx", source=src, line_width=1, color="#e377c2",
                 legend_label="RX")
        fig.line("t", "tx", source=src, line_width=1, color="#17becf",
                 legend_label="TX")
        _attach_anchored_hover(fig, src,
            [("t (ms)", "@t{0.00}"),
             ("RX (GiB/s)", "@rx{0.00}"),
             ("TX (GiB/s)", "@tx{0.00}")])
        peak_nvl_bidi = meta.get("peak_nvlink_bidi_gibps")
        if peak_nvl_bidi:
            _clamp_and_mark_peak(fig, peak_nvl_bidi,
                                  label=f"Max NVLink BW: {peak_nvl_bidi:.1f} GiB/s (Bi-directional)")
        _move_legend_outside(fig)
        panels.append(fig)

    return panels, cds_by_key


def build_system_panels(sys_frame, x_range):
    """Build CPU + memory panels from a projected system frame.

    Returns `(panels, cds_by_key)`.
    """
    panels = []
    cds_by_key = {}
    data = sys_frame["data"]
    meta = sys_frame["meta"]
    tracked = meta.get("tracked_processes", [])

    if data["sys_cpu_total"] is not None:
        src = ColumnDataSource(data["sys_cpu_total"])
        cds_by_key["sys_cpu_total"] = src
        fig = _empty_figure("CPU Utilization (system-wide)", x_range, y_label="%")
        fig.varea(x="t", y1=0,            y2="y_user_top", source=src,
                  color="#4c72b0", alpha=0.7, legend_label="User")
        fig.varea(x="t", y1="y_user_top", y2="y_sys_top",  source=src,
                  color="#dd8452", alpha=0.7, legend_label="System")
        fig.varea(x="t", y1="y_sys_top",  y2="y_iow_top",  source=src,
                  color="#c44e52", alpha=0.7, legend_label="IOWait")
        fig.line("t", "total", source=src, color="black",
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

    if data["sys_cpu_proc"] is not None:
        src = ColumnDataSource(data["sys_cpu_proc"])
        cds_by_key["sys_cpu_proc"] = src
        pids = meta["pids"]
        fig = _empty_figure("CPU Utilization (per process)", x_range, y_label="%")
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            color = REGION_COLORS[i % 8]
            label = _pid_label(tracked, pid)
            fig.line("t", f"pid_{pid}_sum", source=src, line_width=1, color=color,
                     legend_label=f"{label} (usr+sys)")
            fig.line("t", f"pid_{pid}_user", source=src, line_width=0.8,
                     alpha=0.5, line_dash="dashed", color=color,
                     legend_label=f"{label} (usr)")
            tt.append((f"{label} usr+sys", f"@pid_{pid}_sum{{0.0}}%"))
            tt.append((f"{label} usr",     f"@pid_{pid}_user{{0.0}}%"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(1.0, meta["cpu_proc_max"]) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if data["sys_mem_total"] is not None:
        src = ColumnDataSource(data["sys_mem_total"])
        cds_by_key["sys_mem_total"] = src
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
        total_gib = meta.get("total_ram_gib", 0.0)
        if total_gib > 0:
            _clamp_and_mark_peak(fig, total_gib,
                                  label=f"Total: {total_gib:.0f} GiB")
        _move_legend_outside(fig)
        panels.append(fig)

    if data["sys_mem_proc"] is not None:
        src = ColumnDataSource(data["sys_mem_proc"])
        cds_by_key["sys_mem_proc"] = src
        pids = meta["pids"]
        fig = _empty_figure("Process Memory (RSS)", x_range, y_label="GiB")
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            label = _pid_label(tracked, pid)
            fig.line("t", f"pid_{pid}_rss", source=src, line_width=1,
                     color=REGION_COLORS[i % 8],
                     legend_label=f"{label} RSS")
            tt.append((f"{label} RSS", f"@pid_{pid}_rss{{0.00}} GiB"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(0.01, meta.get("mem_proc_max_gib", 0.0)) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    return panels, cds_by_key


def build_disk_panels(disk_frame, x_range):
    """Build per-device + per-process disk panels from a projected disk frame.

    Returns `(panels, cds_by_key)`.
    """
    panels = []
    cds_by_key = {}
    data = disk_frame["data"]
    meta = disk_frame["meta"]
    devs = meta.get("devs", [])
    tracked = meta.get("tracked_processes", [])

    if data["disk_dev_bw"] is not None:
        src = ColumnDataSource(data["disk_dev_bw"])
        cds_by_key["disk_dev_bw"] = src
        fig = _empty_figure("Disk Bandwidth (per device)", x_range, y_label="MiB/s")
        for palette_idx, dev in enumerate(devs):
            color = REGION_COLORS[palette_idx % 8]
            fig.line("t", f"{dev}_rd", source=src, line_width=1.2, color=color,
                     legend_label=f"{dev} read")
            fig.line("t", f"{dev}_wr", source=src, line_width=1.0,
                     line_dash="dashed", color=color,
                     legend_label=f"{dev} write")
        tt = [("t (ms)", "@t{0.00}")]
        for dev in devs:
            tt.append((f"{dev} read",  f"@{dev}_rd{{0.00}} MiB/s"))
            tt.append((f"{dev} write", f"@{dev}_wr{{0.00}} MiB/s"))
        _attach_anchored_hover(fig, src, tt)
        bw_max = meta.get("dev_bw_max_mibps", 0.0)
        if bw_max > 0:
            upper = max(0.01, bw_max) * YLIM_HEADROOM
            fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if data["disk_proc_bw"] is not None:
        src = ColumnDataSource(data["disk_proc_bw"])
        cds_by_key["disk_proc_bw"] = src
        pids = meta["pids"]
        fig = _empty_figure("Process Disk IO", x_range, y_label="MiB/s")
        tt = [("t (ms)", "@t{0.00}")]
        for i, pid in enumerate(pids):
            color = REGION_COLORS[i % 8]
            label = _pid_label(tracked, pid)
            fig.line("t", f"pid_{pid}_rd", source=src, line_width=1, color=color,
                     legend_label=f"{label} read")
            fig.line("t", f"pid_{pid}_wr", source=src, line_width=1,
                     line_dash="dashed", color=color,
                     legend_label=f"{label} write")
            tt.append((f"{label} read",  f"@pid_{pid}_rd{{0.00}} MiB/s"))
            tt.append((f"{label} write", f"@pid_{pid}_wr{{0.00}} MiB/s"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(0.01, meta.get("proc_bw_max_mibps", 0.0)) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    if data["disk_dev_q"] is not None:
        src = ColumnDataSource(data["disk_dev_q"])
        cds_by_key["disk_dev_q"] = src
        fig = _empty_figure("Disk Queue Depth (per device)", x_range,
                             y_label="depth")
        tt = [("t (ms)", "@t{0.00}")]
        for i, dev in enumerate(devs):
            color = REGION_COLORS[i % 8]
            fig.line("t", f"{dev}_rdq", source=src, line_width=1, color=color,
                     legend_label=f"{dev} read Q")
            fig.line("t", f"{dev}_wrq", source=src, line_width=1,
                     line_dash="dashed", color=color,
                     legend_label=f"{dev} write Q")
            tt.append((f"{dev} read Q",  f"@{dev}_rdq{{0}}"))
            tt.append((f"{dev} write Q", f"@{dev}_wrq{{0}}"))
        _attach_anchored_hover(fig, src, tt)
        upper = max(1.0, meta.get("q_max", 0.0)) * YLIM_HEADROOM
        fig.y_range = Range1d(start=0, end=upper, bounds=(0, upper))
        _move_legend_outside(fig)
        panels.append(fig)

    return panels, cds_by_key


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


def _build_layout(frame, gpu_trace, sys_trace, disk_trace, events,
                   global_t0, session_meta, *, live=False):
    """Construct the full Bokeh layout from a projected frame.

    `frame` is the dict returned by `LiveState.project_all()` (or an
    equivalent built directly via `live_tail.project_*` calls in static
    mode). Returns `(layout, cds_by_key)`. The trace handles are passed
    through purely for the title line (device name, etc.) and for the
    initial xmax computation; the layout itself only consumes `frame`.
    """
    from bokeh.models import Div

    xmax_ms = lt.global_xmax_ms(gpu_trace, sys_trace, disk_trace,
                                  events, global_t0)
    if xmax_ms <= 0:
        xmax_ms = 1.0
    # In live mode the data extends past the initial xmax. Give the upper
    # bound enough headroom that pan/zoom doesn't immediately bottom out,
    # while the static path keeps the snug 1.8x bound it had before.
    x_upper_bound = xmax_ms * (10.0 if live else 1.8)
    shared_x = Range1d(start=0, end=xmax_ms, bounds=(0, x_upper_bound))

    cds_by_key = {}

    annot_data = {k: frame["data"].get(k) for k in ("annot_regions",
                                                       "annot_events")}
    annot, annot_cds = build_annot_panel(annot_data, shared_x)
    cds_by_key.update(annot_cds)

    gpu_frame = {"data": {k: frame["data"].get(k)
                           for k in ("gpu_sm_util", "gpu_warps", "gpu_dram",
                                     "gpu_pcie", "gpu_pcie_cum", "gpu_nvlink")},
                  "meta": frame["meta"].get("gpu", {})}
    gpu_panels, gpu_cds = build_gpu_panels(gpu_frame, shared_x)
    cds_by_key.update(gpu_cds)

    sys_frame = {"data": {k: frame["data"].get(k)
                           for k in ("sys_cpu_total", "sys_cpu_proc",
                                     "sys_mem_total", "sys_mem_proc")},
                  "meta": frame["meta"].get("sys", {})}
    sys_panels, sys_cds = build_system_panels(sys_frame, shared_x)
    cds_by_key.update(sys_cds)

    disk_frame = {"data": {k: frame["data"].get(k)
                            for k in ("disk_dev_bw", "disk_proc_bw",
                                      "disk_dev_q")},
                   "meta": frame["meta"].get("disk", {})}
    disk_panels, disk_cds = build_disk_panels(disk_frame, shared_x)
    cds_by_key.update(disk_cds)

    panels = [annot] + gpu_panels + sys_panels + disk_panels

    # Shared synced crosshair across every panel.
    crosshair = CrosshairTool(overlay=Span(
        dimension="height", line_dash="dashed",
        line_color="#666666", line_width=1, line_alpha=0.6))
    for fig in panels:
        fig.add_tools(crosshair)

    title_lines = [f"<b>Full-System Profile</b> — {session_meta.hostname}"]
    if live:
        title_lines[0] += " (live)"
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
    layout = column(title_div, *panels)
    return layout, cds_by_key


def _project_static(gpu_trace, sys_trace, disk_trace, events, t0):
    """Run the four projection helpers and merge their outputs into the
    same shape `LiveState.project_all` returns. Lets `_build_layout`
    treat static and live entrypoints identically.
    """
    gpu = lt.project_gpu(gpu_trace, t0)
    sysm = lt.project_system(sys_trace, t0)
    disk = lt.project_disk(disk_trace, t0)
    annot = lt.project_events(events, gpu_trace, t0,
                                REGION_COLORS, EVENT_COLORS)
    return {
        "data": {**gpu["data"], **sysm["data"], **disk["data"], **annot["data"]},
        "meta": {"gpu": gpu["meta"], "sys": sysm["meta"], "disk": disk["meta"]},
    }


def _run_live(args):
    """Bokeh-server entrypoint: tail the .pb files and stream new samples
    into the running document on every poll tick.
    """
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.server.server import Server

    state = lt.LiveState(args.metadata, REGION_COLORS, EVENT_COLORS)
    print(f"[live] Waiting for {args.metadata} (timeout {args.live_bootstrap_timeout_s}s)…")
    if not state.wait_for_metadata(args.live_bootstrap_timeout_s):
        print(f"[live] Timed out waiting for {args.metadata}.", file=sys.stderr)
        sys.exit(1)
    state.initial_load()
    print(f"[live] Session: {state.session_meta.hostname} "
          f"@ {state.session_meta.start_iso8601}")
    print(f"[live] Probes:  {[p.output_file for p in state.session_meta.probes]}")

    def make_doc(doc):
        frame = state.project_all()
        layout, cds = _build_layout(frame, state.gpu_trace, state.sys_trace,
                                      state.disk_trace, state.events,
                                      state.t0, state.session_meta, live=True)
        # Each browser tab gets its own CDS registry — `state.cds` would
        # be shared across tabs and confuse the per-tick replace logic.
        # Wrap tick() so the callback closes over this connection's CDS.
        per_doc_cds = dict(cds)

        def tick():
            if not state._drain_into_traces():
                return
            f = state.project_all()
            for key, cols in f["data"].items():
                if cols is None:
                    continue
                src = per_doc_cds.get(key)
                if src is None:
                    continue
                src.data = {k: list(v) if isinstance(v, np.ndarray) else v
                             for k, v in cols.items()}

        doc.add_root(layout)
        doc.add_periodic_callback(tick, args.poll_interval_ms)
        doc.title = "cupti_profiler — Bokeh visualization (live)"

    app = Application(FunctionHandler(make_doc))
    origins = [f"{args.host}:{args.port}", f"localhost:{args.port}",
                f"127.0.0.1:{args.port}"]
    server = Server({"/": app}, address=args.host, port=args.port,
                    allow_websocket_origin=origins)
    server.start()
    url = f"http://localhost:{args.port}/"
    print(f"[live] Serving on {url}  (Ctrl-C to stop)")
    try:
        server.io_loop.add_callback(server.show, "/")
    except Exception:
        pass  # show() opens a browser; ignore failures (e.g. headless server).
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print("\n[live] Shutting down.")


def _run_static(args):
    gpu_trace, sys_trace, disk_trace, events, session_meta = _load_all(args.metadata)
    print(f"Session: {session_meta.hostname} @ {session_meta.start_iso8601}")
    print(f"Probes:  {[p.output_file for p in session_meta.probes]}")

    global_t0 = lt.global_t0_from_traces(gpu_trace, sys_trace, disk_trace)
    frame = _project_static(gpu_trace, sys_trace, disk_trace, events, global_t0)
    layout, _ = _build_layout(frame, gpu_trace, sys_trace, disk_trace,
                                events, global_t0, session_meta, live=False)

    page_title = "cupti_profiler — Bokeh visualization"
    html = file_html(layout, INLINE, title=page_title)
    html_bytes = html.encode("utf-8")

    with open(args.output, "wb") as f:
        f.write(html_bytes)
    print(f"Saved interactive plot to {args.output}")

    if not args.no_serve:
        _serve_html(html_bytes, port=args.port, host=args.host)


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
    parser.add_argument("--live", action="store_true",
                        help="Tail .pb files and stream new samples into a "
                             "running Bokeh server (auto-refreshes the page).")
    parser.add_argument("--poll-interval-ms", type=int, default=1000,
                        help="Live mode: how often to read new bytes "
                             "(default: %(default)s).")
    parser.add_argument("--live-bootstrap-timeout-s", type=float, default=30.0,
                        help="Live mode: how long to wait for "
                             "session_metadata.pb to appear (default: %(default)s).")
    args = parser.parse_args()

    if args.live:
        _run_live(args)
    else:
        _run_static(args)


if __name__ == "__main__":
    main()
