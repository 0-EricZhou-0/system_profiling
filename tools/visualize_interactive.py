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
import gpu_metrics_pb2  # noqa: E402
import system_metrics_pb2  # noqa: E402
import disk_metrics_pb2  # noqa: E402
import session_metadata_pb2  # noqa: E402

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
import metric_suffix  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402

from bokeh.application import Application  # noqa: E402
from bokeh.application.handlers.function import FunctionHandler  # noqa: E402
from bokeh.embed import file_html  # noqa: E402
from bokeh.layouts import column  # noqa: E402
from bokeh.models import ColumnDataSource, HoverTool, Span  # noqa: E402
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
                  projector: TraceProjector) -> str:
    """Compact legend label. Hover tooltips can carry the long form."""
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
        info = projector.gpu_info.get(int(key))
        if info and info.device_name:
            return f"{base}  [GPU {key}: {info.device_name}]"
        return f"{base}  [GPU {key}]"
    return base


# ---------------------------------------------------------------------------
# Panel building (Bokeh)
# ---------------------------------------------------------------------------

# Bokeh color palette — Category10 has 10 distinct colors; cycle past that.
_PALETTE = list(Category10[10])


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
        width=1200, height=240,
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        active_drag="xpan", active_scroll="xwheel_zoom",
        output_backend="webgl",
    )
    if x_range is not None:
        fig_kwargs["x_range"] = x_range
    fig = figure(**fig_kwargs)

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
                 legend_label=_series_label(series, projector))

    # Peak reference line.
    if peak_hint is not None and peak_hint > 0:
        scaled_peak = scale_fn(peak_hint)
        fig.add_layout(Span(location=scaled_peak, dimension="width",
                            line_color="red", line_dash="dashed",
                            line_alpha=0.6, line_width=1.5))

    # Y range overrides.
    if panel.y_min != 0.0 or panel.y_max != 0.0:
        if panel.y_min != 0.0:
            fig.y_range.start = panel.y_min
        if panel.y_max != 0.0:
            fig.y_range.end = panel.y_max

    # Hover — anchored at the panel bottom edge so it doesn't follow
    # any single glyph (works even when legend items are hidden).
    fig.add_tools(HoverTool(
        tooltips=[("t", "@x{0.000}s"), ("y", "@y{0.000}")],
        mode="vline",
    ))

    # Click-to-hide legend entries (only present if a glyph added one).
    if fig.legend:
        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"
        fig.legend.label_text_font_size = "8pt"

    return fig, cds_by_key


# ---------------------------------------------------------------------------
# Static rendering path
# ---------------------------------------------------------------------------

def _render_static(
    projector: TraceProjector,
    layout: metric_layout.PanelLayout,
    catalog_index: dict[str, mc_pb.MetricDescriptor],
    out_path: Path,
    title: str,
) -> None:
    proj = projector.project()
    if not proj:
        _log("no samples — nothing to render")
        return
    t0_ns = min(int(ts[0]) for ts, _ in proj.values() if ts.size > 0)
    series_keys = list(proj.keys())

    # Promote synthesized GPU descriptors.
    for (fqn, _) in series_keys:
        if fqn not in catalog_index:
            catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

    figs = []
    shared_x = None
    for panel in layout.panels:
        series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
        if not series:
            continue
        fig, _ = _build_panel(panel, series, projector, proj, t0_ns,
                              x_range=shared_x)
        if shared_x is None:
            shared_x = fig.x_range
        figs.append(fig)

    layout_root = column(figs, sizing_mode="stretch_width")
    html = file_html(layout_root, INLINE, title=title)
    out_path.write_text(html)
    _log(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KiB)")


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

    httpd = socketserver.TCPServer((host, port), _Handler)
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
    args = parser.parse_args()

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
    _ingest_probes(projector, meta, metadata_path)

    catalog_index = metric_catalog.build_index(catalog)
    out_path = Path(args.output).resolve()
    title = f"Profile — {meta.hostname}  {meta.start_iso8601}"
    _log("rendering")
    _render_static(projector, layout, catalog_index, out_path, title)

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
        width=1200, height=240,
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        active_drag="xpan", active_scroll="xwheel_zoom",
        output_backend="webgl",
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
        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"
        fig.legend.label_text_font_size = "8pt"

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
        info = projector.gpu_info.get(int(scope_key))
        if info and info.device_name:
            label = f"{base}  [GPU {scope_key}: {info.device_name}]"
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
