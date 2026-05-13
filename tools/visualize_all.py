#!/usr/bin/env python3
"""Render a full-system profile (gpu_metrics.pb + system_metrics.pb +
disk_metrics.pb) to a single tall PNG, driven by the MetricCatalog +
PanelLayout pbtxts.

Usage:
    python tools/visualize_all.py profiling_output/session_metadata.pb \\
        -o profile.png

The catalog (FQN → type/unit/peak/scope) is inlined into
session_metadata.pb at run time; the panel layout (which FQNs go on
which subplot) defaults to configs/visualizer_panels.pbtxt. Both can
be overridden via --catalog / --panel-layout.

GPU FQNs that aren't in the static catalog are synthesized from
suffix parsing (see tools/metric_layout.py :: synthesize_descriptor).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()

def _log(msg: str) -> None:
    """[HH:MM:SS.mmm] (+X.XXXs) <msg>"""
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


# ---------------------------------------------------------------------------
# Smoothing — O(N) cumsum boxcar
# ---------------------------------------------------------------------------

def _smooth(vals: np.ndarray, k: int) -> np.ndarray:
    """Edge-aware boxcar via cumsum (divide by actual window size at
    the edges). O(N)."""
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
# Peak resolution at panel level
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
    # No panel-level peak — fall back to descriptor's own.
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


def _render_panel(
    ax,
    panel,
    series_list: list[metric_layout.ResolvedSeries],
    projector: TraceProjector,
    projection: dict,
    sample_freq_hz: float,
    smooth_window_s: float,
    t0_ns: int,
) -> None:
    ax.set_title(panel.title)
    ax.grid(True, alpha=0.3)

    unit = panel.unit_override if panel.unit_override != mc_pb.UNIT_UNSPECIFIED \
        else series_list[0].descriptor.unit
    peak_hint = _resolve_panel_peak(panel, series_list[0].descriptor, projector)
    scale_fn, ylabel = _format_unit_axis(unit, peak_hint)
    ax.set_ylabel(ylabel)

    for i, series in enumerate(series_list):
        color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
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

    if peak_hint is not None and peak_hint > 0:
        ax.axhline(scale_fn(peak_hint), color="red", linestyle="--",
                   linewidth=0.8, alpha=0.6,
                   label=f"peak = {scale_fn(peak_hint):.2g} {ylabel}")

    if panel.y_min != 0.0 or panel.y_max != 0.0:
        cur = ax.get_ylim()
        ax.set_ylim(panel.y_min if panel.y_min != 0.0 else cur[0],
                    panel.y_max if panel.y_max != 0.0 else cur[1])

    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _resolve_path(metadata_path: Path, p: str) -> Path:
    """Probe paths are stored relative to the suite's CWD. Try several
    candidates."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    cands = [Path.cwd() / pp, metadata_path.parent / pp.name, metadata_path.parent / pp]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


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
            traces = _read_delimited(out, gpu_metrics_pb2.GPUMetricsTrace)
            for t in traces:
                projector.ingest_gpu(t)
            sample_freqs["gpu"] = probe.sampling_frequency_hz
            _log(f"  gpu:    {len(traces)} flushes, "
                 f"{sum(len(t.samples) for t in traces)} samples")
        elif probe.kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
            traces = _read_delimited(out, system_metrics_pb2.SystemMetricsTrace)
            for t in traces:
                projector.ingest_system(t)
            sample_freqs["system"] = probe.sampling_frequency_hz
            _log(f"  system: {len(traces)} flushes, "
                 f"{sum(len(t.system_samples) for t in traces)} sys + "
                 f"{sum(len(t.process_samples) for t in traces)} proc samples")
        elif probe.kind == session_metadata_pb2.PROBE_KIND_DISK:
            traces = _read_delimited(out, disk_metrics_pb2.DiskMetricsTrace)
            for t in traces:
                projector.ingest_disk(t)
            sample_freqs["disk"] = probe.sampling_frequency_hz
            _log(f"  disk:   {len(traces)} flushes, "
                 f"{sum(len(t.device_samples) for t in traces)} dev + "
                 f"{sum(len(t.process_samples) for t in traces)} proc samples")
        # PROBE_KIND_EVENTS is timeline data — not rendered by this tool.
    return sample_freqs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", help="Path to session_metadata.pb")
    parser.add_argument("-o", "--output", default="profile.png",
                        help="Output PNG path (default: profile.png)")
    parser.add_argument("--catalog", default=None,
                        help="Override MetricCatalog pbtxt (default: inlined "
                             "in session_metadata.pb)")
    parser.add_argument("--panel-layout", default=None,
                        help="Override PanelLayout pbtxt (default: "
                             "configs/visualizer_panels.pbtxt)")
    parser.add_argument("--smooth-window-s", type=float, default=0.05,
                        help="Boxcar smoothing window in seconds. 0 = none. "
                             "Default: 0.05.")
    parser.add_argument("--panel-height", type=float, default=2.6,
                        help="Per-panel height (in) (default: 2.6)")
    parser.add_argument("--panel-width", type=float, default=14.0,
                        help="Figure width (in) (default: 14.0)")
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
    sample_freqs = _ingest_probes(projector, meta, metadata_path)

    _log("projecting")
    proj = projector.project()
    if not proj:
        _log("no samples — nothing to plot")
        return 1

    t0_ns = min(int(ts[0]) for ts, _ in proj.values() if ts.size > 0)

    catalog_index = metric_catalog.build_index(catalog)
    # Promote synthesized GPU descriptors so panel glob matching works.
    for (fqn, _) in proj.keys():
        if fqn not in catalog_index:
            catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

    series_keys = list(proj.keys())
    resolved: list[tuple] = []
    for panel in layout.panels:
        series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
        if not series:
            _log(f"  skip panel {panel.title!r}  (no matching series)")
            continue
        resolved.append((panel, series))
    if not resolved:
        _log("no panels resolved any series")
        return 1
    _log(f"rendering {len(resolved)} panels")

    n = len(resolved)
    fig, axes = plt.subplots(
        n, 1, figsize=(args.panel_width, args.panel_height * n),
        squeeze=False, sharex=True,
    )
    axes = axes.flatten()
    for ax, (panel, series_list) in zip(axes, resolved):
        if panel.scope == mc_pb.SCOPE_GPU:
            f = sample_freqs.get("gpu", 0)
        elif panel.scope == mc_pb.SCOPE_DEVICE:
            f = sample_freqs.get("disk", 0)
        else:
            f = sample_freqs.get("system", 0)
        _render_panel(ax, panel, series_list, projector, proj,
                      sample_freq_hz=f, smooth_window_s=args.smooth_window_s,
                      t0_ns=t0_ns)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Profile — {meta.hostname}  {meta.start_iso8601}", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    out_path = Path(args.output).resolve()
    _log(f"saving to {out_path}")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
