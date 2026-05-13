#!/usr/bin/env python3
"""Render a GPU-only profile (gpu_metrics.pb) to a PNG.

Used by workflows that drive GpuProfiler directly without the suite
(examples/gemm_profiling.cu). For full-system runs that produce a
session_metadata.pb, use tools/visualize_all.py instead.

    python tools/visualize_single.py -i gpu_metrics.pb -o gpu.png

The MetricCatalog is loaded from --catalog (default:
configs/metric_catalog.pbtxt next to the repo). The panel layout is
filtered to SCOPE_GPU panels only.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "generated" / "proto"))

from google.protobuf.internal.decoder import _DecodeVarint32  # noqa: E402

import metric_catalog_pb2 as mc_pb  # noqa: E402
import gpu_metrics_pb2  # noqa: E402

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402

# Reuse visualize_all's rendering helpers — same matplotlib output.
import visualize_all as _va  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default="gpu_metrics.pb",
                        help="Path to gpu_metrics.pb (default: gpu_metrics.pb)")
    parser.add_argument("-o", "--output", default="gpu_profile.png",
                        help="Output PNG path (default: gpu_profile.png)")
    parser.add_argument("--catalog", default=None,
                        help="Override MetricCatalog pbtxt (default: "
                             "configs/metric_catalog.pbtxt)")
    parser.add_argument("--panel-layout", default=None,
                        help="Override PanelLayout pbtxt (default: "
                             "configs/visualizer_panels.pbtxt)")
    parser.add_argument("--smooth-window-s", type=float, default=0.01,
                        help="Boxcar smoothing window in seconds. 0 = none.")
    parser.add_argument("--panel-height", type=float, default=2.6)
    parser.add_argument("--panel-width", type=float, default=14.0)
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 1

    catalog_path = Path(args.catalog) if args.catalog \
        else _HERE.parent / "configs" / "metric_catalog.pbtxt"
    catalog = metric_catalog.load_catalog(catalog_path)

    layout_path = Path(args.panel_layout) if args.panel_layout \
        else _HERE.parent / "configs" / "visualizer_panels.pbtxt"
    layout = metric_layout.load_panel_layout(layout_path)

    projector = TraceProjector(catalog)
    _va._log(f"reading {in_path}")
    traces = _va._read_delimited(in_path, gpu_metrics_pb2.GPUMetricsTrace)
    for t in traces:
        projector.ingest_gpu(t)
    sample_freq = traces[0].header.sampling_frequency_hz if traces else 0
    _va._log(f"  {len(traces)} flushes, "
             f"{sum(len(t.samples) for t in traces)} samples, "
             f"sampling_freq={sample_freq} Hz")

    proj = projector.project()
    if not proj:
        _va._log("no samples")
        return 1
    t0_ns = min(int(ts[0]) for ts, _ in proj.values() if ts.size > 0)

    catalog_index = metric_catalog.build_index(catalog)
    for (fqn, _) in proj.keys():
        if fqn not in catalog_index:
            catalog_index[fqn] = metric_layout.synthesize_descriptor(fqn)

    # Filter to GPU-only panels.
    series_keys = list(proj.keys())
    resolved = []
    for panel in layout.panels:
        if panel.scope != mc_pb.SCOPE_GPU:
            continue
        series = metric_layout.resolve_panel_series(panel, catalog_index, series_keys)
        if not series:
            continue
        resolved.append((panel, series))
    if not resolved:
        _va._log("no GPU panels matched any series — check --panel-layout")
        return 1
    _va._log(f"rendering {len(resolved)} GPU panels")

    n = len(resolved)
    fig, axes = plt.subplots(
        n, 1, figsize=(args.panel_width, args.panel_height * n),
        squeeze=False, sharex=True,
    )
    axes = axes.flatten()
    for ax, (panel, series_list) in zip(axes, resolved):
        _va._render_panel(ax, panel, series_list, projector, proj,
                          sample_freq_hz=sample_freq,
                          smooth_window_s=args.smooth_window_s,
                          t0_ns=t0_ns)
    axes[-1].set_xlabel("time (s)")

    info = next(iter(projector.gpu_info.values()), None)
    title = f"GPU profile — {info.device_name} ({info.chip_name})" if info else "GPU profile"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    out_path = Path(args.output).resolve()
    _va._log(f"saving to {out_path}")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    _va._log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
