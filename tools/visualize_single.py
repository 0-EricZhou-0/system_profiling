#!/usr/bin/env python3
"""Visualize GPU PM sampling metrics from protobuf output."""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add generated/proto/ to path for _pb2 modules
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, "generated", "proto"))

import gpu_metrics_pb2

# === Configurable constants ===
YLIM_HEADROOM = 1.05       # y-axis max = max_resource * this factor
MAX_LABEL_X_OFFSET = 0.02 # label x position in axes fraction (0=left edge, 1=right edge)
SMOOTH_WINDOW_US = 100    # smoothing window in microseconds (None or 0 to disable)
REGION_COLORS = [          # bold, high-contrast colors for region annotations (cycled)
    "#d62728",  # red
    "#2ca02c",  # green
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]


def smooth(data, interval_us, window_us):
    """Apply uniform moving-average smoothing to a 1-D or 2-D array.

    Args:
        data: 1-D array (n_samples,) or 2-D array (n_samples, n_metrics).
        interval_us: sampling interval in microseconds.
        window_us: smoothing window in microseconds. 0 = no smoothing.

    Returns:
        Smoothed array of the same shape.
    """
    if not window_us:
        return data
    k = max(1, int(round(window_us / interval_us)))
    if k <= 1:
        return data
    kernel = np.ones(k) / k
    if data.ndim == 1:
        return np.convolve(data, kernel, mode="same")
    # 2-D: smooth each column independently
    out = np.empty_like(data)
    for col in range(data.shape[1]):
        out[:, col] = np.convolve(data[:, col], kernel, mode="same")
    return out

def load_trace(path):
    """Load a protobuf trace file.

    Supports both length-delimited streaming format (multiple varint-prefixed
    messages concatenated) and legacy single-message format. When multiple
    messages are present, samples and regions are merged and metadata is taken
    from the first message.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Try length-delimited (streaming) format first
    traces = []
    offset = 0
    while offset < len(data):
        # Decode varint (up to 5 bytes for uint32)
        shift = 0
        msg_len = 0
        varint_bytes = 0
        while offset + varint_bytes < len(data):
            b = data[offset + varint_bytes]
            msg_len |= (b & 0x7F) << shift
            varint_bytes += 1
            shift += 7
            if (b & 0x80) == 0:
                break
        else:
            break  # incomplete varint at end of file

        msg_start = offset + varint_bytes
        msg_end = msg_start + msg_len

        if msg_end > len(data) or msg_len == 0:
            break  # incomplete or zero-length message

        trace = gpu_metrics_pb2.GpuMetricsTrace()
        trace.ParseFromString(data[msg_start:msg_end])
        traces.append(trace)
        offset = msg_end

    # If we successfully parsed at least one delimited message covering all data, merge them
    if traces and offset == len(data):
        if len(traces) == 1:
            return traces[0]
        merged = gpu_metrics_pb2.GpuMetricsTrace()
        merged.CopyFrom(traces[0])
        merged.ClearField("samples")
        merged.ClearField("regions")
        for t in traces:
            merged.samples.extend(t.samples)
            merged.regions.extend(t.regions)
        return merged

    # Fallback: legacy single-message format (no length prefix)
    trace = gpu_metrics_pb2.GpuMetricsTrace()
    trace.ParseFromString(data)
    return trace

def main():
    parser = argparse.ArgumentParser(description="Visualize GPU PM sampling metrics from protobuf output.")
    parser.add_argument("-i", "--input", default="gpu_metrics.pb", help="Input protobuf file (default: gpu_metrics.pb)")
    parser.add_argument("-o", "--output", default="gpu_metrics.png", help="Output figure file (default: gpu_metrics.png)")
    args = parser.parse_args()

    pb_path = args.input
    out_path = args.output

    trace = load_trace(pb_path)
    n_samples = len(trace.samples)
    n_metrics = len(trace.metric_names)

    print(f"Device: {trace.device_name} ({trace.chip_name})")
    print(f"Sampling frequency: {trace.sampling_frequency_hz} Hz")
    print(f"Samples: {n_samples}, Metrics: {n_metrics}")
    for i, name in enumerate(trace.metric_names):
        print(f"  [{i}] {name}")

    # Extract data, skipping sample 0 (initialization artifact with bogus duration)
    samples_list = list(trace.samples)
    if n_samples > 1:
        samples_list = samples_list[1:]
        n_samples -= 1
        print(f"  (skipped first sample — initialization artifact)")

    timestamps_ns = np.array([s.start_timestamp_ns for s in samples_list], dtype=np.float64)
    t0 = timestamps_ns[0]
    time_ms = (timestamps_ns - t0) / 1e6  # convert to ms from start

    values = np.zeros((n_samples, n_metrics))
    for i, sample in enumerate(samples_list):
        for j, v in enumerate(sample.values):
            values[i, j] = v

    # Smooth raw data before plotting
    if SMOOTH_WINDOW_US is not None and SMOOTH_WINDOW_US > 0:
        freq_hz = trace.sampling_frequency_hz
        interval_us = 1e6 / freq_hz if freq_hz > 0 else 100
        k = max(1, int(round(SMOOTH_WINDOW_US / interval_us)))
        print(f"Smoothing: {SMOOTH_WINDOW_US} us window = {k} samples @ {freq_hz} Hz")
        values = smooth(values, interval_us, SMOOTH_WINDOW_US)

    metric_names = list(trace.metric_names)

    # Build index map
    idx = {name: i for i, name in enumerate(metric_names)}

    # Check which metric groups are available
    has_sm_util = ("sm__cycles_active.avg" in idx and "sm__cycles_active.max" in idx
                   and "sm__cycles_elapsed.avg" in idx)
    has_warp = "sm__warps_active.avg" in idx and "sm__warps_active.max" in idx
    has_dram = ("dram__read_throughput.avg.pct_of_peak_sustained_elapsed" in idx and
                "dram__write_throughput.avg.pct_of_peak_sustained_elapsed" in idx)

    n_panels = sum([has_sm_util, has_warp, has_dram])
    if n_panels == 0:
        print("No recognized metrics to plot")
        return

    # Build layout: thin timeline strip at top + metric panels below
    regions = list(trace.regions)
    has_timeline = len(regions) > 0
    height_ratios = ([1] if has_timeline else []) + [7] * n_panels
    n_rows = len(height_ratios)
    fig, all_axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_panels + (0.7 if has_timeline else 0)),
                                  sharex=True, gridspec_kw={"height_ratios": height_ratios})
    if n_rows == 1:
        all_axes = [all_axes]

    # Separate timeline axis from metric axes
    if has_timeline:
        timeline_ax = all_axes[0]
        axes = list(all_axes[1:])
    else:
        timeline_ax = None
        axes = list(all_axes)

    panel = 0

    # Panel 1: SM Utilization (avg & max across SMs)
    if has_sm_util:
        ax = axes[panel]
        elapsed = values[:, idx["sm__cycles_elapsed.avg"]]
        sm_avg = values[:, idx["sm__cycles_active.avg"]]
        sm_max = values[:, idx["sm__cycles_active.max"]]
        util_avg = np.where(elapsed > 0, sm_avg / elapsed * 100, 0)
        util_max = np.where(elapsed > 0, sm_max / elapsed * 100, 0)

        ax.plot(time_ms, util_max, linewidth=0.8, color="C0", alpha=0.5, label="max (busiest SM)")
        ax.plot(time_ms, util_avg, linewidth=1.0, color="C0", label="avg (across all SMs)")
        ax.axhline(100, color="black", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, 100, "Max SM Util: 100%",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylabel("SM Utilization (%)")
        ax.set_ylim(0, 100 * YLIM_HEADROOM)
        ax.legend(loc="upper center", ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Panel 2: Active Warps per cycle (normalized by cycles_elapsed)
    if has_warp:
        ax = axes[panel]
        elapsed = values[:, idx["sm__cycles_elapsed.avg"]]
        warp_avg = values[:, idx["sm__warps_active.avg"]]
        warp_max = values[:, idx["sm__warps_active.max"]]
        # Normalize: warp-cycles → warps per cycle
        occ_avg = np.where(elapsed > 0, warp_avg / elapsed, 0)
        occ_max = np.where(elapsed > 0, warp_max / elapsed, 0)

        ax.plot(time_ms, occ_max, linewidth=0.8, color="C1", alpha=0.5, label="max (busiest SM)")
        ax.plot(time_ms, occ_avg, linewidth=1.0, color="C1", label="avg (across all SMs)")
        # Max warps per SM = 2048 threads / 32 threads per warp = 64
        max_warps_per_sm = 64
        ax.axhline(max_warps_per_sm, color="black", linewidth=1.2, linestyle=":",
                   alpha=0.7)
        ax.text(MAX_LABEL_X_OFFSET, max_warps_per_sm, f"Max Active Warps: {max_warps_per_sm}",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
        ax.set_ylabel("Active Warps / Cycle")
        ax.set_ylim(0, max_warps_per_sm * YLIM_HEADROOM)
        ax.legend(loc="upper center", ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Panel 3: DRAM Read/Write Throughput in GB/s (converted from pct_of_peak)
    if has_dram:
        ax = axes[panel]
        peak_bw = trace.peak_dram_bw_gbps  # theoretical peak in GB/s
        pct_to_gbps = peak_bw / 100.0 if peak_bw > 0 else 1.0

        rd_avg = values[:, idx["dram__read_throughput.avg.pct_of_peak_sustained_elapsed"]] * pct_to_gbps
        wr_avg = values[:, idx["dram__write_throughput.avg.pct_of_peak_sustained_elapsed"]] * pct_to_gbps

        has_dram_max = ("dram__read_throughput.max.pct_of_peak_sustained_elapsed" in idx and
                        "dram__write_throughput.max.pct_of_peak_sustained_elapsed" in idx)
        if has_dram_max:
            rd_max = values[:, idx["dram__read_throughput.max.pct_of_peak_sustained_elapsed"]] * pct_to_gbps
            wr_max = values[:, idx["dram__write_throughput.max.pct_of_peak_sustained_elapsed"]] * pct_to_gbps
            ax.plot(time_ms, rd_max, linewidth=0.8, color="C2", alpha=0.4, label="Read max")
            ax.plot(time_ms, wr_max, linewidth=0.8, color="C3", alpha=0.4, label="Write max")

        ax.plot(time_ms, rd_avg, linewidth=1.0, color="C2", label="Read avg")
        ax.plot(time_ms, wr_avg, linewidth=1.0, color="C3", label="Write avg")

        # Draw peak DRAM bandwidth line
        if peak_bw > 0:
            ax.axhline(peak_bw, color="black", linewidth=1.2, linestyle=":",
                       alpha=0.7)
            ax.text(MAX_LABEL_X_OFFSET, peak_bw, f"Max DRAM BW: {peak_bw:.0f} GB/s",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=7, color="black", fontweight="bold")
            ax.set_ylim(0, peak_bw * YLIM_HEADROOM)

        ax.set_ylabel("DRAM Throughput (GB/s)")
        ax.legend(loc="upper center", ncol=10, fontsize=7, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Draw region annotations
    if regions:
        for ri, region in enumerate(regions):
            r_start_ms = (region.start_timestamp_ns - t0) / 1e6
            r_end_ms = (region.end_timestamp_ns - t0) / 1e6
            dur_ms = r_end_ms - r_start_ms
            color = REGION_COLORS[ri % len(REGION_COLORS)]

            # Shaded spans + dashed lines on metric panels
            for ax in axes:
                ax.axvspan(r_start_ms, r_end_ms, alpha=0.08, color=color)
                ax.axvline(r_start_ms, color=color, linewidth=0.6, linestyle="--", alpha=0.6)
                ax.axvline(r_end_ms, color=color, linewidth=0.6, linestyle="--", alpha=0.6)

            # Timeline strip: colored bar
            if timeline_ax is not None:
                timeline_ax.barh(0, dur_ms, left=r_start_ms, height=0.8,
                                 color=color, alpha=0.25, edgecolor=color, linewidth=0.8)

            # 45-degree label above all graphs, anchored at region start
            # Use the top of the first metric axis as anchor point
            top_ax = timeline_ax if timeline_ax is not None else axes[0]
            # Transform: data x → display → figure coords
            label_text = f"{region.name} ({dur_ms:.1f} ms)"
            top_ax.annotate(label_text,
                            xy=(r_start_ms, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, 4), textcoords="offset points",
                            fontsize=7, color=color, fontweight="bold",
                            rotation=45, rotation_mode="anchor",
                            ha="left", va="bottom",
                            annotation_clip=False)

        print(f"Regions: {len(regions)}")
        for r in regions:
            dur_ms = (r.end_timestamp_ns - r.start_timestamp_ns) / 1e6
            print(f"  {r.name}: {dur_ms:.1f} ms")

    # Configure timeline strip
    if timeline_ax is not None:
        timeline_ax.set_ylim(-0.5, 0.5)
        timeline_ax.set_yticks([])
        timeline_ax.spines["top"].set_visible(False)
        timeline_ax.spines["right"].set_visible(False)
        timeline_ax.spines["left"].set_visible(False)
        timeline_ax.tick_params(bottom=False, labelbottom=False)
        timeline_ax.set_ylabel("Region", fontsize=8)

    # Force x-axis to start at 0 and add more tick marks
    all_axes[-1].set_xlim(left=0)
    all_axes[-1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    all_axes[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes[-1].set_xlabel("Time (ms from start)")

    timing_method = trace.region_timing_method or "N/A"
    fig.suptitle(
        f"CUPTI PM Sampling @ {trace.sampling_frequency_hz} Hz \u2014 "
        f"{trace.device_name} ({trace.chip_name})\n"
        f"{n_samples} samples over {time_ms[-1]:.1f} ms"
        f"  |  Region timing: {timing_method}",
        fontsize=12
    )
    fig.tight_layout()
    # Extra top margin for 45-degree region labels + suptitle
    if has_timeline:
        fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
