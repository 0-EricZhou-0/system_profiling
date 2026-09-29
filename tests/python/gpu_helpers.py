"""Helpers for the GPU probe tests: run a ProfilerSuite with the GPU probe
in a child Python process (a fresh CUPTI session each time) and read the
GPU trace back.
"""

import json
import os
import subprocess
import sys

import gpu_metrics_pb2
from tracing_helpers import read_frames

PY = sys.executable
METRICS = ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
           "dram__read_throughput.avg.pct_of_peak_sustained_elapsed"]

# Child: run $CHILD_PRE (if set), configure + start the suite from argv[1]
# (JSON config), then run argv[2] (Python body: sleep, signal itself,
# ...). `suite` is in scope.
CHILD = """
import json, os, signal, sys, time
import cupti_profiler as cp
exec(os.environ.get("CHILD_PRE", ""))
cfg = json.loads(sys.argv[1])
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
print("started", flush=True)
exec(sys.argv[2])
"""


def gpu_config(tmp_path, **gpu):
    g = {"enabled": True, "sampling_frequency_hz": 1000, "metrics": METRICS,
         "output_file": "gpu_metrics.pb"}
    g.update(gpu)
    return {"output_dir": str(tmp_path), "gpu": g, "system": {"enabled": False},
            "disk": {"enabled": False}, "events": {"enabled": False}}


def run_child(cfg, body, timeout=120, env=None):
    """Run CHILD; returns (returncode, stdout, stderr)."""
    p = subprocess.run([PY, "-c", CHILD, json.dumps(cfg), body], capture_output=True,
                       text=True, timeout=timeout, env={**os.environ, **(env or {})})
    return p.returncode, p.stdout, p.stderr


def gpu_frames(tmp_path):
    return read_frames(tmp_path / "gpu_metrics.pb", gpu_metrics_pb2.GPUMetricsTrace)


def sample_times(frames, gpu_index=0):
    return [s.timestamp_ns for f in frames for s in f.samples if s.gpu_index == gpu_index]


def decode_stats(frames, gpu_index=0):
    """The run's decode totals (the last frame's)."""
    last = None
    for f in frames:
        for d in f.decode_stats:
            if d.gpu_index == gpu_index:
                last = d
    return last


def warnings(stderr):
    return [l for l in stderr.splitlines() if l.startswith("[cupti-profiler] warning:")]


def full_config(tmp_path, mode, flush_ms=5000):
    """GPU (1 kHz, 1 s decode) + System + Disk (100 Hz, `mode`) + Events,
    every probe flushing every `flush_ms`."""
    return {
        "output_dir": str(tmp_path),
        "gpu": {"enabled": True, "sampling_frequency_hz": 1000, "metrics": METRICS,
                "decode_interval_ms": 1000, "flush_interval_ms": flush_ms,
                "output_file": "gpu_metrics.pb"},
        "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": flush_ms,
                   "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "disk": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": flush_ms,
                 "output_file": "disk_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "events": {"enabled": True, "flush_interval_ms": flush_ms, "output_file": "events.pb"},
    }


def last_times_steady(tmp_path):
    """Latest sample time of each trace, on the steady clock (ns); None
    when a trace has no samples."""
    import disk_metrics_pb2
    import system_metrics_pb2
    out = {}
    g = gpu_frames(tmp_path)
    ts = sample_times(g)
    if ts:
        a = g[-1].header.anchors
        out["gpu"] = ts[-1] - a.cupti_reference_ns + a.steady_clock_reference_ns
    else:
        out["gpu"] = None
    for name, cls in (("system", system_metrics_pb2.SystemMetricsTrace),
                      ("disk", disk_metrics_pb2.DiskMetricsTrace)):
        fr = read_frames(tmp_path / f"{name}_metrics.pb", cls)
        t = [s.timestamp_ns for f in fr for s in f.process_samples]
        out[name] = max(t) if t else None
    return out
