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

# Child: configure + start the suite from argv[1] (JSON config), then run
# argv[2] (Python body: sleep, signal itself, ...). `suite` is in scope.
CHILD = """
import json, os, signal, sys, time
import cupti_profiler as cp
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
