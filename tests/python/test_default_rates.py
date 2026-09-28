"""Default sampling rates: GPU 100 Hz, System and Disk 100 Hz when the
rate is unset (0), in the library (both modes); the vLLM example samples
the GPU at 1 kHz and System/Disk at 100 Hz. Needs a GPU with PM sampling.
"""

import importlib.util
import os

import pytest

import disk_metrics_pb2
import system_metrics_pb2
from gpu_helpers import METRICS, gpu_frames, run_child
from tracing_helpers import read_frames

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.parametrize("mode", [1, 2], ids=["legacy", "sidecar"])
def test_library_defaults(tmp_path, mode):
    cfg = {"output_dir": str(tmp_path), "events": {"enabled": False},
           "gpu": {"enabled": True, "metrics": METRICS, "output_file": "gpu_metrics.pb"},
           "system": {"enabled": True, "output_file": "system_metrics.pb",
                      "processes": [{"pid": 0}], "mode": mode},
           "disk": {"enabled": True, "output_file": "disk_metrics.pb",
                    "processes": [{"pid": 0}], "mode": mode}}
    rc, out, err = run_child(cfg, "time.sleep(2.5); suite.stop()")
    assert rc == 0, err
    g = gpu_frames(tmp_path)
    s = read_frames(tmp_path / "system_metrics.pb", system_metrics_pb2.SystemMetricsTrace)
    d = read_frames(tmp_path / "disk_metrics.pb", disk_metrics_pb2.DiskMetricsTrace)
    assert g[0].header.sampling_frequency_hz == 100
    assert s[0].header.sampling_frequency_hz == 100
    assert d[0].header.sampling_frequency_hz == 100
    # GPU samples are really 10 ms apart.
    ts = [x.timestamp_ns for f in g for x in f.samples]
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    assert abs(gaps[len(gaps) // 2] - 10_000_000) < 100_000, gaps[len(gaps) // 2]
    # System ticks: ~100 per second.
    n = sum(len(f.system_samples) for f in s)
    assert 180 <= n <= 260, n


def test_vllm_example_defaults():
    spec = importlib.util.spec_from_file_location(
        "ex", os.path.join(REPO, "examples", "vllm_serving_profiling.py"))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    a = ex.parser().parse_args([])
    assert (a.gpu_hz, a.system_hz, a.disk_hz, a.flush_ms) == (1000, 100, 100, 5000)
    cfg = ex.suite_config(ex.parser().parse_args(["--gpu"]))
    assert cfg["gpu"]["sampling_frequency_hz"] == 1000
    assert "max_samples" not in cfg["gpu"]      # sized by the library
