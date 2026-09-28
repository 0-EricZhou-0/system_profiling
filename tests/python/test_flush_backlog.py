"""A flush thread that cannot keep up (each flush takes longer than the
flush interval) is reported on stderr: the first time at once, then at
most one summary line per report period, then a final summary at stop.
The count is in the trace (FlushStats.slow_flushes) and no sample is
dropped. The slow disk is simulated with a test-only hook. System, Disk
and GPU probes (the GPU one needs a GPU with PM sampling).
"""

import disk_metrics_pb2
import gpu_metrics_pb2
import pytest
import system_metrics_pb2

from gpu_helpers import METRICS, run_child
from tracing_helpers import read_frames

OFF = {"enabled": False}
# probe -> (config section, name in the warning, flush ms, delay ms, run s, trace, count samples)
PROBES = {
    "system": ("system", "System probe", 200, 300, 7.0, "system_metrics.pb",
               system_metrics_pb2.SystemMetricsTrace, lambda f: len(f.system_samples), 100),
    "disk":   ("disk", "Disk probe", 200, 300, 7.0, "disk_metrics.pb",
               disk_metrics_pb2.DiskMetricsTrace, lambda f: len(f.process_samples), 100),
    "gpu":    ("gpu", "GPU probe", 1000, 1500, 9.0, "gpu_metrics.pb",
               gpu_metrics_pb2.GPUMetricsTrace, lambda f: len(f.samples), 1000),
}


def _config(tmp_path, probe, flush_ms):
    cfg = {"output_dir": str(tmp_path), "gpu": OFF, "system": OFF, "disk": OFF, "events": OFF}
    if probe == "gpu":
        cfg["gpu"] = {"enabled": True, "sampling_frequency_hz": 1000, "decode_interval_ms": 500,
                      "flush_interval_ms": flush_ms, "metrics": METRICS, "output_file": "gpu_metrics.pb"}
    else:
        cfg[probe] = {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": flush_ms,
                      "output_file": f"{probe}_metrics.pb", "processes": [{"pid": 0}], "mode": 1}
    return cfg


@pytest.mark.parametrize("probe", PROBES)
def test_slow_flush_warned_rate_limited(tmp_path, probe):
    _, name, flush_ms, delay_ms, run_s, fname, cls, count, hz = PROBES[probe]
    body = f"cp._native._testing_set_flush_delay_ms({delay_ms})\ntime.sleep({run_s})\nsuite.stop()\n"
    pre = "cp._native._testing_set_backlog_report_period_ms(2000)"
    rc, out, err = run_child(_config(tmp_path, probe, flush_ms), body, env={"CHILD_PRE": pre})
    assert rc == 0, err
    w = [l for l in err.splitlines() if l.startswith(f"[cupti-profiler] warning: {name}")]
    print("\n".join(w))
    assert "a flush took" in w[0] and f"longer than the flush interval ({flush_ms} ms)" in w[0], w
    summaries = [l for l in w[1:] if f"slower than the {flush_ms} ms interval in the last" in l]
    # At most one summary per 2 s.
    assert 1 <= len(summaries) <= run_s / 2 + 1, w
    assert "flush summary:" in w[-1], w
    assert len(w) == 1 + len(summaries) + 1, w

    frames = read_frames(tmp_path / fname, cls)
    slow = max(fs.slow_flushes for f in frames for fs in f.flush_stats)
    assert slow >= 3, slow
    n = sum(count(f) for f in frames)
    assert n > run_s * hz * 0.85, n        # nothing dropped
