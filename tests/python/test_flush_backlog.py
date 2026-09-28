"""A flush thread that cannot keep up (each flush takes longer than the
flush interval) is reported on stderr: the first time at once, then at
most one summary line per report period, then a final summary at stop.
The count is in the trace (FlushStats.slow_flushes) and no sample is
dropped. The slow disk is simulated with a test-only hook.
"""

import json

from gpu_helpers import run_child
import system_metrics_pb2
from tracing_helpers import read_frames

SLOW = """
cp._native._testing_set_flush_delay_ms(300)
time.sleep(7.0)
suite.stop()
"""


def test_slow_flush_warned_rate_limited(tmp_path):
    cfg = {"output_dir": str(tmp_path), "gpu": {"enabled": False}, "disk": {"enabled": False},
           "events": {"enabled": False},
           "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 200,
                      "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": 1}}
    pre = "cp._native._testing_set_backlog_report_period_ms(2000)"
    rc, out, err = run_child(cfg, SLOW, env={"CHILD_PRE": pre})
    assert rc == 0, err
    w = [l for l in err.splitlines() if l.startswith("[cupti-profiler] warning: System probe")]
    print("\n".join(w))
    assert "a flush took" in w[0] and "longer than the flush interval (200 ms)" in w[0], w
    summaries = [l for l in w[1:] if "slower than the 200 ms interval in the last" in l]
    # ~14 slow flushes in 7 s; at most one summary per 2 s.
    assert 1 <= len(summaries) <= 4, w
    assert "flush summary:" in w[-1], w
    assert len(w) == 1 + len(summaries) + 1, w

    frames = read_frames(tmp_path / "system_metrics.pb", system_metrics_pb2.SystemMetricsTrace)
    slow = max(fs.slow_flushes for f in frames for fs in f.flush_stats)
    assert slow >= 8, slow
    ticks = sum(len(f.system_samples) for f in frames)
    assert ticks > 7.0 * 100 * 0.9, ticks        # nothing dropped
