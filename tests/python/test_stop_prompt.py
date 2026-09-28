"""stop() does not wait out any probe's interval: every sample, flush and
decode thread waits on an interruptible primitive, in-process (LEGACY)
and in the sidecar (SIDECAR). Needs a GPU with PM sampling.
"""

import json

import pytest

from gpu_helpers import METRICS, run_child

TIMED_STOP = """
time.sleep(2.5)
t = time.monotonic(); suite.stop(); stop_s = time.monotonic() - t
print(json.dumps({"stop_s": stop_s}), flush=True)
"""


@pytest.mark.parametrize("mode", [1, 2], ids=["legacy", "sidecar"])
def test_stop_is_prompt(tmp_path, mode):
    cfg = {
        "output_dir": str(tmp_path),
        "gpu": {"enabled": True, "sampling_frequency_hz": 1000, "metrics": METRICS,
                "decode_interval_ms": 1000, "flush_interval_ms": 5000,
                "output_file": "gpu_metrics.pb"},
        "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                   "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "disk": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                 "output_file": "disk_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "events": {"enabled": True, "flush_interval_ms": 5000, "output_file": "events.pb"},
    }
    rc, out, err = run_child(cfg, TIMED_STOP)
    assert rc == 0, err
    r = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    assert r["stop_s"] < 0.2, f"stop() took {r['stop_s']:.3f} s"
