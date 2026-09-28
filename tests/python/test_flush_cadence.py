"""Flush cadence: every probe flushes every flush_interval_ms, with 0 (or
unset) meaning 5 s — no probe buffers the whole run until stop() — and
each flush reaches the kernel as one write, not a series of 8 KiB ones.
Needs a GPU with PM sampling.
"""

import json

import pytest

from gpu_helpers import METRICS, run_child

FILES = ["gpu_metrics.pb", "system_metrics.pb", "disk_metrics.pb", "events.pb"]


def _config(tmp_path, mode):
    return {
        "output_dir": str(tmp_path),
        "gpu": {"enabled": True, "sampling_frequency_hz": 1000, "metrics": METRICS,
                "output_file": "gpu_metrics.pb"},
        "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 0,
                   "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "disk": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 0,
                 "output_file": "disk_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
        "events": {"enabled": True, "flush_interval_ms": 0, "output_file": "events.pb"},
    }


SIZES = """
suite.get_event_profiler().get_generic_tracker().mark_event("start")
def sizes():
    d = cfg["output_dir"]
    return {f: os.path.getsize(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else 0
            for f in %r}
time.sleep(3.0); early = sizes()
time.sleep(3.5); late = sizes()
print(json.dumps({"early": early, "late": late}), flush=True)
suite.stop()
""" % FILES


@pytest.mark.parametrize("mode", [1, 2], ids=["legacy", "sidecar"])
def test_every_probe_flushes_every_5s_by_default(tmp_path, mode):
    rc, out, err = run_child(_config(tmp_path, mode), SIZES)
    assert rc == 0, err
    r = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    assert all(v == 0 for v in r["early"].values()), f"flushed before 5 s: {r['early']}"
    assert all(v > 0 for v in r["late"].values()), f"not flushed by 6.5 s: {r['late']}"


# Per-thread write syscalls of the flush threads, after two flushes.
SYSCW = """
time.sleep(11.0)
w = {}
for tid in os.listdir("/proc/self/task"):
    comm = open(f"/proc/self/task/{tid}/comm").read().strip()
    if comm.endswith("-flush"):
        io = dict(l.split(": ") for l in open(f"/proc/self/task/{tid}/io").read().splitlines())
        w[comm] = int(io["syscw"])
print(json.dumps(w), flush=True)
suite.stop()
"""


def test_one_write_per_flush(tmp_path):
    rc, out, err = run_child(_config(tmp_path, 1), SYSCW)
    assert rc == 0, err
    w = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    assert set(w) >= {"cupti-gpu-flush", "cupti-sys-flush", "cupti-dsk-flush"}, w
    # Two flushes each (at 5 s and 10 s); one write per flush, plus slack
    # for a stdout line that fills the stdio buffer.
    for name in ("cupti-gpu-flush", "cupti-sys-flush", "cupti-dsk-flush"):
        assert 1 <= w[name] <= 3, w
