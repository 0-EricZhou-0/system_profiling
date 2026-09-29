"""SIDECAR: a crash signal (SIGSEGV here) to the sidecar makes it flush
its probes (best effort, bounded) before it dies of the signal; its
stop signals (TERM, INT, HUP, ...) already did. No GPU needed.
"""

import json
import subprocess
import sys
import time

import disk_metrics_pb2
import system_metrics_pb2
from tracing_helpers import read_frames

CHILD = r"""
import json, os, sys, time
import cupti_profiler as cp
cfg = json.loads(sys.argv[1])
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
print("started", flush=True)
exec(sys.argv[2])
"""


def config(tmp_path, mode):
    return {"output_dir": str(tmp_path), "gpu": {"enabled": False}, "events": {"enabled": False},
            "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                       "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
            "disk": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                     "output_file": "disk_metrics.pb", "processes": [{"pid": 0}], "mode": mode}}


class Child:
    def __init__(self, cfg, body):
        self.p = subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(cfg), body],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.lines = []

    def wait_for(self, text, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            line = self.p.stdout.readline()
            if not line:
                break
            self.lines.append(line.strip())
            if text in line:
                return line.strip()
        raise AssertionError(f"no {text!r}: {self.lines}")

    def finish(self, timeout=60):
        out, err = self.p.communicate(timeout=timeout)
        self.lines += out.splitlines()
        return self.p.returncode, err


def last_system_ts(tmp_path):
    fr = read_frames(tmp_path / "system_metrics.pb", system_metrics_pb2.SystemMetricsTrace)
    ts = [s.timestamp_ns for f in fr for s in f.system_samples]
    return max(ts) if ts else None


def test_sidecar_flushes_on_a_crash_signal(tmp_path):
    body = ("def sidecar():\n"
            "    for t in os.listdir('/proc/self/task'):\n"
            "        for c in open(f'/proc/self/task/{t}/children').read().split():\n"
            "            if 'sidecar' in os.readlink(f'/proc/{c}/exe'):\n"
            "                return int(c)\n"
            "time.sleep(3.5)\n"
            "print(json.dumps({'t_sig': time.monotonic_ns(), 'sc': sidecar()}), flush=True)\n"
            "os.kill(sidecar(), 11)\n"
            "time.sleep(2.0)\n"
            "suite.stop()\n")
    c = Child(config(tmp_path, mode=2), body)
    c.wait_for("started")
    r = json.loads(c.wait_for("t_sig"))
    rc, err = c.finish()
    assert rc == 0, err[-2000:]
    assert "killed by signal 11" in err, err[-2000:]           # the sidecar died of it
    last = last_system_ts(tmp_path)
    assert last is not None and last > r["t_sig"] - 200_000_000, "the sidecar did not flush"
    fr = read_frames(tmp_path / "disk_metrics.pb", disk_metrics_pb2.DiskMetricsTrace)
    assert any(f.process_samples for f in fr), "disk trace not flushed"
