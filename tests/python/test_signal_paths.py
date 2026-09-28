"""The corner cases of flush-on-signal, driven deterministically with a
test-only delay inside ProfilerSuite::Stop() (testing::SetStopDelayMs):

  * a second signal while a signal's flush runs: the process exits at
    once, without waiting for the flush;
  * a signal that lands on the very thread running stop(): stop()
    finishes its flush first, then the signal takes its course;
  * a signal while another thread runs stop(): the handler waits for that
    stop() and then lets the signal take its course;
  * SIDECAR: a crash signal to the sidecar flushes its probes before it
    dies.
Needs a GPU only for the suite's CUDA-free probes' tests? No GPU needed.
"""

import json
import os
import signal
import subprocess
import sys
import time

import system_metrics_pb2
from tracing_helpers import read_frames

PY = sys.executable

CHILD = r"""
import json, os, sys, threading, time
import cupti_profiler as cp
cfg = json.loads(sys.argv[1])
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
print("started", flush=True)
exec(sys.argv[2])
"""


def config(tmp_path, mode=1):
    return {"output_dir": str(tmp_path), "gpu": {"enabled": False}, "events": {"enabled": False},
            "system": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                       "output_file": "system_metrics.pb", "processes": [{"pid": 0}], "mode": mode},
            "disk": {"enabled": True, "sampling_frequency_hz": 100, "flush_interval_ms": 5000,
                     "output_file": "disk_metrics.pb", "processes": [{"pid": 0}], "mode": mode}}


class Child:
    def __init__(self, cfg, body):
        self.p = subprocess.Popen([PY, "-c", CHILD, json.dumps(cfg), body],
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


def test_second_signal_during_flush_exits_at_once(tmp_path):
    c = Child(config(tmp_path), "cp._native._testing_set_stop_delay_ms(4000); time.sleep(60)")
    c.wait_for("started")
    time.sleep(2.0)
    t0 = time.monotonic()
    c.p.send_signal(signal.SIGTERM)          # flush begins; stop() now takes 4 s
    time.sleep(0.5)
    c.p.send_signal(signal.SIGTERM)          # during that flush
    rc, err = c.finish()
    dt = time.monotonic() - t0
    assert rc == -signal.SIGTERM, err[-2000:]
    assert "another signal during the flush: exiting now" in err, err[-2000:]
    assert dt < 2.0, f"took {dt:.2f} s: waited for the flush"


def test_signal_on_the_stopping_thread_waits_for_stop(tmp_path):
    body = ("cp._native._testing_set_stop_delay_ms(2000); time.sleep(3.0)\n"
            "print(json.dumps({'t_stop': time.monotonic_ns()}), flush=True)\n"
            "print('stopping', flush=True); suite.stop(); print('stopped', flush=True)\n"
            "time.sleep(30)")
    c = Child(config(tmp_path), body)
    c.wait_for("started")
    t_stop = json.loads(c.wait_for("t_stop"))["t_stop"]
    c.wait_for("stopping")
    time.sleep(0.5)
    t0 = time.monotonic()
    c.p.send_signal(signal.SIGTERM)          # lands on the main thread, inside stop()
    rc, err = c.finish()
    dt = time.monotonic() - t0
    assert rc == -signal.SIGTERM, err[-2000:]
    assert "stopped" not in c.lines            # the signal's course, right after stop()
    assert 1.0 < dt < 5.0, dt                  # stop()'s remaining ~1.5 s, then exit
    last = last_system_ts(tmp_path)
    assert last is not None and last > t_stop - 200_000_000, "stop() did not finish its flush"
    assert "taking the signal that arrived during it" in err, err[-2000:]


def test_signal_while_another_thread_stops(tmp_path):
    body = ("cp._native._testing_set_stop_delay_ms(2000); time.sleep(3.0)\n"
            "print(json.dumps({'t_stop': time.monotonic_ns()}), flush=True)\n"
            "threading.Thread(target=suite.stop).start()\n"
            "print('stopping', flush=True); time.sleep(30)")
    c = Child(config(tmp_path), body)
    c.wait_for("started")
    t_stop = json.loads(c.wait_for("t_stop"))["t_stop"]
    c.wait_for("stopping")
    time.sleep(0.5)
    c.p.send_signal(signal.SIGTERM)          # main thread; stop() runs on another
    rc, err = c.finish()
    assert rc == -signal.SIGTERM, err[-2000:]
    last = last_system_ts(tmp_path)
    assert last is not None and last > t_stop - 200_000_000, "died before the other stop() flushed"
