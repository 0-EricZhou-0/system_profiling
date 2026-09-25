"""Shared helpers for the descendant-tracking tests: run a suite in
LEGACY or SIDECAR mode, spawn small Python process trees, and read the
system/disk traces back.
"""

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import time

import cupti_profiler as cp
import disk_metrics_pb2
import system_metrics_pb2

MODES = {"legacy": 1, "sidecar": 2}   # SystemProbeMode
PY = sys.executable

# Preamble for every process a test spawns: emit() prints one JSON line
# on stdout (inherited down the tree, so descendants can report too).
PRELUDE = """
import json, os, subprocess, sys, threading, time
PY = sys.executable
def emit(**kw):
    print(json.dumps(kw), flush=True)
def sleeper(seconds):
    return subprocess.Popen([PY, "-c", "import time; time.sleep(%r)" % seconds])
"""


def read_frames(path, cls):
    if not os.path.exists(path):
        return []
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        n = shift = 0
        while True:
            b = data[i]
            i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        msg = cls()
        msg.ParseFromString(data[i:i + n])
        i += n
        out.append(msg)
    return out


def system_frames(tmp_path):
    return read_frames(tmp_path / "system_metrics.pb", system_metrics_pb2.SystemMetricsTrace)


def disk_frames(tmp_path):
    return read_frames(tmp_path / "disk_metrics.pb", disk_metrics_pb2.DiskMetricsTrace)


def tracked(frames):
    """pid -> the last TrackedProcessV2 seen for it across all frames."""
    out = {}
    for f in frames:
        for tp in f.tracked_processes:
            out[tp.pid] = tp
    return out


def sample_times(frames, pid):
    return sorted(s.timestamp_ns for f in frames for s in f.process_samples if s.pid == pid)


def suite_config(tmp_path, mode, discovery=None, processes=(), disk=False,
                 hz=100, flush_ms=200):
    procs = [{"pid": p, "alias": a} for p, a in processes]
    cfg = {
        "output_dir": str(tmp_path),
        "gpu": {"enabled": False},
        "events": {"enabled": False},
        "system": {
            "enabled": True,
            "sampling_frequency_hz": hz,
            "flush_interval_ms": flush_ms,
            "output_file": "system_metrics.pb",
            "processes": procs,
            "mode": MODES[mode],
        },
        "disk": {"enabled": False},
    }
    if disk:
        cfg["disk"] = {
            "enabled": True,
            "sampling_frequency_hz": 20,
            "flush_interval_ms": flush_ms,
            "output_file": "disk_metrics.pb",
            "processes": procs,
            "mode": MODES[mode],
        }
    if discovery is not None:
        cfg["process_discovery"] = discovery
    return cfg


@contextlib.contextmanager
def running_suite(tmp_path, mode, **kw):
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, suite_config(tmp_path, mode, **kw))
    suite.start()
    try:
        yield suite
    finally:
        suite.stop()


class Tree:
    """A Python root process running BODY after PRELUDE. Its stdout
    (shared by every descendant that inherits it) carries JSON lines."""

    def __init__(self, body):
        self.proc = subprocess.Popen([PY, "-c", PRELUDE + body],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.pid = self.proc.pid
        self._buf = b""
        self.extra_pids = []   # descendants to kill at cleanup

    def read(self, key, timeout=10.0):
        """Next JSON line containing `key` (earlier lines are dropped)."""
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if key in msg:
                    return msg
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {key!r} line from the process tree")
            r, _, _ = select.select([fd], [], [], left)
            if r:
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise EOFError(f"process tree closed stdout before {key!r}")
                self._buf += chunk

    def close(self):
        for pid in self.extra_pids + [self.pid]:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=5)
        for pid in self.extra_pids:
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(pid, os.WNOHANG)


@contextlib.contextmanager
def tree(body):
    t = Tree(body)
    try:
        yield t
    finally:
        t.close()
