"""A process's memory read while it exits is not its memory. Once the
kernel has torn an exiting process's memory down, /proc/<pid>/statm
reads RSS 0 while its pidfd still says alive; the System probe checks
the exit evidence (the same as for unreadable /proc files: stat gone,
state Z/X, PF_EXITING) and writes that sample's memory as NaN (missing),
its CPU kept. A process that frees its memory while alive keeps its real
drop. Both modes."""

import math
import subprocess
import sys
import time

import pytest

import metric_catalog_pb2
from tracing_helpers import running_suite, system_frames

HZ = 100
MODES = ["legacy", "sidecar"]
MIB = 1024 * 1024
BIG_EXIT = ("import os, time\n"
            "b = bytearray(4 << 30)\n"
            "for i in range(0, len(b), 4096): b[i] = 1\n"
            "print('ready', flush=True)\n"
            "time.sleep(0.4)\n"
            "os._exit(0)\n")
FREE_THEN_EXIT = ("import os, time\n"
                  "b = bytearray(1 << 30)\n"
                  "for i in range(0, len(b), 4096): b[i] = 1\n"
                  "print('ready', flush=True)\n"
                  "time.sleep(0.5)\n"
                  "del b\n"                       # a live process frees its memory
                  "time.sleep(1.0)\n"
                  "os._exit(0)\n")


def _rss(tmp_path, code, mode):
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    assert p.stdout.readline().strip() == b"ready"
    with running_suite(tmp_path, mode, processes=[(p.pid, "x")], hz=HZ, flush_ms=200):
        p.wait()
        time.sleep(0.3)
    frames = system_frames(tmp_path)
    cols = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    c_rss = cols[metric_catalog_pb2.SCOPE_PROCESS].index("proc__rss_bytes")
    c_cpu = cols[metric_catalog_pb2.SCOPE_PROCESS].index("proc__cycles_active.sum.per_second")
    samples = sorted((x for f in frames for x in f.process_samples if x.pid == p.pid),
                     key=lambda x: x.timestamp_ns)
    return [x.values[c_rss] for x in samples], [x.values[c_cpu] for x in samples]


@pytest.mark.parametrize("mode", MODES)
def test_exit_window_zero_is_missing_not_zero(tmp_path, mode):
    rss, cpu = _rss(tmp_path, BIG_EXIT, mode)
    assert rss and max(v for v in rss if not math.isnan(v)) > 3000 * MIB
    assert not [v for v in rss if v == 0.0], "an RSS of 0 read during the exit reached the trace"
    assert any(math.isnan(v) for v in rss), "premise: a sample was taken during the teardown"
    assert all(math.isfinite(v) for v in cpu)                  # CPU kept


@pytest.mark.parametrize("mode", MODES)
def test_memory_freed_while_alive_keeps_its_drop(tmp_path, mode):
    rss, _cpu = _rss(tmp_path, FREE_THEN_EXIT, mode)
    finite = [v for v in rss if not math.isnan(v)]
    top = max(finite)
    i = finite.index(top)
    after = finite[i:]
    assert top > 900 * MIB
    low = [v for v in after if v < 200 * MIB]
    assert len(low) >= 50, "the drop to a small RSS, a second long, is in the trace"
    assert all(v > 0 for v in low) and finite[-1] > 0
