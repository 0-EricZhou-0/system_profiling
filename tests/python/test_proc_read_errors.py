"""A per-PID /proc file that cannot be read is told apart by why, and
warned about at a bounded rate.

  * A live process whose /proc/<pid>/io this process may not read (not
    dumpable here): a warning naming the cause; the process-table row
    records it (io_unreadable_since_ns / _ticks); no I/O sample — missing,
    not zero.
  * A process that is exiting: its /proc/<pid>/io fails with EACCES from
    the kernel's teardown of its memory until it is reaped (measured:
    ~0.5 s for 8 GB mapped, while its pidfd still says alive). That is its
    exit: no warning, nothing recorded, and it is removed once, with its
    end time, by the pidfd path as before.
  * Any other error (injected with the test hook): warned with the
    errno's text. /proc/<pid>/statm (System probe) likewise: its memory
    values are NaN in those samples, its CPU still sampled.

Warnings are rate-limited per (tracked process, warning type): at most
one a second; the ones suppressed are counted into the next ("(N
suppressed)") and into a last line when the process stops being tracked
or the probe stops, so every failed read is accounted for. The state is
per tracked process and goes with it.

Both modes: LEGACY probes run here, SIDECAR probes in the sidecar (hooks
armed there from the environment it inherits). The warnings go to
stderr, the sidecar's included (it inherits ours).
"""

import errno
import math
import re
import subprocess
import sys
import time

import pytest

import metric_catalog_pb2
from cupti_profiler import _native
from tracing_helpers import disk_frames, running_suite, system_frames, tracked, tree

HZ = 100
MODES = ["legacy", "sidecar"]
NON_DUMPABLE = ("import ctypes, time; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); "   # PR_SET_DUMPABLE
                "print('ready', flush=True); time.sleep(60)")
SLEEPER = "import time; print('ready', flush=True); time.sleep(60)"
# 4 GiB touched, then exit: the kernel's teardown of that memory takes a
# few hundred ms, during which /proc/<pid>/io gives EACCES.
BIG_EXIT = ("import os, time\n"
            "b = bytearray(4 << 30)\n"
            "for i in range(0, len(b), 4096): b[i] = 1\n"
            "print('ready', flush=True)\n"
            "time.sleep(0.5)\n"
            "os._exit(0)\n")
SUPPRESSED = re.compile(r"\((\d+) suppressed")


def _spawn(code):
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    assert p.stdout.readline().strip() == b"ready"
    return p


def _kill(*ps):
    for p in ps:
        p.kill()
        p.wait()


def _lines(err, pid, file):
    return [l for l in err.splitlines() if f"cannot read /proc/{pid}/{file} " in l]


def _accounted(lines):
    """Failed reads a pid's warning lines account for: one per line that
    went out when due, plus every count of suppressed ones."""
    due = [l for l in lines if "suppressed; " not in l]   # not a final summary line
    return len(due) + sum(int(m.group(1)) for l in lines for m in [SUPPRESSED.search(l)] if m)


def _col(frames, fqn):
    cols = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    return cols[metric_catalog_pb2.SCOPE_PROCESS].index(fqn)


def _suite(tmp_path, mode, procs, **kw):
    return running_suite(tmp_path, mode, processes=[(p.pid, f"p{i}") for i, p in enumerate(procs)],
                         hz=HZ, disk=True, disk_hz=HZ, flush_ms=200, **kw)


@pytest.mark.parametrize("mode", MODES)
def test_non_dumpable_warned_at_most_once_a_second_and_recorded(tmp_path, capfd, mode):
    p = _spawn(NON_DUMPABLE)
    try:
        with _suite(tmp_path, mode, [p]):
            time.sleep(3.5)
    finally:
        _kill(p)
    lines = _lines(capfd.readouterr().err, p.pid, "io")
    due = [l for l in lines if "suppressed; " not in l]
    assert 3 <= len(due) <= 5, due                         # ~1 a second over 3.5 s
    assert "not dumpable" in due[0] and "missing from the trace" in due[0], due[0]
    assert not SUPPRESSED.search(due[0]) and all(SUPPRESSED.search(l) for l in due[1:]), due
    d = disk_frames(tmp_path)
    row = tracked(d)[p.pid]
    assert row.io_unreadable_since_ns > 0 and row.io_unreadable_ticks >= 250
    assert _accounted(lines) == row.io_unreadable_ticks, (lines, row.io_unreadable_ticks)
    assert not [s for f in d for s in f.process_samples if s.pid == p.pid]
    # statm stays readable (mode 0444): memory sampled, nothing recorded
    s = system_frames(tmp_path)
    assert tracked(s)[p.pid].mem_unreadable_ticks == 0
    rss = [x.values[_col(s, "proc__rss_bytes")] for f in s for x in f.process_samples
           if x.pid == p.pid]
    assert rss and all(v > 0 for v in rss)


@pytest.mark.parametrize("mode", MODES)
def test_two_processes_are_limited_independently(tmp_path, capfd, mode):
    a, b = _spawn(NON_DUMPABLE), _spawn(NON_DUMPABLE)
    try:
        with _suite(tmp_path, mode, [a, b]):
            time.sleep(3.5)
    finally:
        _kill(a, b)
    err = capfd.readouterr().err
    for p in (a, b):
        due = [l for l in _lines(err, p.pid, "io") if "suppressed; " not in l]
        assert 3 <= len(due) <= 5, (p.pid, due)


def _inject(monkeypatch, mode, pid, specs):
    if mode == "sidecar":
        monkeypatch.setenv("CUPTI_PROFILER_TEST_READ_ERROR",
                           ",".join(f"{probe}:{pid}:{e}" for probe, e in specs))
    else:
        for probe, e in specs:
            _native._testing_set_read_error(pid, probe, e)


def _clear():
    for probe in ("system", "disk"):
        _native._testing_set_read_error(0, probe, 0)


@pytest.mark.parametrize("mode", MODES)
def test_two_warning_types_on_one_process_are_limited_independently(tmp_path, capfd,
                                                                    monkeypatch, mode):
    p = _spawn(SLEEPER)
    try:
        _inject(monkeypatch, mode, p.pid, [("disk", errno.EIO), ("system", errno.EACCES)])
        with _suite(tmp_path, mode, [p]):
            time.sleep(3.5)
    finally:
        _clear()
        _kill(p)
    err = capfd.readouterr().err
    io = [l for l in _lines(err, p.pid, "io") if "suppressed; " not in l]
    statm = [l for l in _lines(err, p.pid, "statm") if "suppressed; " not in l]
    assert 3 <= len(io) <= 5 and 3 <= len(statm) <= 5, (io, statm)
    assert "Input/output error" in io[0] and "not dumpable" not in io[0], io[0]
    assert statm[0].startswith("[System]"), statm[0]
    s = system_frames(tmp_path)
    row = tracked(s)[p.pid]
    assert row.mem_unreadable_since_ns > 0 and row.mem_unreadable_ticks >= 250
    samples = [x for f in s for x in f.process_samples if x.pid == p.pid]
    assert len(samples) >= 250
    assert all(math.isnan(x.values[_col(s, "proc__rss_bytes")]) for x in samples)
    assert all(math.isfinite(x.values[_col(s, "proc__cycles_active.sum.per_second")])
               for x in samples)
    drow = tracked(disk_frames(tmp_path))[p.pid]
    assert drow.io_unreadable_ticks >= 250 and drow.mem_unreadable_ticks == 0


@pytest.mark.parametrize("mode", MODES)
def test_exiting_process_is_no_permission_failure(tmp_path, capfd, mode):
    p = _spawn(BIG_EXIT)
    zombie_ns = None
    with _suite(tmp_path, mode, [p]):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                state = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()[0]
            except OSError:
                break
            if state == "Z":
                zombie_ns = time.monotonic_ns()       # the trace clock
                break
            time.sleep(0.002)
        time.sleep(0.3)                  # a zombie, as the probes see it
        p.wait()
        time.sleep(0.3)
    assert zombie_ns, "premise: the child became a zombie while traced"
    err = capfd.readouterr().err
    assert not _lines(err, p.pid, "io"), err
    for frames in (disk_frames(tmp_path), system_frames(tmp_path)):
        rows = [tp for f in frames for tp in f.tracked_processes if tp.pid == p.pid]
        assert [tp.removed for tp in rows].count(True) == 1, "removed exactly once"
        last = rows[-1]
        assert last.removed and last.io_unreadable_ticks == 0 and last.mem_unreadable_ticks == 0
        # seen gone by the pidfd within a tick or two of becoming a zombie
        assert zombie_ns - 50_000_000 <= last.end_time_ns <= zombie_ns + 100_000_000, \
            (last.end_time_ns - zombie_ns) / 1e6


MANY_NON_DUMPABLE = """
import ctypes
kids = []
for i in range(12):
    kids.append(subprocess.Popen([PY, "-c",
        "import ctypes, time; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); time.sleep(0.4)"]))
    time.sleep(0.1)
for k in kids:
    k.wait()
emit(done=len(kids))
sys.stdin.readline()
"""


@pytest.mark.parametrize("mode", MODES)
def test_warning_state_goes_with_the_processes(tmp_path, capfd, monkeypatch, mode):
    """Twelve short-lived non-dumpable descendants, each warned about:
    once they are gone, no warning state is left for them."""
    if mode == "sidecar":
        monkeypatch.setenv("CUPTI_PROFILER_TEST_REPORT_WARN_STATE", "1")
    with tree(MANY_NON_DUMPABLE) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], hz=HZ, disk=True,
                           disk_hz=HZ, flush_ms=100,
                           discovery={"enabled": True, "scan_interval_ms": 20}):
            peak = 0
            deadline = time.monotonic() + 15
            done = None
            while done is None and time.monotonic() < deadline:
                if mode == "legacy":
                    peak = max(peak, _native._testing_warn_state_size())
                try:
                    done = t.read("done", timeout=0.05)
                except TimeoutError:
                    pass
            time.sleep(0.6)                   # their removals flushed
            left = _native._testing_warn_state_size() if mode == "legacy" else None
    err = capfd.readouterr().err
    warned = {int(m.group(1)) for m in re.finditer(r"cannot read /proc/(\d+)/io ", err)}
    assert len(warned) >= 8, warned                  # most children were read and warned about
    if mode == "legacy":
        assert peak >= 1 and left == 0, (peak, left)
    else:
        assert "[testing] disk warn state at stop: 0" in err, err[-2000:]


@pytest.mark.parametrize("mode", MODES)
def test_a_new_registration_of_the_same_pid_is_a_new_key(tmp_path, capfd, mode):
    """Removed and added again: the same PID number is a new tracked
    process, warned about at once, not counted into the old one's rate."""
    p = _spawn(NON_DUMPABLE)
    try:
        with _suite(tmp_path, mode, [p]) as suite:
            time.sleep(0.3)
            suite.remove_tracked_process(p.pid)
            time.sleep(0.3)                    # the removal flushed
            suite.add_tracked_process(p.pid, "again")
            time.sleep(0.4)
    finally:
        _kill(p)
    due = [l for l in _lines(capfd.readouterr().err, p.pid, "io") if "suppressed; " not in l]
    assert len(due) == 2 and not SUPPRESSED.search(due[1]), due
