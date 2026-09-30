"""SIDECAR mode: the sidecar process's lifecycle and its control pipes.

The sidecar is found as the child of this process whose comm starts with
"cupti-profiler" (the binary is cupti-profiler-sidecar; comm keeps 15
characters).
"""

import contextlib
import json
import os
import signal
import subprocess
import threading
import time

import pytest

import cupti_profiler as cp
from tracing_helpers import (PY, running_suite, sample_times, suite_config,
                             system_frames, tracked, tree)


def sidecar_pids(parent=None):
    parent = parent or os.getpid()
    out = []
    for tid in os.listdir(f"/proc/{parent}/task"):
        try:
            kids = open(f"/proc/{parent}/task/{tid}/children").read().split()
        except OSError:
            continue
        for k in kids:
            try:
                comm = open(f"/proc/{k}/comm").read().strip()
            except OSError:
                continue
            if comm.startswith("cupti-profiler"):
                out.append(int(k))
    return out


def pipe_fds(pid):
    """fd -> 'pipe:[inode]' for every pipe the process holds."""
    out = {}
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("pipe:"):
            out[int(fd)] = link
    return out


def test_sidecar_pipes_not_inherited(tmp_path):
    # A program this process execs after Configure() (close_fds=False, as
    # with os.posix_spawn or a C++ host's fork+exec) must not inherit the
    # host's ends of the sidecar's pipes: an inherited write end of the
    # control pipe keeps it open after the host exits, so the sidecar
    # never sees EOF.
    with running_suite(tmp_path, "sidecar"):
        [sc] = sidecar_pids()
        control = {v for fd, v in pipe_fds(sc).items() if fd in (3, 4, 5)}
        child = subprocess.Popen(["sleep", "30"], close_fds=False)
        try:
            inherited = control & set(pipe_fds(child.pid).values())
        finally:
            child.kill()
            child.wait()
    assert len(control) == 3, f"sidecar fds 3/4/5 should be its three pipes: {pipe_fds(sc)}"
    assert not inherited, f"an exec'd child of the host holds the sidecar's pipes: {inherited}"


# ---------------------------------------------------------------------------
# Lifecycle: every way the sidecar can be told to stop ends with the final
# flush on disk and exit status 0.

def proc_state(pid):
    """(state, wait status) of a process from /proc/<pid>/stat, or None if
    it is gone. The wait status (field 52) is meaningful for a zombie."""
    try:
        text = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    rest = text[text.rindex(")") + 2:].split()
    return rest[0], int(rest[49])


def wait_until(pred, timeout, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def host_samples(tmp_path, pid):
    return sample_times(system_frames(tmp_path), pid)


def test_sidecar_handshake(tmp_path, capfd):
    me = os.getpid()
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, suite_config(tmp_path, "sidecar", processes=[(0, "self")]))
    try:
        [sc] = sidecar_pids()
        argv = open(f"/proc/{sc}/cmdline").read().split("\0")
        assert f"--host-pid={me}" in argv, argv
        suite.start()
        time.sleep(0.3)
    finally:
        suite.stop()
    assert proc_state(sc) is None, "the sidecar must have exited and been reaped by stop()"
    assert len(host_samples(tmp_path, me)) >= 10
    err = capfd.readouterr().err
    assert "MSG_STOP received" in err and "clean shutdown, exit 0" in err, err


def test_sidecar_add_remove(tmp_path):
    with tree("sys.stdin.readline()") as t, running_suite(tmp_path, "sidecar") as suite:
        time.sleep(0.2)
        suite.add_tracked_process(t.pid, "mid")
        time.sleep(0.5)
        t_remove = time.monotonic_ns()
        suite.remove_tracked_process(t.pid)
        time.sleep(0.6)   # > 2 flushes: marker once, then dropped
    frames = system_frames(tmp_path)
    ts = sample_times(frames, t.pid)
    assert len(ts) >= 20, len(ts)
    assert ts[-1] <= t_remove + 30_000_000, "samples after the removal"
    flags = [tp.removed for f in frames for tp in f.tracked_processes if tp.pid == t.pid]
    assert flags.count(True) == 1 and flags[-1] is True, flags
    assert tracked(frames).get(t.pid).alias == "mid"


def test_sidecar_sigterm_flushes(tmp_path):
    # flush_interval 3 s and SIGTERM at ~1 s: every sample on disk came
    # from the sidecar's final flush.
    me = os.getpid()
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, suite_config(tmp_path, "sidecar", processes=[(0, "self")],
                                           flush_ms=3000))
    suite.start()
    try:
        [sc] = sidecar_pids()
        time.sleep(1.0)
        t_kill = time.monotonic_ns()
        os.kill(sc, signal.SIGTERM)
        st = wait_until(lambda: (proc_state(sc) or ("gone", 0))[0] == "Z" and proc_state(sc), 10)
        assert st and st[0] == "Z", f"sidecar did not exit after SIGTERM: {proc_state(sc)}"
        assert os.WIFEXITED(st[1]) and os.WEXITSTATUS(st[1]) == 0, \
            f"sidecar died instead of stopping: wait status {st[1]:#x}"
        ts = host_samples(tmp_path, me)
    finally:
        suite.stop()
    assert len(ts) >= 50, f"only {len(ts)} samples on disk: the final flush is missing"
    assert ts[-1] >= t_kill - 50_000_000, \
        f"last sample {(t_kill - ts[-1]) / 1e6:.0f} ms before SIGTERM"


PARENT = """
import json, os, sys, time
import cupti_profiler as cp
cfg, holder = json.loads(sys.argv[1]), sys.argv[2] == "1"
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
me = os.getpid()
sc = [int(k) for t in os.listdir(f"/proc/{me}/task")
      for k in open(f"/proc/{me}/task/{t}/children").read().split()
      if open(f"/proc/{k}/comm").read().startswith("cupti-profiler")]
out = {"sidecar": sc[0]}
if holder:
    # A forked child keeps every fd, O_CLOEXEC or not: the control pipe
    # stays open after this process dies, so the sidecar never sees EOF.
    pid = os.fork()
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    out["holder"] = pid
print(json.dumps(out), flush=True)
time.sleep(60)
"""


@pytest.mark.parametrize("holder", [False, True], ids=["eof", "pipe_held_open"])
def test_sidecar_parent_dies(tmp_path, holder):
    cfg = suite_config(tmp_path, "sidecar", processes=[(0, "host")], flush_ms=3000)
    log = open(tmp_path / "host.err", "w+")
    host = subprocess.Popen([PY, "-c", PARENT, json.dumps(cfg), "1" if holder else "0"],
                            stdout=subprocess.PIPE, stderr=log, text=True)
    info = {}
    try:
        line = ""
        while not line.startswith("{"):   # the library logs to stdout too
            line = host.stdout.readline()
            assert line, "host exited before reporting the sidecar"
        info = json.loads(line)
        sc = info["sidecar"]
        time.sleep(1.0)
        t_kill = time.monotonic_ns()
        host.kill()
        host.wait()
        gone = wait_until(lambda: (proc_state(sc) or ("gone",))[0] in ("gone", "Z"), 15)
        assert gone, f"sidecar {sc} still running {15} s after its host was killed"
    finally:
        host.kill()
        host.wait()
        if "holder" in info:
            with contextlib.suppress(ProcessLookupError):
                os.kill(info["holder"], signal.SIGKILL)
    ts = host_samples(tmp_path, host.pid)
    assert len(ts) >= 50, f"only {len(ts)} samples on disk: the final flush is missing"
    assert ts[-1] >= t_kill - 50_000_000, \
        f"last sample {(t_kill - ts[-1]) / 1e6:.0f} ms before the host died"
    log.seek(0)
    err = log.read()
    assert "clean shutdown, exit 0" in err, err[-2000:]
    if holder:
        assert "host process exited" in err, err[-2000:]


def test_configure_from_short_thread(tmp_path):
    # The sidecar is forked by whichever thread calls configure(). That
    # thread exiting must not stop it (PR_SET_PDEATHSIG would).
    me = os.getpid()
    suite = cp.ProfilerSuite()
    th = threading.Thread(target=lambda: (
        cp.configure_suite(suite, suite_config(tmp_path, "sidecar", processes=[(0, "self")])),
        suite.start()))
    th.start()
    th.join()
    try:
        [sc] = sidecar_pids()
        time.sleep(1.0)
        st = proc_state(sc)
        assert st and st[0] != "Z", "the sidecar stopped when its creating thread exited"
        t_stop = time.monotonic_ns()
    finally:
        suite.stop()
    ts = host_samples(tmp_path, me)
    assert ts and ts[-1] >= t_stop - 50_000_000, "sampling ended before stop()"


def test_sidecar_timestamps_share_host_clock(tmp_path):
    # No clock handshake exists: the sidecar's samples must already be on
    # this process's steady clock (CLOCK_MONOTONIC). Marker: at T, touch a
    # fresh 64 MiB buffer; the first sidecar sample showing this process's
    # RSS up by 32 MiB must land after T and within the fill time plus
    # two system ticks.
    import metric_catalog_pb2
    me = os.getpid()
    tick = 10_000_000
    marks = []
    with running_suite(tmp_path, "sidecar", processes=[(0, "self")]):
        time.sleep(0.5)
        for _ in range(3):
            t0 = time.monotonic_ns()
            buf = b"\x01" * (64 << 20)
            marks.append((t0, time.monotonic_ns()))
            time.sleep(0.3)
            del buf
            time.sleep(0.3)
    frames = system_frames(tmp_path)
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    col = fqns[metric_catalog_pb2.SCOPE_PROCESS].index("proc__rss_bytes")
    rss = sorted((s.timestamp_ns, s.values[col]) for f in frames for s in f.process_samples
                 if s.pid == me)
    for t0, t1 in marks:
        base = max(v for ts, v in rss if t0 - 200_000_000 <= ts < t0)
        edge = next(ts for ts, v in rss if ts >= t0 - 100_000_000 and v > base + (32 << 20))
        assert t0 <= edge <= t1 + 2 * tick, \
            f"RSS marker seen at {(edge - t0) / 1e6:+.1f} ms (fill took {(t1 - t0) / 1e6:.1f} ms)"


@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_start_failure_raises(tmp_path, mode):
    # A system probe whose output file cannot be opened used to run with
    # no data and one stderr line (in the sidecar's stderr, under SIDECAR).
    # A directory where the file should be: a plain name that cannot be
    # opened for writing.
    cfg = suite_config(tmp_path, mode, processes=[(0, "self")])
    (tmp_path / "system_metrics.pb").mkdir()
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, cfg)
    with pytest.raises(RuntimeError, match="ProbeStartFailed"):
        suite.start()
    suite.stop()   # still fine after a failed start
    assert sidecar_pids() == [], "the failed sidecar must have been reaped"


def test_sidecar_cpus_pins_every_thread(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[-1]
    cfg = suite_config(tmp_path, "sidecar", processes=[(0, "self")], disk=True,
                       discovery={"enabled": True, "scan_interval_ms": 50})
    cfg["sidecar_cpus"] = [cpu]
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, cfg)
    suite.start()
    try:
        time.sleep(0.3)
        [sc] = sidecar_pids()
        tids = [int(t) for t in os.listdir(f"/proc/{sc}/task")]
        masks = {t: os.sched_getaffinity(t) for t in tids}
    finally:
        suite.stop()
    # main + system sample/flush + disk sample/flush + discovery
    assert len(tids) >= 6, tids
    assert all(m == {cpu} for m in masks.values()), masks
    assert sorted(os.sched_getaffinity(0)) == allowed, "the host's affinity must not change"


def test_sidecar_cpus_unusable_fails_configure(tmp_path):
    allowed = os.sched_getaffinity(0)
    outside = [c for c in range(os.cpu_count()) if c not in allowed]
    for cpus in ([outside[0]] if outside else []) + [4096]:
        cfg = suite_config(tmp_path, "sidecar")
        cfg["sidecar_cpus"] = [cpus] if isinstance(cpus, int) else cpus
        suite = cp.ProfilerSuite()
        with pytest.raises(RuntimeError, match="SidecarAffinityFailed"):
            cp.configure_suite(suite, cfg)
        assert sidecar_pids() == [], "the refused sidecar must have been reaped"
