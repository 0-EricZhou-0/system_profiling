"""Auto-reaped children.

A parent whose SIGCHLD is SIG_IGN has its children reaped by the kernel
as they exit (do_notify_parent -> exit_notify -> release_task), without
wait_task_zombie, which is the only place a child's I/O and CPU are
added to its parent. So nothing is folded: subtracting the child's I/O
from the parent's (as for a parent that waits, test_reap_io.py) would
remove I/O that was never added. The probes read the parent's SigIgn
when they find the reap, and then do not subtract; the IoReapAdjustment
and the CpuTail are marked autoreaped.

The root ignores SIGCHLD and forks a child (no exec: zeroed counters, no
I/O before it is found), which does known I/O and CPU, idles so that a
reading follows, reports its own counters and exits. Right after the
child is gone, the root does I/O of its own, spread over many sample
intervals: a subtraction at the reap would take it out of the root.
"""

import time

import pytest

from test_reap_io import COUNTERS, KiB, MiB, series, total
from test_cpu_head_tail import cpu_accounting
from tracing_helpers import disk_frames, running_suite, system_frames, tracked, tree

MODES = ["legacy", "sidecar"]
HZ = 50
ON = {"enabled": True, "scan_interval_ms": 50}

ROOT = """
import signal
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}
def churn(path, n):
    chunk = os.urandom(1 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    for _ in range(n):
        os.write(fd, chunk)
        time.sleep(0.004)
    os.fsync(fd)
    os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    while os.read(fd, 1 << 20):
        time.sleep(0.004)
    os.close(fd)
    os.unlink(path)
i0 = io()
emit(ready=1)
d, spin = sys.stdin.readline().split()
pid = os.fork()
if pid == 0:
    time.sleep(0.5)
    end = time.process_time() + float(spin)
    while time.process_time() < end:
        pass
    churn(os.path.join(d, "child.dat"), 64)
    time.sleep(0.5)
    emit(child=os.getpid(), io=io(), cpu_ns=time.process_time_ns())
    os._exit(0)
while os.path.exists(f"/proc/{pid}"):   # auto-reaped: never a zombie to wait for
    time.sleep(0.001)
churn(os.path.join(d, "root.dat"), 32)
time.sleep(0.5)
t = os.times()
emit(done=pid, i0=i0, i1=io(), children_cpu=t.children_user + t.children_system)
sys.stdin.readline()
"""


def run(tmp_path, mode, spin):
    with tree(ROOT) as t:
        t.read("ready")
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], disk=True,
                           hz=HZ, disk_hz=HZ, discovery=ON):
            time.sleep(0.2)
            t.proc.stdin.write(f"{tmp_path} {spin}\n".encode())
            t.proc.stdin.flush()
            child = t.read("child", 60)
            done = t.read("done", 60)
            time.sleep(0.2)
    return t.pid, child, done


@pytest.mark.parametrize("mode", MODES)
def test_autoreaped_child_io_not_subtracted(tmp_path, mode):
    root, child, done = run(tmp_path, mode, 0)
    frames = disk_frames(tmp_path)
    kid, kid_io = child["child"], child["io"]
    raw = {k: done["i1"][k] - done["i0"][k] for k in COUNTERS}
    # Premise: the child did its I/O, and the kernel folded NONE of it into
    # the root (the root's own is 32 MiB, the child's 64).
    assert kid_io["wchar"] >= 64 * MiB and kid_io["rchar"] >= 64 * MiB, kid_io
    assert 32 * MiB <= raw["wchar"] < 40 * MiB, (raw, "the child's I/O was folded: not auto-reaped")
    rs, ks = series(frames, root), series(frames, kid)
    assert len(rs["rchar"]) > 20 and len(ks["rchar"]) > 20, "too few samples"

    adjs = [a for f in frames for a in f.io_reap_adjustments]
    assert len(adjs) == 1, f"expected one IoReapAdjustment: {adjs}"
    [a] = adjs
    assert a.parent_pid == root and [c.pid for c in a.children] == [kid], a
    assert a.autoreaped and a.children[0].autoreaped, a
    assert not a.ambiguous and not a.children[0].ambiguous, a
    failures = []
    for k in COUNTERS:
        # Nothing subtracted: the root's sample at the record is its whole
        # raw delta, the remainder.
        rem = getattr(a.remainder, k)
        at = [b for ts, b in rs[k] if ts == a.timestamp_ns]
        if len(at) != 1 or abs(at[0] - rem) > 1 + 1e-9 * rem:
            failures.append(("sample at the record", k, at, rem))
        # Root + child totals equal the kernel's: the root's raw delta (its
        # own I/O only) plus the child's own report.
        truth = raw[k] + kid_io[k]
        traced = total(rs[k]) + total(ks[k])
        print(f"{mode} {k:22s} kernel {truth / MiB:8.3f} MiB  traced {traced / MiB:8.3f} MiB"
              f"  (root {total(rs[k]) / MiB:.3f} of {raw[k] / MiB:.3f}, child {total(ks[k]) / MiB:.3f})")
        if abs(traced - truth) > 256 * KiB:
            failures.append(("tree total", k, truth, round(traced)))
    assert not failures, failures


@pytest.mark.parametrize("mode", MODES)
def test_autoreaped_child_cpu_tail_marked(tmp_path, mode):
    root, child, done = run(tmp_path, mode, 1.0)
    frames = system_frames(tmp_path)
    kid, truth = child["child"], child["cpu_ns"] / 1e9
    # Premise: the root's cutime + cstime never grew (auto-reaped).
    assert done["children_cpu"] == 0, done
    head, samples, tails, _ = cpu_accounting(frames, kid)
    assert len(tails) == 1 and list(tails[0].pids) == [kid], tails
    [c] = tails
    assert c.parent_pid == root and c.autoreaped, c
    assert c.cpu_after_last_sample_ns == 0, c   # no counter holds it: not a measurement
    # The child idled after its CPU, so its samples hold all of it: the
    # tree's CPU for it equals its own clock.
    print(f"{mode}: head {head:.3f} + samples {samples:.3f} = {head + samples:.3f} s vs truth {truth:.3f} s")
    assert tracked(frames)[kid].pid == kid
    assert abs(head + samples - truth) <= 0.04 + 0.02 * truth
