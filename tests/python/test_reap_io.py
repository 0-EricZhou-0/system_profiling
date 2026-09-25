"""Reaped children's I/O.

When a process reaps a child, the kernel adds the child's lifetime I/O to
the parent's own /proc/<pid>/io counters. With both tracked, the child's
I/O would be counted twice: in its own samples, and again as a jump in the
parent's at the reap. The disk probe subtracts the child's last reading
from the parent's delta at the reap and records an IoReapAdjustment, so a
process's samples carry its own I/O only.

The root forks a child (fork, no exec: the child starts with zeroed
counters and does no I/O before it is found), which writes, fsyncs, reads
back and deletes 64 MiB, idles so that a reading follows, reports its own
/proc/self/io and exits. The root reaps it either at once (blocked in
waitpid: the reap lands inside the tick that sees the exit) or after
leaving it a zombie for a while (the reap lands between ticks). The root
reads its own /proc/self/io before and after: the kernel's view of the
whole tree.
"""

import time

import pytest

import metric_catalog_pb2
from tracing_helpers import disk_frames, running_suite, tree

MODES = ["legacy", "sidecar"]
MiB = 1 << 20
KiB = 1 << 10
HZ = 50
ON = {"enabled": True, "scan_interval_ms": 50}

COUNTERS = {
    "rchar":                 "proc__io_rchar.sum.per_second",
    "wchar":                 "proc__io_wchar.sum.per_second",
    "read_bytes":            "proc__io_read_bytes.sum.per_second",
    "write_bytes":           "proc__io_write_bytes.sum.per_second",
    "cancelled_write_bytes": "proc__io_cancelled_write_bytes.sum.per_second",
}

ROOT = """
def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}
def zombie(pid):
    return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
i0 = io()
emit(ready=1)
d, linger = sys.stdin.readline().split()
pid = os.fork()
if pid == 0:
    time.sleep(0.5)
    path = os.path.join(d, "child.dat")
    chunk = os.urandom(1 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    for _ in range(64):
        os.write(fd, chunk)
    os.fsync(fd)
    os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    while os.read(fd, 1 << 20):
        pass
    os.close(fd)
    os.unlink(path)
    time.sleep(0.5)
    emit(child=os.getpid(), io=io())
    os._exit(0)
if float(linger) > 0:
    while not zombie(pid):
        time.sleep(0.005)
    time.sleep(float(linger))
os.waitpid(pid, 0)
time.sleep(0.5)
emit(reaped=pid, i0=i0, i1=io())
sys.stdin.readline()
"""


def run(tmp_path, mode, discovery, linger):
    with tree(ROOT) as t:
        t.read("ready")
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], disk=True,
                           disk_hz=HZ, discovery=discovery):
            time.sleep(0.2)
            t.proc.stdin.write(f"{tmp_path} {linger}\n".encode())
            t.proc.stdin.flush()
            child = t.read("child", 30)
            done = t.read("reaped", 30)
            time.sleep(0.2)
    return t.pid, child, done, disk_frames(tmp_path)


def series(frames, pid):
    """counter -> [(timestamp_ns, bytes in the interval ending there)]. The
    first sample's interval (from the seeding reading, not emitted) is taken
    as the nominal period; the processes here are idle then."""
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    cols = fqns[metric_catalog_pb2.SCOPE_PROCESS]
    rows = sorted((s.timestamp_ns, list(s.values)) for f in frames for s in f.process_samples
                  if s.pid == pid)
    out = {}
    for counter, fqn in COUNTERS.items():
        j = cols.index(fqn)
        prev, xs = None, []
        for ts, v in rows:
            dt = (ts - prev) / 1e9 if prev is not None else 1.0 / HZ
            xs.append((ts, v[j] * dt))
            prev = ts
        out[counter] = xs
    return out


def total(xs):
    return sum(b for _, b in xs)


@pytest.mark.parametrize("linger", [0, 0.3], ids=["reaped-at-once", "zombie-first"])
@pytest.mark.parametrize("mode", MODES)
def test_reaped_traced_child_io_subtracted(tmp_path, mode, linger):
    root, child, done, frames = run(tmp_path, mode, ON, linger)
    kid = child["child"]
    kid_io = child["io"]
    raw = {k: done["i1"][k] - done["i0"][k] for k in COUNTERS}
    # Premise: the child did the I/O, and the kernel folded it into the root.
    assert kid_io["wchar"] >= 64 * MiB and kid_io["rchar"] >= 64 * MiB, kid_io
    assert raw["wchar"] >= kid_io["wchar"], (raw, kid_io)
    rs, ks = series(frames, root), series(frames, kid)
    assert len(rs["rchar"]) > 20 and len(ks["rchar"]) > 20, "too few samples"

    adjs = [a for f in frames for a in f.io_reap_adjustments]
    assert len(adjs) == 1, f"expected one IoReapAdjustment: {adjs}"
    [a] = adjs
    assert a.parent_pid == root and [c.pid for c in a.children] == [kid], a
    last = a.children[0].last_seen
    failures = []
    for k in COUNTERS:
        seen = getattr(last, k)
        # What was subtracted is the child's I/O, up to its report (a
        # /proc/self/io read and one printed line, either side of its last
        # reading).
        if abs(kid_io[k] - seen) > 64 * KiB:
            failures.append(("last_seen", k, kid_io[k], seen))
        rem = getattr(a.remainder, k)
        if not 0 <= rem <= 64 * KiB:
            failures.append(("remainder", k, rem))
        # The parent's sample at the adjustment is the remainder, so the
        # raw /proc delta is reconstructable: sample + last_seen.
        at = [b for ts, b in rs[k] if ts == a.timestamp_ns]
        if len(at) != 1 or abs(at[0] - rem) > 1 + 1e-9 * rem:
            failures.append(("sample at the adjustment", k, at, rem))
        # No jump: the root does no I/O of its own beyond a few lines.
        biggest = max(b for _, b in rs[k])
        if biggest > 256 * KiB:
            failures.append(("jump in the parent", k, round(biggest / MiB, 3)))
        # Parent + child totals equal the kernel's.
        traced = total(rs[k]) + total(ks[k])
        print(f"{mode} {k:22s} kernel {raw[k] / MiB:8.3f} MiB  traced {traced / MiB:8.3f} MiB"
              f"  (root {total(rs[k]) / MiB:.3f}, child {total(ks[k]) / MiB:.3f})")
        if abs(traced - raw[k]) > 256 * KiB:
            failures.append(("tree total", k, raw[k], round(traced)))
    assert not failures, failures


@pytest.mark.parametrize("mode", MODES)
def test_untraced_child_io_stays_in_parent(tmp_path, mode):
    # Discovery off: the child is never tracked, so its I/O is counted
    # once, in the parent that reaped it. Nothing is subtracted.
    root, child, done, frames = run(tmp_path, mode, None, 0)
    kid_io = child["io"]
    assert not [a for f in frames for a in f.io_reap_adjustments]
    assert not series(frames, child["child"])["rchar"], "premise: the child is not tracked"
    rs = series(frames, root)
    raw = {k: done["i1"][k] - done["i0"][k] for k in COUNTERS}
    failures = []
    for k in COUNTERS:
        traced = total(rs[k])
        print(f"{mode} {k:22s} kernel {raw[k] / MiB:8.3f} MiB  root {traced / MiB:8.3f} MiB"
              f"  child {kid_io[k] / MiB:8.3f} MiB")
        if abs(traced - raw[k]) > 256 * KiB or traced + 64 * KiB < kid_io[k]:
            failures.append((k, raw[k], kid_io[k], round(traced)))
    assert not failures, failures
