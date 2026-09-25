"""Reap chains: a tracked process P reaps its tracked child C, then exits
and is reaped by the tracked G, all within one sampling interval.

The kernel's reap folds the reaped process's own I/O and everything it
had reaped into the reaper (kernel/exit.c, wait_task_zombie), so G's
sample holds P's and C's I/O: both must be subtracted there. Whether P
reaped C is not always knowable — had P exited first, C would have been
re-parented and its I/O would be elsewhere — so the disk probe decides
with the host's orphan reaper (adopt_orphans()): C reaped by it -> not in
G; reaper on and the tree under the host -> P reaped C; otherwise the
record is flagged ambiguous and C is not subtracted.

G forks P, P forks C. C writes, fsyncs, reads back and deletes 64 MiB, P
does 16 MiB of its own, and each lingers so that a reading follows. Then
either P reaps C and exits at once and G reaps P at once ("chain": the
reaps land within ~1 ms, far inside one 500 ms interval), or P exits
first, C is orphaned and exits once re-parented, and G leaves P a zombie
for a while before reaping it ("orphan"). G reads its own /proc/self/io
before and after: the kernel's view of the whole tree.

PR_SET_CHILD_SUBREAPER cannot be undone, so every run happens in a fresh
launcher process (as in test_adopt_orphans.py).
"""

import json
import subprocess

import pytest

import metric_catalog_pb2
from tracing_helpers import MODES as MODE_IDS
from tracing_helpers import PY, disk_frames, suite_config, tracked

MODES = list(MODE_IDS)
MiB = 1 << 20
KiB = 1 << 10
HZ = 2          # a 500 ms interval: a ~1 ms reap chain lands inside one
ON = {"enabled": True, "scan_interval_ms": 10}

COUNTERS = {
    "rchar":                 "proc__io_rchar.sum.per_second",
    "wchar":                 "proc__io_wchar.sum.per_second",
    "read_bytes":            "proc__io_read_bytes.sum.per_second",
    "write_bytes":           "proc__io_write_bytes.sum.per_second",
    "cancelled_write_bytes": "proc__io_cancelled_write_bytes.sum.per_second",
}

G_CODE = r"""
import json, os, sys, time
ORPHAN = sys.argv[1] == "orphan"
D = sys.argv[2]
def emit(**kw):
    print(json.dumps(kw), flush=True)
def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}
def zombie(pid):
    return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
def churn(name, mib):
    path = os.path.join(D, name)
    chunk = os.urandom(1 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    for _ in range(mib):
        os.write(fd, chunk)
    os.fsync(fd)
    os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    while os.read(fd, 1 << 20):
        pass
    os.close(fd)
    os.unlink(path)
i0 = io()
emit(ready=os.getpid())
sys.stdin.readline()
r, w = os.pipe()
p = os.fork()
if p == 0:
    time.sleep(0.3)                  # P: discovered and read before C exists
    c = os.fork()
    if c == 0:
        time.sleep(0.8)              # C: discovered and read
        churn("c.dat", 64)
        time.sleep(1.5)              # a reading after the I/O
        if ORPHAN:
            parent = os.getppid()
            os.write(w, b"x")        # P exits now, without reaping C
            while os.getppid() == parent:
                time.sleep(0.001)
        emit(child=os.getpid(), ppid=os.getppid(), io=io())
        os._exit(0)
    churn("p.dat", 16)
    if ORPHAN:
        os.read(r, 1)
    else:
        os.waitpid(c, 0)             # reap C, then exit at once
    os._exit(0)
if ORPHAN:                           # leave P a zombie while C is re-parented
    while not zombie(p):
        time.sleep(0.005)
    time.sleep(1.5)
os.waitpid(p, 0)
time.sleep(1.5)
emit(done=1, p=p, i0=i0, i1=io())
sys.stdin.readline()
"""

# An untracked subreaper that starts G and reaps whatever it adopts at
# once: C's orphan goes here, not to the launcher.
X_CODE = r"""
import ctypes, os, subprocess, sys
ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)   # PR_SET_CHILD_SUBREAPER
subprocess.Popen([sys.executable, "-c"] + sys.argv[1:])
while True:
    try:
        os.wait()
    except ChildProcessError:
        break
"""

LAUNCHER = r"""
import json, os, subprocess, sys, time
import cupti_profiler as cp

cfg, g_code, scenario, d, adopt, via_x, x_code = json.loads(sys.argv[1])
if adopt:
    cp.adopt_orphans()
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
argv = [sys.executable, "-c", g_code, scenario, d]
if via_x:
    argv = [sys.executable, "-c", x_code, g_code, scenario, d]
proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
lines = []
def read(key):
    while True:
        line = proc.stdout.readline()
        if not line:
            raise EOFError(key)
        msg = json.loads(line)
        lines.append(msg)
        if key in msg:
            return msg
g = read("ready")["ready"]
suite.add_tracked_process(g, "g", track_descendants=True)
time.sleep(0.6)
proc.stdin.write(b"go\n"); proc.stdin.flush()
child = read("child")
done = read("done")
time.sleep(0.6)
suite.stop()
proc.stdin.write(b"exit\n"); proc.stdin.flush()
proc.wait(timeout=10)
print("RESULT " + json.dumps({"me": os.getpid(), "g": g, "child": child, "done": done}), flush=True)
"""


def run(tmp_path, mode, scenario, adopt, via_x=False):
    cfg = suite_config(tmp_path, mode, discovery=ON, disk=True, disk_hz=HZ, flush_ms=200)
    args = [cfg, G_CODE, scenario, str(tmp_path), adopt, via_x, X_CODE]
    run = subprocess.run([PY, "-c", LAUNCHER, json.dumps(args)],
                         capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr[-4000:]
    res = json.loads(next(l for l in run.stdout.splitlines()
                          if l.startswith("RESULT "))[len("RESULT "):])
    return res, disk_frames(tmp_path)


def series(frames, pid):
    """counter -> [(timestamp_ns, bytes in the interval ending there)]."""
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


def check(res, frames, want_children, expect_c_in_g):
    """want_children: [(pid, reaped_by, ambiguous)] of the one adjustment,
    at G. expect_c_in_g: whether C's I/O is in G's kernel counters."""
    g, p, c = res["g"], res["done"]["p"], res["child"]["child"]
    c_io = res["child"]["io"]
    raw = {k: res["done"]["i1"][k] - res["done"]["i0"][k] for k in COUNTERS}
    table = tracked(frames)
    # Premises: all three traced and read; C did the I/O.
    assert c_io["wchar"] >= 64 * MiB and c_io["rchar"] >= 64 * MiB, c_io
    for pid in (g, p, c):
        assert pid in table and table[pid].HasField("io_before_tracking"), (pid, table.keys())
    ser = {pid: series(frames, pid) for pid in (g, p, c)}
    assert len(ser[p]["rchar"]) >= 3 and len(ser[c]["rchar"]) >= 3, "too few samples"

    adjs = [a for f in frames for a in f.io_reap_adjustments]
    got = sorted((ch.pid, ch.reaped_by, ch.ambiguous) for a in adjs for ch in a.children)
    assert len(adjs) == 1 and adjs[0].parent_pid == g, \
        f"premise or fix: expected one adjustment at G {g}, got {[(a.parent_pid, [c.pid for c in a.children]) for a in adjs]}"
    [a] = adjs
    assert got == sorted(want_children), (got, want_children)
    assert a.ambiguous == any(amb for _, _, amb in want_children), a

    failures = []
    for k in COUNTERS:
        rem = getattr(a.remainder, k)
        subtracted = sum(getattr(ch.last_seen, k) for ch in a.children if not ch.ambiguous)
        unsub = sum(getattr(ch.last_seen, k) for ch in a.children if ch.ambiguous)
        # The adjustment's sample is the remainder: raw delta reconstructable.
        at = [b for ts, b in ser[g][k] if ts == a.timestamp_ns]
        if len(at) != 1 or abs(at[0] - max(rem, 0)) > 1 + 1e-9 * abs(rem):
            failures.append(("sample at the adjustment", k, at, rem))
        # What is left is G's own I/O plus the children's after their last
        # readings (a few printed lines), plus an ambiguous child's I/O if
        # it was really in G.
        extra = unsub if (expect_c_in_g and unsub) else 0
        if not -1 <= rem - extra <= 64 * KiB:
            failures.append(("remainder", k, rem, extra))
        # last_seen = io_before_tracking + the samples (the I/O head).
        for ch in a.children:
            head = getattr(table[ch.pid].io_before_tracking, k)
            seen = getattr(ch.last_seen, k)
            if abs(head + total(ser[ch.pid][k]) - seen) > 2 + 1e-9 * seen:
                failures.append(("head + samples != last_seen", ch.pid, k, head,
                                 total(ser[ch.pid][k]), seen))
        # Each process's samples + heads against the kernel's view of G's
        # tree (C only if its I/O went into G).
        traced = total(ser[g][k]) + total(ser[p][k]) + getattr(table[p].io_before_tracking, k)
        c_traced = total(ser[c][k]) + getattr(table[c].io_before_tracking, k)
        if expect_c_in_g:
            traced += c_traced
            if any(amb for _, _, amb in want_children):
                traced -= unsub   # counted twice by design: not subtracted
        print(f"{k:22s} kernel(G tree) {raw[k] / MiB:8.3f} MiB  traced {traced / MiB:8.3f} MiB  "
              f"subtracted {subtracted / MiB:.3f}  remainder {rem / MiB:.3f}")
        if abs(traced - raw[k]) > 256 * KiB:
            failures.append(("tree total", k, raw[k], round(traced)))
    assert not failures, failures


@pytest.mark.parametrize("mode", MODES)
def test_chain_subtracted_at_grandparent(tmp_path, mode):
    # Reaper on, tree under the launcher: P reaped C -> both subtracted at G.
    res, frames = run(tmp_path, mode, "chain", adopt=True)
    g, p, c = res["g"], res["done"]["p"], res["child"]["child"]
    check(res, frames, [(p, g, False), (c, p, False)], expect_c_in_g=True)


@pytest.mark.parametrize("mode", MODES)
def test_chain_ambiguous_without_reaper(tmp_path, mode):
    # Reaper off: whether P reaped C cannot be known -> C listed as
    # ambiguous, not subtracted (it did go into G here, so G's remainder
    # holds it).
    res, frames = run(tmp_path, mode, "chain", adopt=False)
    g, p, c = res["g"], res["done"]["p"], res["child"]["child"]
    check(res, frames, [(p, g, False), (c, p, True)], expect_c_in_g=True)


@pytest.mark.parametrize("mode", MODES)
def test_orphan_reaped_by_launcher_not_subtracted(tmp_path, mode):
    # P exits first; C is adopted and reaped by the launcher's reaper: its
    # I/O is in the launcher, not in G. Discriminating only when the disk
    # probe first looks at C after the reaper took it (reaper latency
    # ~10-20 ms against a 500 ms interval); otherwise C is seen as the
    # launcher's zombie and dropped by the older ppid rule.
    res, frames = run(tmp_path, mode, "orphan", adopt=True)
    g, p = res["g"], res["done"]["p"]
    assert res["child"]["ppid"] == res["me"], f"premise: C adopted by {res['child']['ppid']}"
    check(res, frames, [(p, g, False)], expect_c_in_g=False)


@pytest.mark.parametrize("mode", MODES)
def test_orphan_outside_launcher_tree_ambiguous(tmp_path, mode):
    # Reaper on, but G is started by an untracked subreaper X: the orphan
    # C goes to X, which reaps it at once. The launcher's reaper does not
    # cover this tree, so C is ambiguous (and correctly not subtracted).
    res, frames = run(tmp_path, mode, "orphan", adopt=True, via_x=True)
    g, p, c = res["g"], res["done"]["p"], res["child"]["child"]
    assert res["child"]["ppid"] not in (res["me"], g, p), res["child"]
    check(res, frames, [(p, g, False), (c, p, True)], expect_c_in_g=False)
