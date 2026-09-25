"""The I/O head: TrackedProcessV2.io_before_tracking, the analogue of
cpu_before_tracking_ns. Every tracked process — listed root or discovered
— records its five /proc/<pid>/io counters at its first reading, so its
counters at any later reading equal io_before_tracking + its samples up to
there, with nothing to reconstruct.

The root reads 8 MiB before it is tracked; once tracked it forks a child
that writes and reads 16 MiB at once (partly before its first reading),
and both report their own /proc/self/io after lingering long enough for a
reading to follow.
"""

import time

import pytest

import metric_catalog_pb2
from tracing_helpers import disk_frames, running_suite, tracked, tree

MODES = ["legacy", "sidecar"]
MiB = 1 << 20
KiB = 1 << 10
HZ = 20
ON = {"enabled": True, "scan_interval_ms": 20}

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
d = sys.stdin.readline().strip()
path = os.path.join(d, "pre.dat")
with open(path, "wb") as f:
    f.write(os.urandom(8 << 20))
with open(path, "rb") as f:
    while f.read(1 << 20):
        pass
emit(ready=1)
sys.stdin.readline()                  # tracked now
pid = os.fork()
if pid == 0:
    p = os.path.join(d, "child.dat")
    chunk = os.urandom(1 << 20)
    with open(p, "wb") as f:
        for _ in range(16):
            f.write(chunk)
    with open(p, "rb") as f:
        while f.read(1 << 20):
            pass
    time.sleep(0.6)
    emit(child=os.getpid(), io=io())
    time.sleep(0.6)                   # a reading after the report
    os._exit(0)
time.sleep(1.5)
emit(root=os.getpid(), io=io())
time.sleep(0.6)
emit(settled=1)
sys.stdin.readline()
"""


def sums(frames, pid):
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    cols = fqns[metric_catalog_pb2.SCOPE_PROCESS]
    rows = sorted((s.timestamp_ns, list(s.values)) for f in frames for s in f.process_samples
                  if s.pid == pid)
    # (timestamp, bytes) per counter; each sample's interval ends at its
    # timestamp and starts at the previous reading.
    out = {k: [] for k in COUNTERS}
    prev = None
    for ts, v in rows:
        for k, fqn in COUNTERS.items():
            dt = (ts - prev) / 1e9 if prev is not None else 1.0 / HZ
            out[k].append((ts, v[cols.index(fqn)] * dt))
        prev = ts
    return out


@pytest.mark.parametrize("mode", MODES)
def test_io_before_tracking(tmp_path, mode):
    with tree(ROOT) as t:
        t.proc.stdin.write(f"{tmp_path}\n".encode())
        t.proc.stdin.flush()
        t.read("ready")
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], disk=True,
                           disk_hz=HZ, discovery=ON):
            time.sleep(0.3)
            t.proc.stdin.write(b"go\n")
            t.proc.stdin.flush()
            child = t.read("child", 30)
            root = t.read("root", 30)
            t.read("settled", 30)
        t.proc.stdin.write(b"x\n")
        t.proc.stdin.flush()
    frames = disk_frames(tmp_path)
    table = tracked(frames)
    failures = []
    for who, rep in (("root", root), ("child", child)):
        pid = rep[who]
        assert pid in table, f"{who} {pid} not tracked"
        tp = table[pid]
        assert tp.HasField("io_before_tracking"), f"{who}: no io_before_tracking"
        s = sums(frames, pid)
        for k in COUNTERS:
            head = getattr(tp.io_before_tracking, k)
            # Counters at the report = head + samples up to the report; the
            # report itself (one line, one /proc read) is the only slack.
            upto = head + sum(b for _, b in s[k])
            print(f"{mode} {who:5s} {k:22s} head {head / MiB:8.3f} MiB  head+samples "
                  f"{upto / MiB:8.3f} MiB  /proc {rep['io'][k] / MiB:8.3f} MiB")
            if not -64 * KiB <= upto - rep["io"][k] <= 64 * KiB:
                failures.append((who, k, head, upto, rep["io"][k]))
        if who == "root":
            # The 8 MiB read before tracking is in the head, not a sample.
            if tp.io_before_tracking.rchar < 8 * MiB:
                failures.append(("root head", tp.io_before_tracking.rchar))
            if max((b for _, b in s["rchar"]), default=0) >= 8 * MiB:
                failures.append(("root head folded into a sample",))
    assert not failures, failures
