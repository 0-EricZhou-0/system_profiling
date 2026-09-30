"""A reap chain through a traced parent that was never read.

G (the listed root) forks P, which makes itself non-dumpable, so its
/proc/<pid>/io cannot be read; P forks C, which makes itself dumpable
again and does known I/O. P reaps C and exits; G reaps P. The kernel
folds C's I/O into P and P's into G, so G's reading holds C's I/O while
C's own samples hold it too. The chain walk has no record for P (never
read), so it cannot subtract C at G: not fixed, but detected when the
walk stops at P — one `[cupti-profiler] warning:` naming C, P and G, and
an IoReapChainBreak in the disk trace. The CPU tail has no such gap.
"""

import re
import time

import pytest

from test_reap_io import COUNTERS, MiB, series, total
from tracing_helpers import disk_frames, running_suite, tree

MODES = ["legacy", "sidecar"]
ON = {"enabled": True, "scan_interval_ms": 50}

ROOT = """
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}
emit(ready=1)
d = sys.stdin.readline().strip()
p = os.fork()
if p == 0:
    libc.prctl(4, 0, 0, 0, 0)                  # PR_SET_DUMPABLE 0: /proc/<P>/io unreadable
    time.sleep(0.5)
    c = os.fork()
    if c == 0:
        libc.prctl(4, 1, 0, 0, 0)              # C is readable
        time.sleep(0.5)
        path = os.path.join(d, "c.dat")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        for _ in range(16):
            os.write(fd, os.urandom(1 << 20))
        os.close(fd)
        os.unlink(path)
        time.sleep(0.5)
        emit(c=os.getpid(), io=io())
        os._exit(0)
    os.waitpid(c, 0)
    emit(p=os.getpid(), c_reaped=c)
    os._exit(0)
os.waitpid(p, 0)
time.sleep(0.5)
emit(done=p)
sys.stdin.readline()
"""


def run(tmp_path, mode, capfd):
    with tree(ROOT) as t:
        t.read("ready")
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], disk=True,
                           disk_hz=50, discovery=ON):
            time.sleep(0.2)
            t.proc.stdin.write(f"{tmp_path}\n".encode())
            t.proc.stdin.flush()
            child = t.read("c", 30)
            par = t.read("p", 30)
            t.read("done", 30)
            time.sleep(0.3)
    return t.pid, par["p"], child, capfd.readouterr().err


@pytest.mark.parametrize("mode", MODES)
def test_chain_through_unread_parent_is_warned_and_recorded(tmp_path, mode, capfd):
    g, p, child, err = run(tmp_path, mode, capfd)
    c = child["c"]
    frames = disk_frames(tmp_path)
    # Premise: P was traced but never read; C was read.
    gs, ps, cs = series(frames, g), series(frames, p), series(frames, c)
    assert not ps["wchar"], "premise: P's I/O was read"
    assert total(cs["wchar"]) >= 15 * MiB, "premise: C's I/O was not sampled"
    tp = {t.pid for f in frames for t in f.tracked_processes}
    assert p in tp and c in tp, f"premise: P and C are traced ({p}, {c}, {sorted(tp)})"

    warnings = [l for l in err.splitlines()
                if l.startswith("[cupti-profiler] warning:") and "reap chain" in l]
    assert len(warnings) == 1, err
    [w] = warnings
    assert re.search(rf"PID {c} was reaped by traced PID {p}\b", w), w
    assert f"traced PID {g} " in w and "double-counted" in w, w

    breaks = [b for f in frames for b in f.io_reap_chain_breaks]
    assert [(b.pid, b.missing_pid, b.absorbed_by) for b in breaks] == [(c, p, g)], breaks
    # What the record warns about: C's I/O is in G's samples a second time.
    print(f"{mode}: G wchar {total(gs['wchar']) / MiB:.3f} MiB, C wchar {total(cs['wchar']) / MiB:.3f} MiB")
    assert total(gs["wchar"]) >= 15 * MiB
