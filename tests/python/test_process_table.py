"""The trace's process table (TrackedProcessV2) and exit detection.

Every tracked PID — listed root or discovered, with descendant tracking
on or off — is watched by the probe itself through a pidfd. A process
that exits is emitted once more with removed=true and its end time, and
is never sampled again. With discovery off (the default) nothing else
watches a root, so these tests run with discovery off unless they are
about discovered processes.

Each row also names the process: comm (followed across renames, with a
history), the kernel's start time on the trace clock (10 ms resolution),
parent, kind (discovered or not) and label.
"""

import os
import sys
import time

import pytest

from tracing_helpers import (disk_frames, running_suite, sample_times,
                             system_frames, tracked, tree)

MODES = ["legacy", "sidecar"]
ON = {"enabled": True, "scan_interval_ms": 50}
HZ = 100
TICK_NS = 1_000_000_000 // HZ
MS = 1_000_000

# Waits for a line on stdin, reports the time, and exits without being
# reaped (the test reaps it later): the probe must see a zombie as gone.
EXITS_ON_LINE = """
emit(ready=time.monotonic_ns())
sys.stdin.readline()
emit(exiting=time.monotonic_ns())
os._exit(0)
"""


def rows(frames, pid):
    """(frame index, TrackedProcessV2) for every frame that lists pid."""
    return [(i, tp) for i, f in enumerate(frames) for tp in f.tracked_processes if tp.pid == pid]


def check_exit(frames, pid, exiting_ns, what):
    r = rows(frames, pid)
    assert r, f"{what}: {pid} never tracked"
    removed = [(i, tp) for i, tp in r if tp.removed]
    assert len(removed) == 1, f"{what}: removed=true must appear exactly once: " \
                              f"{[(i, tp.removed) for i, tp in r]}"
    i_removed, tp = removed[0]
    assert all(i < i_removed for i, _ in r if i != i_removed), \
        f"{what}: still listed after its removal marker"
    assert i_removed < len(frames) - 1, f"{what}: no flush after the removal marker"
    assert tp.end_time_ns > 0, f"{what}: removed after an exit but no end time"
    late = (tp.end_time_ns - exiting_ns) / MS
    # Seen gone at the first tick after the exit; os._exit and the
    # kernel's teardown take a few ms, scheduling a few more.
    assert 0 < late <= TICK_NS / MS + 40, f"{what}: end time {late:.1f} ms after exit"
    ts = sample_times(frames, pid)
    assert ts, f"{what}: never sampled"
    assert max(ts) < tp.end_time_ns, f"{what}: sampled after its end time"
    return tp, late


@pytest.mark.parametrize("mode", MODES)
def test_root_exit_detected_without_discovery(tmp_path, mode):
    # Gap C: with discovery off, only the probe can notice that a listed
    # root has exited. It must be removed with an end time, and never
    # sampled after it, even while it is an unreaped zombie.
    with tree(EXITS_ON_LINE) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], disk=True,
                           disk_hz=HZ, hz=HZ, flush_ms=100):
            t.read("ready")
            time.sleep(0.3)
            t.proc.stdin.write(b"exit\n")
            t.proc.stdin.flush()
            exiting = t.read("exiting")["exiting"]
            time.sleep(0.5)                      # still a zombie here
            zombie = open(f"/proc/{t.pid}/stat").read().split(")")[-1].split()[0]
            t.proc.wait()
            time.sleep(0.4)                      # several flushes after
    assert zombie == "Z", f"premise: the root should be an unreaped zombie, was {zombie}"
    for what, frames in (("system", system_frames(tmp_path)), ("disk", disk_frames(tmp_path))):
        _, late = check_exit(frames, t.pid, exiting, what)
        print(f"{mode} {what}: end time {late:.1f} ms after the exit")


@pytest.mark.parametrize("mode", MODES)
def test_remove_tracked_process_still_works(tmp_path, mode):
    with tree(EXITS_ON_LINE) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")],
                           hz=HZ, flush_ms=100) as suite:
            t.read("ready")
            time.sleep(0.3)
            removed_at = time.monotonic_ns()
            suite.remove_tracked_process(t.pid)
            time.sleep(0.4)
        alive = t.proc.poll() is None
    assert alive, "premise: the root is still running when removed"
    frames = system_frames(tmp_path)
    r = rows(frames, t.pid)
    flags = [tp.removed for _, tp in r]
    assert flags.count(True) == 1 and flags[-1], flags
    assert r[-1][1].end_time_ns == 0, "removed by request, not by exit: no end time"
    late = [ts for ts in sample_times(frames, t.pid) if ts > removed_at + 2 * TICK_NS]
    assert not late, f"sampled {len(late)} times after its removal"


RENAMES = """
import ctypes
libc = ctypes.CDLL("libc.so.6")
def rename(name):
    libc.prctl(15, name.encode(), 0, 0, 0)          # PR_SET_NAME
c = subprocess.Popen([PY, "-c", '''
import ctypes, json, sys, time
sys.stdin.readline()
ctypes.CDLL("libc.so.6").prctl(15, b"renamed-kid", 0, 0, 0)
print(json.dumps({"kid_renamed": time.monotonic_ns()}), flush=True)
sys.stdin.readline()
'''], stdin=subprocess.PIPE)
emit(child=c.pid, root_comm=open("/proc/self/comm").read().strip())
for line in sys.stdin:
    if line.startswith("kid"):
        c.stdin.write(b"go\\n"); c.stdin.flush()
    elif line.startswith("root"):
        rename("renamed-root")
        emit(root_renamed=time.monotonic_ns())
"""


@pytest.mark.parametrize("mode", MODES)
def test_rename_followed(tmp_path, mode, capfd):
    # A discovered child renames itself AFTER discovery found it: its
    # alias and comm must follow, with the rename in comm_history. A root
    # that renames keeps its listed alias; only comm follows.
    with tree(RENAMES) as t:
        with running_suite(tmp_path, mode, discovery=ON, disk=True, hz=HZ,
                           processes=[(t.pid, "root")], flush_ms=100):
            first = t.read("child")
            child, root_comm = first["child"], first["root_comm"]
            t.extra_pids.append(child)
            time.sleep(0.5)                            # discovered, and flushed as found
            t.proc.stdin.write(b"kid\n"); t.proc.stdin.flush()
            kid_renamed = t.read("kid_renamed")["kid_renamed"]
            t.proc.stdin.write(b"root\n"); t.proc.stdin.flush()
            root_renamed = t.read("root_renamed")["root_renamed"]
            time.sleep(0.5)
    err = capfd.readouterr().err
    for what, frames in (("system", system_frames(tmp_path)), ("disk", disk_frames(tmp_path))):
        r = rows(frames, child)
        assert r, f"{what}: child never discovered"
        orig = r[0][1].comm
        assert orig and orig != "renamed-kid"
        assert r[0][1].alias == f"root/{orig}", f"{what}: premise — found before the rename"
        last = r[-1][1]
        assert last.alias == "root/renamed-kid", f"{what}: alias did not follow: {last.alias}"
        assert last.comm == "renamed-kid" and last.label == "root" and last.discovered
        hist = [(h.comm, h.timestamp_ns) for h in last.comm_history]
        assert [c for c, _ in hist] == [orig, "renamed-kid"], hist
        seen_after = (hist[1][1] - kid_renamed) / MS
        # (kid_renamed is stamped just after the prctl, so a probe tick in
        # between may see the rename a hair "before" it.)
        assert -5 < seen_after <= 100 + TICK_NS / MS + 50, \
            f"{what}: rename seen {seen_after:.0f} ms after it happened"

        root = rows(frames, t.pid)[-1][1]
        assert root.alias == "root" and root.label == "root" and not root.discovered
        assert root.comm == "renamed-root", f"{what}: root comm {root.comm}"
        assert [h.comm for h in root.comm_history] == [root_comm, "renamed-root"]
        assert root.comm_history[1].timestamp_ns > root_renamed - 5 * MS
        print(f"{mode} {what}: child rename seen after {seen_after:.0f} ms")

    # The tree is logged as it grows: pid, comm, parent pid and comm.
    line = f"[discovery] + {child} {orig} (parent {t.pid} {root_comm})"
    assert line in err, f"missing {line!r} in stderr:\n{err[-3000:]}"


@pytest.mark.parametrize("mode", MODES)
def test_process_table_fields(tmp_path, mode):
    # Start time: the kernel's (10 ms ticks since boot, on CLOCK_BOOTTIME)
    # converted to the trace clock, so it falls between the test's spawn
    # and the process's first line, within one tick. Parent and kind are
    # recorded for roots and discovered processes alike.
    body = """
emit(ready=time.monotonic_ns())
c = sleeper(30)
emit(child=c.pid, child_spawned=time.monotonic_ns())
sys.stdin.readline()
"""
    spawned = time.monotonic_ns()
    with tree(body) as t:
        ready = t.read("ready")["ready"]
        msg = t.read("child")
        child = msg["child"]
        t.extra_pids.append(child)
        with running_suite(tmp_path, mode, discovery=ON, hz=HZ,
                           processes=[(t.pid, "root")]):
            time.sleep(0.5)
    seen = tracked(system_frames(tmp_path))
    root, kid = seen[t.pid], seen[child]
    assert spawned - 10 * MS <= root.start_time_ns <= ready + 10 * MS, \
        (root.start_time_ns - spawned) / MS
    assert ready - 10 * MS <= kid.start_time_ns <= msg["child_spawned"] + 10 * MS, \
        (kid.start_time_ns - ready) / MS
    assert root.parent_pid == os.getpid() and not root.discovered and root.label == "root"
    assert kid.parent_pid == t.pid and kid.discovered and kid.label == "root"
    exe = os.path.basename(sys.executable)[:15]
    assert root.comm == exe and kid.comm == exe, (root.comm, kid.comm, exe)
    assert root.end_time_ns == 0 and kid.end_time_ns == 0
