"""Descendant tracking (ProcessDiscoveryConfig / track_descendants).

Every test runs in both LEGACY (discovery thread in this process) and
SIDECAR (discovery thread in the sidecar, settings and per-root override
sent over the pipe). The trees are small Python processes; each reports
its PIDs as JSON lines on the stdout it shares with its descendants.

Guarantee under test: a process that is a child of a tracked process for
at least one full scan interval is discovered, and once discovered it is
tracked until it exits, regardless of reparenting.
"""

import os
import time

import pytest

from tracing_helpers import (disk_frames, running_suite, sample_times,
                             system_frames, tracked, tree)

MODES = ["legacy", "sidecar"]
ON = {"enabled": True, "scan_interval_ms": 50}

# root -> one sleeping child
ONE_CHILD = """
c = sleeper(30)
emit(child=c.pid)
sys.stdin.readline()
"""

# root -> child -> grandchild
GRANDCHILD = """
c = subprocess.Popen([PY, "-c", '''
import json, subprocess, sys
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
print(json.dumps({"grandchild": g.pid}), flush=True)
g.wait()
'''])
emit(child=c.pid)
sys.stdin.readline()
"""


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("enabled,override,expect_child", [
    (False, None,  False),   # knob off: only listed PIDs
    (True,  False, False),   # knob on, root opts out
    (False, True,  True),    # knob off, root opts in
])
def test_discovery_off(tmp_path, mode, enabled, override, expect_child):
    disc = {"enabled": enabled, "scan_interval_ms": 50}
    with tree(ONE_CHILD) as t, running_suite(tmp_path, mode, discovery=disc) as suite:
        child = t.read("child")["child"]
        t.extra_pids.append(child)
        suite.add_tracked_process(t.pid, "root", track_descendants=override)
        time.sleep(0.6)   # >= 10 scan intervals
    seen = tracked(system_frames(tmp_path))
    if expect_child:
        assert set(seen) == {t.pid, child}, seen
        assert seen[child].discovered and seen[child].parent_pid == t.pid
    else:
        assert set(seen) == {t.pid}, f"only the listed PID may be traced: {sorted(seen)}"
    assert not seen[t.pid].discovered and seen[t.pid].parent_pid == 0


@pytest.mark.parametrize("mode", MODES)
def test_direct_child(tmp_path, mode):
    # Root listed in the config (inherits enabled=True); both probes get
    # the discovered child, because one setting drives both.
    with tree(ONE_CHILD) as t:
        child = t.read("child")["child"]
        t.extra_pids.append(child)
        with running_suite(tmp_path, mode, discovery=ON, disk=True,
                           processes=[(t.pid, "root")]):
            time.sleep(0.6)
    for name, frames in (("system", system_frames(tmp_path)), ("disk", disk_frames(tmp_path))):
        seen = tracked(frames)
        assert set(seen) == {t.pid, child}, (name, sorted(seen))
        c = seen[child]
        assert c.discovered and c.parent_pid == t.pid, (name, c)
        assert c.alias.startswith("root/") and len(c.alias) > len("root/"), (name, c.alias)
    assert len(sample_times(system_frames(tmp_path), child)) >= 10


@pytest.mark.parametrize("mode", MODES)
def test_grandchild_recursive(tmp_path, mode):
    with tree(GRANDCHILD) as t, running_suite(tmp_path, mode, discovery=ON) as suite:
        child = t.read("child")["child"]
        grandchild = t.read("grandchild")["grandchild"]
        t.extra_pids += [child, grandchild]
        suite.add_tracked_process(t.pid, "root")
        time.sleep(0.6)
    seen = tracked(system_frames(tmp_path))
    assert set(seen) == {t.pid, child, grandchild}, sorted(seen)
    assert seen[child].parent_pid == t.pid
    assert seen[grandchild].parent_pid == child and seen[grandchild].discovered
    assert seen[grandchild].alias.startswith("root/")


@pytest.mark.parametrize("mode", MODES)
def test_grandchild_direct_only(tmp_path, mode):
    disc = dict(ON, direct_children_only=True)
    with tree(GRANDCHILD) as t, running_suite(tmp_path, mode, discovery=disc) as suite:
        child = t.read("child")["child"]
        grandchild = t.read("grandchild")["grandchild"]
        t.extra_pids += [child, grandchild]
        suite.add_tracked_process(t.pid, "root")
        time.sleep(0.6)
    seen = tracked(system_frames(tmp_path))
    assert set(seen) == {t.pid, child}, sorted(seen)
    assert grandchild not in seen


@pytest.mark.parametrize("mode", MODES)
def test_child_from_worker_thread(tmp_path, mode):
    # The worker thread forks the child and stays alive waiting on it,
    # so the child hangs off the WORKER's children file, not the main
    # thread's.
    body = """
def work():
    c = sleeper(30)
    emit(child=c.pid, tid=threading.get_native_id())
    c.wait()
threading.Thread(target=work, daemon=True).start()
sys.stdin.readline()
"""
    with tree(body) as t, running_suite(tmp_path, mode, discovery=ON) as suite:
        msg = t.read("child")
        child, tid = msg["child"], msg["tid"]
        t.extra_pids.append(child)
        suite.add_tracked_process(t.pid, "root")
        time.sleep(0.6)
        # The premise: the main thread's file does not list the child.
        main = open(f"/proc/{t.pid}/task/{t.pid}/children").read().split()
        worker = open(f"/proc/{t.pid}/task/{tid}/children").read().split()
    assert str(child) not in main and str(child) in worker, (main, worker)
    seen = tracked(system_frames(tmp_path))
    assert child in seen and seen[child].parent_pid == t.pid, sorted(seen)


# Double fork: A forks B and exits; B, reparented, forks C. B's argv[1]
# is A's PID; A's argv[1] is B's code.
B_CODE = """
import json, os, subprocess, sys, time
a = int(sys.argv[1])
while os.getppid() == a:
    time.sleep(0.01)
print(json.dumps({"reparented": os.getppid(), "t": time.monotonic_ns()}), flush=True)
c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
print(json.dumps({"c": c.pid}), flush=True)
c.wait()
time.sleep(0.3)
"""
A_CODE = """
import json, os, subprocess, sys, time
time.sleep(0.4)
b = subprocess.Popen([sys.executable, "-c", sys.argv[1], str(os.getpid())])
print(json.dumps({"b": b.pid}), flush=True)
time.sleep(0.5)
os._exit(0)
"""


@pytest.mark.parametrize("mode", MODES)
def test_double_fork(tmp_path, mode):
    # root -> A -> B. A lives long enough to be discovered and to have B
    # discovered as its child, then exits: B is reparented out of the
    # root's tree. B keeps being tracked, and B's child C — forked only
    # AFTER the reparenting, so not reachable from the root — is found
    # because discovery scans every tracked process.
    body = f"""
a = subprocess.Popen([PY, "-c", {A_CODE!r}, {B_CODE!r}])
emit(a=a.pid)
a.wait()
sys.stdin.readline()
"""
    with tree(body) as t, running_suite(tmp_path, mode, discovery=ON) as suite:
        suite.add_tracked_process(t.pid, "root")
        a = t.read("a")["a"]
        b = t.read("b")["b"]
        t.extra_pids.append(b)
        rep = t.read("reparented")
        c = t.read("c")["c"]
        t.extra_pids.append(c)
        time.sleep(0.6)
    frames = system_frames(tmp_path)
    seen = tracked(frames)
    assert rep["reparented"] != a
    assert a in seen and seen[a].parent_pid == t.pid
    assert b in seen and seen[b].parent_pid == a, sorted(seen)
    after = [ts for ts in sample_times(frames, b) if ts > rep["t"]]
    assert len(after) >= 20, f"B not sampled after reparenting: {len(after)} samples"
    assert c in seen, f"C (child of reparented B) not discovered: {sorted(seen)}"
    assert seen[c].parent_pid == b and seen[c].discovered


@pytest.mark.parametrize("mode", MODES)
def test_short_lived_child(tmp_path, mode):
    # 30 children, each alive ~30 ms, against a 100 ms scan: most are
    # missed, some are caught. Whatever is caught must be a real child,
    # and must be removed again — nothing stale, no crash.
    body = """
kids = []
for _ in range(30):
    p = os.fork()
    if p == 0:
        time.sleep(0.03)
        os._exit(0)
    kids.append(p)
    os.waitpid(p, 0)
    time.sleep(0.02)
emit(kids=kids)
sys.stdin.readline()
"""
    disc = {"enabled": True, "scan_interval_ms": 100}
    # Frequent flushes give a removal more chances to land between a
    # flush's snapshot and its commit, where it was once dropped without
    # its removed=true marker. That window is short, so this only raises
    # the odds of noticing a regression; it does not guarantee it.
    with tree(body) as t, running_suite(tmp_path, mode, discovery=disc, flush_ms=20) as suite:
        suite.add_tracked_process(t.pid, "root")
        kids = set(t.read("kids", timeout=20)["kids"])
        time.sleep(0.6)   # several flushes after the last child exited
    frames = system_frames(tmp_path)
    seen = tracked(frames)
    caught = set(seen) - {t.pid}
    print(f"caught {len(caught)} of {len(kids)} short-lived children")
    assert caught <= kids, f"discovered PIDs that were never children: {caught - kids}"
    last = {tp.pid: tp for tp in frames[-1].tracked_processes}
    stale = [p for p, tp in last.items() if tp.discovered and not tp.removed]
    assert not stale, f"exited children still tracked: {stale}"
    for p in caught:
        assert seen[p].removed, f"{p} was discovered but never marked removed"


@pytest.mark.parametrize("mode", MODES)
def test_exit_marks_removed(tmp_path, mode):
    body = """
c = sleeper(0.6)
emit(child=c.pid)
c.wait()
emit(exited=time.monotonic_ns())
sys.stdin.readline()
"""
    with tree(body) as t, running_suite(tmp_path, mode, discovery=ON, flush_ms=200) as suite:
        suite.add_tracked_process(t.pid, "root")
        child = t.read("child")["child"]
        t.read("exited")
        time.sleep(1.2)   # >= 5 flushes after the exit
    frames = system_frames(tmp_path)
    states = [next((tp.removed for tp in f.tracked_processes if tp.pid == child), None)
              for f in frames]
    present = [i for i, s in enumerate(states) if s is not None]
    assert present, "child never tracked"
    removed_at = [i for i, s in enumerate(states) if s is True]
    assert len(removed_at) == 1, f"removed=true must appear exactly once: {states}"
    assert all(s is False for s in states[present[0]:removed_at[0]]), states
    assert all(s is None for s in states[removed_at[0] + 1:]), f"dropped after removal: {states}"
    assert removed_at[0] < len(frames) - 1, "no flush after the removal marker"
    assert sample_times(frames, child), "child tracked but never sampled"


@pytest.mark.parametrize("mode", MODES)
def test_pid_reuse_guard(tmp_path, mode, monkeypatch):
    # Synthetic /proc: root R's children file lists X and Y. X's stat
    # names R as its parent; Y's names a PID nobody tracks, as a PID
    # recycled by an unrelated process would. All three are real, live
    # processes (pidfds and the probes use the real kernel).
    with tree("sys.stdin.readline()") as r, tree("sys.stdin.readline()") as x, \
         tree("sys.stdin.readline()") as y:
        fake = tmp_path / "proc"

        def mk(pid, ppid, comm, children=None):
            d = fake / str(pid)
            (d / "task" / str(pid)).mkdir(parents=True)
            fields = ["S", str(ppid)] + ["0"] * 17 + ["12345"] + ["0"] * 20
            (d / "stat").write_text(f"{pid} ({comm}) {' '.join(fields)}\n")
            (d / "comm").write_text(comm + "\n")
            (d / "task" / str(pid) / "children").write_text(
                "".join(f"{c} " for c in (children or [])))

        mk(r.pid, os.getpid(), "fake-root", children=[x.pid, y.pid])
        mk(x.pid, r.pid, "fake-child")
        mk(y.pid, 999999, "fake-reused")
        monkeypatch.setenv("CUPTI_PROFILER_PROC_ROOT", str(fake))
        with running_suite(tmp_path, mode, discovery=ON,
                           processes=[(r.pid, "root")]):
            time.sleep(0.5)
    frames = system_frames(tmp_path)
    seen = tracked(frames)
    assert x.pid in seen and seen[x.pid].parent_pid == r.pid, sorted(seen)
    assert seen[x.pid].alias == "root/fake-child"
    assert y.pid not in seen, "a PID whose parent is not tracked was accepted"
    stats = frames[-1].discovery_stats
    assert stats.rejected >= 1 and stats.discovered == 1, stats


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("interval_ms", [50, 200])
def test_scan_interval(tmp_path, mode, interval_ms):
    hz = 100
    body = """
sys.stdin.readline()
c = subprocess.Popen([PY, "-c", "import json, time; print(json.dumps({'born': time.monotonic_ns()}), flush=True); time.sleep(30)"])
emit(child=c.pid)
sys.stdin.readline()
"""
    disc = {"enabled": True, "scan_interval_ms": interval_ms}
    with tree(body) as t, running_suite(tmp_path, mode, discovery=disc, hz=hz) as suite:
        suite.add_tracked_process(t.pid, "root")
        t0 = time.monotonic()
        time.sleep(1.0)
        t.proc.stdin.write(b"go\n")
        t.proc.stdin.flush()
        child = t.read("child")["child"]
        t.extra_pids.append(child)
        born = t.read("born")["born"]
        time.sleep(1.0)
        elapsed = time.monotonic() - t0
    frames = system_frames(tmp_path)
    stats = frames[-1].discovery_stats
    assert stats.scan_interval_ns == interval_ms * 1_000_000
    expected = elapsed * 1000 / interval_ms
    assert 0.75 * expected <= stats.scans <= 1.25 * expected + 2, (stats.scans, expected)
    assert 0 < stats.scan_p50_ns <= stats.scan_p99_ns <= stats.scan_max_ns, stats
    assert stats.discovered == 1 and stats.exited == 0, stats
    # Cadence also bounds discovery latency: found within one interval,
    # first sample one sample tick later (plus scheduling slack).
    first = sample_times(frames, child)[0]
    latency_ms = (first - born) / 1e6
    print(f"scans {stats.scans} (expected ~{expected:.0f}); p50 {stats.scan_p50_ns/1e3:.1f} us, "
          f"p99 {stats.scan_p99_ns/1e3:.1f} us, max {stats.scan_max_ns/1e3:.1f} us; "
          f"child first sampled {latency_ms:.0f} ms after birth")
    assert latency_ms <= interval_ms + 2 * 1000 / hz + 50, latency_ms
