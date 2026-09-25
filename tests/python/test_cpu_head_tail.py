"""Head and tail CPU of discovered processes.

A process found by descendant tracking mid-run has CPU from its fork to
its first sample (the head, TrackedProcessV2.cpu_before_discovery_ns) and
from its last sample to its exit (the tail, a CpuTail measured on the
parent that reaped it). With both, a discovered process's whole CPU is
accounted for:

    head + sum over samples + tail == the CPU it used (its own clock)

Each spinner reports its own process CPU clock as the last thing it does
(os._exit right after), which is the ground truth.

Tolerances: the tail comes from the parent's cutime+cstime, which is in
USER_HZ ticks (10 ms) and truncated per field (two fields, two readings);
a running thread's clock is folded in at scheduler ticks. Hence 40 ms + 2%.
"""

import time

import pytest

import metric_catalog_pb2
from tracing_helpers import running_suite, system_frames, tracked, tree

MODES = ["legacy", "sidecar"]
CPU_FQN = "proc__cycles_active.sum.per_second"

# A process that burns SECONDS of CPU, then reports its own CPU clock.
SPINNER = """
import json, os, sys, time
end = time.process_time() + float(sys.argv[1])
while time.process_time() < end:
    pass
print(json.dumps({"spinner": os.getpid(), "cpu_ns": time.process_time_ns()}), flush=True)
os._exit(0)
"""

# Root: on each stdin line "<n> <seconds>", start n spinners at once, reap
# them together once all have exited, and report.
ROOT = """
SPIN = %r
def zombie(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True
for line in sys.stdin:
    n, secs = line.split()
    ps = [subprocess.Popen([PY, "-c", SPIN, secs]) for _ in range(int(n))]
    while not all(zombie(p.pid) for p in ps):
        time.sleep(0.005)
    for p in ps:
        p.wait()
    emit(reaped=[p.pid for p in ps])
""" % SPINNER


def cpu_accounting(frames, pid):
    """(head_s, samples_s, tails) for a discovered pid: tails are the
    CpuTail messages listing it."""
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    col = fqns[metric_catalog_pb2.SCOPE_PROCESS].index(CPU_FQN)
    sys_ts = sorted(s.timestamp_ns for f in frames for s in f.system_samples)
    samples = sorted((s.timestamp_ns, s.values[col]) for f in frames for s in f.process_samples
                     if s.pid == pid)
    total, prev, per = 0.0, None, []
    for ts, pct in samples:
        if prev is None:   # the baseline was seeded one sample-loop iteration earlier
            prev = max(t for t in sys_ts if t < ts)
        per.append((pct, ts - prev))
        total += pct / 100.0 * (ts - prev) / 1e9
        prev = ts
    head = tracked(frames)[pid].cpu_before_discovery_ns / 1e9
    tails = [c for f in frames for c in f.cpu_tails if pid in c.pids]
    return head, total, tails, per


def spin(t, n, secs, timeout=30):
    t.proc.stdin.write(f"{n} {secs}\n".encode())
    t.proc.stdin.flush()
    truths = {}
    while len(truths) < n:
        m = t.read("spinner", timeout)
        truths[m["spinner"]] = m["cpu_ns"] / 1e9
    reaped = t.read("reaped", timeout)["reaped"]
    assert sorted(reaped) == sorted(truths)
    return truths


@pytest.mark.parametrize("mode", MODES)
def test_cpu_head_recorded_once(tmp_path, mode):
    # Scan every 1 s: the first scan runs at start(), the spinner is born
    # ~0.1 s later and is found by the second scan, having spun ~0.85 s.
    with tree(ROOT) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")],
                           discovery={"enabled": True, "scan_interval_ms": 1000}):
            time.sleep(0.1)
            truths = spin(t, 1, 1.6)
            time.sleep(0.3)
    frames = system_frames(tmp_path)
    [(pid, truth)] = truths.items()
    head, samples, tails, per = cpu_accounting(frames, pid)
    tail = sum(c.cpu_after_last_sample_ns for c in tails) / 1e9
    print(f"head {head:.3f} + samples {samples:.3f} + tail {tail:.3f} = "
          f"{head + samples + tail:.3f} s vs truth {truth:.3f} s")
    assert tracked(frames)[t.pid].cpu_before_discovery_ns == 0, "a listed root has no head"
    assert head >= 0.5, f"head {head:.3f} s: the CPU before discovery was not recorded"
    # Never a first-interval spike: one thread, one core, plus one late
    # scheduler tick (10 ms at CONFIG_HZ=100).
    for pct, dt in per:
        assert pct <= 100.0 * (dt + 10_000_000) / dt, f"cpu_pct {pct:.0f}% over {dt / 1e6:.1f} ms"
    assert abs(head + samples + tail - truth) <= 0.04 + 0.02 * truth


@pytest.mark.parametrize("mode", MODES)
def test_cpu_tail_attributed(tmp_path, mode):
    # 2 Hz sampling: up to 0.5 s of each spinner's CPU falls after its last
    # sample. Three spinners, one after another, each reaped in its own
    # interval, so each tail is its own.
    with tree(ROOT) as t:
        with running_suite(tmp_path, mode, hz=2, processes=[(t.pid, "root")],
                           discovery={"enabled": True, "scan_interval_ms": 50}):
            time.sleep(0.2)
            truths = {}
            for _ in range(3):
                truths.update(spin(t, 1, 1.2))
                time.sleep(0.7)
            time.sleep(0.6)
    frames = system_frames(tmp_path)
    for pid, truth in truths.items():
        head, samples, tails, _ = cpu_accounting(frames, pid)
        assert len(tails) == 1 and list(tails[0].pids) == [pid], \
            f"expected one CpuTail of its own for {pid}: {tails}"
        assert tails[0].parent_pid == t.pid
        tail = tails[0].cpu_after_last_sample_ns / 1e9
        print(f"{pid}: head {head:.3f} + samples {samples:.3f} + tail {tail:.3f} = "
              f"{head + samples + tail:.3f} s vs truth {truth:.3f} s")
        assert abs(head + samples + tail - truth) <= 0.04 + 0.02 * truth


@pytest.mark.parametrize("mode", MODES)
def test_cpu_tail_siblings_reported_per_parent(tmp_path, mode):
    # Two children of one parent reaped in the same interval: their tails
    # cannot be told apart, so one CpuTail lists both with the combined
    # tail — no guessed split.
    with tree(ROOT) as t:
        with running_suite(tmp_path, mode, hz=2, processes=[(t.pid, "root")],
                           discovery={"enabled": True, "scan_interval_ms": 50}):
            time.sleep(0.2)
            truths = spin(t, 2, 1.0)
            time.sleep(1.2)
    frames = system_frames(tmp_path)
    tails = [c for f in frames for c in f.cpu_tails]
    assert len(tails) == 1 and sorted(tails[0].pids) == sorted(truths), tails
    accounted = sum(sum(cpu_accounting(frames, pid)[:2]) for pid in truths)
    accounted += tails[0].cpu_after_last_sample_ns / 1e9
    truth = sum(truths.values())
    print(f"accounted {accounted:.3f} s vs truth {truth:.3f} s")
    assert abs(accounted - truth) <= 0.06 + 0.02 * truth


# Root -> middle -> spinner. On each stdin line "<seconds> <linger>", the
# root starts a middle process that runs one spinner, reaps it, then lives
# on for <linger> s before reporting its own CPU clock and exiting.
MIDDLE = """
import json, os, subprocess, sys, time
SPIN = %r
s = subprocess.Popen([sys.executable, "-c", SPIN, sys.argv[1]])
s.wait()
time.sleep(float(sys.argv[2]))
print(json.dumps({"middle": os.getpid(), "cpu_ns": time.process_time_ns()}), flush=True)
os._exit(0)
""" % SPINNER

NESTED = """
MID = %r
for line in sys.stdin:
    secs, linger = line.split()
    m = subprocess.Popen([PY, "-c", MID, secs, linger])
    m.wait()
    emit(reaped=[m.pid])
""" % MIDDLE


@pytest.mark.parametrize("mode", MODES)
def test_cpu_tail_excludes_reaped_grandchildren(tmp_path, mode):
    # The middle process reaps the spinner long before it exits itself, so
    # when the root reaps the middle, the root's cutime grows by the
    # middle's CPU AND the spinner's (a reaped child's CPU folds into its
    # parent's cutime). The middle's tail must subtract the spinner's CPU,
    # which its own samples and tail already account for.
    with tree(NESTED) as t:
        with running_suite(tmp_path, mode, hz=20, processes=[(t.pid, "root")],
                           discovery={"enabled": True, "scan_interval_ms": 50}):
            time.sleep(0.2)
            t.proc.stdin.write(b"0.8 0.6\n")
            t.proc.stdin.flush()
            spinner = t.read("spinner", 30)
            middle = t.read("middle", 30)
            t.read("reaped", 30)
            time.sleep(0.6)
    frames = system_frames(tmp_path)
    for pid, truth in ((spinner["spinner"], spinner["cpu_ns"] / 1e9),
                       (middle["middle"], middle["cpu_ns"] / 1e9)):
        head, samples, tails, _ = cpu_accounting(frames, pid)
        tail = sum(c.cpu_after_last_sample_ns for c in tails) / 1e9
        print(f"{pid}: head {head:.3f} + samples {samples:.3f} + tail {tail:.3f} = "
              f"{head + samples + tail:.3f} s vs truth {truth:.3f} s")
        assert abs(head + samples + tail - truth) <= 0.04 + 0.02 * truth


@pytest.mark.parametrize("mode", MODES)
def test_cpu_tail_emitted_once(tmp_path, mode):
    # Sampling at 100 Hz, scanning every 1 s: after the spinner is reaped,
    # up to a second of sample ticks pass before discovery marks it removed.
    # Its tail is still emitted exactly once.
    with tree(ROOT) as t:
        with running_suite(tmp_path, mode, hz=100, processes=[(t.pid, "root")],
                           discovery={"enabled": True, "scan_interval_ms": 1000}):
            time.sleep(0.1)
            truths = spin(t, 1, 1.5)
            time.sleep(1.5)
    frames = system_frames(tmp_path)
    [pid] = truths
    tails = [c for f in frames for c in f.cpu_tails if pid in c.pids]
    assert len(tails) == 1, f"{len(tails)} CpuTail messages for {pid}: {tails}"
