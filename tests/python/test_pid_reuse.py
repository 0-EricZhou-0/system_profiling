"""A PID number that is reused names a different process.

The kernel only reuses a number after the whole PID space wraps, so the
test makes it happen on demand: it runs the suite inside a new user +
PID namespace (`unshare -Urpf --mount-proc`, unprivileged), where it may
write /proc/sys/kernel/ns_last_pid, and so can hand a dead process's
number to the next process it spawns.

Tracked process A burns 2 s of CPU, sleeps, and is killed. Its number is given
to B, which spins at 100% of a core:

  * not re-added: B must never be sampled — A's entry was removed with
    its end time, and no entry names B;
  * re-added right away (while A's removal has not been flushed yet):
    B gets its own entry, registered once A's is gone, with its own
    start time, and its first samples are its own CPU — not deltas
    against A's last baseline.
"""

import json
import os
import subprocess
import sys

import pytest

from tracing_helpers import MODES, sample_times, system_frames

HERE = os.path.dirname(os.path.abspath(__file__))
UNSHARE = ["unshare", "-Urpf", "--mount-proc"]
MS = 1_000_000

# Runs as PID 1 of the new namespace. argv: outdir mode readd
HELPER = r"""
import json, os, subprocess, sys, time
sys.path.insert(0, sys.argv[4])
from tracing_helpers import running_suite
outdir, mode, readd = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
PY = sys.executable
SPIN = "while True: pass"
BURN = ("import time\nt = time.process_time()\n"
        "while time.process_time() - t < 2.0: pass\ntime.sleep(60)")
res = {}
# Flushes 1 s apart: A's removal is still unflushed when B is re-added.
with running_suite(outdir, mode, hz=100, flush_ms=1000) as suite:
    a = subprocess.Popen([PY, "-c", BURN])
    suite.add_tracked_process(a.pid, "A")
    time.sleep(2.5)                            # A burned 2 s, now sleeps
    # A may be gone (and seen gone by the probe) any time after kill():
    # a_kill bounds its end time from below, a_dead (reaped) from above.
    res["a_kill"] = time.monotonic_ns()
    a.kill(); a.wait()
    res["a"] = a.pid
    res["a_dead"] = time.monotonic_ns()
    for _ in range(5):
        with open("/proc/sys/kernel/ns_last_pid", "w") as f:
            f.write(str(a.pid - 1))
        res["b_spawn"] = time.monotonic_ns()
        b = subprocess.Popen([PY, "-c", SPIN])
        if b.pid == a.pid:
            break
        b.kill(); b.wait()
    res["b"] = b.pid
    if readd:
        # B may be sampled (as B) as soon as the add reaches the probe,
        # before this call returns (SIDECAR: the sidecar acks, then
        # samples): readd_call bounds "nobody's number" from above.
        res["readd_call"] = time.monotonic_ns()
        suite.add_tracked_process(b.pid, "B")
        res["readd"] = time.monotonic_ns()
    time.sleep(2.5)            # B's entry may wait up to one flush (1 s)
    b.kill(); b.wait()
print("RESULT " + json.dumps(res), flush=True)
"""


def namespaces_available():
    try:
        return subprocess.run(UNSHARE + ["true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_in_namespace(tmp_path, mode, readd):
    if not namespaces_available():
        pytest.skip("unprivileged user + PID namespaces are not available here")
    p = subprocess.run(UNSHARE + [sys.executable, "-c", HELPER, str(tmp_path), mode,
                                  "1" if readd else "0", HERE],
                       capture_output=True, text=True, timeout=60)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert p.returncode == 0 and line, f"rc {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    res = json.loads(line[len("RESULT "):])
    assert res["b"] == res["a"], f"premise: B did not get A's number: {res}"
    return res


def entries(frames, pid):
    """Distinct registrations of pid, in order, each as its last row."""
    out = []
    for f in frames:
        for tp in f.tracked_processes:
            if tp.pid != pid:
                continue
            if out and out[-1].label == tp.label and out[-1].start_time_ns == tp.start_time_ns:
                out[-1] = tp
            else:
                out.append(tp)
    return out


def cpu_samples(frames, pid):
    col = next(list(s.fqns).index("proc__cycles_active.sum.per_second")
               for f in frames for s in f.scope_metric_names
               if "proc__cycles_active.sum.per_second" in s.fqns)
    return sorted((s.timestamp_ns, s.values[col]) for f in frames for s in f.process_samples
                  if s.pid == pid)


@pytest.mark.parametrize("mode", list(MODES))
def test_reused_pid_not_sampled_under_old_entry(tmp_path, mode):
    res = run_in_namespace(tmp_path, mode, readd=False)
    frames = system_frames(tmp_path)
    regs = entries(frames, res["a"])
    assert len(regs) == 1 and regs[0].label == "A", [(r.label, r.removed) for r in regs]
    a = regs[0]
    assert a.removed and a.end_time_ns > 0, a
    # A was dead and reaped at a_dead (B got its number right after); the
    # probe notices at its next tick, so A's end time may be a little
    # later — but no sample of the number may be from after a_dead.
    ts = sample_times(frames, res["a"])
    assert ts, "A never sampled"
    after = [t for t in ts if t > res["a_dead"]]
    assert not after, f"{len(after)} samples of the reused number after A died"
    assert a.end_time_ns - res["a_dead"] <= 50 * MS


@pytest.mark.parametrize("mode", list(MODES))
def test_reused_pid_readded_is_a_new_entry(tmp_path, mode):
    res = run_in_namespace(tmp_path, mode, readd=True)
    frames = system_frames(tmp_path)
    regs = entries(frames, res["a"])
    assert [r.label for r in regs] == ["A", "B"], [(r.label, r.removed, r.start_time_ns)
                                                   for r in regs]
    a, b = regs
    assert a.removed and res["a_kill"] < a.end_time_ns <= res["a_dead"] + 50 * MS, a
    assert res["b_spawn"] - 10 * MS <= b.start_time_ns <= res["readd"] + 10 * MS, \
        (b.start_time_ns - res["b_spawn"]) / MS
    # Between A's death and the call that registers B, the number is nobody's.
    gap = [t for t in sample_times(frames, res["a"]) if res["a_dead"] < t <= res["readd_call"]]
    assert not gap, f"{len(gap)} samples between A's death and B's registration"
    # One number, two processes, never listed in the same flush.
    for f in frames:
        assert sum(tp.pid == res["a"] for tp in f.tracked_processes) <= 1
    # B spins from birth: every sample of it is ~100% of a core. A delta
    # against A's last baseline (2 s of CPU, more than B can have used
    # before it is registered) would read 0.
    b_samples = [(ts, v) for ts, v in cpu_samples(frames, res["a"]) if ts > res["readd"]]
    assert len(b_samples) >= 50, len(b_samples)
    low = [(round((ts - res["b_spawn"]) / MS), round(v)) for ts, v in b_samples if v < 50]
    print(f"{mode}: B sampled {len(b_samples)} times; first "
          f"{[round(v) for _, v in b_samples[:5]]} % of a core")
    assert not low, f"B samples below 50% (ms after spawn, %): {low[:10]}"
