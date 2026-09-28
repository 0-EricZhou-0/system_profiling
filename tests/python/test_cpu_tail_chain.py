"""CPU reap chains: a tracked process P reaps its busy tracked child C and
exits at once, and the tracked G reaps P at once, all within one sample
interval (a `sh -c` running a compiler behaves like this).

G's cutime then grows by P's whole CPU *including C's*, which P had
reaped; C's CPU up to its last sample is already in C's own samples, so
P's tail must not count it again. Whether P reaped C (rather than exiting
first and leaving it to a subreaper or init) is the reap-chain rule the
disk probe uses: with the host's orphan reaper (adopt_orphans()) and the
tree under the host, P reaped C and C's CPU is taken out of P's tail
(CpuTail.chain_pids); otherwise C is listed in CpuTail.ambiguous_pids and
nothing is guessed.

The kernel's total for the tree is G's own CPU plus its reaped children's
(os.times()). PR_SET_CHILD_SUBREAPER cannot be undone, so every run
happens in a fresh launcher process.
"""

import json
import subprocess

import pytest

import metric_catalog_pb2
from tracing_helpers import MODES as MODE_IDS
from tracing_helpers import PY, suite_config, system_frames, tracked

MODES = list(MODE_IDS)
HZ = 5            # a 200 ms interval: P's reap-and-exit (~1 ms) lands inside one
ON = {"enabled": True, "scan_interval_ms": 10}
CPU_FQN = "proc__cycles_active.sum.per_second"
C_SECONDS = 1.0

G_CODE = r"""
import json, os, sys, time
def emit(**kw):
    print(json.dumps(kw), flush=True)
emit(ready=os.getpid())
sys.stdin.readline()
p = os.fork()
if p == 0:
    time.sleep(0.5)                  # P: discovered and read before C exists
    c = os.fork()
    if c == 0:
        time.sleep(0.5)              # C: discovered and read, then burns CPU
        end = time.process_time() + %f
        while time.process_time() < end:
            pass
        os._exit(0)
    os.waitpid(c, 0)                 # reap C, then exit at once
    os._exit(0)
os.waitpid(p, 0)                     # reap P at once
t = os.times()
emit(done=1, p=p, c=None, kernel_s=t.user + t.system + t.children_user + t.children_system)
sys.stdin.readline()
""" % C_SECONDS

LAUNCHER = r"""
import json, os, subprocess, sys, time
import cupti_profiler as cp
cfg, g_code, adopt = json.loads(sys.argv[1])
if adopt:
    cp.adopt_orphans()
suite = cp.ProfilerSuite()
cp.configure_suite(suite, cfg)
suite.start()
proc = subprocess.Popen([sys.executable, "-c", g_code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def read(key):
    while True:
        msg = json.loads(proc.stdout.readline())
        if key in msg:
            return msg
g = read("ready")["ready"]
suite.add_tracked_process(g, "g", track_descendants=True)
time.sleep(0.8)
proc.stdin.write(b"go\n"); proc.stdin.flush()
done = read("done")
time.sleep(1.0)
suite.stop()
proc.stdin.write(b"exit\n"); proc.stdin.flush()
proc.wait(timeout=10)
print("RESULT " + json.dumps({"g": g, "done": done}), flush=True)
"""


def run(tmp_path, mode, adopt):
    cfg = suite_config(tmp_path, mode, discovery=ON, hz=HZ, flush_ms=200)
    r = subprocess.run([PY, "-c", LAUNCHER, json.dumps([cfg, G_CODE, adopt])],
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stderr[-4000:]
    res = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))[7:])
    return res, system_frames(tmp_path)


def trace_cpu(frames, pids):
    """head + samples + tails (credited to their first pid) of `pids`, s."""
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    col = fqns[metric_catalog_pb2.SCOPE_PROCESS].index(CPU_FQN)
    sys_ts = sorted(s.timestamp_ns for f in frames for s in f.system_samples)
    table = tracked(frames)
    total = 0.0
    for pid in pids:
        total += table[pid].cpu_before_tracking_ns / 1e9
        rows = sorted((s.timestamp_ns, s.values[col]) for f in frames for s in f.process_samples
                      if s.pid == pid)
        prev = None
        for ts, pct in rows:
            if prev is None:
                prev = max(t for t in sys_ts if t < ts)
            total += pct / 100.0 * (ts - prev) / 1e9
            prev = ts
    tails = [c for f in frames for c in f.cpu_tails if c.pids and c.pids[0] in pids]
    return total + sum(c.cpu_after_last_sample_ns for c in tails) / 1e9, tails


def descendants(frames, g):
    table = tracked(frames)
    out, todo = [g], [g]
    while todo:
        x = todo.pop()
        kids = [p for p, tp in table.items() if tp.parent_pid == x and tp.discovered]
        out += kids
        todo += kids
    return out


@pytest.mark.parametrize("mode", MODES)
def test_chain_not_counted_twice(tmp_path, mode):
    res, frames = run(tmp_path, mode, adopt=True)
    g, p = res["g"], res["done"]["p"]
    pids = descendants(frames, g)
    assert p in pids and len(pids) == 3, pids            # G, P and C were all traced
    c = next(x for x in pids if x not in (g, p))
    got, tails = trace_cpu(frames, pids)
    kernel = res["done"]["kernel_s"]
    ptail = [t for t in tails if p in t.pids]
    assert ptail and c in ptail[0].chain_pids, [(list(t.pids), list(t.chain_pids)) for t in tails]
    # Kernel: G + P + C. Tolerance as test_cpu_head_tail: 40 ms + 2%.
    assert abs(got - kernel) <= 0.04 + 0.02 * kernel, (got, kernel)


@pytest.mark.parametrize("mode", MODES)
def test_chain_without_reaper_is_ambiguous(tmp_path, mode):
    res, frames = run(tmp_path, mode, adopt=False)
    g, p = res["g"], res["done"]["p"]
    pids = descendants(frames, g)
    c = next(x for x in pids if x not in (g, p))
    got, tails = trace_cpu(frames, pids)
    ptail = [t for t in tails if p in t.pids]
    assert ptail and c in ptail[0].ambiguous_pids and not ptail[0].chain_pids, \
        [(list(t.pids), list(t.chain_pids), list(t.ambiguous_pids)) for t in tails]
    # Not subtracted: C's CPU is in P's tail too (documented, flagged).
    assert got - res["done"]["kernel_s"] > 0.8 * C_SECONDS
