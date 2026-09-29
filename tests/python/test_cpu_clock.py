"""Per-PID CPU accounting (proc__cycles_active.sum.per_second).

Drives the SystemProfiler through ProfilerSuite against a child process
with a known CPU workload and compares the trace with the child's own
ground truth. The child blocks on stdin until the profiler has seeded
its baseline, so the trace covers the whole workload.

Granularity: another process's running threads are credited CPU time
at scheduler ticks (CONFIG_HZ; 4 ms at 250 Hz) and context switches,
so one 10 ms sample can be off by up to a tick per running thread.
Nothing is lost — it lands in the next sample — so the assertions are
on sums over many samples, and single samples are bounded by one tick.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import time

import cupti_profiler as cp
import metric_catalog_pb2
import system_metrics_pb2

CPU_FQN = "proc__cycles_active.sum.per_second"
HZ = 100
# Largest scheduler tick we allow for (CONFIG_HZ=100); 250 Hz is 4 ms.
MAX_KERNEL_TICK_NS = 10_000_000

# Child preamble: wait for "go", then run BODY, which must set `result`.
# The JSON line is printed after the workload, then the child idles for
# TAIL_S so the profiler takes a few more ticks before it exits.
_CHILD = """
import json, os, sys, threading, time
sys.stdin.readline()
t0 = os.times(); w0 = time.monotonic_ns()
{body}
t1 = os.times(); w1 = time.monotonic_ns()
result.update(cpu_s=(t1.user + t1.system) - (t0.user + t0.system),
              start_ns=w0, end_ns=w1)
print(json.dumps(result), flush=True)
time.sleep({tail})
"""


def _read_frames(path, cls):
    data = path.read_bytes()
    i = 0
    while i < len(data):
        n = shift = 0
        while True:
            b = data[i]
            i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        msg = cls()
        msg.ParseFromString(data[i:i + n])
        i += n
        yield msg


def _process_cpu(path, pid):
    """Return (ts_ns, cpu_pct, dt_ns) for every sample of `pid`, in order.

    dt_ns is the interval the sample's cpu_pct was computed over: the
    previous sample of the same PID, or — for the first sample — the
    system tick that seeded the PID's baseline (same sample-loop
    iteration, same timestamp).
    """
    frames = list(_read_frames(path, system_metrics_pb2.SystemMetricsTrace))
    fqns = {}
    for f in frames:
        for s in f.scope_metric_names:
            fqns[s.scope] = list(s.fqns)
    col = fqns[metric_catalog_pb2.SCOPE_PROCESS].index(CPU_FQN)
    sys_ts = sorted(s.timestamp_ns for f in frames for s in f.system_samples)
    samples = sorted((s.timestamp_ns, s.values[col])
                     for f in frames for s in f.process_samples if s.pid == pid)
    out = []
    prev = None
    for ts, pct in samples:
        if prev is None:
            prev = max(t for t in sys_ts if t < ts)
        out.append((ts, pct, ts - prev))
        prev = ts
    return out


def _profile_child(tmp_path, body, tail_s=0.3, after_exit_s=0.1):
    """Run BODY in a child tracked by the SystemProfiler at HZ.

    Returns (child result dict, per-sample list from _process_cpu, child pid).
    """
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(body=textwrap.dedent(body), tail=tail_s)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    suite = cp.ProfilerSuite()
    try:
        cp.configure_suite(suite, {
            "output_dir": str(tmp_path),
            "gpu":    {"enabled": False},
            "disk":   {"enabled": False},
            "events": {"enabled": False},
            "system": {
                "enabled": True,
                "sampling_frequency_hz": HZ,
                "flush_interval_ms": 200,
                "output_file": "system_metrics.pb",
                "processes": [{"pid": child.pid, "alias": "child"}],
            },
        })
        suite.start()
        time.sleep(0.1)                 # several ticks: baseline is seeded
        child.stdin.write("go\n")
        child.stdin.flush()
        line = child.stdout.readline()
        child.wait(timeout=60)
        time.sleep(after_exit_s)
    finally:
        suite.stop()
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 0, f"child failed rc={child.returncode}"
    result = json.loads(line)
    return result, _process_cpu(tmp_path / "system_metrics.pb", child.pid), child.pid


def _traced_cpu_s(samples):
    return sum(pct / 100.0 * dt / 1e9 for _, pct, dt in samples)


def test_cpu_counts_exited_threads(tmp_path):
    # 300 threads run one after another, each burning 2 ms of its own
    # CPU and exiting — well under one 10 ms tick, so most of them are
    # born and gone between two samples and never appear in
    # /proc/<pid>/task. Only the process CPU clock keeps their time.
    result, samples, _ = _profile_child(tmp_path, """
        def burn():
            end = time.thread_time() + 0.002
            while time.thread_time() < end:
                pass
        for _ in range(300):
            t = threading.Thread(target=burn)
            t.start()
            t.join()
        result = {}
    """)
    traced_s = _traced_cpu_s(samples)
    truth_s = result["cpu_s"]
    print(f"trace {traced_s:.3f} s vs ground truth {truth_s:.3f} s")
    # os.times() is CLK_TCK-quantized (10 ms per reading, two readings),
    # and the trace also covers the child's idle wait before and after
    # the workload. 5% + 30 ms bounds that; observed error is 0-20 ms.
    # The per-thread schedstat walk this replaced reported 0.05-0.13 s
    # against ~0.63 s here.
    assert truth_s > 0.5, f"workload too small to be meaningful: {truth_s:.3f} s"
    assert abs(traced_s - truth_s) <= 0.05 * truth_s + 0.03, \
        f"trace {traced_s:.3f} s vs ground truth {truth_s:.3f} s"


def test_cpu_multicore(tmp_path):
    # hashlib releases the GIL for large buffers, so four threads run
    # truly in parallel.
    result, samples, _ = _profile_child(tmp_path, """
        import hashlib
        buf = b"x" * (1 << 20)
        def spin():
            h = hashlib.sha256()
            end = time.monotonic() + 2.0
            while time.monotonic() < end:
                h.update(buf)
        ts = [threading.Thread(target=spin) for _ in range(4)]
        for t in ts: t.start()
        for t in ts: t.join()
        result = {}
    """)
    # Samples whose whole interval lies inside the busy window, trimming
    # thread start-up and wind-down. Individual samples swing by a tick
    # per thread (330% / 446% alternating at HZ=250), so assert on the
    # time-weighted average across the window.
    lo = result["start_ns"] + 50_000_000
    hi = result["end_ns"] - 50_000_000
    busy = [(pct, dt) for ts, pct, dt in samples if ts - dt >= lo and ts <= hi]
    assert len(busy) >= 100, f"only {len(busy)} samples in the busy window"
    busy_ns = sum(dt for _, dt in busy)
    avg = sum(pct * dt for pct, dt in busy) / busy_ns
    traced_s = _traced_cpu_s(samples)
    print(f"busy-window average {avg:.1f}% over {len(busy)} samples; "
          f"trace {traced_s:.3f} s vs ground truth {result['cpu_s']:.3f} s")
    # Four saturated cores = 400% of one core. Tick quantization can move
    # at most one tick per thread across each window edge — 2 x 4 x 10 ms
    # over ~1.9 s is 4% even at HZ=100 — and the job's 8 CPUs keep the
    # threads from being starved. Hence 400% +/- 5%.
    assert 380 <= avg <= 420, f"busy-window average {avg:.1f}%"
    # Whole-run ground truth: ~8 CPU-seconds, and the trace agrees.
    assert abs(traced_s - result["cpu_s"]) <= 0.05 * result["cpu_s"] + 0.03, \
        f"trace {traced_s:.3f} s vs ground truth {result['cpu_s']:.3f} s"


def test_cpu_pid_exits_midrun(tmp_path):
    # The child burns CPU briefly and exits (and is reaped) while the
    # profiler keeps running for several more ticks.
    result, samples, pid = _profile_child(tmp_path, """
        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            pass
        result = {}
    """, tail_s=0.0, after_exit_s=0.5)
    assert samples, "no samples for the child at all"
    ncpu = os.cpu_count()
    for ts, pct, dt in samples:
        assert math.isfinite(pct) and 0.0 <= pct <= 100.0 * ncpu, \
            f"absurd cpu_pct {pct} at {ts}"
        # Single-threaded child: one core, plus at most one scheduler
        # tick credited late into this sample (116% seen at HZ=250).
        limit = 100.0 * (dt + MAX_KERNEL_TICK_NS) / dt
        assert pct <= limit, f"cpu_pct {pct:.1f} > {limit:.1f} for a single-threaded child"
    # Once the process is gone its CPU clock cannot be read, and it is
    # skipped — no fabricated samples for the 0.5 s the profiler keeps
    # running after it exited. 150 ms covers interpreter teardown and a
    # tick or two.
    last_ts = samples[-1][0]
    assert last_ts <= result["end_ns"] + 150_000_000, \
        f"sample {(last_ts - result['end_ns']) / 1e6:.0f} ms after the child exited"
    traced_s = _traced_cpu_s(samples)
    print(f"trace {traced_s:.3f} s vs ground truth {result['cpu_s']:.3f} s; "
          f"last sample {(last_ts - result['end_ns']) / 1e6:.0f} ms after workload end")
    assert abs(traced_s - result["cpu_s"]) <= 0.05 * result["cpu_s"] + 0.03, \
        f"trace {traced_s:.3f} s vs ground truth {result['cpu_s']:.3f} s"
