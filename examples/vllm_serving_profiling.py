"""Profile a live vLLM server, process by process.

The launcher (this script) hosts a ProfilerSuite with the System and Disk
probes in SIDECAR mode, starts `vllm serve`, and tracks the server's whole
process tree: the API server it spawns, the EngineCore that renames itself
after fork, the multiprocessing helper, and the short-lived workers of
startup (compilers, `ldconfig`, ...). The server is registered right after
Popen, before it is ready, so startup is traced too. Then it waits for
/v1/models, drives batches of mixed-length requests, stops the server, and
renders the trace with tools/visualize_all.py.

Nothing here is specific to a machine: the model, the vLLM executable, the
port, the output directory, the sampling rates and the discovery scan
interval are arguments. Where the model weights live, the CUDA libraries,
and anything else vLLM needs come from the environment you run this in
(see docs/examples/vllm_serving.md).

Run (with the library importable, e.g. after `pip install .`):
    python examples/vllm_serving_profiling.py
    python examples/vllm_serving_profiling.py --vllm /path/to/env/bin/vllm \\
        --model Qwen/Qwen3.5-0.8B --output-dir vllm_profile --load-seconds 90
Everything after `--` goes to `vllm serve` unchanged:
    python examples/vllm_serving_profiling.py -- --max-model-len 8192
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import cupti_profiler as cp

HERE = os.path.dirname(os.path.abspath(__file__))
VISUALIZER = os.path.normpath(os.path.join(HERE, "..", "tools", "visualize_all.py"))
PANELS = os.path.normpath(os.path.join(HERE, "..", "configs", "vllm_serving_panels.pbtxt"))

# GPU PM sampling metrics, when --gpu is given (device-wide counters).
GPU_METRICS = [
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.per_cycle_active",
    "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
]

# Request mix: prompt lengths (in words, ~1 token each) and output lengths.
PROMPT_WORDS = [32, 128, 512, 1536, 3000]
OUTPUT_TOKENS = [16, 64, 256]
# Concurrency of each load batch: a ramp up and down, so the per-process
# series move with the load.
BATCH_CONCURRENCY = [4, 16, 64, 64, 16, 4]
VOCAB = ("time year people way day man thing woman life child world school state family "
         "student group country problem hand part place case week company system program "
         "question work government number night point home water room mother area money "
         "story fact month lot right study book eye job word business issue side kind head "
         "house service friend father power hour game line end member law car city").split()


def physical_block_devices():
    """Whole block devices of this machine, for the per-device disk panel."""
    try:
        names = sorted(os.listdir("/sys/block"))
    except OSError:
        return []
    return [n for n in names if not n.startswith(("loop", "ram", "zram", "sr", "fd"))]


def suite_config(args):
    discovery = {"enabled": True, "direct_children_only": False,
                 "scan_interval_ms": args.scan_interval_ms}
    cfg = {
        "output_dir": args.output_dir,
        "gpu": {"enabled": False},
        "events": {"enabled": True, "flush_interval_ms": 1000, "output_file": "events.pb"},
        "system": {
            "enabled": True,
            "sampling_frequency_hz": args.system_hz,
            "flush_interval_ms": args.flush_ms,
            "output_file": "system_metrics.pb",
            "mode": 2,                           # SIDECAR
        },
        "disk": {
            "enabled": True,
            "sampling_frequency_hz": args.disk_hz,
            "flush_interval_ms": args.flush_ms,
            "output_file": "disk_metrics.pb",
            "devices": args.disk_device or physical_block_devices(),
            "mode": 2,                           # SIDECAR
        },
        "process_discovery": discovery,
    }
    if args.gpu:
        cfg["gpu"] = {
            "enabled": True,
            "device_indices": [args.gpu_device],
            "sampling_frequency_hz": args.gpu_hz,
            "hw_buffer_size": 512 * 1024 * 1024,
            "max_samples": 50000,
            "metrics": GPU_METRICS,
            "flush_interval_ms": 5000,
            "output_file": "gpu_metrics.pb",
        }
    return cfg


def http_json(url, body=None, timeout=5.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base, server, timeout_s):
    """Poll /v1/models until the server answers. Returns the model id."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"vllm serve exited with code {server.returncode} before it was ready")
        try:
            return http_json(f"{base}/v1/models", timeout=2.0)["data"][0]["id"]
        except (urllib.error.URLError, ConnectionError, OSError, KeyError, IndexError, ValueError):
            time.sleep(1.0)
    raise SystemExit(f"vllm serve not ready after {timeout_s} s")


def drive_load(base, model, seconds, tracker, seed):
    """Batches of mixed-length completions, BATCH_CONCURRENCY workers each,
    for `seconds` in total. Every request has a fresh random prompt (so the
    prefix cache does not short-cut it) and ignore_eos, so its output length
    is fixed. Each batch is a region in the event trace."""
    rng = random.Random(seed)
    per_batch = seconds / len(BATCH_CONCURRENCY)
    done = failed = 0
    for i, conc in enumerate(BATCH_CONCURRENCY):
        stop_at = time.monotonic() + per_batch
        lock = threading.Lock()
        counts = [0, 0]

        def worker(wid):
            wrng = random.Random(rng.random() + wid)
            while time.monotonic() < stop_at:
                body = {"model": model,
                        "prompt": " ".join(wrng.choices(VOCAB, k=wrng.choice(PROMPT_WORDS))),
                        "max_tokens": wrng.choice(OUTPUT_TOKENS),
                        "ignore_eos": True, "temperature": 0.0}
                ok = 1
                try:
                    http_json(f"{base}/v1/completions", body, timeout=120.0)
                except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                    ok = 0
                with lock:
                    counts[0 if ok else 1] += 1

        rid = tracker.begin_region(f"batch {i + 1}: {conc} concurrent")
        with cf.ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(worker, range(conc)))
        tracker.end_region(rid)
        done += counts[0]
        failed += counts[1]
        print(f"  batch {i + 1}/{len(BATCH_CONCURRENCY)}: concurrency {conc}, "
              f"{counts[0]} requests ok, {counts[1]} failed", flush=True)
    return done, failed


def stop_server(server, grace_s=60.0):
    """SIGTERM the server (vLLM shuts its workers down), then SIGKILL its
    whole process group if it has not exited in time."""
    if server.poll() is not None:
        return
    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def read_process_table(output_dir):
    """The trace's process table (system probe): pid -> latest
    TrackedProcessV2, plus the comm history and per-process CPU seconds
    (head + samples + exit tail)."""
    from cupti_profiler.proto import system_metrics_pb2
    from google.protobuf.internal.decoder import _DecodeVarint32

    path = os.path.join(output_dir, "system_metrics.pb")
    data = open(path, "rb").read()
    frames, pos = [], 0
    while pos < len(data):
        n, pos = _DecodeVarint32(data, pos)
        f = system_metrics_pb2.SystemMetricsTrace()
        f.ParseFromString(data[pos:pos + n])
        pos += n
        frames.append(f)

    table, samples, ticks = {}, {}, []
    col = None
    for f in frames:
        for tp in f.tracked_processes:
            table[tp.pid] = tp                    # the latest flush wins
        for s in f.scope_metric_names:
            if "proc__cycles_active.sum.per_second" in s.fqns:
                col = list(s.fqns).index("proc__cycles_active.sum.per_second")
        ticks.extend(s.timestamp_ns for s in f.system_samples)
        for s in f.process_samples:
            samples.setdefault(s.pid, []).append((s.timestamp_ns, s.values[col]))
    ticks.sort()
    cpu = {pid: tp.cpu_before_tracking_ns / 1e9 for pid, tp in table.items()}
    import bisect
    for pid, xs in samples.items():
        xs.sort()
        prev = None
        for ts, pct in xs:
            if prev is None:                      # its baseline: the tick before
                k = bisect.bisect_left(ticks, ts) - 1
                prev = ticks[k] if k >= 0 else ts
            cpu[pid] = cpu.get(pid, 0.0) + pct / 100.0 * (ts - prev) / 1e9
            prev = ts
    for f in frames:
        for t in f.cpu_tails:
            # A tail shared by several children is credited to the first.
            cpu[t.pids[0]] = cpu.get(t.pids[0], 0.0) + t.cpu_after_last_sample_ns / 1e9
    return table, cpu


def print_process_table(table, cpu, t0_ns):
    print(f"\n{'pid':>8} {'parent':>8}  {'kind':<10} {'start s':>8} {'end s':>8} {'CPU s':>8}  comm history")
    for pid, tp in sorted(table.items(), key=lambda kv: (kv[1].start_time_ns, kv[0])):
        kind = "discovered" if tp.discovered else "root"
        start = (tp.start_time_ns - t0_ns) / 1e9 if tp.start_time_ns else float("nan")
        end = f"{(tp.end_time_ns - t0_ns) / 1e9:8.1f}" if tp.end_time_ns else "   alive"
        names = " -> ".join(c.comm for c in tp.comm_history) or tp.comm
        print(f"{pid:>8} {tp.parent_pid:>8}  {kind:<10} {start:8.1f} {end} {cpu.get(pid, 0.0):8.2f}  {names}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="model to serve (default: %(default)s)")
    ap.add_argument("--vllm", default="vllm",
                    help="vllm executable, a path or a name on PATH (default: %(default)s)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--output-dir", default="vllm_serving_profile",
                    help="trace directory (default: %(default)s)")
    ap.add_argument("--png", default=None, help="figure path (default: <output-dir>/vllm_serving.png)")
    ap.add_argument("--system-hz", type=int, default=50)
    ap.add_argument("--disk-hz", type=int, default=50)
    ap.add_argument("--flush-ms", type=int, default=1000)
    ap.add_argument("--scan-interval-ms", type=int, default=100,
                    help="descendant discovery scan interval (default: %(default)s)")
    ap.add_argument("--disk-device", action="append",
                    help="block device for the per-device disk panel; repeatable "
                         "(default: every whole block device in /sys/block)")
    ap.add_argument("--gpu", action="store_true",
                    help="also run GPU PM sampling in this launcher (device-wide counters)")
    ap.add_argument("--gpu-device", type=int, default=0)
    ap.add_argument("--gpu-hz", type=int, default=500)
    ap.add_argument("--load-seconds", type=float, default=90.0,
                    help="duration of the request load, split into %d batches (default: %%(default)s)"
                         % len(BATCH_CONCURRENCY))
    ap.add_argument("--ready-timeout", type=float, default=900.0,
                    help="seconds to wait for /v1/models (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="skip visualize_all.py")
    ap.add_argument("vllm_args", nargs="*", help="after `--`: extra arguments for `vllm serve`")
    args = ap.parse_args()

    vllm = shutil.which(args.vllm) or args.vllm
    os.makedirs(args.output_dir, exist_ok=True)
    base = f"http://{args.host}:{args.port}"

    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, suite_config(args))
    suite.start()
    tracker = suite.get_event_profiler().get_generic_tracker()
    t0_ns = time.monotonic_ns()   # the trace clock (steady_clock)

    cmd = [vllm, "serve", args.model, "--host", args.host, "--port", str(args.port),
           "--seed", str(args.seed), *args.vllm_args]
    log_path = os.path.join(args.output_dir, "vllm.log")
    print("launching:", " ".join(cmd), f"(log: {log_path})", flush=True)
    server = None
    try:
        with open(log_path, "w") as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Right away, before the server is ready: startup's short-lived
        # workers are children of this tree for only seconds.
        suite.add_tracked_process(server.pid, "vllm", track_descendants=True)
        tracker.mark_event("vllm serve spawned")
        startup = tracker.begin_region("startup")
        model = wait_ready(base, server, args.ready_timeout)
        tracker.end_region(startup)
        tracker.mark_event("server ready")
        print(f"ready after {(time.monotonic_ns() - t0_ns) / 1e9:.1f} s; "
              f"load for {args.load_seconds:.0f} s", flush=True)
        ok, failed = drive_load(base, model, args.load_seconds, tracker, args.seed)
        print(f"load done: {ok} requests ok, {failed} failed", flush=True)
        time.sleep(2.0)                           # an idle stretch before shutdown
        tracker.mark_event("shutdown")
    finally:
        if server is not None:
            stop_server(server)
            time.sleep(0.5)                       # let the probes see the exits
        suite.stop()

    table, cpu = read_process_table(args.output_dir)
    print_process_table(table, cpu, t0_ns)

    if not args.no_render and not os.path.exists(VISUALIZER):
        print(f"not rendering: {VISUALIZER} not found (run from a checkout of the repository)")
    elif not args.no_render:
        png = args.png or os.path.join(args.output_dir, "vllm_serving.png")
        subprocess.run([sys.executable, VISUALIZER,
                        os.path.join(args.output_dir, "session_metadata.pb"), "-o", png,
                        "--panel-layout", PANELS], check=True)
        print("figure:", png)


if __name__ == "__main__":
    main()
