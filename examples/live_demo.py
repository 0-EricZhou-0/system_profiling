"""Long-running demo workload for the `--live` visualizer.

Mirrors the suite-driven ProfilerSuite setup of
examples/full_system_profiling.py, but the workload is intentionally
paced — duration-timed phases (not iteration-counted) with explicit
host-side sleeps between them — so the live Bokeh page has a clearly
animated stream of activity to observe. Each phase opens a Generic-
and/or GPU-domain region so the regions strip lights up in real time.

Run together with:
    python tools/visualize_interactive.py --live \\
        profiling_output/session_metadata.pb

Default total runtime: ~50 seconds. Override with `--duration-s`.
"""

import argparse
import os
import time

import torch

import cupti_profiler as cp


# Sibling layout: examples/live_demo.py → ../configs/example.pbtxt
DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "configs", "example.pbtxt")
)


def _busy_gemm(stream, A, B, C, N: int, deadline: float):
    """Run NxN matmul on `stream` until `deadline`. Reuses pre-allocated
    flat buffers `A`/`B`/`C` reshaped to NxN."""
    a = A[: N * N].view(N, N)
    b = B[: N * N].view(N, N)
    c = C[: N * N].view(N, N)
    iters = 0
    with torch.cuda.stream(stream):
        while time.monotonic() < deadline:
            torch.matmul(a, b, out=c)
            iters += 1
    return iters


def _busy_vecadd(stream, A, B, C, vec_n: int, deadline: float):
    a = A[:vec_n]
    b = B[:vec_n]
    c = C[:vec_n]
    iters = 0
    with torch.cuda.stream(stream):
        while time.monotonic() < deadline:
            torch.add(a, b, out=c)
            iters += 1
    return iters


def run_paced_workload(gpu_tracker: cp.EventTracker,
                        host_tracker: cp.EventTracker,
                        device_index: int,
                        total_duration_s: float) -> None:
    torch.cuda.set_device(device_index)

    stream = torch.cuda.Stream(device=device_index)
    gpu_tracker.set_stream(int(stream.cuda_stream))

    setup_id = host_tracker.begin_region("workload setup")

    # Pre-allocate so phase boundaries reflect just compute + sleeps,
    # not allocator chatter. 4096x4096 = 64 MiB; vecAdd uses 256 MiB.
    max_gemm_elems = 4096 * 4096
    vec_n = 64 * 1024 * 1024
    max_elems = max(max_gemm_elems, vec_n)
    with torch.cuda.stream(stream):
        A = torch.empty(max_elems, dtype=torch.float32, device="cuda").fill_(0.5)
        B = torch.empty(max_elems, dtype=torch.float32, device="cuda").fill_(0.5)
        C = torch.empty(max_elems, dtype=torch.float32, device="cuda")

    host_tracker.end_region(setup_id)
    host_tracker.mark_event("workload begin")

    # Each step is (label, kind, duration_share, idle_after_s).
    # Shares are scaled to total_duration_s; idle_after_s holds host-side
    # time.sleep AFTER the compute, intentionally inflating the run so the
    # live visualizer shows clear flat regions between bursts.
    plan = [
        ("warmup 1024",       "gemm",   1024, 1.0, 1.0),
        ("ramp 2048",         "gemm",   2048, 1.0, 1.0),
        ("peak 4096",         "gemm",   4096, 1.5, 2.0),
        ("idle (host sleep)", "idle",      0, 0.0, 3.0),
        ("vecAdd (mem-bound)","vecadd",    0, 1.5, 1.0),
        ("mixed compute",     "gemm",   2048, 1.0, 1.0),
        ("cooldown 1024",     "gemm",   1024, 1.0, 1.0),
        ("idle 512",          "gemm",    512, 1.0, 0.0),
    ]

    total_share = sum(s for *_rest, s, _idle in
                       [(p[0], p[1], p[2], p[3], p[4]) for p in plan]) or 1.0
    total_idle  = sum(idle for *_rest, idle in plan)
    compute_budget = max(1.0, total_duration_s - total_idle)

    print(f"Plan: {len(plan)} phases, ~{compute_budget:.1f}s compute + "
          f"{total_idle:.1f}s idle, target total {total_duration_s:.0f}s")

    overall_t0 = time.monotonic()

    for label, kind, N, share, idle_after in plan:
        phase_secs = compute_budget * (share / total_share) if share > 0 else 0.0
        deadline = time.monotonic() + phase_secs

        if kind == "idle":
            # Host-side sleep wrapped in a Generic-domain region so the
            # regions strip shows a clear flat band where no GPU work is
            # happening. `idle_after` becomes the band's width.
            print(f"  {time.monotonic() - overall_t0:6.1f}s  "
                  f"[host-only sleep] {label}: {idle_after:.1f}s")
            rid = host_tracker.begin_region(label)
            time.sleep(idle_after)
            host_tracker.end_region(rid)
            continue
        if kind == "gemm" and phase_secs > 0:
            rid = gpu_tracker.begin_region(label)
            iters = _busy_gemm(stream, A, B, C, N, deadline)
            gpu_tracker.end_region(rid)
            print(f"  {time.monotonic() - overall_t0:6.1f}s  "
                  f"[GEMM N={N}] {label}: {iters} iters in ~{phase_secs:.1f}s")
        elif kind == "vecadd" and phase_secs > 0:
            rid = gpu_tracker.begin_region(label)
            iters = _busy_vecadd(stream, A, B, C, vec_n, deadline)
            gpu_tracker.end_region(rid)
            print(f"  {time.monotonic() - overall_t0:6.1f}s  "
                  f"[vecAdd]   {label}: {iters} iters in ~{phase_secs:.1f}s")

        if idle_after > 0:
            host_tracker.mark_event(f"{label} done; sleeping {idle_after:.1f}s")
            time.sleep(idle_after)

    stream.synchronize()
    del A, B, C
    torch.cuda.empty_cache()
    host_tracker.mark_event("workload end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to .pbtxt config file (default: %(default)s)")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="CUDA device index (default: %(default)s)")
    parser.add_argument("--duration-s", type=float, default=50.0,
                        help="Approximate total wall-clock runtime (default: %(default)ss). "
                             "Compute phases scale to fill (duration - host-sleep budget).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch — cannot run this example.")

    suite = cp.ProfilerSuite()
    suite.load_config(args.config)
    suite.configure()
    suite.start()
    suite.get_event_profiler().get_generic_tracker().mark_event("suite start")

    print(f"Live URL hint: in another terminal,\n"
          f"  python tools/visualize_interactive.py --live "
          f"$(dirname $(realpath -m profiling_output/session_metadata.pb))/session_metadata.pb")

    t0 = time.monotonic()
    run_paced_workload(
        suite.get_event_profiler().get_gpu_tracker(),
        suite.get_event_profiler().get_generic_tracker(),
        args.device,
        args.duration_s,
    )
    elapsed = time.monotonic() - t0
    print(f"\nWorkload completed in {elapsed:.2f} seconds")

    suite.stop()


if __name__ == "__main__":
    main()
