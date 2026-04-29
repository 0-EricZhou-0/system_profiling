"""Full-system profiling driven from Python via the pybind11 wrapper.

Mirror of examples/full_system_profiling.cu — same ProfilerSuite setup,
same GEMM ramp + vecAdd workload, same shared configs/example.pbtxt.
The C++ version drives cuBLAS directly; this version drives torch on a
torch-managed CUDA stream that we hand to the GPU-domain EventTracker.

Default config: ../configs/example.pbtxt (resolved relative to this file
so the script works regardless of the current working directory).

Run:
    PYTHONPATH=build/python:generated/proto \
        python examples/full_system_profiling.py
    PYTHONPATH=... python examples/full_system_profiling.py -c my.pbtxt

For an example of building the config in Python (no .pbtxt) via
cupti_profiler.configure_suite(), see tests/python/test_basic.py.
"""

import argparse
import os
import time

import torch

import cupti_profiler as cp


# Sibling layout: examples/full_system_profiling.py → ../configs/example.pbtxt
DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "configs", "example.pbtxt")
)


def run_gemm_workload(gpuTracker: cp.EventTracker,
                      genericTracker: cp.EventTracker,
                      device_index: int) -> None:
    """Mirror of RunGemmWorkload in full_system_profiling.cu, expressed via
    torch.matmul (cuBLAS under the hood) and torch.add. All compute is
    scheduled onto a torch-managed CUDA stream whose raw cudaStream_t
    handle is forwarded to the GPU-domain EventTracker via set_stream().
    """
    torch.cuda.set_device(device_index)

    # Hand torch's stream to the C++ side as an integer pointer. The
    # binding interprets the int as cudaStream_t, so begin_region /
    # end_region record cudaEvents on this exact stream.
    stream = torch.cuda.Stream(device=device_index)
    gpuTracker.set_stream(int(stream.cuda_stream))

    setup_id = genericTracker.begin_region("workload setup")

    phases = [
        ( 512, 200, "warmup 512"),
        (1024, 150, "ramp-up 1024"),
        (2048, 100, "medium 2048"),
        (4096,  50, "peak 4096"),
        (2048, 100, "ramp-down 2048"),
        (1024, 150, "cool-down 1024"),
        ( 512, 200, "idle 512"),
    ]
    max_n = max(n for n, _, _ in phases)
    max_gemm_elems = max_n * max_n           # 4096 × 4096 = 16M floats = 64 MiB

    vec_n = 64 * 1024 * 1024                  # 64M floats = 256 MiB
    max_elems = max(max_gemm_elems, vec_n)

    # Allocate three flat buffers; reshape per-phase. This mirrors the .cu
    # which mallocs once at maxBytes and reuses for every phase.
    with torch.cuda.stream(stream):
        A = torch.empty(max_elems, dtype=torch.float32, device="cuda")
        B = torch.empty(max_elems, dtype=torch.float32, device="cuda")
        C = torch.empty(max_elems, dtype=torch.float32, device="cuda")
        A.fill_(0.5)
        B.fill_(0.5)

    genericTracker.end_region(setup_id)
    genericTracker.mark_event("workload begin")

    for N, iterations, label in phases:
        print(f"  Phase: {label} ({iterations} iterations)")
        a = A[: N * N].view(N, N)
        b = B[: N * N].view(N, N)
        c = C[: N * N].view(N, N)
        rid = gpuTracker.begin_region(label)
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                torch.matmul(a, b, out=c)
        gpuTracker.end_region(rid)

    vec_iters = 200
    print(f"  Phase: vecAdd ({vec_iters} iterations, "
          f"{vec_n * 4 // (1024 * 1024)} MiB/iter)")
    a_flat = A[:vec_n]
    b_flat = B[:vec_n]
    c_flat = C[:vec_n]
    rid = gpuTracker.begin_region("vecAdd (mem-bound)")
    with torch.cuda.stream(stream):
        for _ in range(vec_iters):
            torch.add(a_flat, b_flat, out=c_flat)
    gpuTracker.end_region(rid)

    stream.synchronize()
    del A, B, C
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to .pbtxt config file (default: %(default)s)")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="CUDA device index (default: %(default)s)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch — cannot run this example.")

    suite = cp.ProfilerSuite()
    suite.load_config(args.config)
    suite.configure()
    suite.start()
    suite.get_event_profiler().get_generic_tracker().mark_event("suite start")

    t0 = time.time()
    run_gemm_workload(
        suite.get_event_profiler().get_gpu_tracker(),
        suite.get_event_profiler().get_generic_tracker(),
        args.device,
    )
    elapsed = time.time() - t0

    print(f"\nWorkload completed in {elapsed:.2f} seconds")
    print(f"Config used: {args.config}")

    suite.stop()


if __name__ == "__main__":
    main()
