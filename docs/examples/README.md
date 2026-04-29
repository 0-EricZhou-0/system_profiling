# Examples

Three programs ship with the library. The two `full_system_profiling`
examples (C++ and Python) load the **same** [`configs/example.pbtxt`](../../configs/example.pbtxt)
by default, with the path resolved relative to each source file's own
directory — so running them from any working directory just works. The
GPU-only `gemm_profiling` example keeps a smaller, single-file footprint
for quick metric studies.

| Example | Language | Scope | Default config | Output files |
|---|---|---|---|---|
| [`gemm_profiling`](gemm_profiling.md) | C++ | GPU only | Hardcoded in C++; `-o` picks output file | 1 (`gpu_metrics.pb`) |
| [`full_system_profiling`](full_system_profiling.md) | C++ | GPU + CPU/Mem + Disk + Events | `configs/example.pbtxt` (resolved via `__FILE__`) | 5 (incl. `events.pb`, `session_metadata.pb`) |
| [`full_system_profiling.py`](full_system_profiling_python.md) | Python | GPU + CPU/Mem + Disk + Events | `configs/example.pbtxt` (resolved via `__file__`) | 5 (same set as above) |

## Picking one

- **GPU only, want to tweak metrics in code, minimal moving parts** → `gemm_profiling`
- **Whole-machine correlated trace, config-file driven C++** → `full_system_profiling.cu`
- **Whole-machine trace driven from a Python workload (PyTorch / JAX / serving)** → `full_system_profiling.py`

All three link against the same `libcupti_profiler.so`; the Python
example goes through the pybind11 wrapper at `python/binding.cpp`.
`ProfilerSuite` is just composition over the individual profilers in
every case.

For an example of building the config in Python without a `.pbtxt`
(via `cupti_profiler.configure_suite(suite, {...})`), see
[`tests/python/test_basic.py`](../../tests/python/test_basic.py).
