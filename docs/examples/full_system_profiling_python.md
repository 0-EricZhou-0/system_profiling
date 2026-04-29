# `full_system_profiling.py`

Python counterpart to [`full_system_profiling.cu`](full_system_profiling.md).
Drives the same `ProfilerSuite` end to end through the pybind11 wrapper,
loading the same `.pbtxt` config, and runs the same GEMM ramp + `vecAdd`
workload — but expressed via [`torch`](https://pytorch.org/) instead of
direct cuBLAS.

Source: [`examples/full_system_profiling.py`](../../examples/full_system_profiling.py)

## What it does

Mirrors `RunGemmWorkload` from the `.cu` example one-for-one:

1. `torch.cuda.set_device(deviceIndex)`.
2. Allocate a `torch.cuda.Stream`; pass `int(stream.cuda_stream)` (the
   raw `cudaStream_t` pointer as int) to `gpuTracker.set_stream(...)`.
3. Generic-domain region `workload setup` brackets tensor allocation.
4. Seven GEMM phases on the same stream — same N values and iteration
   counts as the C++ reference: `warmup 512` (200 iters), `ramp-up
   1024` (150), `medium 2048` (100), `peak 4096` (50), `ramp-down 2048`
   (100), `cool-down 1024` (150), `idle 512` (200). Each phase calls
   `torch.matmul(a, b, out=c)` in a loop, all bracketed with
   `gpuTracker.begin_region(label) / end_region(rid)`.
5. `vecAdd (mem-bound)` phase — 200 iterations of `torch.add(a, b, out=c)`
   on a 256 MiB working set.
6. `stream.synchronize()` at the end so all events resolve before
   `suite.stop()` runs the final flush.

Because the workload is real GPU compute on the same stream the
EventTracker watches, the emitted `events.pb` regions and the GPU PM
samples in `gpu_metrics.pb` align — the SM utilization and DRAM
bandwidth panels in the visualization should look qualitatively
identical to the C++ run.

## Profiler setup (shared `.pbtxt`)

The default config path is computed from `__file__`, so the script
picks up [`configs/example.pbtxt`](../../configs/example.pbtxt)
regardless of the current working directory:

```python
DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "configs", "example.pbtxt"))
```

The C++ side honors this through the same code path the `.cu` example
uses (`ProfilerSuite::LoadConfig` → `Impl::ApplyParsedConfig` in
`lib/src/profiler_suite.cpp`).

If you'd rather build the config in Python (no `.pbtxt`), use
`cupti_profiler.configure_suite(suite, {...})` — see
[`tests/python/test_basic.py`](../../tests/python/test_basic.py) for a
working example.

## Run

Two equivalent ways to expose `cupti_profiler` to the script —
either install it into the active env, or point `PYTHONPATH` at the
staged build directory.

```bash
# Option A: install into the active env (recommended).
# Run once; thereafter just `python examples/full_system_profiling.py`.
pip install -e . --no-build-isolation

# Option B: build-tree PYTHONPATH (no install).
PYTHONPATH=build/python:generated/proto \
    python examples/full_system_profiling.py

# Override the config:
python examples/full_system_profiling.py -c my.pbtxt

# Pick a different device:
python examples/full_system_profiling.py -d 1
```

See [`docs/integration.md`](../integration.md) for how a sibling
project should depend on this repo (submodule + `pip install -e .` is
the recommended path).

Output: five `.pb` files in the `output_dir` set by the pbtxt
(`profiling_output/` by default), interpreted relative to whatever
directory you launched the script from.

## Visualize

```bash
python tools/visualize_all.py \
    profiling_output/session_metadata.pb \
    -o profiling_output/profile.png
```

The visualizer takes only the manifest and discovers each per-probe
file from its `probes` list.

## Dependencies

Beyond what `requirements.txt` already pulls for the wrapper itself
(pybind11, protobuf, etc.), this example needs:

- **`torch`** with CUDA support matching your driver. Install via the
  appropriate index URL, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## When to use

- Wrapping the profiler around a Python workload (PyTorch / JAX
  training loop, vLLM serving, dataloader benchmark) without writing
  any C++.
- Producing a Python-driven trace that's directly comparable to the
  cuBLAS-driven C++ trace — same regions, same phases.
- Quick experimentation in a notebook: `import cupti_profiler` →
  `suite.load_config(...)` → `suite.start() / suite.stop()`.

For the C++ reference of the same workflow, see
[`full_system_profiling`](full_system_profiling.md).

## Type stubs

`build/python/cupti_profiler/{__init__,_native,_stream}.pyi` are produced
on every `cmake --build build` run (when `pybind11-stubgen` is installed
— it's listed in [`requirements.txt`](../../requirements.txt)). Add
`build/python` and `generated/proto` to your IDE's interpreter path so
autocomplete and type checking work without an install step.
