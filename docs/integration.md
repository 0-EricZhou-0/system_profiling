# Using cupti-profiler from another project

This page covers how a sibling project should depend on this repository.
The recommended approach for most users is **#1: submodule + `pip install -e .`**;
the others are listed for completeness.

## 1. Submodule + `pip install -e .` (recommended)

Add this repo as a git submodule and install it as an editable Python
package into the consumer's active environment. After install, anything
running in that env can `import cupti_profiler` from any working
directory — no `PYTHONPATH`, no `LD_LIBRARY_PATH`.

```bash
# ---------- one-time setup in the consuming repo ----------
git submodule add git@…/cupti-profiler third_party/cupti-profiler
git submodule update --init --recursive

# Activate whatever Python env the consumer project uses:
conda activate my-project           # or: source .venv/bin/activate

# Install build-time deps (only the editable install needs these — a
# regular install pulls them in via build isolation automatically):
pip install scikit-build-core pybind11 pybind11-stubgen grpcio-tools

# Install cupti_profiler into the env. -e = editable, so re-running
# cmake --build inside the submodule picks up local edits without
# reinstalling. Drop -e for a one-shot install.
pip install -e third_party/cupti-profiler --no-build-isolation
```

Now in any Python file or notebook in the consumer project:

```python
import cupti_profiler as cp

suite = cp.ProfilerSuite()
cp.configure_suite(suite, {
    "output_dir": "my_run/",
    "events": {"enabled": True, "flush_interval_ms": 200,
               "output_file": "events.pb"},
    "gpu":    {"enabled": True, "device_index": 0,
               "sampling_frequency_hz": 10_000,
               "metrics": ["sm__cycles_active.avg",
                           "sm__cycles_elapsed.avg"],
               "output_file": "gpu_metrics.pb"},
    # ... system / disk if you want them ...
})
suite.start()
# ... your workload, with begin_region / end_region / mark_event ...
suite.stop()
```

Reading the `.pb` files back uses the protobuf modules that ship inside
the package:

```python
from cupti_profiler.proto import events_pb2, session_metadata_pb2
```

### What gets installed

`pip install` lands the following inside `<env>/lib/python*/site-packages/cupti_profiler/`:

```
cupti_profiler/
├── __init__.py             pure-Python wrapper (configure_suite, etc.)
├── __init__.pyi            type stubs forwarding parameter names + docstrings
├── _stream.py              CudaStream context manager
├── _stream.pyi
├── _native.cpython-…so     pybind11 extension (the C++ binding)
├── _native.pyi             stubs introspected from the extension
├── libcupti_profiler.so    bundled — _native.so finds it via $ORIGIN rpath
└── proto/
    ├── __init__.py
    ├── events_pb2.py
    ├── session_metadata_pb2.py
    ├── profiler_config_pb2.py
    ├── gpu_metrics_pb2.py
    ├── system_metrics_pb2.py
    └── disk_metrics_pb2.py
```

### Editing the submodule

Editable install (`-e`) means the package's import location points at
the install tree, but C++ source changes still need a rebuild:

```bash
# After editing a .cpp file in third_party/cupti-profiler/:
pip install -e third_party/cupti-profiler --no-build-isolation
# (or just rerun cmake --build build inside the submodule and reinstall)
```

Pure-Python edits inside `python/cupti_profiler/*.py` likewise require
a reinstall, because scikit-build-core copies them into site-packages
at install time rather than redirecting via `.pth`. If you want live
edits to the pure-Python side, work directly in the submodule and use
the dev workflow (#3 below) instead of `pip install -e .`.

### Updating the submodule

```bash
cd third_party/cupti-profiler
git fetch && git checkout <ref>     # or `git pull` on the desired branch
cd ../..
pip install -e third_party/cupti-profiler --no-build-isolation
```

## 2. Build a wheel and `pip install` it (for cross-machine deployment)

If consumers run on different machines and shouldn't rebuild from
source, build a wheel once and distribute it:

```bash
cd third_party/cupti-profiler
pip install build
python -m build --wheel
# → dist/cupti_profiler-0.1.0-cp311-cp311-linux_x86_64.whl

# On the consumer machine (matching Python ABI + CUDA driver):
pip install cupti_profiler-…whl
```

Same internal layout as #1, but the build only happens once. Multi-CUDA
or multi-Python support means producing one wheel per target.

## 3. Submodule + `PYTHONPATH` (no install — for hacking on the wrapper itself)

If you're actively iterating on the C++ binding or the pure-Python
shim, the lowest-friction loop is the dev workflow this repo uses:

```bash
cd third_party/cupti-profiler
cmake -S . -B build
cmake --build build -j

# In the consumer project, prepend the build's staged package dir to
# PYTHONPATH (no install needed):
PYTHONPATH=third_party/cupti-profiler/build/python:$PYTHONPATH \
    python my_script.py
```

Pros: every `cmake --build build` is reflected immediately, no
`pip install` round-trip. Cons: every consumer of this snapshot needs
the same `PYTHONPATH` shim.

## Picking one

| Situation | Use |
|---|---|
| Consumer is another Python project on the same host, same env | **#1 submodule + `pip install -e .`** |
| Consumer runs on different machines / CI / Docker images | **#2 build a wheel** |
| You're actively developing the wrapper itself | **#3 PYTHONPATH** |

For C++ consumers (sibling projects that link `libcupti_profiler.so`
directly), use CMake `find_package` after `cmake --install` — orthogonal
to the Python story above; ask if you need a worked example.
