# Tools

Auxiliary command-line utilities that ship alongside `libcupti_profiler.so`.
The C++ tool lives under [`tools/src/`](../../tools/src/) and is built by
the same CMake invocation that builds the library; the three Python
visualizers live directly under [`tools/`](../../tools/) and are used to
read the `.pb` files the runtime emits.

| Tool | Purpose | Best for |
|---|---|---|
| [`list_pm_metrics`](#list_pm_metrics) | Enumerate CUPTI PM-samplable metrics on the local GPU | Picking metric names for `gpu` config blocks |
| [`visualize_single.py`](#visualize_singlepy) | Static PNG of a single `gpu_metrics.pb` | GPU-only studies (kernel micro-benchmarks, metric exploration) |
| [`visualize_all.py`](#visualize_allpy) | Static PNG of a full-system run (`session_metadata.pb` driven) | Sharing a snapshot, embedding in slides / reports |
| [`visualize_interactive.py`](#visualize_interactivepy) | Interactive Bokeh HTML of the same data, with built-in HTTP server | Drill-down hover, zoom, range-select; remote viewing over SSH tunnel |

## `list_pm_metrics`

Source: [`tools/src/list_pm_metrics/list_pm_metrics.cpp`](../../tools/src/list_pm_metrics/list_pm_metrics.cpp)

Enumerates the CUPTI metric catalog for **PM Sampling** on a target
device. The PM-sampling subset is narrower than the full metric list
shown by `ncu --query-metrics` — it only includes counters that fit in
one pass and that the PM unit can stream at high rate.

```bash
./build/tools/src/list_pm_metrics/list_pm_metrics            # device 0, all metrics
./build/tools/src/list_pm_metrics/list_pm_metrics -d 1       # device 1
./build/tools/src/list_pm_metrics/list_pm_metrics dram       # filter by substring
./build/tools/src/list_pm_metrics/list_pm_metrics --sub      # also print sub-metrics
```

Use it to discover valid metric names before adding `metrics:` entries
to a `.pbtxt` config or to the `metrics` field on a Python
`GpuProfilerConfig`.

## `visualize_single.py`

GPU-only renderer. Takes one `gpu_metrics.pb` and emits a static PNG
with the GPU panels (SM utilization, warps/cycle, DRAM, PCIe, NVLink).
Pairs with the [`gemm_profiling`](../examples/gemm_profiling.md) example.

```bash
python tools/visualize_single.py -i gpu_metrics.pb -o gpu_metrics.png
```

No system/disk/event panels; for those use `visualize_all.py` or
`visualize_interactive.py` against a `session_metadata.pb` instead.

## `visualize_all.py`

Full-system static renderer. Takes a `session_metadata.pb` and
auto-discovers each per-probe file from the manifest's `probes` list.
Produces a single tall PNG with every panel matching the metric and
event probes that were enabled.

```bash
python tools/visualize_all.py profiling_output/session_metadata.pb \
    -o full_profile.png
```

Panels (auto-skip when the corresponding probe is disabled):
event timeline strip → region timeline strip → GPU SM Util →
Active Warps / Cycle → DRAM → PCIe → Cumulative PCIe → NVLink →
CPU (system + per-PID) → Memory (system + per-PID) →
Disk (per-device + per-PID + queue depth) → Flush-rate panels →
Write-rate footer.

Many layout knobs are exposed as constants near the top of the file
(`PANEL_HEIGHT_*`, `SPACING_*`, `FIG_*`, `YLABEL_*`,
`SPACING_TITLE_FORST_PANEL`).

## `visualize_interactive.py`

Bokeh-based interactive renderer, same input contract as
`visualize_all.py`. Output is a single self-contained HTML file (~2 MB
with BokehJS bundled inline) plus a built-in HTTP server.

```bash
# Build + serve on http://localhost:8000
python tools/visualize_interactive.py profiling_output/session_metadata.pb

# Custom port + custom output file path:
python tools/visualize_interactive.py session_metadata.pb \
    -o /tmp/profile.html --port 9000

# Build only — don't host:
python tools/visualize_interactive.py session_metadata.pb --no-serve

# Bind to localhost only (default is 0.0.0.0 = all interfaces):
python tools/visualize_interactive.py session_metadata.pb --host 127.0.0.1
```

What you get:

- **Synced pan + wheel-zoom** across every panel via a shared X axis
  (mouse wheel = zoom time on whichever panel the cursor is over).
- **Synced dashed crosshair**: a vertical gray line follows the
  cursor's horizontal position and is mirrored on every panel.
- **Combined hover tooltips** anchored at the bottom edge of each
  plot — one popup per panel listing every co-plotted series' value
  at the cursor's x. Triggers regardless of which legend entries you
  hide.
- **Click-to-hide legend entries** (legends are docked to the right
  of each panel so they don't cover glyphs).
- **Y-axis clamps** with dashed reference lines at the theoretical
  peak (100% SM Util, 64 active warps, peak DRAM / PCIe / NVLink BW,
  install RAM total).
- **X-axis clamp**: pan/zoom is bounded to `[0, xmax + 0.8 × duration]`
  so you can scroll past the data tail but not into infinity.

Panel set is identical to `visualize_all.py` — see the table in that
section.

### Viewing from a remote server

The script binds `--host 0.0.0.0` by default, but the easiest way to
view it from a laptop SSH'd into the box is local port forwarding:

```bash
# On the laptop, in a new terminal:
ssh -L 8000:localhost:8000 user@remote-host
# → open http://localhost:8000/ in your browser
```

VS Code / Cursor's "Remote - SSH" auto-detects the listening port and
forwards it; check the **Ports** panel at the bottom.

## Dependencies

All Python tools share the same dependency set, declared in the repo
root [`requirements.txt`](../../requirements.txt). Quick install:

```bash
pip install -r requirements.txt
```

`visualize_all.py` needs `numpy + matplotlib`; `visualize_interactive.py`
adds `bokeh + pandas` (a Bokeh transitive dep).
