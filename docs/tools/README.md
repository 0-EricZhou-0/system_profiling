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

All three Python renderers are **descriptor-driven**: they consume a
`MetricCatalog` (inlined into `session_metadata.pb` by the suite, or
loaded via `--catalog PATH`) and a `PanelLayout` pbtxt
(`configs/visualizer_panels.pbtxt` by default, override via
`--panel-layout PATH`). The catalog declares each FQN's type / unit /
peak / scope; the panel layout declares which FQN globs go on which
subplot. To add or remove panels, edit the pbtxt — no Python code
changes needed. See [`metric-model.md`](../metric-model.md) for the
type system and [`cupti-fqn-suffixes.md`](../cupti-fqn-suffixes.md)
for the full table of `.per_cycle_active` / `.pct_of_peak_*` /
`.per_second` suffixes plus the auto-title fallback used when a
panel omits `title:`.

## `visualize_single.py`

GPU-only renderer. Takes a `gpu_metrics.pb` and emits a static PNG
with the catalog/layout-resolved GPU panels. Pairs with
[`gemm_profiling`](../examples/gemm_profiling.md) which drives
`GpuProfiler` directly and doesn't emit a `session_metadata.pb`.

```bash
python tools/visualize_single.py -i gpu_metrics.pb -o gpu_metrics.png
# Optional overrides:
python tools/visualize_single.py -i gpu_metrics.pb \
    --catalog       configs/metric_catalog.pbtxt \
    --panel-layout  configs/visualizer_panels.pbtxt \
    --smooth-window-s 0.01
```

## `visualize_all.py`

Full-system static renderer. Takes a `session_metadata.pb` and
auto-discovers each per-probe file from the manifest's `probes` list.
Walks the panel layout and emits one matplotlib subplot per panel that
has matching series.

```bash
python tools/visualize_all.py profiling_output/session_metadata.pb \
    -o full_profile.png
# Override the inlined catalog or default panel layout:
python tools/visualize_all.py session_metadata.pb \
    --catalog       my_catalog.pbtxt \
    --panel-layout  my_layout.pbtxt \
    --smooth-window-s 0.05
```

Panels in the default layout (auto-skipped when no series matches):
SM Util → Active Warps/Cycle → DRAM Bandwidth → PCIe Bandwidth →
NVLink Bandwidth → CPU Utilization → System Memory → Per-PID CPU →
Per-PID Resident Memory → Per-PID I/O → Disk Bandwidth → Disk Queue
Depth.

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

# Live mode — tail the .pb files and stream new samples into a running
# Bokeh server. Open the URL in a browser; new data appears every
# poll-interval-ms (default 1s). Run alongside an active workload.
python tools/visualize_interactive.py --live \
    /tmp/run/profiling_output/session_metadata.pb
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

### Live mode (`--live`)

Adding `--live` switches the script from a one-shot static HTML render to
a `bokeh.server` that **tails the `.pb` files as the profiler is still
writing to them** and refreshes every `--poll-interval-ms` (default
`1000`). Workflow:

```bash
# Terminal A — start the workload (anything that drives a ProfilerSuite)
./build/examples/full_system_profiling -c configs/example.pbtxt

# Terminal B — point the live visualizer at the same output_dir's manifest
python tools/visualize_interactive.py --live \
    profiling_output/session_metadata.pb \
    --port 8000 --poll-interval-ms 1000
# → http://localhost:8000/
```

How it works:

- **`session_metadata.pb` is written at `Start()`**, not just `Stop()`.
  The live visualizer reads it as soon as the run begins to discover
  which probe files to tail and the inlined `MetricCatalog`. If you
  launch the visualizer before the workload, it polls (up to
  `--live-bootstrap-timeout-s`, default 30 s) for the manifest to
  appear.
- **Each tick** (`--poll-interval-ms`) reads only the bytes appended
  since the last tick using a strict offset-aware varint reader, parses
  the new `*Trace` messages, ingests them into a shared `TraceProjector`,
  and **streams** the new tail slice into each existing
  `ColumnDataSource` via `cds.stream(...)`. Bookmark per-series so we
  never re-send rows the browser already has.
- **Mid-run PID join**: when
  `suite.add_tracked_process(pid)` is called from your workload, the
  next flush carries the new PID in `tracked_processes[]`. The
  visualizer detects it (via
  `projector.new_scope_keys_since_last_call()`), allocates a new line
  glyph + CDS on every matching panel, and starts streaming the PID's
  samples in.
- **Mid-run PID removal**: `suite.remove_tracked_process(pid)` flips
  `TrackedProcessV2.removed=true` on the PID's last appearance in the
  trace. The visualizer renders a dotted gray vertical marker on the
  series at that t and stops appending further rows.
- **Pan/zoom is preserved across ticks** — the user's current view
  isn't reset when new samples arrive.

Caveats:

- **One Python process per page.** Closing the browser does not stop
  the server; Ctrl-C in Terminal B does.
- **Long runs**: per-tick delta streaming scales linearly in the
  *new* rows since the last tick, not the full run length, so this
  works for arbitrarily long sessions. The Bokeh client still has to
  re-render the canvas every tick; for many-minute runs at 10 kHz,
  bump `--poll-interval-ms` to keep the client responsive.

Disk I/O is the only non-obvious permission gotcha — see
[*Permissions for per-PID I/O*](../system-guide.md#permissions-for-per-pid-i-o).

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

`visualize_all.py` needs `numpy + matplotlib + protobuf`;
`visualize_interactive.py` adds `bokeh + tornado` (Bokeh transitive).

The renderers also depend on the shared catalog + projector modules
under `tools/`:
- `tools/metric_catalog.py` — `MetricCatalog` loader and peak resolver.
- `tools/metric_layout.py` — `PanelLayout` loader, FQN globbing, GPU
  FQN suffix-inference for catalog gaps.
- `tools/metric_projector.py` — `TraceProjector` (proto traces → per-
  `(fqn, scope_key)` ndarray caches).
- `tools/live_tail.py` — `TraceTail` (offset-aware tail) + `LiveCoordinator`
  (drives the Bokeh server's periodic callback). Used only by
  `--live` mode.
