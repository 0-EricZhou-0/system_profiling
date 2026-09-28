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
    --catalog       lib/data/metric_catalog.pbtxt \
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
    --smooth-window-s 0.05 \
    --display-hz    100
```

Quality / size knobs:

- `--smooth-window-s <s>` boxcar-smooths every `smoothable` metric over
  the given window (per-probe kernel size). `0` (default) = no
  smoothing. Cumulative companion panels stay raw so their integrals
  remain faithful.
- `--fit-axis-to-data` (off by default): a panel whose ceiling (the
  Peak line) is more than 5× the largest plotted value — the data would
  fill under a fifth of an axis stretched to it — gets a y-axis sized to
  its data and the ceiling written as `Peak: <value> (off-scale)` instead
  of drawn (the full-system figure's NVLink panel: a few KiB/s under a
  279 GiB/s ceiling). Panels whose ceiling is near their data are
  unchanged. Off, every panel reaches its Peak as before. Also on
  `visualize_interactive.py` and passed through by the vLLM example.
- `--unit-scale-factor <F>` (default 2, must be ≥ 1): the byte-unit
  threshold — an axis or footer rate uses the largest prefix P
  (KiB, MiB, GiB, TiB) with its largest value ≥ F × P (see "Byte units"
  below); 1 switches as soon as a value reaches the prefix. Also on
  `visualize_interactive.py` (static and live mode) and passed through by
  `examples/vllm_serving_profiling.py`.
- `--display-hz <Hz>` stride-decimates every series to the given
  display rate after smoothing. `0` (default) keeps the raw sampling
  rate. Useful for cutting render time on high-frequency GPU traces.

**Legends** sit above each panel, outside the plot, under the panel
title, in as many columns as the widest entry allows. A panel with more
than ten series (a cold vLLM start tracks ~160 processes) lists the ten
most active — by the time integral of the value; on a cumulative panel,
by run total — and one `+k more` entry; the rest are drawn in light
grey, so every colour in a legend names one line (ten = the length of
the colour cycle). A process keeps one colour in every panel, the
busiest processes getting distinct colours first; within a panel the
listed entries never share a colour. Legend entries name the series or
process only, on cumulative companions too (the values are on the
axis). Every series, listed or not, is in `<output>.legend.txt`.
In a panel with several metrics per process (the per-process I/O
panels) or per disk device (disk bandwidth: read and write), the
legend's first row is the line-style key alone (`── IO rchar (sum)  - -
IO wchar (sum)`), the processes (devices) on the rows under it, each
entry naming the process (device) only.

**Process timeline.** Directly under the Region strip, on the same time
axis: one bar per tracked process (processes only — threads are not
traced) from its start to its exit, or to the end of the trace if it was
still running, in the process's colour from the per-process panels.
Every process is labelled with its latest name: `comm (pid)` inside its
bar where that fits (else `comm` alone); otherwise `comm (pid)` in rows
under the lanes, joined to its bar by a thin grey leader drawn beneath
the bars. The outside labels never overlap one another or a bar: they
are spread along the axis, at least 8 pt apart and the first row clear
of the lowest lane by more than half a lane (`tools/label_spread.py`,
the same routine the event and region strips use) in as many rows as it takes — up to 16 —
for each to sit within 8% of the axis width of its bar; a burst of
short-lived compilers (a cold vLLM start, ~90 processes in a few
seconds) becomes a few rows of labels near the burst.
Listed roots are outlined solid, orphans — discovered processes whose
parent is not in the trace — dashed. A thin line with a dot on the
parent's bar marks each fork: from the parent's bar at the child's start
to the child's bar (the parent is the one recorded when the child was
found, so a reparented process still links to the process that forked
it). The bars are packed into the fewest lanes possible: in start order,
each takes a lane free at its start — the one nearest its parent's — and
a new lane opens only when none is free, so the lane count is the most
processes alive at one instant. From the trace's process table (the
System probe's): pid, parent, comm history, start (10 ms ticks) and end.

**Colours and line order.** A process has one colour on the whole page
(its PID's; per-process panels tell a process's metrics apart by line
style). Every other series takes its colour from its *base metric* —
entity, counter and submetric, plus the device or GPU it is of — one hue
per base metric, handed out in layout order from the tab10 cycle and
continuing across panels (SM activity blue, warps orange, DRAM read
green, DRAM write red, ...). Statistic variants of one metric share its
hue: `.max` in the full colour, `.avg` tinted halfway to white, `.min`
three quarters, `.sum` shaded a third toward black; a statistic alone in
its panel keeps the full colour. Within a panel two base metrics never
share a hue (the more active keeps it). Series are drawn from the most
active to the least, so a smaller one lies on top: the lighter `.avg`
band over the full-colour `.max` one (max ≥ avg, so max shows from avg up
to max). Series lines are 0.63 pt (the grey "+k more" 0.42 pt; the Bokeh
page 0.84 / 0.56 px).

**A process's end on memory panels.** While the kernel tears an exiting
process's memory down, `/proc/<pid>/statm` reads RSS 0 though its pidfd
still says alive. The System probe checks the exit evidence at that
reading (`/proc/<pid>/stat` gone, zombie, or `PF_EXITING`) and records
the sample's memory as missing (NaN) instead of 0 — its CPU is kept; a
process that frees its memory while alive keeps its real drop. On the
per-process memory panels (bytes, e.g. RSS), each process that exited
ends in a dashed vertical line in its colour from 0 up to its last
measured value.

**Cumulative companions.** A layout panel with `aggregation:
PANEL_AGGREGATION_INTEGRATE` gets a companion under it plotting ∫ y dt of
each of its series (trapezoid rule, full-resolution data), one line per
series as in the panel above: nothing is summed across processes or
devices. The disk bandwidth panel and its companion, in both shipped
layouts, draw each device in one colour (the same in every disk panel),
read solid and write dashed.

**Byte units.** Every bytes and bytes/s axis — rates, gauges and
cumulative panels, in both renderers — takes its unit from the largest
value actually plotted on it (after any smoothing or decimation asked
for; the Peak line does not choose it), with a threshold factor F
(`--unit-scale-factor`, default 2) so it does not switch too early; with
F = 2: ≥ 2 TiB → TiB, ≥ 2 GiB → GiB, ≥ 2 MiB → MiB,
≥ 2 KiB → KiB, else B (a 1.5 GiB peak reads as 1,536 MiB); rates the same
with `/s`; an empty or all-zero panel B. The Peak line is drawn as before
and labelled in the axis's unit. The write-rate footer uses the same rule,
each value in its own unit (`tools/units.py`).

**Statistic labels.** A metric that is a statistic over its entity's
instances says which in its legend, from the FQN's rollup:
`sm__cycles_active.avg…` reads "Active Cycles (avg)" (the mean over the
SMs), `.max` "(max)" (the busiest SM), `.sum` "(sum)" (e.g. a
process's CPU summed over its threads). Metrics without a rollup
(`mem__used_bytes`) carry none.

**Peak line.** A panel with a known ceiling — from the layout
(`peak_constant`, `peak_from_gpu_info`, …) or the catalog's `peak` —
draws a dotted line at it labelled **`Peak: <value> <unit>`** (100 %,
the GPU's peak DRAM / PCIe / NVLink bandwidth, installed RAM, …) and
caps the y-axis 10% above it. It is the hardware or configured
ceiling, not the largest value in the data.

Panels in the default layout (auto-skipped when no series matches):
SM Util → Active Warps/Cycle → DRAM Bandwidth → PCIe Bandwidth →
NVLink Bandwidth → CPU Utilization → System Memory → Per-PID CPU →
Per-PID Resident Memory → Per-PID I/O (syscall layer) → Per-PID I/O
(storage layer) → Disk Bandwidth → Disk Queue Depth.

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

# Dark theme + downsample for faster first paint:
python tools/visualize_interactive.py session_metadata.pb \
    --theme dark --display-hz 100 --smooth-window-s 0.01

# Live mode — tail the .pb files and stream new samples into a running
# Bokeh server. Open the URL in a browser; new data appears every
# poll-interval-ms (default 1s). Run alongside an active workload.
python tools/visualize_interactive.py --live \
    /tmp/run/profiling_output/session_metadata.pb
```

Flag reference (selected; full list via `--help`):

- `--theme {light,dark}` — `dark` applies Bokeh's `dark_minimal` to
  every plot and flips the page background, loading overlay, and
  sticky-strip fills to match; the page's own marks (fold headers,
  Peak labels, the timeline's outlines, fork links and outside labels,
  the line-style key's swatches) take the theme's text colour, so they
  stay legible on the dark background. Default `light`.
- `--render-backend {canvas,webgl,svg}` — output backend per figure.
  Default `canvas`: ~4-5× faster first paint than `webgl` at our
  trace volume (some GPU drivers stall on WebGL `ReadPixels`). `webgl`
  wins on pan/zoom repaint smoothness.
- `--smooth-window-s <s>` / `--display-hz <Hz>` — same semantics as
  `visualize_all.py`.

What you get:

- **Process timeline pinned** in the sticky band, under the event and
  region strips (same x-range as the panels), as in `visualize_all.py`;
  hover a bar for the process's every name, pid, parent, kind, start and
  end. It keeps its full height (a cold vLLM start: 21 lanes and their
  label rows make the band tall; fold the timeline with its ▾ to give the
  space back). Zoomed, a bar's label sits in the visible part of the bar
  and is hidden when it no longer fits there. Every figure on the page
  has the same plot frame (left edge and width) and reserves the same
  right border (room for the widest legend), so a time is at the same x
  in the strips, the timeline and every panel, and the page has one
  right edge. In a window too narrow for all of it, every frame narrows
  alike.
- **Foldable panels**: each panel, and the timeline, has a header with
  its title and a ▾/▸ control; collapsed, only the header row is left.
  *Collapse all* / *Expand all* sit at the top of the sticky band. Works
  in the saved HTML without a server (not remembered across reloads).
- **Room to scroll the last panel up**: a window's height of empty page
  (the page background, either theme) follows the last panel and the
  footer, so the last panel can sit right under the sticky band.
- **Keys** (ignored while typing in a text field; the toolbar's tools
  are unchanged):

  | key | does |
  |---|---|
  | `r` or `0` | reset zoom to the whole trace |
  | `=` or `+` | zoom in 2x around the centre (every panel: they share one x-range) |
  | `-` | zoom out 2x |
  | `←` / `→` | pan 10% of the visible span (`Shift`: 50%) |
  | `c` | collapse / expand all panels |
  | `?` | show / hide this key list |
- **Sticky event + region strips** pinned at the top of the page; the
  metric panels below scroll past behind them. The two strips share
  one continuous opaque band with a dashed separator at the bottom.
- **Gesture conventions** that match TensorBoard / NSYS / NCU:
    - plain mouse scroll → page scroll
    - **ctrl + scroll** → cursor-anchored x-axis zoom
    - **click + drag** → box-zoom rectangle (release zooms to that
      x-region)
  Pan is still available via the toolbar's pan button on the left.
- **Unified hover popup** anchored at the bottom edge of each panel:
  one tooltip per panel listing every co-plotted series's value at
  the cursor x (interpolated where sampling rates differ). Triggers
  regardless of which legend entries are hidden.
- **Same series styling as `visualize_all.py`** (shared code,
  `tools/panel_legend.py`): one colour per process across the page,
  panels and process timeline alike; a line style per metric in
  panels with several metrics per process, with a legend of processes
  plus line styles; discovered processes labelled `child of <pid>`;
  statistic labels; legend entries that name the series only.
- **Click-to-hide legend entries**; a panel's legend sits **to the
  right** of its plot, one entry per compact row, capped as in
  `visualize_all.py` (ten entries plus `+k more`, which hides or shows
  all the grey lines at once); a line-style key (several metrics per
  process or device) stays on one row above the plot. The plot frame has
  a fixed size, so a legend never squeezes it. (The PNG keeps its
  legends above the panels.)
- **Y-axis clamps** with dashed reference lines at the theoretical
  peak (100% SM Util, `max_warps_per_sm` for Active Warps, peak DRAM /
  PCIe / NVLink BW, installed RAM total), labelled `Peak: …` as in
  `visualize_all.py`; legend labels name the statistic ("(avg)", …) the
  same way.
- **X-axis clamp**: pan/zoom is bounded to
  `[0, t_end + 0.4 × current_window_length]`, recomputed live as you
  zoom. The trace stays at ≥60% of the viewport even when you scroll
  past the tail. `WheelZoomTool.maintain_focus=False` so cursor-
  anchored zoom near an edge that already touches a bound absorbs the
  remaining zoom on the other side instead of being dropped.
- **Loading overlay** with elapsed-seconds counter + CSS-driven
  indeterminate bar that animates even while the JS thread is busy in
  Bokeh hydration; hides as soon as the layout is in the DOM.

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
