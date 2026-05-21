# Metric model

This document describes the metric type system this profiler inherits from
**CUPTI PerfWorks** (Counter / Ratio / Throughput) and proposes how the
non-GPU metric streams — CPU, memory, disk — can be mapped into the same
abstraction so post-processing (smoothing, unit conversion, peak
normalization, panel grouping) can be written **once and shared across all
probes**.

The first half is reference material describing what CUPTI actually
exposes. The second half is the proposed generalization that drives the
unified post-processing scheme.

---

## Part 1 — CUPTI's model (authoritative)

CUPTI does not model metrics as OO subclasses. The Profiler Host API
(`cuptiProfilerHost*`) flattens everything to a single discriminator
enum + opaque host object + name-keyed lookups:

```c title:"cupti_profiler_host.h"
typedef enum {
    CUPTI_METRIC_TYPE_COUNTER    = 0,
    CUPTI_METRIC_TYPE_RATIO      = 1,
    CUPTI_METRIC_TYPE_THROUGHPUT = 2,
} CUpti_MetricType;
```

A base metric is a `const char*` name (`"sm__cycles_active"`,
`"dram__throughput"`, …). The catalog is discovered by walking the enum:

| Query | Returns |
| ----- | ------- |
| `cuptiProfilerHostGetBaseMetrics(type)` | every base metric of that type |
| `cuptiProfilerHostGetMetricProperties(name)` | declared type, description, unit |
| `cuptiProfilerHostGetSubMetrics(name, type)` | the **legal** suffixes for *that specific* base metric |

A fully-qualified metric name is built by string concatenation:

```text ln:false
<entity>__<counter>[.<rollup>][.<submetric>]
   sm    cycles_active   avg   pct_of_peak_sustained_elapsed
```

- **entity** — hardware unit being measured (`sm`, `smsp`, `dram`, `gpc`,
  `lts`, `l1tex`, `pcie`, `nvlrx`, `nvltx`, `fbpa`, `gr`, …).
- **counter** — the raw event (`cycles_active`, `cycles_elapsed`,
  `warps_active`, `read_bytes`, …).
- **rollup** — aggregation across instances of the entity (across all SMs,
  all DRAM partitions, …).
- **submetric** — post-rollup transformation (rate, percent of peak, …).

The rollup and submetric layers are conceptually distinct but are
flattened into one suffix string. **What's legal for a given metric is
enumerated by `cuptiProfilerHostGetSubMetrics`** — not by Cartesian
product. `tools/src/list_pm_metrics/list_pm_metrics.cpp:80-108` walks
this enumeration to print the catalog.

### 1.1 — Counter (`CUPTI_METRIC_TYPE_COUNTER`)

Raw hardware event count per sample window. Largest type — on H100, ~860
base counters.

**Rollups** (aggregation across instances of the entity):

| Suffix | Meaning |
| ------ | ------- |
| `.sum` | Sum across all instances |
| `.avg` | Mean across instances |
| `.min` | Minimum-valued instance |
| `.max` | Maximum-valued instance |

**Submetric transforms** (applied after the rollup):

| Suffix | Meaning |
| ------ | ------- |
| *(none)* | Raw count for the sample window |
| `.per_second` | Count ÷ wall-clock seconds → rate (Hz, B/s, …) |
| `.per_cycle_active` | Count ÷ **active** cycles of the entity |
| `.per_cycle_elapsed` | Count ÷ **elapsed** cycles |
| `.per_cycle_region` | Per-cycle inside a profiling region (range profiler only) |
| `.peak_sustained` | Theoretical peak the HW can hold indefinitely |
| `.peak_sustained_active` / `.peak_sustained_elapsed` | Peak conditioned on active vs elapsed cycles |
| `.peak_burst` | Theoretical peak for a single cycle |
| `.pct_of_peak_sustained_{active,elapsed}` | Value as % of the corresponding peak |
| `.pct_of_peak_burst_{active,elapsed}` | Same against burst peak |

`peak_*` suffixes by themselves evaluate to a hardware **constant** — they
are useful only as denominators inside `.pct_of_peak_*`.

### 1.2 — Ratio (`CUPTI_METRIC_TYPE_RATIO`)

Pre-built **dimensionless** quantity. Already normalized, so there is no
rollup-across-instances step:

| Suffix | Meaning |
| ------ | ------- |
| `.ratio` | Raw value (typically 0 → 1, occasionally unbounded) |
| `.pct` | `ratio × 100` |
| `.max_rate` | The denominator's max — the value at peak (constant) |

Example: `smsp__average_warps_active_per_cycle_active.ratio` — warp
occupancy fraction; H100 max is 64.

### 1.3 — Throughput (`CUPTI_METRIC_TYPE_THROUGHPUT`)

Pre-built composite, already `(counter / peak)`. The base metric encodes
both numerator and denominator (`sm__throughput`, `dram__throughput`,
`lts__throughput`, …). The suffix picks the aggregation and which peak:

| Suffix | Meaning |
| ------ | ------- |
| `.avg.pct_of_peak_sustained_{active,elapsed}` | Mean as % of sustained peak |
| `.max.pct_of_peak_sustained_{active,elapsed}` | Per-window max instance as % |
| `.avg.pct_of_peak_burst_{active,elapsed}` | Mean against burst peak |
| `.max.pct_of_peak_burst_{active,elapsed}` | Max against burst peak |

Throughputs have a narrower legal suffix set than counters
(`.per_second` is not exposed — the throughput already encodes a
rate-vs-peak comparison).

### 1.4 — Quick reference

| Type | What it is | Rollups | Submetric transforms | Example |
| ---- | ---------- | ------- | -------------------- | ------- |
| **Counter** | Raw event count | `.sum` `.avg` `.min` `.max` | `.per_second`, `.per_cycle_{active,elapsed,region}`, `.peak_sustained{,_active,_elapsed}`, `.peak_burst`, `.pct_of_peak_{sustained,burst}_{active,elapsed}` | `dram__read_bytes.sum.per_second` |
| **Ratio** | Pre-normalized scalar | — | `.ratio` `.pct` `.max_rate` | `smsp__average_warps_active_per_cycle_active.ratio` |
| **Throughput** | Pre-built counter / peak | `.avg` `.max` (sometimes `.sum`) | `.pct_of_peak_{sustained,burst}_{active,elapsed,region}` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |

> [!NOTE]
> After `cuptiProfilerHostEvaluateToGpuValues` returns, every metric is a
> `double`. The type discriminator only matters for **suffix validation**
> at config time and **interpretation** at plot time. That's why
> `metric_names` is carried on every `GpuMetricsTrace` — without the
> name, the units of a column of doubles are unknowable.

---

## Part 2 — Generalization, as implemented

> [!NOTE]
> The original draft of this section was a design proposal; the
> repository now implements it. The wire format, the catalog, the
> renderer pipeline, and the suite-level mid-run PID API are all
> live as of branch `feature/unified-metric-model`. Subsection
> bodies still read as "what we adopt" — they describe the current
> behavior, not a future plan.

`SystemMetricsTrace` and `DiskMetricsTrace` previously used hand-rolled
field names that looked nothing like CUPTI's `<entity>__<counter>.<suffix>`
scheme. They've been redesigned to share substructures
(`Sample`, `ProcessSample`, `DeviceSample`, `GPUSample`,
`TrackedProcessV2`, `TraceHeader`, `FlushStats`,
`ScopeMetricNames`) defined in `proto/metric_sample.proto`, so a single
post-processor handles every probe.

### 2.1 — Goal

Every metric — GPU, CPU, memory, disk — should resolve to a record of:

```python title:"target post-processing contract"
@dataclass(frozen=True)
class MetricDescriptor:
    fqn: str                  # "dram__read_bytes.sum.per_second"
    metric_type: MetricType   # COUNTER | RATIO | THROUGHPUT
    entity: str               # "dram"
    counter: str              # "read_bytes"
    rollup: str | None        # "sum"
    submetric: str | None     # "per_second"
    unit: Unit                # COUNT | BYTES | BYTES_PER_SEC | PCT | RATIO | HZ | CYCLES | REQUESTS
    peak: float | None        # absolute peak value (for pct normalization / dashed ref line)
    scope: Scope              # SYSTEM | DEVICE("nvme0n1") | PROCESS(pid) | GPU(idx)
```

This is everything the renderer (`visualize_all.py`,
`visualize_interactive.py`) needs to know to:
- Pick the y-axis label (from `unit`)
- Pick the y-axis range and dashed peak line (from `peak`)
- Group panels (by `entity`)
- Group series within a panel (by `scope`)
- Decide smoothing window applicability (boxcar over a counter is fine;
  over a ratio is also fine; over a peak-normalized throughput needs
  the same kernel size — uniform across types)

### 2.2 — FQN scheme for non-CUPTI metrics

Same `<entity>__<counter>[.<rollup>][.<submetric>]` grammar as CUPTI.
Where the existing CUPTI suffixes fit, they're reused verbatim. Where
they don't, the orthography is extended.

**Entities** (new namespace, doesn't collide with CUPTI's `sm`/`dram`/…):

| Entity | Meaning | Scope dimension |
| ------ | ------- | --------------- |
| `cpu` | The full CPU subsystem | system-wide |
| `core` | A single logical CPU core (future — currently we only sample aggregate) | per-core |
| `mem` | System memory | system-wide |
| `proc` | Per-process counters (any kind) | per-PID |
| `disk` | A block device | per-device |

**Scope** lives outside the FQN. Two metrics that differ only in PID or
device get the **same** FQN; the `MetricDescriptor.scope` field
disambiguates. This keeps the FQN catalog small.

### 2.3 — Mapping `SystemMetricsTrace`

Today's `CPUSystemSample`:

| Legacy field | Catalog FQN | Type | Unit | Peak |
| ------------- | ------------ | ---- | ---- | ---- |
| `total_utilization_pct` | `cpu__cycles_busy.avg.pct_of_peak_sustained_elapsed` | Throughput | PCT | 100 |
| `user_pct` | `cpu__cycles_user.avg.pct_of_peak_sustained_elapsed` | Throughput | PCT | 100 |
| `system_pct` | `cpu__cycles_kernel.avg.pct_of_peak_sustained_elapsed` | Throughput | PCT | 100 |
| `iowait_pct` | `cpu__cycles_iowait.avg.pct_of_peak_sustained_elapsed` | Throughput | PCT | 100 |

Today's `MemorySystemSample`:

| Legacy field | Catalog FQN | Type | Unit | Peak |
| ------------- | ------------ | ---- | ---- | ---- |
| `total_bytes` | `mem__capacity_bytes` | Counter | BYTES | — *(this **is** the peak for `used_bytes`)* |
| `used_bytes` | `mem__used_bytes` | Counter | BYTES | `total_bytes` |
| `available_bytes` | `mem__available_bytes` | Counter | BYTES | `total_bytes` |
| `buffers_bytes` | `mem__buffers_bytes` | Counter | BYTES | `total_bytes` |
| `cached_bytes` | `mem__cached_bytes` | Counter | BYTES | `total_bytes` |

Today's `CPUProcessSample` and `MemoryProcessSample`:

| Legacy field | Catalog FQN | Type | Unit | Peak | Scope |
| ------------- | ------------ | ---- | ---- | ---- | ----- |
| `cpu_pct` | `proc__cycles_active.sum.per_second` | Counter | PCT (of one core) | `100 × ncpus` | `PROCESS(pid)` |
| `rss_bytes` | `proc__rss_bytes` | Counter | BYTES | `mem.capacity_bytes` | `PROCESS(pid)` |
| `vms_bytes` | `proc__vms_bytes` | Counter | BYTES | — | `PROCESS(pid)` |
| `shared_bytes` | `proc__shared_bytes` | Counter | BYTES | `mem.capacity_bytes` | `PROCESS(pid)` |

> Per-PID `cpu_pct` is "% of one CPU" — modelling it as a Counter with
> `.sum.per_second` semantics (not pre-normalized) is more faithful: a
> 4-thread saturating process reports `400`, exactly what `.sum` across
> four cores would yield. The value is derived from `/proc/<pid>/schedstat`
> field 1 (`sum_exec_runtime` in ns), giving nanosecond precision rather
> than the 10 ms `CLK_TCK` quantization that `utime/stime` would impose.
> The trade-off is that we no longer split per-PID time into user /
> kernel / iowait — schedstat reports total on-CPU time only.

### 2.4 — Mapping `DiskMetricsTrace`

Today's `DiskDeviceSample`:

| Legacy field | Catalog FQN | Type | Unit | Peak | Scope |
| ------------- | ------------ | ---- | ---- | ---- | ----- |
| `read_bytes_per_sec` | `disk__read_bytes.sum.per_second` | Counter | BYTES_PER_SEC | device max-rated read BW (if discoverable) | `DEVICE("nvme0n1")` |
| `write_bytes_per_sec` | `disk__write_bytes.sum.per_second` | Counter | BYTES_PER_SEC | device max-rated write BW | `DEVICE("nvme0n1")` |
| `read_queue_depth` | `disk__read_inflight` | Counter | REQUESTS | — | `DEVICE("nvme0n1")` |
| `write_queue_depth` | `disk__write_inflight` | Counter | REQUESTS | — | `DEVICE("nvme0n1")` |

Today's `DiskProcessSample`:

| Legacy field | Catalog FQN | Type | Unit | Peak | Scope |
| ------------- | ------------ | ---- | ---- | ---- | ----- |
| `read_bytes_per_sec` | `proc__io_rchar.sum.per_second` | Counter | BYTES_PER_SEC | — | `PROCESS(pid)` |
| `write_bytes_per_sec` | `proc__io_wchar.sum.per_second` | Counter | BYTES_PER_SEC | — | `PROCESS(pid)` |

> The per-PID counters are syscall-layer (`rchar` / `wchar`), not
> block-layer (`read_bytes` / `write_bytes`). Encoding the source in the
> counter name (`io_rchar`, not `read_bytes`) makes that explicit and
> matches the existing system-guide warning that the two aren't
> comparable.

### 2.5 — Generic post-processing pipeline

With every metric carrying a `MetricDescriptor`, post-processing collapses
to three stages, none of which need probe-specific code:

```text ln:false
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ (timestamp,  │ →  │ resample / smooth│ →  │ render (unit,    │
│  value)[]    │    │ (kernel from CLI │    │  peak, scope)    │
│ per FQN+scope│    │  in seconds)     │    │                  │
└──────────────┘    └──────────────────┘    └──────────────────┘
```

- **Resample / smooth**: a single boxcar implementation runs over the
  raw `(ts_ns, value)` series of every FQN. The kernel size is the same
  for all metric types — what differs is interpretation, not the
  numeric op.
- **Render**: `unit` picks the y-axis formatter; `peak` (if present)
  picks the dashed reference line; `entity` picks the panel; `scope`
  picks the series-within-panel color/legend label.

The two current renderers (`visualize_all.py`, `visualize_interactive.py`)
already implement a special-cased version of this — but with one
`build_*_panels` function per probe. Once metrics carry `MetricDescriptor`
they collapse to a single `build_panel_for_entity(entity, descriptors,
samples)` call.

### 2.6 — Where the descriptors come from

Two options, not mutually exclusive:

1. **Statically declared in the C++ profilers** — each probe owns a
   `static const MetricDescriptor catalog[] = { … }` table and emits FQNs
   into the proto messages alongside the values. This is the most
   faithful to CUPTI's path (the GPU profiler already does this — every
   `GpuMetricsTrace` carries `metric_names[]`).
2. **Lifted from the proto schema in Python** — a sibling
   `tools/metric_catalog.py` defines the mapping from
   `(SystemMetricsTrace, "user_pct")` → `MetricDescriptor`. Cheaper to
   land (no proto schema change) but means the C++ and Python sides can
   drift.

Option 1 requires a proto schema bump — adding `repeated string
metric_names = N` to each non-GPU trace and emitting the FQN of each
field. Option 2 is what `visualize_all.py` already does implicitly.

### 2.7 — Open design choices

These should be settled before implementing — flagging them rather than
deciding unilaterally:

- **Do non-GPU traces grow a `metric_names[]` field on the wire?** Adding
  it makes the format self-describing (a sufficiently new visualizer can
  read a file from a future version that added a metric); leaving it out
  keeps the wire format stable and lifts the catalog into Python.
- **Should `proc__rss_bytes` etc. live under entity `proc` (per-PID
  unification) or under `mem`/`cpu` with a per-PID rollup?** Entity-
  based grouping (`proc__*`) is what the current renderer effectively
  uses (one "per-PID" panel row). The CUPTI orthography would push us
  toward `mem__rss.avg.per_pid_*` — but that's inventing rollups CUPTI
  doesn't define. Stay with entity `proc`.
- **`peak` for `disk__read_bytes`** — there is no clean `/sys/block`
  source for "vendor max sequential read BW". Either leave `peak = None`
  (renderer auto-scales) or sample once at startup (sysfs
  `queue/max_sectors_kb` × `queue/nr_requests` is an upper bound on
  in-flight bytes, not throughput).
- **`peak` for `cpu` rollups** — the natural denominator is
  `cycles_elapsed × ncpus`, which makes the `.pct_of_peak_sustained_elapsed`
  framing exact. Setting `peak = 100` per panel is simpler and matches
  what `/proc/stat` already gives us.

---

## See also

- [`cupti-fqn-suffixes.md`](cupti-fqn-suffixes.md) — full table of FQN suffixes (`.per_cycle_active`, `.pct_of_peak_sustained_elapsed`, `.per_second`, …) with the math CUPTI applies and the resulting unit. Also covers the auto-title fallback used when a panel in `visualizer_panels.pbtxt` omits `title:`.
- [`system-guide.md` § GPU metrics reference](system-guide.md#gpu-metrics-reference) — the operational view (what to put in `metrics:`).
- [`system-guide.md` § System & disk metrics reference](system-guide.md#system--disk-metrics-reference) — the current (pre-unification) field-by-field listing.
- [`tools/README.md` § `list_pm_metrics`](tools/README.md#list_pm_metrics) — enumerate the live CUPTI catalog on the local GPU.
- Nsight Compute *Profiling Guide → Metrics Reference* (NVIDIA docs) — canonical descriptions of every entity and counter.
