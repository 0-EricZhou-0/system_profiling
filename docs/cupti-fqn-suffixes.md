# CUPTI FQN suffix reference

CUPTI PerfWorks names metrics with a fully-qualified format

```
<entity>__<counter>[.<rollup>][.<submetric>]
```

The suffixes (`.sum`, `.avg.per_cycle_active`, `.pct_of_peak_sustained_elapsed`,
`.per_second`, …) are **not just labels** — they instruct CUPTI to apply
specific math to the raw hardware counts during metric evaluation. This
page lists every suffix this codebase understands, what CUPTI does with
it, and the resulting unit that the visualizer formats against.

## Where the transform happens

Our code does **not** do any of the suffix math. Per sample:

1. CUPTI's per-device counter-data image holds raw hardware counts —
   one row per sample range, one column per counter.
2. Our decode thread
   ([`lib/src/profiler_host_internal.cpp:79`](../lib/src/profiler_host_internal.cpp))
   passes the **suffixed FQN string** straight to
   `cuptiProfilerHostEvaluateToGpuValues(...)`.
3. CUPTI parses the suffix and applies the corresponding math against
   the counters in the data image — sums or averages across instances
   (for `.sum`/`.avg`), divides by `sm__cycles_active.sum` for
   `.per_cycle_active`, divides by `peak_sustained × elapsed cycles`
   and multiplies by 100 for `.pct_of_peak_sustained_elapsed`, etc.
4. The resulting `double` lands in `SamplerRange.metricValues[i]` and
   flows through to `GPUSample.values[i]` unchanged.

So: **the suffix decides what the value means; the table below maps
each suffix to its resulting unit and to a human-readable label
fragment used when the visualizer auto-derives a panel title.**

## The table

### Rollup (between `__counter` and `.submetric`)

When a counter has multiple instances on chip (e.g. one per SM, one per
DRAM channel), the rollup reduces across them.

| Suffix | CUPTI math                  | Unit       | Auto-title hint |
|--------|-----------------------------|------------|-----------------|
| `.sum` | sum across instances        | as raw     | `(sum)`         |
| `.avg` | mean across instances       | as raw     | `(avg)`         |
| `.min` | min across instances        | as raw     | `(min)`         |
| `.max` | max across instances        | as raw     | `(max)`         |

### Normalization submetric (after rollup)

The most-used suffixes. CUPTI divides the rolled-up value by some
reference and (optionally) scales by 100.

| Suffix                              | CUPTI math                                              | Unit         | Auto-title hint                  |
|-------------------------------------|---------------------------------------------------------|--------------|----------------------------------|
| `.per_second`                       | rolled / elapsed-wall-seconds                           | `BYTES_PER_SEC` if counter is byte-y, else `HZ` | `/ s`                  |
| `.per_cycle_active`                 | rolled / `sm__cycles_active.sum`                        | `COUNT`      | `/ active cycle`                 |
| `.per_cycle_elapsed`                | rolled / `sm__cycles_elapsed.sum`                       | `COUNT`      | `/ elapsed cycle`                |
| `.pct_of_peak_sustained_active`     | 100 × rolled / (peak_sustained_active × active cycles)  | `PCT`        | `(% of sustained peak, active)`  |
| `.pct_of_peak_sustained_elapsed`    | 100 × rolled / (peak_sustained_elapsed × elapsed cycles)| `PCT`        | `(% of sustained peak)`          |
| `.pct_of_peak_burst_active`         | 100 × rolled / (peak_burst_active × active cycles)      | `PCT`        | `(% of burst peak, active)`      |
| `.pct_of_peak_burst_elapsed`        | 100 × rolled / (peak_burst_elapsed × elapsed cycles)    | `PCT`        | `(% of burst peak)`              |
| `.pct`                              | already a 0–100 percent                                 | `PCT`        | `%`                              |
| `.ratio`                            | already a 0–1 ratio                                     | `RATIO`      | `ratio`                          |

Byte-y counters: anything whose `counter` name contains `bytes` or
`throughput` is treated as a byte counter. `.per_second` on a byte-y
counter resolves to `UNIT_BYTES_PER_SEC` and gets a `MiB/s` / `GiB/s`
axis; on every other counter it resolves to `UNIT_HZ`.

### Peak inquiry (returns the peak itself)

These don't depend on a sample — they return the hardware peak that
the percentage-of-peak forms divide against. Useful for sanity-checking
or for raw bandwidth normalization in custom panels.

| Suffix                       | Returns                                       |
|------------------------------|-----------------------------------------------|
| `.peak_sustained`            | peak per second (or per work unit)            |
| `.peak_sustained_active`     | peak per active cycle                         |
| `.peak_sustained_elapsed`    | peak per elapsed cycle                        |
| `.peak_burst`                | burst peak                                    |
| `.peak_burst_active`         | burst peak per active cycle                   |
| `.peak_burst_elapsed`        | burst peak per elapsed cycle                  |

All peak inquiries resolve to `UNIT_COUNT` in the catalog by default —
override `unit_override` on the panel if you need different formatting.

### No submetric

The bare `<entity>__<counter>[.<rollup>]` form returns the raw
(possibly rolled-up) count. Unit is `UNIT_COUNT`. Useful for
debugging; most analyses prefer one of the normalized forms above.

## Examples

| FQN                                                       | Reads as                                    | Unit          |
|-----------------------------------------------------------|---------------------------------------------|---------------|
| `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`     | SM utilization (%)                          | `PCT`         |
| `sm__warps_active.avg.per_cycle_active`                   | Active warps per active cycle (avg)         | `COUNT`       |
| `dram__read_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM read bandwidth (% of peak)             | `PCT`         |
| `pcie__read_bytes.sum.per_second`                         | PCIe bytes received per wall-second (total) | `BYTES_PER_SEC` |
| `nvltx__bytes.sum.per_second`                             | NVLink TX bytes per wall-second (total)     | `BYTES_PER_SEC` |
| `lts__t_sectors.sum`                                      | L2 sector transactions (sum, total count)   | `COUNT`       |

## How the codebase uses this table

- [`tools/metric_suffix.py`](../tools/metric_suffix.py) — encodes
  the full table. Exposes `unit_for(counter, submetric)` and
  `label_for(entity, counter, rollup, submetric)`.
- [`tools/metric_layout.py :: synthesize_descriptor`](../tools/metric_layout.py)
  — falls back on `unit_for` and `label_for` when an FQN has no
  catalog entry (the typical case for chip-specific GPU metrics).
- [`tools/visualize_all.py :: _panel_title`](../tools/visualize_all.py),
  [`tools/visualize_interactive.py :: _panel_title`](../tools/visualize_interactive.py) —
  if a panel in `visualizer_panels.pbtxt` omits `title:`, the
  renderer auto-derives it from the first matched series using
  `label_for`.

## See also

- [`metric-model.md`](metric-model.md) — the underlying type system
  (Counter / Ratio / Throughput, FQN structure).
- [`tools/README.md` § list_pm_metrics](tools/README.md#list_pm_metrics)
  — enumerate the live CUPTI catalog on the local GPU.
- NVIDIA Nsight Compute *Profiling Guide → Metrics Reference* —
  canonical descriptions of every PerfWorks entity and counter.
