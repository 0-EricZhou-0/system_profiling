"""Generic descriptor-driven trace projector.

Replaces the per-domain `prepare_*` / `_extend_*_cache` / `project_*`
helpers that used to live in visualize_all.py and live_tail.py. The
projector consumes the new wire format (GPUMetricsTrace,
SystemMetricsTrace, DiskMetricsTrace) and exposes a uniform
(fqn, scope_key) -> (ts_array, vals_array) projection for the
renderers.

Live-mode bookkeeping:
  - `new_scope_keys_since_last_call()` reports (Scope, scope_key)
    pairs that first appeared since the previous call, so the Bokeh
    server can dynamically allocate a new ColumnDataSource + glyph
    when a tracked PID joins mid-run.
  - `pending_removals()` reports TrackedProcessV2 entries whose
    `removed=true` marker has arrived. The renderer draws a "removed"
    tick on the corresponding series.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import numpy as np

import metric_catalog_pb2 as _mc

from metric_catalog import HostMeta, MetricCatalog, build_index


# Type aliases.
ScopeKey = Hashable                 # int for PID/GPU, str for device, None for SYSTEM
SeriesKey = tuple[str, ScopeKey]    # (FQN, scope_key)


# ---------------------------------------------------------------------------
# Internal cache shape
# ---------------------------------------------------------------------------

@dataclass
class _SeriesCache:
    """Per-series sample buffer. Python lists during ingest (O(1)
    append); converted to numpy at project() time."""
    ts:   list[int]   = field(default_factory=list)
    vals: list[float] = field(default_factory=list)


@dataclass
class GpuDeviceInfo:
    """Mirror of proto/metric_sample.proto :: GPUDeviceInfo for plot-time
    peak resolution against `Panel.peak_from_gpu_info`."""
    device_name: str = ""
    chip_name: str = ""
    peak_dram_bw_bytes_per_s: float = 0.0
    peak_pcie_bw_bytes_per_s: float = 0.0
    peak_nvlink_bw_bytes_per_s: float = 0.0
    max_warps_per_sm: int = 0


@dataclass
class TrackedProcess:
    """Mirror of TrackedProcessV2 for renderer-side display."""
    pid: int = 0
    alias: str = ""
    removed: bool = False


# ---------------------------------------------------------------------------
# TraceProjector
# ---------------------------------------------------------------------------

class TraceProjector:
    """Ingests *MetricsTrace protos, exposes (fqn, scope_key)-keyed
    projections, and tracks per-Scope key churn for live-mode delta
    rendering."""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog
        self.descriptors = build_index(catalog)

        # (fqn, scope_key) -> _SeriesCache
        self._caches: dict[SeriesKey, _SeriesCache] = {}

        # Per-scope set of seen scope_keys; used to compute deltas.
        self._seen_keys: dict[int, set[ScopeKey]] = defaultdict(set)
        self._new_keys_buffer: dict[int, list[ScopeKey]] = defaultdict(list)
        self._pending_removals_buffer: dict[int, list[ScopeKey]] = defaultdict(list)
        self._seen_removals: set[tuple[int, ScopeKey]] = set()

        # Host metadata + per-GPU info — populated as traces arrive.
        self.host = HostMeta()
        self.gpu_info: dict[int, GpuDeviceInfo] = {}

        # Tracked PID metadata across both system + disk probes,
        # keyed by pid so the visualizer can render a consistent
        # legend label regardless of which probe surfaced the PID.
        self.tracked_processes: dict[int, TrackedProcess] = {}

        # FQN -> source probe ("gpu" / "system" / "disk"). Lets the
        # visualizer group panels by which probe emitted their series.
        # Populated as each ScopeMetricNames is seen on ingest.
        self.fqn_to_probe: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def _note_scope_key(self, scope: int, key: ScopeKey) -> None:
        if key not in self._seen_keys[scope]:
            self._seen_keys[scope].add(key)
            self._new_keys_buffer[scope].append(key)

    def _record_removal(self, scope: int, key: ScopeKey) -> None:
        token = (scope, key)
        if token in self._seen_removals:
            return
        self._seen_removals.add(token)
        self._pending_removals_buffer[scope].append(key)

    def _absorb_header(self, header) -> None:
        # Last writer wins — they're constant for a run.
        if header.hostname:
            self.host = HostMeta(hostname=header.hostname,
                                 cpu_count=header.host_cpu_count or self.host.cpu_count)
        elif header.host_cpu_count and not self.host.cpu_count:
            self.host = HostMeta(hostname=self.host.hostname,
                                 cpu_count=header.host_cpu_count)

    def _scope_fqns(self, trace, scope: int) -> list[str]:
        for smn in trace.scope_metric_names:
            if smn.scope == scope:
                return list(smn.fqns)
        return []

    def _record_probe_source(self, trace, probe: str) -> None:
        for smn in trace.scope_metric_names:
            for fqn in smn.fqns:
                # First-writer wins; in practice an FQN is owned by one
                # probe, so this is just defensive.
                self.fqn_to_probe.setdefault(fqn, probe)

    def _ingest_samples_uniform(
        self,
        samples,
        fqns: list[str],
        scope: int,
        scope_key_fn: Callable[[Any], ScopeKey],
    ) -> None:
        """Common path for every {Sample,ProcessSample,DeviceSample,
        GPUSample}: each sample contributes one (ts, value) pair per
        FQN, into the cache keyed by (fqn, scope_key_fn(sample))."""
        if not fqns:
            return
        n = len(fqns)
        for s in samples:
            key = scope_key_fn(s)
            self._note_scope_key(scope, key)
            ts = s.timestamp_ns
            for i in range(n):
                cache = self._caches.get((fqns[i], key))
                if cache is None:
                    cache = _SeriesCache()
                    self._caches[(fqns[i], key)] = cache
                cache.ts.append(ts)
                cache.vals.append(s.values[i])

    def _absorb_tracked_processes(self, entries) -> None:
        for e in entries:
            tp = self.tracked_processes.get(e.pid)
            if tp is None:
                tp = TrackedProcess(pid=e.pid, alias=e.alias, removed=e.removed)
                self.tracked_processes[e.pid] = tp
            else:
                # Alias may be set on first appearance only; later
                # flushes can flip `removed`.
                if e.alias:
                    tp.alias = e.alias
                tp.removed = tp.removed or e.removed
            if e.removed:
                self._record_removal(_mc.SCOPE_PROCESS, e.pid)

    def ingest_gpu(self, trace) -> None:
        self._absorb_header(trace.header)
        self._record_probe_source(trace, "gpu")
        for g in trace.tracked_gpus:
            self.gpu_info[g.device_index] = GpuDeviceInfo(
                device_name=g.device_name,
                chip_name=g.chip_name,
                peak_dram_bw_bytes_per_s=g.peak_dram_bw_bytes_per_s,
                peak_pcie_bw_bytes_per_s=g.peak_pcie_bw_bytes_per_s,
                peak_nvlink_bw_bytes_per_s=g.peak_nvlink_bw_bytes_per_s,
                max_warps_per_sm=int(g.max_warps_per_sm),
            )
        fqns = self._scope_fqns(trace, _mc.SCOPE_GPU)
        if not fqns:
            return
        # GPU samples are timestamped in CUPTI's own clock domain. The
        # trace header carries an anchor pair (`steady_clock_reference_ns`,
        # `cupti_reference_ns`) captured at the same instant, so we can
        # convert into the same `steady_clock` number space the system
        # and disk probes use — necessary for shared X-axis plotting.
        anchors = trace.header.anchors
        if anchors.cupti_reference_ns:
            cupti_to_steady = (int(anchors.steady_clock_reference_ns)
                               - int(anchors.cupti_reference_ns))
        else:
            cupti_to_steady = 0
        n = len(fqns)
        for s in trace.samples:
            ts = int(s.timestamp_ns) + cupti_to_steady
            key = int(s.gpu_index)
            self._note_scope_key(_mc.SCOPE_GPU, key)
            for i in range(n):
                cache = self._caches.get((fqns[i], key))
                if cache is None:
                    cache = _SeriesCache()
                    self._caches[(fqns[i], key)] = cache
                cache.ts.append(ts)
                cache.vals.append(s.values[i])

    def ingest_system(self, trace) -> None:
        self._absorb_header(trace.header)
        self._record_probe_source(trace, "system")
        self._absorb_tracked_processes(trace.tracked_processes)
        sys_fqns = self._scope_fqns(trace, _mc.SCOPE_SYSTEM)
        proc_fqns = self._scope_fqns(trace, _mc.SCOPE_PROCESS)
        self._ingest_samples_uniform(
            trace.system_samples, sys_fqns, _mc.SCOPE_SYSTEM,
            scope_key_fn=lambda s: None,
        )
        self._ingest_samples_uniform(
            trace.process_samples, proc_fqns, _mc.SCOPE_PROCESS,
            scope_key_fn=lambda s: int(s.pid),
        )

    def ingest_disk(self, trace) -> None:
        self._absorb_header(trace.header)
        self._record_probe_source(trace, "disk")
        self._absorb_tracked_processes(trace.tracked_processes)
        dev_fqns = self._scope_fqns(trace, _mc.SCOPE_DEVICE)
        proc_fqns = self._scope_fqns(trace, _mc.SCOPE_PROCESS)
        self._ingest_samples_uniform(
            trace.device_samples, dev_fqns, _mc.SCOPE_DEVICE,
            scope_key_fn=lambda s: str(s.device_name),
        )
        self._ingest_samples_uniform(
            trace.process_samples, proc_fqns, _mc.SCOPE_PROCESS,
            scope_key_fn=lambda s: int(s.pid),
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self) -> dict[SeriesKey, tuple[np.ndarray, np.ndarray]]:
        """Snapshot every (fqn, scope_key) cache as numpy arrays.
        Caches stay as Python lists internally so subsequent ingest()s
        keep appending in O(1); calling project() repeatedly returns
        progressively longer arrays."""
        out: dict[SeriesKey, tuple[np.ndarray, np.ndarray]] = {}
        for key, cache in self._caches.items():
            if not cache.ts:
                continue
            ts  = np.asarray(cache.ts,   dtype=np.uint64)
            val = np.asarray(cache.vals, dtype=np.float64)
            out[key] = (ts, val)
        return out

    def lookup_first_value(self, fqn: str) -> float | None:
        """Resolve `MetricDescriptor.peak_ref`: returns the first
        observed value of `fqn` (any scope key). None if no samples for
        that FQN have arrived. Used by metric_catalog.resolve_peak().
        """
        for (cache_fqn, _), cache in self._caches.items():
            if cache_fqn == fqn and cache.vals:
                return float(cache.vals[0])
        return None

    # ------------------------------------------------------------------
    # Live-mode delta hooks
    # ------------------------------------------------------------------

    def new_scope_keys_since_last_call(self) -> dict[int, list[ScopeKey]]:
        """Returns and clears the per-Scope list of scope_keys that
        first appeared since the previous call. The Bokeh live mode
        uses this to lazily allocate new series glyphs."""
        out = {scope: keys for scope, keys in self._new_keys_buffer.items() if keys}
        self._new_keys_buffer = defaultdict(list)
        return out

    def pending_removals(self) -> dict[int, list[ScopeKey]]:
        """Returns and clears the per-Scope list of scope_keys that
        were marked `removed=true` since the previous call. The renderer
        draws a removal marker on the corresponding series."""
        out = {scope: keys for scope, keys in self._pending_removals_buffer.items() if keys}
        self._pending_removals_buffer = defaultdict(list)
        return out
