"""Live-tail coordinator for visualize_interactive --live mode.

The Bokeh server hosts one `LiveCoordinator` and registers a periodic
callback that calls `coordinator.tick(doc)` every poll-interval. On
each tick the coordinator:

  1. Tails the per-probe .pb files (length-delimited GPUMetricsTrace /
     SystemMetricsTrace / DiskMetricsTrace) — reads only bytes appended
     since the previous tick.
  2. Feeds parsed traces into the shared TraceProjector.
  3. For every existing ColumnDataSource, streams in the rows that
     appeared since the previous tick (Bokeh diff send, not full
     replace).
  4. For every (Scope, scope_key) that first appeared this tick (e.g.
     a PID added via suite.add_tracked_process() mid-run), allocates a
     new glyph + CDS on the matching panels.
  5. For every PID flagged TrackedProcessV2.removed=true this tick,
     draws a vertical Span marker on each of that PID's series figures.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32

from bokeh.models import ColumnDataSource, Span

# Sibling tools.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "generated" / "proto"))

import metric_catalog_pb2 as mc_pb  # noqa: E402
import gpu_metrics_pb2  # noqa: E402
import system_metrics_pb2  # noqa: E402
import disk_metrics_pb2  # noqa: E402
import session_metadata_pb2  # noqa: E402

import metric_catalog  # noqa: E402
import metric_layout  # noqa: E402
from metric_projector import TraceProjector  # noqa: E402


# ---------------------------------------------------------------------------
# Offset-aware tail reader
# ---------------------------------------------------------------------------

@dataclass
class TraceTail:
    """One length-delimited proto stream tailer.

    Maintains a byte offset so each `read_new_messages()` call returns
    only the messages appended since the previous call. Partial
    trailing messages (writer mid-flush) are buffered until the next
    tick — `offset` is only advanced past complete messages.
    """
    path: Path
    proto_class: type
    offset: int = 0
    _pending: bytes = b""

    def read_new_messages(self) -> list:
        """Read bytes appended to the file since the last call. Parse
        every complete message; leave partial trailing bytes in
        `_pending` for the next call. Idempotent if nothing changed."""
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size <= self.offset:
            return []

        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read(size - self.offset)
        if not chunk:
            return []

        buf = self._pending + chunk
        out = []
        pos = 0
        while pos < len(buf):
            try:
                msg_size, after_varint = _DecodeVarint32(buf, pos)
            except IndexError:
                # Truncated varint — wait for the rest.
                break
            end = after_varint + msg_size
            if end > len(buf):
                # Incomplete payload — wait.
                break
            m = self.proto_class()
            m.ParseFromString(bytes(buf[after_varint:end]))
            out.append(m)
            pos = end

        # Advance file offset by the bytes we read this round; carry
        # the unconsumed tail forward as `_pending` so the next call
        # re-parses it together with newly-appended bytes.
        self.offset += len(chunk)
        self._pending = bytes(buf[pos:])
        return out


# ---------------------------------------------------------------------------
# LiveCoordinator
# ---------------------------------------------------------------------------

# Bokeh palette for newly-allocated glyphs.
from bokeh.palettes import Category10
_PALETTE = list(Category10[10])


@dataclass
class _PanelEntry:
    """One panel registered with the coordinator."""
    panel: object                    # panels_pb2.Panel
    figure: object                   # bokeh.plotting.figure
    scale_fn: Callable[[np.ndarray], np.ndarray]
    palette_idx: int = 0             # next color index for new series


class LiveCoordinator:
    """Drives the projector + Bokeh CDS streaming for --live mode."""

    def __init__(
        self,
        catalog: mc_pb.MetricCatalog,
        layout: metric_layout.PanelLayout,
        metadata_path: Path,
        meta: session_metadata_pb2.SessionMetadata,
        log: Callable[[str], None],
        series_factory: Callable[..., ColumnDataSource],
        panel_factory: Callable[..., tuple],
        t0_ns: int | None,
    ):
        self.catalog = catalog
        self.layout = layout
        self.metadata_path = metadata_path
        self.meta = meta
        self.log = log
        # Bokeh-side hooks — supplied by visualize_interactive.py so
        # this module doesn't need to know how panels are built.
        self.series_factory = series_factory   # (panel_entry, series, ts_s, vals) -> CDS
        self.panel_factory = panel_factory     # (panel, series_list, projector, projection, t0_ns) -> (fig, scale_fn)
        self.t0_ns = t0_ns                     # may be None until first sample arrives

        self.projector = TraceProjector(catalog)
        self.catalog_index = metric_catalog.build_index(catalog)

        # One TraceTail per active probe.
        self.tails: list[tuple[int, TraceTail]] = []
        for probe in meta.probes:
            out = self._resolve_path(probe.output_file)
            if probe.kind == session_metadata_pb2.PROBE_KIND_GPU:
                self.tails.append((probe.kind,
                                   TraceTail(out, gpu_metrics_pb2.GPUMetricsTrace)))
            elif probe.kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
                self.tails.append((probe.kind,
                                   TraceTail(out, system_metrics_pb2.SystemMetricsTrace)))
            elif probe.kind == session_metadata_pb2.PROBE_KIND_DISK:
                self.tails.append((probe.kind,
                                   TraceTail(out, disk_metrics_pb2.DiskMetricsTrace)))

        # Per-(fqn, scope_key) CDS + rows-already-streamed bookmark.
        self.cds_by_series: dict[tuple, ColumnDataSource] = {}
        self.streamed_count: dict[tuple, int] = {}

        # Registered panels, in render order.
        self.panels: list[_PanelEntry] = []

        # Per-PID removal-marker bookkeeping (don't re-draw the same
        # vertical span twice).
        self._removal_markers_drawn: set[tuple[int, object]] = set()

    # ------------------------------------------------------------------
    # Path resolution + bookkeeping
    # ------------------------------------------------------------------

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        for c in (Path.cwd() / pp, self.metadata_path.parent / pp.name,
                  self.metadata_path.parent / pp):
            if c.exists():
                return c
        return Path.cwd() / pp

    def register_panel(self, entry: _PanelEntry) -> None:
        self.panels.append(entry)

    def register_series(self, key: tuple, cds: ColumnDataSource) -> None:
        self.cds_by_series[key] = cds
        self.streamed_count[key] = len(cds.data.get("x", []))

    # ------------------------------------------------------------------
    # Periodic tick — called by Bokeh's add_periodic_callback
    # ------------------------------------------------------------------

    def tick(self) -> None:
        new_traces = 0
        for kind, tail in self.tails:
            for trace in tail.read_new_messages():
                new_traces += 1
                if kind == session_metadata_pb2.PROBE_KIND_GPU:
                    self.projector.ingest_gpu(trace)
                elif kind == session_metadata_pb2.PROBE_KIND_SYSTEM:
                    self.projector.ingest_system(trace)
                elif kind == session_metadata_pb2.PROBE_KIND_DISK:
                    self.projector.ingest_disk(trace)
        if new_traces == 0:
            return

        # If we didn't have a t0 anchor yet, take the earliest sample
        # we've seen across any series.
        if self.t0_ns is None:
            for cache in self.projector._caches.values():
                if cache.ts:
                    self.t0_ns = cache.ts[0] if self.t0_ns is None \
                                  else min(self.t0_ns, cache.ts[0])

        # 1. Stream new rows into existing CDSs.
        self._stream_new_rows()
        # 2. Allocate glyphs for newly-seen (scope, key) pairs.
        self._absorb_new_scope_keys()
        # 3. Draw removal markers for newly-removed PIDs.
        self._draw_pending_removals()

    def _series_native(self, series_key: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Return the full (ts_ns, vals) arrays for a (fqn, scope_key)
        from the projector's caches without touching numpy if we're
        going to just slice."""
        cache = self.projector._caches.get(series_key)
        if cache is None or not cache.ts:
            return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64))
        return (np.asarray(cache.ts, dtype=np.uint64),
                np.asarray(cache.vals, dtype=np.float64))

    def _stream_new_rows(self) -> None:
        if self.t0_ns is None:
            return
        for series_key, cds in self.cds_by_series.items():
            cache = self.projector._caches.get(series_key)
            if cache is None or not cache.ts:
                continue
            seen = self.streamed_count.get(series_key, 0)
            n = len(cache.ts)
            if n <= seen:
                continue
            new_ts = np.asarray(cache.ts[seen:n], dtype=np.int64)
            new_vals = np.asarray(cache.vals[seen:n], dtype=np.float64)
            self.streamed_count[series_key] = n

            # Apply the panel's scale_fn — same one used at initial
            # build. The CDS's "y" column carries display values, not
            # raw values, so the rest of the panel (peak line, axis
            # range) stays consistent.
            entry = self._owning_panel(series_key)
            scale_fn = entry.scale_fn if entry is not None else (lambda v: v)

            time_s = (new_ts - self.t0_ns) / 1e9
            cds.stream(dict(x=time_s, y=scale_fn(new_vals)))

    def _owning_panel(self, series_key: tuple) -> _PanelEntry | None:
        """Which panel registered this series. Linear search but
        panel count is small; OK for the live-tick cadence."""
        for entry in self.panels:
            fqn, scope_key = series_key
            for fig in [entry.figure]:
                # Map series_key back to panel via re-glob match.
                if metric_layout.fnmatch.fnmatchcase(fqn, entry.panel.series_glob):
                    desc = self.catalog_index.get(fqn) \
                        or metric_layout.synthesize_descriptor(fqn)
                    if desc.scope == entry.panel.scope:
                        return entry
        return None

    def _absorb_new_scope_keys(self) -> None:
        new_keys = self.projector.new_scope_keys_since_last_call()
        if not new_keys:
            return
        # For each (scope, [key, key, ...]) walk every panel of that
        # scope and add a glyph if a matching FQN cache now exists.
        for scope, keys in new_keys.items():
            for key in keys:
                for entry in self.panels:
                    if entry.panel.scope != scope:
                        continue
                    self._maybe_add_series_to_panel(entry, key)

    def _maybe_add_series_to_panel(self, entry: _PanelEntry,
                                   scope_key) -> None:
        # Walk every (fqn, scope_key) in the projector's caches and
        # add a glyph for any that match this panel's glob and that
        # we haven't already registered.
        for cache_key, cache in self.projector._caches.items():
            fqn, k = cache_key
            if k != scope_key:
                continue
            if cache_key in self.cds_by_series:
                continue
            if not metric_layout.fnmatch.fnmatchcase(fqn, entry.panel.series_glob):
                continue
            desc = self.catalog_index.get(fqn) \
                or metric_layout.synthesize_descriptor(fqn)
            if desc.scope != entry.panel.scope:
                continue
            # Allocate the glyph.
            ts = np.asarray(cache.ts, dtype=np.int64)
            vals = np.asarray(cache.vals, dtype=np.float64)
            time_s = (ts - (self.t0_ns or 0)) / 1e9
            display_vals = entry.scale_fn(vals)
            color = _PALETTE[entry.palette_idx % len(_PALETTE)]
            entry.palette_idx += 1
            cds = self.series_factory(
                entry=entry,
                fqn=fqn,
                scope_key=k,
                color=color,
                ts_s=time_s,
                vals=display_vals,
                descriptor=desc,
                projector=self.projector,
            )
            self.cds_by_series[cache_key] = cds
            self.streamed_count[cache_key] = ts.size
            self.log(f"  + new series [{entry.panel.title}] {fqn}  scope_key={k}")

    def _draw_pending_removals(self) -> None:
        pending = self.projector.pending_removals()
        if not pending:
            return
        # Only PROCESS-scope removals are emitted today.
        for scope, keys in pending.items():
            for key in keys:
                token = (scope, key)
                if token in self._removal_markers_drawn:
                    continue
                self._removal_markers_drawn.add(token)
                # Find figures that currently host this PID and add a
                # vertical Span at "now" (the most recent t in the
                # series' CDS).
                marker_x: float | None = None
                figures: list[object] = []
                for (fqn, k), cds in self.cds_by_series.items():
                    if k != key:
                        continue
                    xs = cds.data.get("x", [])
                    if len(xs):
                        marker_x = max(marker_x or float("-inf"), float(xs[-1]))
                    entry = self._owning_panel((fqn, k))
                    if entry is not None and entry.figure not in figures:
                        figures.append(entry.figure)
                if marker_x is None:
                    continue
                for fig in figures:
                    fig.add_layout(Span(
                        location=marker_x, dimension="height",
                        line_color="gray", line_dash="dotted",
                        line_alpha=0.6, line_width=1.5,
                    ))
                self.log(f"  ✗ PID {key} removed at t≈{marker_x:.2f}s")


# ---------------------------------------------------------------------------
# Bootstrap helper — wait for session_metadata.pb to appear
# ---------------------------------------------------------------------------

def wait_for_metadata(path: Path, timeout_s: float,
                       log: Callable[[str], None]) -> session_metadata_pb2.SessionMetadata:
    """Poll for `path` to appear. Returns the parsed SessionMetadata
    or raises TimeoutError after `timeout_s`."""
    deadline = time.time() + timeout_s
    last_log = 0.0
    while True:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    meta = session_metadata_pb2.SessionMetadata()
                    meta.ParseFromString(f.read())
                if meta.probes:
                    return meta
            except Exception:
                pass  # Partial mid-write — retry.
        if time.time() > deadline:
            raise TimeoutError(
                f"session_metadata.pb did not appear at {path} "
                f"within {timeout_s}s — is the suite running?"
            )
        now = time.time()
        if now - last_log >= 2.0:
            log(f"waiting for {path}...")
            last_log = now
        time.sleep(0.1)
