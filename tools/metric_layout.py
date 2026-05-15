"""Panel layout loader + helpers for resolving each panel to a
concrete list of series.

The panel layout (proto/panels.proto) is a pbtxt sidecar that pairs
catalog FQNs with display semantics (title, scope filter, y-range,
peak source). The renderers walk this list and look up matching
series in the TraceProjector's output.

This module also synthesizes a MetricDescriptor from a bare FQN string
when no catalog entry is present — needed for GPU FQNs (which are
chip-specific and not declared in the static catalog).
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from cupti_profiler.proto import panels_pb2 as _pn
    from cupti_profiler.proto import metric_catalog_pb2 as _mc
except ImportError:
    import panels_pb2 as _pn  # noqa: F401
    import metric_catalog_pb2 as _mc

from google.protobuf import text_format


Panel = _pn.Panel
PanelLayout = _pn.PanelLayout
Unit = _mc.Unit
Scope = _mc.Scope
MetricType = _mc.MetricType
MetricDescriptor = _mc.MetricDescriptor


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panel_layout(path: str | os.PathLike) -> PanelLayout:
    """Parse a PanelLayout text-format pbtxt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"panel layout not found: {p}\n"
            f"  Pass --panel-layout PATH or use the example at "
            f"configs/visualizer_panels.pbtxt."
        )
    layout = PanelLayout()
    text_format.Parse(p.read_text(), layout)
    return layout


# ---------------------------------------------------------------------------
# GPU FQN -> MetricDescriptor synthesis
# ---------------------------------------------------------------------------

# Counter-name fragments that indicate the underlying value is bytes
# (and so a `.per_second` suffix means bytes-per-second rather than Hz).
_BYTE_COUNTER_FRAGMENTS = ("bytes", "throughput")


def _infer_unit(entity: str, counter: str, submetric: str) -> int:
    """Best-effort unit inference for an FQN that isn't declared in
    the static catalog. Documented heuristic (see docs/metric-model.md):

      .ratio                    -> RATIO
      .pct                      -> PCT
      .pct_of_peak_*            -> PCT
      .per_second on byte-y     -> BYTES_PER_SEC
      .per_second on cycle-y    -> HZ
      .per_cycle_*              -> COUNT
      everything else           -> COUNT
    """
    s = submetric.lower()
    if s == "ratio":
        return _mc.UNIT_RATIO
    if s == "pct" or s.startswith("pct_of_"):
        return _mc.UNIT_PCT
    if s == "per_second":
        is_bytey = any(frag in counter.lower() for frag in _BYTE_COUNTER_FRAGMENTS)
        return _mc.UNIT_BYTES_PER_SEC if is_bytey else _mc.UNIT_HZ
    if s.startswith("per_cycle"):
        return _mc.UNIT_COUNT
    if s.startswith("peak_"):
        # peak_* by itself evaluates to a constant — render as count.
        return _mc.UNIT_COUNT
    return _mc.UNIT_COUNT


def synthesize_descriptor(fqn: str) -> MetricDescriptor:
    """Build a best-effort MetricDescriptor for an FQN not in the
    catalog. Used for GPU metrics — the chip-specific catalog is
    enumerated at runtime via cuptiProfilerHostGetSubMetrics(), and
    until that's wired into the suite this fallback keeps the
    visualizer functional.

    Parses `<entity>__<counter>[.<rollup>][.<submetric>]` and infers
    the unit from the suffix. The scope is always SCOPE_GPU (the only
    domain that today emits non-catalog FQNs). The type is COUNTER
    unless the suffix clearly indicates RATIO or THROUGHPUT.
    """
    d = MetricDescriptor()
    d.fqn = fqn
    d.scope = _mc.SCOPE_GPU

    if "__" in fqn:
        entity, rest = fqn.split("__", 1)
        d.entity = entity
        parts = rest.split(".")
        d.counter = parts[0]
        if len(parts) >= 2:
            d.rollup = parts[1]
        if len(parts) >= 3:
            # The submetric is the rest joined back — `.pct_of_peak_sustained_elapsed`
            # is one logical suffix even though it has underscores.
            d.submetric = ".".join(parts[2:])
    else:
        d.entity = fqn
        d.counter = fqn

    sm = d.submetric.lower()
    if sm == "ratio":
        d.type = _mc.METRIC_TYPE_RATIO
    elif sm.startswith("pct_of_peak_"):
        d.type = _mc.METRIC_TYPE_THROUGHPUT
    else:
        d.type = _mc.METRIC_TYPE_COUNTER

    d.unit = _infer_unit(d.entity, d.counter, d.submetric)
    d.smoothable = True
    return d


def get_or_synthesize_descriptor(
    catalog_index: dict[str, MetricDescriptor], fqn: str
) -> MetricDescriptor:
    """Lookup-or-synthesize. Cached synthesis would be a nice
    optimization but in practice this is only called once per series
    at render-build time."""
    d = catalog_index.get(fqn)
    if d is not None:
        return d
    return synthesize_descriptor(fqn)


# ---------------------------------------------------------------------------
# Panel -> series resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedSeries:
    """One series within one panel — directly renderable."""
    fqn: str
    scope: int
    scope_key: object              # int / str / None
    descriptor: MetricDescriptor   # catalog entry or synthesized

    @property
    def label(self) -> str:
        """Friendly legend label. Falls through to the FQN."""
        return self.fqn


def resolve_panel_series(
    panel: Panel,
    catalog_index: dict[str, MetricDescriptor],
    series_keys: Iterable[tuple[str, object]],
) -> list[ResolvedSeries]:
    """For one panel, list the series the renderer should draw.

    `series_keys` is whatever (fqn, scope_key) tuples the projector
    has produced; we filter to those whose FQN matches the panel's
    series_glob. If `panel.scope` is set (non-zero) we additionally
    require the descriptor's scope to match — but in practice the
    FQN glob is enough because entity prefixes don't overlap across
    scopes (`cpu__*` is always SCOPE_SYSTEM, `proc__*` is always
    SCOPE_PROCESS, etc.), so leaving `panel.scope` unset works."""
    out: list[ResolvedSeries] = []
    for fqn, scope_key in series_keys:
        if not fnmatch.fnmatchcase(fqn, panel.series_glob):
            continue
        d = get_or_synthesize_descriptor(catalog_index, fqn)
        if panel.scope != _mc.SCOPE_UNSPECIFIED and d.scope != panel.scope:
            continue
        out.append(ResolvedSeries(
            fqn=fqn, scope=d.scope, scope_key=scope_key, descriptor=d,
        ))
    return out
