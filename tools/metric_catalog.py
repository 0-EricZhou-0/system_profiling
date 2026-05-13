"""Python-side wrapper around the MetricCatalog pbtxt.

The renderers (visualize_all.py / visualize_interactive.py) load a
MetricCatalog once at startup and consult it to:

  - Look up a metric's `MetricType`, `Unit`, `Scope`, smoothability,
    and reference peak (Find / iter_by_scope).
  - Resolve a descriptor's peak value at plot time (resolve_peak) by
    dispatching on which field of `MetricDescriptor.peak` is set:
       peak_constant  -> the constant itself
       peak_ref       -> first-value lookup of that FQN in the data
       peak_expr      -> evaluate a named formula against HostMeta

Authoritative source for catalog format: proto/metric_catalog.proto.
The catalog is also inlined into session_metadata.pb at runtime
(SessionMetadata.catalog), so the visualizer can pick it up from the
manifest without needing a separate file dependency.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

# Generated proto bindings — assumed importable (either dev path
# `PYTHONPATH=generated/proto` or pip-installed under
# cupti_profiler.proto). Try both shapes.
try:
    from cupti_profiler.proto import metric_catalog_pb2 as _mc
except ImportError:
    import metric_catalog_pb2 as _mc  # noqa: F401

from google.protobuf import text_format


# Re-export the enum classes for convenience (so callers don't need a
# separate `import metric_catalog_pb2`).
MetricType = _mc.MetricType
Unit = _mc.Unit
Scope = _mc.Scope
MetricDescriptor = _mc.MetricDescriptor
MetricCatalog = _mc.MetricCatalog


# ---------------------------------------------------------------------------
# HostMeta + peak-expr allow-list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostMeta:
    """Per-run host context. Populated from TraceHeader fields the
    profiler writes into every .pb stream."""
    hostname: str = ""
    cpu_count: int = 0


# Named formulas that `MetricDescriptor.peak_expr` and
# `Panel.peak_from_expr` may reference. Allow-list — never eval()
# arbitrary strings (the user prefers no second config format, so this
# is the way to keep the descriptor purely declarative).
PEAK_EXPRS: dict[str, Callable[[HostMeta], float]] = {
    # "% of one CPU core, summed across N cores" -> 100 * ncpus.
    "ncpus_x_100": lambda host: float(host.cpu_count) * 100.0,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_catalog(path: str | os.PathLike) -> MetricCatalog:
    """Parse a MetricCatalog text-format pbtxt file. Returns the
    proto message verbatim — callers index it via the helpers below."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"metric catalog not found: {p}\n"
            f"  Pass --catalog PATH to point at your own MetricCatalog "
            f"pbtxt, or use the example at configs/metric_catalog.pbtxt."
        )
    cat = MetricCatalog()
    text_format.Parse(p.read_text(), cat)
    return cat


def load_catalog_from_session_metadata(session_meta) -> MetricCatalog:
    """Extract the inlined MetricCatalog from a parsed SessionMetadata
    proto. This is the preferred path — the suite writes its active
    catalog into session_metadata.pb at Start()/Stop() so the
    visualizer doesn't need a separate file dependency."""
    if not session_meta.HasField("catalog") or not session_meta.catalog.metrics:
        raise ValueError(
            "session_metadata.pb has no inlined MetricCatalog. The .pb "
            "file was probably written by a profiler version older than "
            "the unified-metric-model branch — re-run the workload with "
            "the current libcupti_profiler.so."
        )
    return session_meta.catalog


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def build_index(catalog: MetricCatalog) -> dict[str, MetricDescriptor]:
    """FQN → descriptor dict for O(1) lookup."""
    return {d.fqn: d for d in catalog.metrics}


def find(catalog: MetricCatalog, fqn: str) -> Optional[MetricDescriptor]:
    """O(N) Find — for one-off lookups. Use build_index() for hot paths."""
    for d in catalog.metrics:
        if d.fqn == fqn:
            return d
    return None


def iter_by_scope(catalog: MetricCatalog, scope: int) -> Iterable[MetricDescriptor]:
    """Yield descriptors whose .scope matches. Stable order — used by
    probes to fix their ScopeMetricNames registry, so don't sort."""
    for d in catalog.metrics:
        if d.scope == scope:
            yield d


# ---------------------------------------------------------------------------
# Peak resolution
# ---------------------------------------------------------------------------

# Sentinel returned by resolve_peak when no peak is declared OR a
# peak_ref / peak_expr can't be satisfied yet — the renderer falls
# back to auto-scale in that case.
NO_PEAK: Optional[float] = None


def resolve_peak(
    descriptor: MetricDescriptor,
    host: HostMeta,
    lookup_first_value: Callable[[str], Optional[float]] | None = None,
) -> Optional[float]:
    """Evaluate the descriptor's `peak` oneof.

    Args:
      descriptor:           the MetricDescriptor whose peak to resolve.
      host:                 captured TraceHeader fields (for peak_expr).
      lookup_first_value:   callable that returns the first sample value
                            of an FQN — used by peak_ref. Pass None to
                            disable peak_ref resolution (e.g. before any
                            samples have arrived).

    Returns the resolved peak as a float, or None if:
      - no peak is declared (oneof unset)
      - peak_ref points at an FQN the data hasn't carried yet
      - peak_expr names a formula not in the allow-list
    """
    which = descriptor.WhichOneof("peak")
    if which is None:
        return NO_PEAK
    if which == "peak_constant":
        return float(descriptor.peak_constant)
    if which == "peak_ref":
        if lookup_first_value is None:
            return NO_PEAK
        return lookup_first_value(descriptor.peak_ref)
    if which == "peak_expr":
        fn = PEAK_EXPRS.get(descriptor.peak_expr)
        if fn is None:
            print(
                f"[metric_catalog] unknown peak_expr {descriptor.peak_expr!r} "
                f"on {descriptor.fqn!r} — falling back to auto-scale.",
                file=sys.stderr,
            )
            return NO_PEAK
        return float(fn(host))
    return NO_PEAK
