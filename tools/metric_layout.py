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

import panels_pb2 as _pn
import metric_catalog_pb2 as _mc

from google.protobuf import text_format

import metric_suffix as _suffix  # sibling under tools/


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

    d.unit = _suffix.unit_for(d.counter, d.submetric)
    # Auto-derive a human-readable description from the suffix table
    # so synthesized descriptors carry something useful for the
    # visualizer's auto-title fallback. Catalog-declared metrics get
    # their own `description` from the pbtxt and this is not used.
    d.description = _suffix.label_for(d.entity, d.counter, d.rollup, d.submetric)
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
        """Long-form pretty label for this series — same fallback chain
        as the visualizers' `_panel_title`:

          1. descriptor `description` (catalog-declared, wins if set)
          2. `metric_suffix.label_for(entity, counter, rollup, submetric)`
          3. raw FQN

        Used for the companion `<output>.legend.txt` and for tooltips.
        The scope-key suffix ("[PID 1234]", "[GPU 0: H100]", …) is
        appended by the visualizer's `_series_label`, not here."""
        d = self.descriptor
        if d.description:
            return d.description
        return _suffix.label_for(d.entity, d.counter, d.rollup, d.submetric) or self.fqn

    @property
    def label_short(self) -> str:
        """Compact legend label — just the pretty counter name. Keeps
        plot legends from spilling across the page. The panel title
        already carries the entity + suffix; the legend only needs to
        differentiate series *within* one panel."""
        return _suffix.pretty_counter(self.descriptor.counter) or self.fqn


def disambiguate_short_labels(
    series_list: list[ResolvedSeries],
) -> dict[tuple[str, object], str]:
    """For one panel's series list, return the minimum-length labels
    that keep every entry distinct.

    Starts from `ResolvedSeries.label_short`. For each group of series
    that collapse to the same short label, widens *only that group*
    by the first axis among {entity, rollup, submetric} that actually
    differs across the colliding series. The scope-key suffix
    ('[GPU 0: H100]', '[PID 1234]') is appended later by the
    visualizer and isn't considered here.

    The result is keyed by `(fqn, scope_key)` — that tuple is the
    visualizer's canonical series identity."""
    labels: dict[tuple[str, object], str] = {
        (s.fqn, s.scope_key): s.label_short for s in series_list
    }
    # Group series by their initial label.
    groups: dict[str, list[ResolvedSeries]] = {}
    for s in series_list:
        groups.setdefault(s.label_short, []).append(s)

    for label, members in groups.items():
        if len(members) < 2:
            continue
        # Pick the first axis on which these series actually differ.
        # Most realistic collisions disambiguate on entity (e.g.
        # nvlrx__bytes vs nvltx__bytes) or rollup (e.g. avg vs max).
        entities  = {m.descriptor.entity  for m in members}
        rollups   = {m.descriptor.rollup  for m in members}
        submetrics = {m.descriptor.submetric for m in members}
        # If every standard descriptor axis matches across the group,
        # the only thing that varies is scope_key (e.g. same FQN
        # measured on multiple GPUs / PIDs / devices). The visualizer
        # appends the scope-key suffix later, so leaving label_short
        # alone keeps the legend terse.
        if len(entities) == 1 and len(rollups) == 1 and len(submetrics) == 1:
            continue
        for s in members:
            d = s.descriptor
            if len(entities) > 1:
                labels[(s.fqn, s.scope_key)] = (
                    f"{_suffix.pretty_entity(d.entity)} {label}"
                )
            elif len(rollups) > 1 and d.rollup:
                labels[(s.fqn, s.scope_key)] = f"{label} ({d.rollup})"
            elif len(submetrics) > 1 and d.submetric:
                labels[(s.fqn, s.scope_key)] = f"{label} [{d.submetric}]"
    return labels


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
