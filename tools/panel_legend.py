"""What a panel draws — colours, line styles, which series its legend
lists and with which labels — shared by both visualizers.

Legends sit above their panel. A panel can hold many series: a cold
vLLM start tracks ~160 processes, most of them short-lived compilers.
A legend then lists the LEGEND_MAX_ENTRIES most active entries and one
"+k more" entry for the rest, which are drawn in OTHER_COLOR so that
every colour still in the legend names one line.

LEGEND_MAX_ENTRIES is the length of the default colour cycle
(matplotlib's tab10, Bokeh's Category10): past ten, colours repeat, so
a further legend entry could not name its line anyway.
"""

from __future__ import annotations

from typing import Hashable, Iterable

import numpy as np

import metric_catalog_pb2 as _mc
import metric_layout


LEGEND_MAX_ENTRIES = 10
OTHER_COLOR = "#cccccc"


def activity(ts_ns: np.ndarray, vals: np.ndarray) -> float:
    """How much a series did over the run: the time integral of |value|
    (trapezoid, value x seconds). A single sample counts by its value.
    Missing values (NaN: e.g. memory of a process whose statm could not
    be read) count as nothing."""
    if vals.size == 0:
        return 0.0
    v = np.nan_to_num(np.abs(vals.astype(np.float64)), nan=0.0)
    if ts_ns.size < 2:
        return float(v.sum())
    dt_s = np.diff(ts_ns.astype(np.int64)) / 1e9
    return float(np.sum(0.5 * (v[:-1] + v[1:]) * dt_s))


def cap(activities: Iterable[tuple[Hashable, float]],
        limit: int = LEGEND_MAX_ENTRIES) -> tuple[list, list]:
    """Split legend keys into (shown, hidden). Everything is shown when
    there are at most `limit`; otherwise the `limit` most active, in
    their original order, and the rest hidden. Ties keep the original
    order."""
    items = list(activities)
    if len(items) <= limit:
        return [k for k, _a in items], []
    order = sorted(range(len(items)), key=lambda i: (-items[i][1], i))
    keep = set(order[:limit])
    shown = [items[i][0] for i in range(len(items)) if i in keep]
    hidden = [items[i][0] for i in range(len(items)) if i not in keep]
    return shown, hidden


def more_label(n_hidden: int) -> str:
    return f"+{n_hidden} more"


# ---------------------------------------------------------------------------
# Colours, line styles and entries of one panel (both renderers)
# ---------------------------------------------------------------------------

# The colour cycle of both renderers: matplotlib's tab10 (its default
# cycle) = Bokeh's Category10[10].
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# A per-PID panel can hold several metrics of one process (the I/O
# panels: rchar and wchar; read_bytes, write_bytes and cancelled). The
# colour stays the PID's, the same in every panel; the line style tells
# its metrics apart: the n-th distinct metric of the panel gets style n
# (0 solid, 1 dashed, 2 dotted, 3 dash-dot; each renderer maps them).
# Panels with a single metric per process keep solid lines.
N_METRIC_STYLES = 4
METRIC_COLOR = "black"   # the line-style entries' colour


def series_label(series, projector, base: str | None = None) -> str:
    """Compact legend label (the panel title carries the entity and
    suffix). `base` overrides `series.label_short`: pass the label from
    `metric_layout.disambiguate_short_labels`."""
    if base is None:
        base = series.label_short
    key = series.scope_key
    if series.scope == _mc.SCOPE_SYSTEM:
        return base
    if series.scope == _mc.SCOPE_PROCESS:
        return f"{base}  [{process_label(projector, key)}]"
    if series.scope == _mc.SCOPE_DEVICE:
        return f"{base}  [{key}]"
    if series.scope == _mc.SCOPE_GPU:
        # Single-GPU runs need no suffix; with several GPUs, the index.
        if len(projector.gpu_info) <= 1:
            return base
        return f"{base}  [GPU {key}]"
    return base


def process_label(projector, key) -> str:
    """'vllm/VLLM::EngineCor (PID 7, child of 6)': discovered processes
    name the tracked parent they were found under, so they read apart
    from the listed roots."""
    tp = projector.tracked_processes.get(int(key))
    found = f", child of {tp.parent_pid}" if tp and tp.discovered else ""
    name = f"{tp.alias} " if tp and tp.alias else ""
    return f"{name}(PID {key}{found})"


# Entities drawn in one colour each, their metrics told apart by line
# style (a process: its rchar and wchar; a disk device: its read and
# write bytes), never summed across entities.
ENTITY_SCOPES = (_mc.SCOPE_PROCESS, _mc.SCOPE_DEVICE)


def metric_styles(series_list) -> dict:
    """(fqn, scope_key) -> line style index (see N_METRIC_STYLES)."""
    fqns = list(dict.fromkeys(s.fqn for s in series_list if s.scope in ENTITY_SCOPES))
    return {(s.fqn, s.scope_key): (fqns.index(s.fqn) % N_METRIC_STYLES
                                   if len(fqns) > 1 and s.scope in ENTITY_SCOPES else 0)
            for s in series_list}


def device_key(s) -> tuple:
    """metric_color_map's key of a disk device's colour."""
    return ("device", s.scope_key)


def pid_color_map(panels, projection) -> dict:
    """PID -> colour, shared by every per-process panel (and the process
    timeline) so a process has one colour throughout. `panels`:
    (panel, series_list, kind) in layout order. Colours go by rank: the
    processes most active in the first per-process metric panel
    (per-process CPU in both shipped layouts) first, the rest in the
    order they appear, so with more processes than colours the busiest
    ones still get distinct colours."""
    seen: list = []
    rank_act: dict = {}
    first_panel = None
    for _p, series_list, kind in panels:
        for s in series_list:
            if s.scope != _mc.SCOPE_PROCESS:
                continue
            pid = int(s.scope_key)
            if pid not in seen:
                seen.append(pid)
            if first_panel is None and kind == "metric":
                first_panel = id(series_list)
            if id(series_list) == first_panel:
                rank_act[pid] = rank_act.get(pid, 0.0) + activity(
                    *projection[(s.fqn, s.scope_key)])
    order = sorted(range(len(seen)),
                   key=lambda i: (seen[i] not in rank_act, -rank_act.get(seen[i], 0.0), i))
    return {seen[i]: COLORS[r % len(COLORS)] for r, i in enumerate(order)}


# Statistic variants of one metric (sm__cycles_active.avg / .max, ...)
# share its hue: max in the full colour, avg tinted halfway to white, min
# three quarters, sum shaded a third toward black (a sum over instances
# is at least their max). Only when a panel holds two or more statistics
# of one base; alone, a statistic has the full colour.
STAT_SHADE = {"max": 0.0, "avg": 0.5, "min": 0.75, "sum": -0.35}


def shade(color: str, f: float) -> str:
    """`color` moved a fraction f toward white (f > 0) or black (f < 0)."""
    c = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x + (1 - x) * f if f >= 0 else x * (1 + f) for x in c]
    return "#" + "".join(f"{round(x * 255):02x}" for x in c)


def metric_base(s) -> tuple:
    """A series' metric without its statistic: entity, counter, submetric
    and the instance it is of (device, GPU; none for whole-system)."""
    d = s.descriptor
    return (d.entity, d.counter, d.submetric,
            None if s.scope == _mc.SCOPE_SYSTEM else s.scope_key)


def metric_color_map(panels) -> dict:
    """Base metric -> hue, for the series that are not per-process, in
    layout order across the whole figure, so a metric keeps its hue and
    the cycle continues from panel to panel (as v0.0.1's figure did). A
    disk device has one colour for all its metrics (key device_key).
    `panels`: (panel, series_list, kind) in layout order."""
    out: dict = {}
    for _p, series_list, _k in panels:
        for s in series_list:
            if s.scope == _mc.SCOPE_DEVICE:
                out.setdefault(device_key(s), COLORS[len(out) % len(COLORS)])
            elif s.scope != _mc.SCOPE_PROCESS:
                out.setdefault(metric_base(s), COLORS[len(out) % len(COLORS)])
    return out


END_LINE_COLOR = "black"   # the PNG's; the Bokeh page uses its theme's ink


def wants_end_lines(kind: str, unit: int) -> bool:
    """Exit end lines go on every cumulative panel and on per-process
    gauges (a level, not a rate: bytes such as RSS, counts)."""
    return kind in ("integrated", "cumulative") or unit in (_mc.UNIT_BYTES, _mc.UNIT_COUNT)


def end_lines(series_list, plotted: dict, projector, p: "Plan") -> list:
    """Each process that exited ends in a dashed vertical line from 0 up
    to its series' last value at that series' last time, so its end reads
    as an end (the series itself stops at its last value; a gauge's last
    samples during the exit carry no value). plotted: (fqn, scope_key) ->
    (ts_ns, values) as drawn (a cumulative panel: the running totals).
    Returns [(key, t_ns, value)] for the per-process series of processes
    the trace saw exit (listed or grey alike)."""
    exited = {r.pid for r in projector.process_table.values() if r.end_time_ns}
    out = []
    for s in series_list:
        k = (s.fqn, s.scope_key)
        if (s.scope != _mc.SCOPE_PROCESS or int(s.scope_key) not in exited
                or k not in p.styles or k not in plotted):
            continue
        ts, vals = plotted[k]
        ok = np.flatnonzero(np.isfinite(np.asarray(vals, dtype=np.float64)))
        if ok.size:
            out.append((k, int(ts[ok[-1]]), float(vals[ok[-1]])))
    return out


def _distinct_colors(keys_by_rank: list, colors: dict) -> dict:
    """Colours for a panel's listed legend keys (most active first), each
    its own: a key whose colour an earlier one already has gets the first
    colour of the cycle not yet used in the panel."""
    out, used = {}, set()
    for k in keys_by_rank:
        c = colors[k]
        if c in used:
            c = next((x for x in COLORS if x not in used), c)
        out[k] = c
        used.add(c)
    return out


class Plan:
    """What a panel draws, before any renderer draws it.

    styles:  (fqn, scope_key) -> (color, style index, listed); a series
             not listed in the legend is drawn in OTHER_COLOR.
    entries: [(label, color, style index, keys)] in legend order, keys =
             the series the entry stands for; line-style entries have
             METRIC_COLOR.
    order:   the series keys in drawing order: the unlisted (grey) ones
             first, then the listed ones from the most active to the
             least, so a smaller series is drawn over a larger one (the
             mean over the SMs over the busiest SM) instead of hidden
             under it.
    """

    def __init__(self, styles: dict, entries: list, amounts: dict):
        self.styles = styles
        self.entries = entries
        keys = list(styles)
        self.order = sorted(keys, key=lambda k: (styles[k][2], -amounts.get(k, 0.0),
                                                 keys.index(k)))


def plan(series_list, projector, projection: dict, pid_colors: dict,
         totals: dict | None = None,
         metric_colors: dict | None = None) -> Plan:
    """Colours, line styles and legend entries of one panel. A metric
    panel ranks its entries by activity; a cumulative companion, given
    `totals` ((fqn, scope_key) -> run total, full resolution), by those.
    Entries name the series or process only; the values are on the axis."""
    label_bases = metric_layout.disambiguate_short_labels(series_list)
    styles_idx = metric_styles(series_list)
    styled = any(v for v in styles_idx.values())
    live = [s for s in series_list if projection[(s.fqn, s.scope_key)][0].size]
    cumulative = totals is not None

    if cumulative:
        amount = {(s.fqn, s.scope_key): totals.get((s.fqn, s.scope_key), 0.0) for s in live}
    else:
        amount = {(s.fqn, s.scope_key): activity(*projection[(s.fqn, s.scope_key)])
                  for s in live}

    colors = {}
    color_idx = 0
    groups: dict = {}                     # base metric (a device) -> its series, not per-process
    for s in live:
        if s.scope == _mc.SCOPE_PROCESS and int(s.scope_key) in pid_colors:
            colors[(s.fqn, s.scope_key)] = pid_colors[int(s.scope_key)]
        elif s.scope == _mc.SCOPE_PROCESS:
            colors[(s.fqn, s.scope_key)] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
        elif s.scope == _mc.SCOPE_DEVICE:
            groups.setdefault(device_key(s), []).append(s)
        else:
            groups.setdefault(metric_base(s), []).append(s)
    # One hue per base metric or device (the figure-wide one), distinct
    # within the panel (the most active group keeps its hue); a metric's
    # statistics shaded.
    hues, used = {}, set()
    for b in sorted(groups, key=lambda b: -sum(amount[(s.fqn, s.scope_key)] for s in groups[b])):
        h = (metric_colors or {}).get(b) or COLORS[len(hues) % len(COLORS)]
        if h in used:
            h = next((c for c in COLORS if c not in used), h)
        hues[b] = h
        used.add(h)
    for b, ss in groups.items():
        several = len({(s.descriptor.rollup or "").lower() for s in ss}) > 1
        for s in ss:
            f = STAT_SHADE.get((s.descriptor.rollup or "").lower(), 0.0) if several else 0.0
            colors[(s.fqn, s.scope_key)] = shade(hues[b], f)

    entries = []
    if styled:
        # One entry per process or device (its colour) and one per metric
        # (its line style), not every entity x metric pair.
        fqn_base = {s.fqn: label_bases[(s.fqn, s.scope_key)] for s in live}
        procs: dict = {}
        for s in live:
            procs.setdefault(s.scope_key, []).append(s)
        proc_act = {k: sum(amount[(s.fqn, k)] for s in ss) for k, ss in procs.items()}
        shown, hidden = cap(list(proc_act.items()))
        listed = set(shown)
        pcolor = _distinct_colors(sorted(shown, key=lambda k: -proc_act[k]),
                                  {k: colors[(procs[k][0].fqn, k)] for k in shown})
        for k, c in pcolor.items():
            for s in procs[k]:
                colors[(s.fqn, k)] = c
        for k in shown:
            name = (process_label(projector, k) if procs[k][0].scope == _mc.SCOPE_PROCESS
                    else str(k))
            entries.append((name, pcolor[k], 0, [(s.fqn, k) for s in procs[k]]))
        if hidden:
            entries.append((more_label(len(hidden)), OTHER_COLOR, 0,
                            [(s.fqn, k) for k in hidden for s in procs[k]]))
        for fqn, st in dict.fromkeys((s.fqn, styles_idx[(s.fqn, s.scope_key)]) for s in live):
            entries.append((fqn_base[fqn], METRIC_COLOR, st,
                            [(s.fqn, s.scope_key) for s in live if s.fqn == fqn]))
        styles = {(s.fqn, s.scope_key): (colors[(s.fqn, s.scope_key)],
                                         styles_idx[(s.fqn, s.scope_key)],
                                         s.scope_key in listed) for s in live}
    else:
        keys = [(s.fqn, s.scope_key) for s in live]
        by_key_scope = {(s.fqn, s.scope_key): s.scope for s in live}
        shown, hidden = cap([(k, amount[k]) for k in keys])
        listed = set(shown)
        # (Metric hues are distinct per group already; processes here.)
        colors.update(_distinct_colors(sorted((k for k in shown if by_key_scope[k] == _mc.SCOPE_PROCESS),
                                              key=lambda k: -amount[k]), colors))
        by_key = {(s.fqn, s.scope_key): s for s in live}
        for k in shown:
            label = series_label(by_key[k], projector, base=label_bases[k])
            entries.append((label, colors[k], styles_idx[k], [k]))
        if hidden:
            entries.append((more_label(len(hidden)), OTHER_COLOR, 0, list(hidden)))
        styles = {k: (colors[k], styles_idx[k], k in listed) for k in keys}
    return Plan(styles, entries, amount)
