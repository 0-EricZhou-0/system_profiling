"""Which series a panel's legend lists, shared by both visualizers.

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


LEGEND_MAX_ENTRIES = 10
OTHER_COLOR = "#cccccc"


def activity(ts_ns: np.ndarray, vals: np.ndarray) -> float:
    """How much a series did over the run: the time integral of |value|
    (trapezoid, value x seconds). A single sample counts by its value."""
    if vals.size == 0:
        return 0.0
    v = np.abs(vals.astype(np.float64))
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
