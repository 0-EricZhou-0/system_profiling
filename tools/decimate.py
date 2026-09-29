"""Per-pixel min/max reduction of a line series for the Bokeh page
(tools/decimate.js is the page's line-for-line port; tests/python/
test_decimate.py runs both on the same inputs).

The page keeps every series' full data and draws a reduced copy: for the
view [start, end] of width w, the samples in [start - w, end + w] (plus
the one just outside each end, so the line reaches the window's edges)
are grouped into columns of width w / cols on a grid anchored at 0 (a pan
that keeps the width keeps the columns). Each column keeps its first and
last sample, its lowest and highest finite one, and its first and last
NaN (a gap: the line stays broken there). Every kept point is a real
sample, and each column's min and max are those of the full data, so the
drawn line is the full one at pixel resolution. When the window holds at
most EXACT_PER_COLUMN samples per column, all of them are kept (drawn
exactly). The series' own first, last, lowest and highest samples are
always drawn too (extremes()), so the drawn data spans the full data.
"""

from __future__ import annotations

import numpy as np

EXACT_PER_COLUMN = 4


def window(start: float, end: float, cols: int) -> tuple[float, float, float]:
    """(lo, hi, column width) for the view [start, end]."""
    w = end - start
    return start - w, end + w, w / cols


def keep(x: np.ndarray, y: np.ndarray, lo: float, hi: float, bw: float) -> np.ndarray:
    """Indices (ascending) of the samples to draw. x ascending."""
    n = x.size
    i0 = max(int(np.searchsorted(x, lo, side="left")) - 1, 0)
    i1 = min(int(np.searchsorted(x, hi, side="right")) + 1, n)
    if i1 <= i0:
        return np.arange(0, dtype=np.int64)
    if i1 - i0 <= EXACT_PER_COLUMN * (hi - lo) / bw:
        return np.arange(i0, i1, dtype=np.int64)
    xs, ys = x[i0:i1], y[i0:i1]
    col = np.floor(xs / bw)
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    ends = np.r_[starts[1:], xs.size]                   # exclusive
    counts = ends - starts
    group = np.repeat(np.arange(starts.size), counts)
    finite = np.isfinite(ys)
    out = [starts, ends - 1]
    for arr, red in ((np.where(finite, ys, np.inf), np.minimum),
                     (np.where(finite, ys, -np.inf), np.maximum)):
        g = red.reduceat(arr, starts)
        hit = (arr == np.repeat(g, counts)) & finite
        pos = np.flatnonzero(hit)
        _, first = np.unique(group[pos], return_index=True)
        out.append(pos[first])                           # first index at the extreme
    nan = np.flatnonzero(~finite)
    if nan.size:
        _, first = np.unique(group[nan], return_index=True)
        _, last = np.unique(group[nan][::-1], return_index=True)
        out += [nan[first], nan[::-1][last]]
    return np.unique(np.concatenate(out)) + i0


def extremes(y: np.ndarray) -> list[int]:
    """The series' first and last sample and its lowest and highest finite
    one (first index at each): always drawn, so the drawn data spans the
    full data's x and y (the page's auto-ranged axes and its reset tool
    see the full extent). Outside the window they are off-screen."""
    n = y.size
    if n == 0:
        return []
    out = [0, n - 1]
    fin = np.isfinite(y)
    if fin.any():
        out += [int(np.argmin(np.where(fin, y, np.inf))), int(np.argmax(np.where(fin, y, -np.inf)))]
    return out


def indices(x: np.ndarray, y: np.ndarray, start: float, end: float, cols: int) -> np.ndarray:
    """Rows to draw for the view [start, end]: keep() plus extremes()."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    lo, hi, bw = window(start, end, cols)
    return np.unique(np.concatenate([keep(x, y, lo, hi, bw), np.asarray(extremes(y), np.int64)]))


def reduce(x: np.ndarray, y: np.ndarray, start: float, end: float, cols: int):
    """(x, y) to draw for the view [start, end]."""
    k = indices(x, y, start, end, cols)
    return np.asarray(x)[k], np.asarray(y)[k]
