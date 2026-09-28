"""The byte-unit rule of both visualizers, for every bytes and bytes/s
axis (rates, gauges, cumulative panels), the write-rate footer and the
run totals.

A unit is chosen from the largest value actually plotted, with a 2x
threshold so it does not switch too early: >= 2 TiB -> TiB, >= 2 GiB ->
GiB, >= 2 MiB -> MiB, >= 2 KiB -> KiB, else B (a 1.5 GiB peak reads as
1536 MiB). Rates the same with "/s". An empty or all-zero series: B.
"""

from __future__ import annotations

import math

import numpy as np

_LADDER = [(1024.0 ** 4, "TiB"), (1024.0 ** 3, "GiB"), (1024.0 ** 2, "MiB"), (1024.0, "KiB")]


def byte_unit(largest: float | None, rate: bool = False) -> tuple[float, str]:
    """(divisor, unit label) for values up to `largest` bytes (or bytes/s)."""
    sfx = "/s" if rate else ""
    if largest is not None and math.isfinite(largest):
        for div, name in _LADDER:
            if largest >= 2 * div:
                return div, name + sfx
    return 1.0, "B" + sfx


def fmt_bytes(v: float, rate: bool = False) -> str:
    """One value in its own unit (the same rule): '1536 MiB', '3.2 GiB/s'."""
    div, name = byte_unit(abs(v), rate)
    x = v / div
    if x == int(x):
        s = f"{int(x):,}"
    elif abs(x) >= 100:
        s = f"{x:,.0f}"
    elif abs(x) >= 1:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
    else:
        s = f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{s} {name}"


def largest(arrays) -> float:
    """Largest finite value over some arrays (0 when there is none)."""
    best = 0.0
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            best = max(best, float(a.max()))
    return best
