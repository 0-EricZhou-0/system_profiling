"""The byte-unit rule of both visualizers, for every bytes and bytes/s
axis (rates, gauges, cumulative panels) and the write-rate footer.

A unit is chosen from the largest value actually plotted, with a
threshold factor F (default 2) so it does not switch too early: >= F TiB
-> TiB, >= F GiB -> GiB, >= F MiB -> MiB, >= F KiB -> KiB, else B (with
F = 2 a 1.5 GiB peak reads as 1536 MiB; F = 1 switches as soon as a value
reaches the prefix). Rates the same with "/s". An empty or all-zero
series: B. F is one process-wide setting (set_scale_factor; the tools'
--unit-scale-factor), so every axis, total and footer of a render agree.
"""

from __future__ import annotations

import math

import numpy as np

_LADDER = [(1024.0 ** 4, "TiB"), (1024.0 ** 3, "GiB"), (1024.0 ** 2, "MiB"), (1024.0, "KiB")]

DEFAULT_SCALE_FACTOR = 2.0
_scale_factor = DEFAULT_SCALE_FACTOR


def set_scale_factor(factor: float) -> None:
    """Set the threshold factor for this process. ValueError unless it is
    a finite number >= 1."""
    global _scale_factor
    f = float(factor)
    if not math.isfinite(f) or f < 1.0:
        raise ValueError(f"unit scale factor must be >= 1 (1 = switch to a prefix as soon "
                         f"as a value reaches it), got {factor!r}")
    _scale_factor = f


def scale_factor() -> float:
    return _scale_factor


# --fit-axis-to-data (opt-in): a ceiling more than this many times the
# largest plotted value is "off-scale" — the data would fill under a fifth
# of an axis stretched to it — so the axis fits the data and the Peak is
# written, not drawn.
OFFSCALE_FACTOR = 5.0


def off_scale(peak: float | None, data_max: float) -> bool:
    """Is this ceiling off-scale for data up to data_max (see OFFSCALE_FACTOR)?"""
    return peak is not None and peak > 0 and peak > OFFSCALE_FACTOR * max(data_max, 0.0)


def byte_unit(largest: float | None, rate: bool = False,
              factor: float | None = None) -> tuple[float, str]:
    """(divisor, unit label) for values up to `largest` bytes (or bytes/s):
    the largest prefix P with largest >= factor x P (factor: the process
    setting unless given)."""
    f = _scale_factor if factor is None else factor
    sfx = "/s" if rate else ""
    if largest is not None and math.isfinite(largest):
        for div, name in _LADDER:
            if largest >= f * div:
                return div, name + sfx
    return 1.0, "B" + sfx


def fmt_bytes(v: float, rate: bool = False, factor: float | None = None) -> str:
    """One value in its own unit (the same rule): '1536 MiB', '3.2 GiB/s'."""
    div, name = byte_unit(abs(v), rate, factor)
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
