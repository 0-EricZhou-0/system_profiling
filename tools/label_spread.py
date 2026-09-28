"""Labels along one axis without overlap, each as close to its anchor as
the others allow. Used by the event / region strips (rotated labels
spread sideways) and by the process timeline's external labels, in both
visualizers.

spread_1d() merges overlapping labels into clusters, centres each
cluster on the mean of its members' wanted positions (the least-squares
placement), keeps clusters inside [lo, hi], and merges again until no
two overlap. If everything fits in [lo, hi] nothing overlaps.

assign_rows() splits labels into as few rows as needed for each row's
labels to fit side by side in [lo, hi], neighbours in x going to
different rows, so a dense burst becomes a few rows of labels near their
anchors instead of one row pushed far away.
"""

from __future__ import annotations


def spread_1d(items: list[tuple[float, float]], lo: float, hi: float,
              pad: float = 0.0) -> list[float]:
    """items: [(wanted centre, width)]; returns the centres, in input
    order, with at least `pad` between neighbours, inside [lo, hi] when
    they fit."""
    n = len(items)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: items[i][0])
    # cluster: [first, last, left edge]; offs[j] = j's left edge - its cluster's left edge
    offs, clusters = [0.0] * n, []

    def width(c):
        a, b = c[0], c[1]
        return offs[b] + items[order[b]][1] - offs[a] + 0.0

    def place(c):
        a, b = c[0], c[1]
        wanted = [items[order[j]][0] - items[order[j]][1] / 2 - offs[j] for j in range(a, b + 1)]
        left = sum(wanted) / len(wanted)
        c[2] = min(max(left, lo), max(lo, hi - width(c)))

    for j in range(n):
        offs[j] = 0.0
        c = [j, j, 0.0]
        place(c)
        clusters.append(c)
        while len(clusters) > 1:
            p, q = clusters[-2], clusters[-1]
            if p[2] + width(p) + pad <= q[2]:
                break
            base = offs[p[1]] + items[order[p[1]]][1] + pad     # q starts here within p
            q0 = offs[q[0]]
            for k in range(q[0], q[1] + 1):
                offs[k] = base + offs[k] - q0
            clusters.pop()
            p[1] = q[1]
            place(p)
    out = [0.0] * n
    for a, b, left in clusters:
        for j in range(a, b + 1):
            out[order[j]] = left + offs[j] + items[order[j]][1] / 2
    return out


def assign_rows(items: list[tuple[float, float]], lo: float, hi: float,
                pad: float = 0.0, max_rows: int = 64,
                max_shift: float | None = None) -> tuple[list[int], list[float]]:
    """items: [(wanted centre, width)]; returns (row of each, centre of
    each): the fewest rows (up to max_rows) whose labels, spread with
    spread_1d, fit in [lo, hi] without overlapping and, with max_shift,
    each lie within max_shift of where it wants to be (more rows, shorter
    leaders); labels taken in x order go to rows round-robin."""
    n = len(items)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: items[i][0])
    span = hi - lo

    def layout(rows):
        row = [0] * n
        for r, i in enumerate(order):
            row[i] = r % rows
        centres = [0.0] * n
        for r in range(rows):
            idx = [i for i in range(n) if row[i] == r]
            for i, c in zip(idx, spread_1d([items[i] for i in idx], lo, hi, pad)):
                centres[i] = c
        return row, centres

    rows = 1
    while True:
        loads = [0.0] * rows
        for r, i in enumerate(order):
            loads[r % rows] += items[i][1] + pad
        fits = max(loads) <= span
        if fits or rows >= max_rows:
            row, centres = layout(rows)
            if (max_shift is None or rows >= max_rows
                    or max(abs(c - items[i][0]) for i, c in enumerate(centres)) <= max_shift):
                return row, centres
        rows += 1
