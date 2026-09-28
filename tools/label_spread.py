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


def place_bar_labels(bars: list[tuple[float, float, list[str]]], lo: float, hi: float,
                     width_units: float, text_width, pad: float = 0.0, max_rows: int = 64,
                     max_shift: float | None = None) -> list:
    """Labels for bars (and points: left == right) seen through the view
    [lo, hi] (data units) drawn width_units wide. bars: [(left, right,
    texts)], texts in order of preference. Each bar gets, in input order:

      ("in", text, x, x, -1)          the first text that fits inside the
                                      bar's visible part (text + pad),
                                      centred on it;
      ("out", texts[0], x, anchor, row)  else off the bar, in label row
                                      `row`, centred at x, its leader to
                                      `anchor` (the visible part's centre);
      None                            the bar is out of view, or its label
                                      finds no room.

    Off-bar labels: assign_rows over the view's width, at most max_rows
    rows, so none overlaps another; when they cannot all fit, those of the
    narrowest visible bars are left out until the rest do. x in data units.
    tools/label_spread.js is the same, line for line, for the Bokeh page."""
    per = width_units / max(hi - lo, 1e-12)
    out: list = [None] * len(bars)
    outside = []                                  # (index, anchor, text, width, visible)
    for i, (left, right, texts) in enumerate(bars):
        if left > hi or right < lo:
            continue
        a, b = max(left, lo), min(right, hi)
        vis = (b - a) * per
        fit = next((t for t in texts if text_width(t) + pad < vis), None)
        if fit is not None:
            out[i] = ("in", fit, (a + b) / 2, (a + b) / 2, -1)
        else:
            outside.append((i, (a + b) / 2, texts[0], text_width(texts[0]), vis))
    if max_rows < 1:
        outside = []                              # no label rows
    while outside:
        items = [((anc - lo) * per, w) for _i, anc, _t, w, _v in outside]
        rows, centres = assign_rows(items, 0.0, width_units, pad, max_rows, max_shift)
        loads = [0.0] * (max(rows) + 1)
        for r, (_c, w) in zip(rows, items):
            loads[r] += w + pad
        if max(loads) <= width_units:
            for (i, anc, text, _w, _v), r, c in zip(outside, rows, centres):
                out[i] = ("out", text, lo + c / per, anc, r)
            break
        # Too many for max_rows: leave out the narrowest visible bar's label.
        outside.remove(min(outside, key=lambda o: (o[4], o[0])))
    return out
