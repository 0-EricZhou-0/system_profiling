"""The Bokeh page draws each long series reduced to its per-pixel-column
first / last / min / max samples (and NaN gaps) over the view ±1 width,
from the full data it keeps (tools/decimate.py; tools/decimate.js is the
page's port). The drawn line is the full one at pixel resolution: every
column's min and max are kept, every kept point is a real sample, gaps
stay gaps, and a sparse window is drawn exactly."""

import json
import os
import random
import shutil
import subprocess

import numpy as np
import pytest

import viz_trace
import decimate

NODE = shutil.which("node")
JS = os.path.join(viz_trace.TOOLS, "decimate.js")


def _series(rng, n):
    x = np.cumsum(rng.uniform(0.0005, 0.002, n)) + rng.uniform(0, 5)
    if rng.random() < 0.3:                          # duplicate timestamps
        x[rng.integers(1, n, n // 50)] = x[rng.integers(1, n, n // 50) - 1]
        x.sort()
    y = rng.normal(50, 20, n)
    if rng.random() < 0.5:
        y = np.round(y)                             # ties
    for _ in range(rng.integers(0, 4)):             # gaps
        a = rng.integers(0, n)
        y[a:a + rng.integers(1, 400)] = np.nan
    return x, y


def _cases(k=120):
    rng = np.random.default_rng(3)
    out = []
    for i in range(k):
        n = int(rng.choice([10, 2000, 30000]))
        x, y = _series(rng, n)
        span = x[-1] - x[0]
        w = span * float(rng.choice([1.2, 0.3, 0.02, 0.001]))
        start = x[0] + float(rng.uniform(-0.2, 1.0)) * span
        out.append((x, y, start, start + w, int(rng.choice([860, 300, 50]))))
    return out


def _cols(x, lo, hi, bw):
    """Column of each sample in [lo, hi]."""
    m = (x >= lo) & (x <= hi)
    return m, np.floor(x / bw)


def test_every_column_keeps_its_min_max_and_gaps_with_real_samples():
    for x, y, s, e, cols in _cases():
        lo, hi, bw = decimate.window(s, e, cols)
        k = decimate.keep(x, y, lo, hi, bw)
        assert np.all(np.diff(k) > 0)
        m, col = _cols(x, lo, hi, bw)
        kept = np.zeros(x.size, bool)
        kept[k] = True
        # Columns wholly inside the window (the two at its edges are cut by
        # it, a view width off-screen).
        for c in np.unique(col[m])[1:-1]:
            g = col == c
            fy, ky = y[g], y[g & kept]
            if np.isfinite(fy).any():
                assert np.nanmin(fy) == np.nanmin(ky) and np.nanmax(fy) == np.nanmax(ky), (c, cols)
            if np.isnan(fy).any():
                assert np.isnan(ky).any(), ("gap lost", c)
            # first and last sample of the column are kept
            idx = np.flatnonzero(g)
            assert kept[idx[0]] and kept[idx[-1]]
        # the samples just outside the window are kept: the line reaches its edges
        before = np.flatnonzero(x < lo)
        after = np.flatnonzero(x > hi)
        if before.size:
            assert kept[before[-1]]
        if after.size:
            assert kept[after[0]]
        # What is drawn: those plus the series' first, last, lowest and
        # highest samples (the full extent), real samples only.
        xr, yr = decimate.reduce(x, y, s, e, cols)
        kk = np.unique(np.r_[k, decimate.extremes(y)]).astype(int)
        assert np.array_equal(xr, x[kk]) and np.array_equal(yr, y[kk], equal_nan=True)
        assert xr[0] == x[0] and xr[-1] == x[-1]
        if np.isfinite(y).any():
            assert np.nanmin(yr) == np.nanmin(y) and np.nanmax(yr) == np.nanmax(y)


def test_reduces_a_dense_window_and_draws_a_sparse_one_exactly():
    x = np.arange(200_000) / 1000.0                # 200 s at 1 kHz
    y = np.sin(x)
    k = decimate.keep(x, y, *decimate.window(0.0, 200.0, 860))
    assert k.size <= 4 * 860 + 8, k.size         # full view: ~4 per column
    lo, hi, bw = decimate.window(100.0, 100.5, 860)  # 0.5 s view, 1500 samples in the window
    k = decimate.keep(x, y, lo, hi, bw)
    assert np.array_equal(k, np.arange(np.searchsorted(x, lo) - 1, np.searchsorted(x, hi, "right") + 1))


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_matches_python(tmp_path):
    cases = _cases()
    want = [[decimate.keep(x, y, *decimate.window(s, e, c)).tolist(), decimate.extremes(y)]
            for x, y, s, e, c in cases]
    inp = tmp_path / "cases.json"
    inp.write_text(json.dumps([dict(x=x.tolist(), y=[None if np.isnan(v) else v for v in y],
                                    s=s, e=e, c=c) for x, y, s, e, c in cases]))
    script = (f"const D = require({json.dumps(JS)});"
              f"const cs = JSON.parse(require('fs').readFileSync({json.dumps(str(inp))}, 'utf8'));"
              "console.log(JSON.stringify(cs.map(c => {"
              " const y = Float64Array.from(c.y, v => v === null ? NaN : v);"
              " const [lo, hi, bw] = D.decimWindow(c.s, c.e, c.c);"
              " return [D.decimKeep(Float64Array.from(c.x), y, lo, hi, bw), D.decimExtremes(y)]; })));")
    got = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True,
                                    check=True, timeout=120).stdout)
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, (i, len(g), len(w))


# ---- the page ------------------------------------------------------------

def _reductions(doc):
    from bokeh.models import CustomJS
    return [m for m in doc.root.select({"type": CustomJS}) if "decimIndices(f.x" in m.code]


def _page(tmp_path, monkeypatch, cols=5, least=20):
    """A synthetic page whose series (101 samples at 10 Hz) count as long:
    the reduction runs with `cols` columns a view."""
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    monkeypatch.setattr(vi, "_DECIMATE_COLS", cols)
    monkeypatch.setattr(vi, "_DECIMATE_MIN", least)
    procs = [viz_trace.proc(10, comm="root", discovered=False, cpu=30.0),
             viz_trace.proc(11, ppid=10, comm="w", start_s=2.0, end_s=7.0, cpu=80.0)]
    meta = viz_trace.write_trace(str(tmp_path / "trace"), procs, disk=True,
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])
    doc = vi.build_static(meta)
    [cb] = _reductions(doc)
    return vi, doc, cb.args["pairs"], cb.args["cols"]


def test_page_draws_reduced_rows_of_the_full_data(tmp_path, monkeypatch):
    vi, doc, pairs, cols = _page(tmp_path, monkeypatch)
    x_end = doc.t_end_s
    kinds = {p[2] for p in pairs}
    assert {"y", "_anchor_y"} <= kinds, kinds          # series and hover anchors
    reduced = 0
    for full, drawn, ycol in pairs:
        f, d = full.data, drawn.data
        if ycol == "ys":
            for fx, fy, dx, dy in zip(f["xs"], f["ys"], d["xs"], d["ys"]):
                wx, wy = decimate.reduce(np.asarray(fx), np.asarray(fy), 0.0, x_end, cols)
                assert np.array_equal(dx, wx) and np.array_equal(dy, wy, equal_nan=True)
            continue
        assert set(f) == set(d)
        k = decimate.indices(np.asarray(f["x"]), np.asarray(f[ycol]), 0.0, x_end, cols)
        assert 0 < len(k) <= len(f["x"])
        reduced += len(k) < len(f["x"])
        # whole rows of the full data: a hover row is real samples
        for c in f:
            assert np.array_equal(np.asarray(d[c]), np.asarray(f[c])[k], equal_nan=True), c
        if ycol == "y":
            fy, dy = np.asarray(f["y"]), np.asarray(d["y"])
            if np.isfinite(fy).any():
                assert np.nanmin(dy) == np.nanmin(fy) and np.nanmax(dy) == np.nanmax(fy)
            assert d["x"][0] == f["x"][0] and d["x"][-1] == f["x"][-1]
    assert reduced >= 3, reduced

def test_page_keeps_legend_and_hover_wiring(tmp_path, monkeypatch):
    """The drawn sources stay the renderers' (legend hide, the hover's
    formatter read them); only their data is reduced."""
    from bokeh.models import CustomJSHover, GlyphRenderer, Legend
    vi, doc, pairs, _ = _page(tmp_path, monkeypatch)
    drawn = {p[1].id for p in pairs}
    rendered = {r.data_source.id for r in doc.root.select({"type": GlyphRenderer})}
    assert drawn <= rendered
    hovers = [h for h in doc.root.select({"type": CustomJSHover})]
    assert hovers and all(h.args["src"].id in drawn for h in hovers)
    legend_rs = {r.id for lg in doc.root.select({"type": Legend}) for it in lg.items for r in it.renderers}
    assert legend_rs, "no legend renderers"


def test_short_series_are_drawn_as_they_are(tmp_path, monkeypatch):
    """At the real thresholds a 101-sample series is not reduced."""
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    procs = [viz_trace.proc(10, comm="root", discovered=False)]
    doc = vi.build_static(viz_trace.write_trace(str(tmp_path / "t"), procs))
    assert not _reductions(doc)
