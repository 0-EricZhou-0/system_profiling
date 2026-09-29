"""Colours of the series that are not per-process: one hue per base
metric (entity, counter, submetric, instance), continuing across the
figure's panels; statistic variants of one metric share it — .max in the
full colour, .avg lighter — and a different metric (DRAM read vs write)
gets its own. Both renderers."""

import colorsys

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)
import panel_legend  # noqa: E402

AVG, MAX = ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__cycles_active.max.pct_of_peak_sustained_elapsed")
RD, WR = ("dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
          "dram__write_throughput.avg.pct_of_peak_sustained_elapsed")


def _hls(c):
    return colorsys.rgb_to_hls(*[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _check(col):
    """col: legend label -> colour."""
    avg, mx = col["Active Cycles (avg)"], col["Active Cycles (max)"]
    ha, la, _ = _hls(avg)
    hm, lm, _ = _hls(mx)
    assert abs(ha - hm) < 0.02, (avg, mx)             # one hue
    assert la > lm + 0.15, (avg, mx)                  # avg lighter
    assert mx == panel_legend.COLORS[0]               # max: the metric's full colour
    rd, wr = col["Read Throughput (avg)"], col["Write Throughput (avg)"]
    assert len({_hls(c)[0] for c in (mx, rd, wr)}) == 3, (mx, rd, wr)   # other metrics: other hues
    assert rd in panel_legend.COLORS and wr in panel_legend.COLORS      # alone: full colour


def _trace(tmp_path):
    return viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 gpu_fqns=[AVG, MAX, RD, WR], gpu_values=[40, 90, 30, 10])


def test_statistics_share_a_hue_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_trace(tmp_path))
    col = {}
    for p, _s, _k, ax in r.panel_axes:
        if p.series_glob.startswith(("sm__", "dram__")):
            leg = ax.get_legend()
            col.update({t.get_text(): h.get_color()
                        for t, h in zip(leg.get_texts(), leg.legend_handles)})
    _check(col)


def test_statistics_share_a_hue_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_trace(tmp_path))
    col = {}
    for p, _k, f in doc.panel_figs:
        if p.series_glob.startswith(("sm__", "dram__")):
            col.update({it.label.value: it.renderers[0].glyph.line_color
                        for it in f.legend[0].items})
    _check(col)


def test_shade():
    assert panel_legend.shade("#000000", 0.5) == "#808080"
    assert panel_legend.shade("#ffffff", -0.5) == "#808080"
    assert panel_legend.shade("#1f77b4", 0.0) == "#1f77b4"
