"""The session's stop (the last sample any probe took) as a dashed
vertical line on every metric panel, rate and cumulative, in both
renderers; not on the event / region strips or the process timeline; no
legend entry; the axis ranges unchanged."""

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)

DUR = 10.0


def _meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, duration_s=DUR, disk=True,
                                 regions=[("load", 1.0, 8.0), ("late", 9.0, 12.0)],
                                 events=[("ready", 2.0)])   # "late" ends after the last sample


def test_stop_line_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_meta(tmp_path))
    kinds = {k for _p, _s, k, _ax in r.panel_axes}
    assert {"metric", "integrated"} <= kinds
    for p, _s, _k, ax in r.panel_axes:
        [ln] = [ln for ln in ax.get_lines() if getattr(ln, "_stop_line", False)]
        assert ln.get_linestyle() == "--" and ln.get_xdata()[0] == pytest.approx(DUR, abs=0.02)
        assert "stop" in [t.get_text().strip() for t in ax.texts]
        leg = ax.get_legend()
        assert leg is None or all("stop" not in t.get_text() for t in leg.get_texts())
    for ax in (r.region_ax, r.event_ax, r.process_ax):
        if ax is not None:
            assert not [ln for ln in ax.get_lines() if getattr(ln, "_stop_line", False)]


def test_stop_line_keeps_the_axes(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    with_line = visualize_all.build_figure(_meta(tmp_path / "a"))
    monkeypatch.setattr(visualize_all, "_mark_stop", lambda axes, stop_s: None)
    without = visualize_all.build_figure(_meta(tmp_path / "b"))
    for (_p, _s, _k, a), (_p2, _s2, _k2, b) in zip(with_line.panel_axes, without.panel_axes):
        assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_stop_line_bokeh(tmp_path, monkeypatch, theme):
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    monkeypatch.setattr(vi, "_THEME", theme)
    doc = vi.build_static(_meta(tmp_path))
    assert {k for _p, k, _f in doc.panel_figs} == {"metric", "cumulative"}
    for _p, _k, f in doc.panel_figs:
        [span] = [a for a in f.center if a.name == "stop-line"]
        assert span.location == pytest.approx(DUR, abs=0.02) and span.dimension == "height"
        assert span.line_dash == "dashed" or list(span.line_dash) == [6]
        assert span.line_color == vi._THEMES[theme]["link"]
        assert [a.text for a in f.center if a.name == "stop-note"] == ["stop"]
        assert all("stop" not in it.label.value for lg in f.legend for it in lg.items)
    for f in list(doc.strips) + [doc.timeline]:
        assert not [a for a in f.center if a.name in ("stop-line", "stop-note")]
