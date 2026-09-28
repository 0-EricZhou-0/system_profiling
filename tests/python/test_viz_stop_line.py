"""The session's stop (the last sample any probe took) on the Bokeh page:
a dashed vertical line and the time after it shaded grey on every metric
panel, rate and cumulative; not on the event / region strips or the
process timeline; no legend entry; the ranges unchanged. The PNG has
neither."""

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)

DUR = 10.0


def _meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, duration_s=DUR, disk=True,
                                 regions=[("load", 1.0, 8.0), ("late", 9.0, 12.0)],
                                 events=[("ready", 2.0)])   # "late" ends after the last sample


def test_no_stop_marks_in_the_png(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    assert not hasattr(visualize_all, "_mark_stop")
    r = visualize_all.build_figure(_meta(tmp_path))
    axes = [ax for *_x, ax in r.panel_axes] + [r.region_ax, r.event_ax, r.process_ax]
    for ax in axes:
        if ax is None:
            continue
        assert not [t for t in ax.texts if t.get_text().strip() == "stop"]
        assert not [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"
                    and len(set(ln.get_xdata())) == 1 and ln.get_xdata()[0] == pytest.approx(DUR, abs=0.02)]
    assert r.panel_axes[0][3].get_xlim()[1] == pytest.approx(12.0, abs=0.02)   # the late region's end


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_stop_line_and_shade_bokeh(tmp_path, monkeypatch, theme):
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    monkeypatch.setattr(vi, "_THEME", theme)
    doc = vi.build_static(_meta(tmp_path))
    t = vi._THEMES[theme]
    assert {k for _p, k, _f in doc.panel_figs} == {"metric", "cumulative"}
    for _p, _k, f in doc.panel_figs:
        [span] = [a for a in f.center if a.name == "stop-line"]
        assert span.location == pytest.approx(DUR, abs=0.02) and span.dimension == "height"
        assert span.line_color == t["link"]
        assert [a.text for a in f.center if a.name == "stop-note"] == ["stop"]
        [shade] = [a for a in f.renderers + f.center if getattr(a, "name", None) == "after-stop"]
        assert type(shade).__name__ == "BoxAnnotation"
        assert shade.left == pytest.approx(DUR, abs=0.02)
        edge = lambda v: (getattr(v, "target", None), getattr(v, "symbol", None))
        assert edge(shade.right) == ("frame", "right")         # to the frame's edge, as it pans
        assert (edge(shade.top), edge(shade.bottom)) == (("frame", "top"), ("frame", "bottom"))
        assert shade.fill_color == t["after_stop"]
        assert all("stop" not in it.label.value for lg in f.legend for it in lg.items)
    for f in list(doc.strips) + [doc.timeline]:
        assert not [a for a in f.renderers + f.center
                    if getattr(a, "name", None) in ("stop-line", "stop-note", "after-stop")]
