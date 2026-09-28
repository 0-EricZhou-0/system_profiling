"""The interactive static page (tools/visualize_interactive.py):
the process timeline pinned in the sticky band with the event and region
strips, scrolling inside the band past a height cap; every panel (and
the timeline) foldable under a header that keeps its title, plus
collapse / expand all; keyboard shortcuts on the shared x-range."""

import pytest

pytest.importorskip("bokeh")

import viz_trace  # noqa: E402
import visualize_interactive as vi  # noqa: E402


def _meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, regions=[("load", 1.0, 8.0)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])


def test_timeline_pinned_and_capped(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    assert doc.band.styles["position"] == "sticky"
    assert doc.timeline_block in doc.band.children
    st = doc.timeline_block.styles
    assert st["max-height"] == f"{vi._TIMELINE_MAX_VH}vh" and st["overflow-y"] == "auto"
    assert doc.timeline.height_policy == "fixed"          # scrolls, not shrunk to the cap


def test_every_panel_foldable_with_title_kept(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    figs = [f for _p, _k, f in doc.panel_figs] + [doc.timeline]
    folded = {id(f) for f, _b, _t in doc.folds}
    assert all(id(f) in folded for f in figs)
    for fig, btn, title in doc.folds:
        assert title and btn.label == "▾ " + title and fig.title.text == ""
        [cb] = btn.js_event_callbacks["button_click"]
        assert "fig.visible = !fig.visible" in cb.code and cb.args["fig"] is fig
    names = [b.label for b in doc.band.children[0].children]
    assert names == ["Collapse all", "Expand all"]


def test_hotkeys_in_the_page(tmp_path):
    html = vi.static_page(_meta(tmp_path))
    doc_xr = "cuptiHotkey"
    assert 'document.addEventListener("keydown"' in html and doc_xr in html
    for key in ('case "r"', 'case "0"', 'case "="', 'case "+"', 'case "-"',
                'case "ArrowLeft"', 'case "ArrowRight"', 'case "c"', 'case "?"'):
        assert key in html, key
    assert 'id="cupti-keys-help"' in html
    assert 't.tagName === "INPUT"' in html                 # not while typing
