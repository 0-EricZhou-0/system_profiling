"""The interactive static page (tools/visualize_interactive.py):
the process timeline pinned in the sticky band with the event and region
strips, at its full height (nothing cut, no inner scroll); every figure
with the same plot frame (left edge, width), so a time is at the same x
everywhere; every panel (and the timeline) foldable under a header that
keeps its title, plus collapse / expand all; keyboard shortcuts on the
shared x-range; timeline bar labels that follow the visible bar; a
window's height of empty page after the last panel."""

import pytest

pytest.importorskip("bokeh")

import viz_trace  # noqa: E402
import visualize_interactive as vi  # noqa: E402


def _meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, regions=[("load", 1.0, 8.0)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])


def test_timeline_pinned_at_full_height(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    assert doc.band.styles["position"] == "sticky"
    assert doc.timeline_block in doc.band.children
    st = dict(doc.timeline_block.styles)
    assert "max-height" not in st and "overflow-y" not in st        # nothing cut, no inner scroll
    assert doc.timeline.frame_height >= 2 * vi._LANE_PX           # its lanes' full height


def test_every_figure_has_the_same_frame(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    figs = doc.strips + [doc.timeline] + [f for _p, _k, f in doc.panel_figs]
    assert {f.min_border_left for f in figs} == {vi._BORDER_LEFT_PX}
    assert {f.frame_width for f in figs} == {vi._FRAME_WIDTH}


def test_timeline_labels_follow_the_visible_bar(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    [lab] = [r for r in doc.timeline.renderers if r.name == "inside-labels"]
    cbs = doc.timeline.x_range.js_property_callbacks
    codes = [cb.code for key in ("change:start", "change:end") for cb in cbs.get(key, [])]
    assert any("alpha" in c and "Math.max(d.left[i], a)" in c for c in codes)
    assert lab.glyph.text_alpha == "alpha" or getattr(lab.glyph.text_alpha, "field", None) == "alpha"


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
    # A key-set range counts as user-set: else the data range snaps back to
    # the full extent when the timeline labels' data follows the zoom.
    assert "xr.have_updated_interactively = true;" in html



def test_two_pages_in_one_process(tmp_path):
    """No model shared between documents (Bokeh: one document per model):
    a second page in the same process builds."""
    for n in range(2):
        assert "cupti-keys-help" in vi.static_page(_meta(tmp_path / str(n)))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_bottom_space_after_the_last_panel(tmp_path, monkeypatch, theme):
    """100vh of page background at the end of the page (after the panels
    and the footer), so the last panel scrolls up under the band."""
    monkeypatch.setattr(vi, "_THEME", theme)
    html = vi.static_page(_meta(tmp_path))
    i = html.find('<div id="cupti-bottom-space"')
    assert i > 0 and html.count('id="cupti-bottom-space"') == 1
    tag = html[i:html.index(">", i) + 1]
    assert "height:100vh" in tag and f"background:{vi._THEMES[theme]['page_bg']}" in tag, tag
    assert i > html.rfind("</pre>") and i > html.rfind("</script>")     # after everything drawn
    assert html[html.index("</div>", i) + len("</div>"):].strip().startswith("</body>")
