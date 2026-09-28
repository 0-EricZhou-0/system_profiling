"""The interactive static page (tools/visualize_interactive.py):
the process timeline pinned in the sticky band with the event and region
strips, at its full height (nothing cut, no inner scroll); every figure
with the same plot frame (left edge, width), so a time is at the same x
everywhere; every panel (and the timeline) foldable under a header that
keeps its title, plus collapse / expand all; keyboard shortcuts on the
shared x-range; timeline, region and event labels re-placed for every
view; the timeline with the panels' tools; half-height strips; a
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


def _strip_meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0)]
    return viz_trace.write_trace(
        str(tmp_path / "t"), procs, gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"],
        regions=[("load", 1.0, 8.0), ("a rather long region name", 8.0, 8.2)],
        events=[("server ready", 1.0), ("shutdown", 9.9)])


def _texts(fig, name):
    [r] = [r for r in fig.renderers if r.name == name]
    return list(r.data_source.data["text"])


def test_labels_re_placed_on_every_view(tmp_path):
    """Timeline, regions and events: one debounced JS relayout (the port of
    label_spread.place_bar_labels) on x-range start and end; the first
    layout is Python's: a label on its bar when it fits, else off it."""
    doc = vi.build_static(_strip_meta(tmp_path))
    events, regions = doc.strips
    cbs = doc.timeline.x_range.js_property_callbacks
    for key in ("timeline", "regions", "events"):
        for change in ("change:start", "change:end"):
            [cb] = [c for c in cbs[change] if c.args.get("key") == key]
            assert "function placeBarLabels" in cb.code and "setTimeout(run, wait)" in cb.code
            assert cb.args["wait"] == vi._RELAYOUT_DEBOUNCE_MS
    assert _texts(regions, "regions-on-bar") == ["load"]
    assert _texts(regions, "regions-off-bar") == ["a rather long region name"]
    assert _texts(events, "events-on-bar") == []                     # a point holds no label
    assert sorted(_texts(events, "events-off-bar")) == ["server ready", "shutdown"]
    assert sorted(_texts(doc.timeline, "timeline-on-bar")) == ["child (11)", "root (10)"]


def test_timeline_has_the_panels_tools(tmp_path):
    """Same tools (so the same toolbar and right-click menu) as a metric
    panel, box zoom on a drag; its key right of the lanes."""
    doc = vi.build_static(_strip_meta(tmp_path))
    tl, panel = doc.timeline, doc.panel_figs[0][2]
    kinds = lambda f: [type(t).__name__ for t in f.tools]
    assert kinds(tl) == kinds(panel) and "BoxZoomTool" in kinds(tl), (kinds(tl), kinds(panel))
    assert tl.toolbar_location == panel.toolbar_location == "left"
    assert type(tl.toolbar.active_drag).__name__ == "BoxZoomTool"
    assert tl.toolbar.active_drag.dimensions == "width"
    [key] = [r for r in tl.right if type(r).__name__ == "Legend"]
    assert not [r for r in tl.above if type(r).__name__ == "Legend"]
    assert [it.label.value for it in key.items][-1] == "fork link (parent -> child)"


def test_strips_half_height(tmp_path):
    """Event and region strips at half their former 70 px: no title row
    (the name is a horizontal label left of the frame)."""
    doc = vi.build_static(_strip_meta(tmp_path))
    assert [s.yaxis[0].axis_label for s in doc.strips] == ["Events", "Regions"]
    for s in doc.strips:
        assert not s.title.text
        assert s.frame_height + s.min_border_top + s.min_border_bottom <= 36
        assert s.frame_width == vi._FRAME_WIDTH and s.min_border_left == vi._BORDER_LEFT_PX


def test_every_panel_foldable_with_title_kept(tmp_path):
    doc = vi.build_static(_meta(tmp_path))
    figs = [f for _p, _k, f in doc.panel_figs] + [doc.timeline]
    folded = {id(f) for f, _b, _t in doc.folds}
    assert all(id(f) in folded for f in figs)
    for fig, btn, title in doc.folds:
        assert title and btn.label == "▾ " + title and fig.title.text == ""
        [cb] = btn.js_event_callbacks["button_click"]
        # folded by CSS display, not fig.visible (a whole-document relayout)
        assert "setFolded(fig, hide)" in cb.code and cb.args["fig"] is fig
        assert ".visible" not in cb.code
    names = [b.label for b in doc.band.children[0].children]
    assert names == ["Collapse all", "Expand all"]
    for b in doc.band.children[0].children:
        [cb] = b.js_event_callbacks["button_click"]
        assert "setFolded(figs[i], collapse)" in cb.code and ".visible" not in cb.code
    html = vi.static_page(_meta(tmp_path / "p"))
    assert 'display: any ? "none" : ""' in html and "model(f).visible" not in html   # the 'c' key
    # Shown again by display "", not by deleting the key (Bokeh keeps a
    # removed style on the element: the panel stayed hidden).
    [cb] = doc.folds[0][1].js_event_callbacks["button_click"]
    assert 'display: hide ? "none" : ""' in cb.code and "delete st" not in cb.code


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


def _io_meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False),
             viz_trace.proc(11, ppid=10, comm="child", start_s=1.0, end_s=5.0),
             viz_trace.proc(12, ppid=99, comm="orphan", start_s=2.0, end_s=6.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, regions=[("load", 1.0, 8.0)],
                                 disk=True, gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_marks_take_the_theme_ink(tmp_path, monkeypatch, theme):
    """Timeline outlines, fork links, outside labels and leaders, the
    line-style key's swatches, the Peak label and the fold headers are in
    the theme's ink (light: as before), never dark marks on the dark page."""
    monkeypatch.setattr(vi, "_THEME", theme)
    t = vi._THEMES[theme]
    doc = vi.build_static(_io_meta(tmp_path))
    tl = doc.timeline
    glyphs = [r.glyph for r in tl.renderers]
    kinds = {type(g).__name__ for g in glyphs}
    assert {"Quad", "Segment", "Text"} <= kinds
    outline = {g.line_color for g in glyphs if type(g).__name__ == "Quad"
               and isinstance(g.line_color, str) and g.line_color != "color"}
    assert outline == {t["ink"], "#bbbbbb"}, outline            # root / orphan; "discovered" swatch grey
    segs = {g.line_color for g in glyphs if type(g).__name__ == "Segment"}
    assert segs == {t["link"], t["leader"]}, segs               # fork links, leaders
    texts = {g.text_color for g in glyphs if type(g).__name__ == "Text"}
    assert texts == {"white", t["label"]}, texts                # inside the bars, outside
    io = [f for p, k, f in doc.panel_figs if p.series_glob.startswith("proc__io_?char") and k == "metric"]
    [key] = [lg for lg in io[0].above if type(lg).__name__ == "Legend"]
    assert {it.renderers[-1].glyph.line_color for it in key.items} == {t["ink"]}
    peaks = [lb for _p, _k, f in doc.panel_figs for lb in f.center
             if type(lb).__name__ == "Label" and lb.text.startswith("Peak:")]
    assert peaks and {lb.text_color for lb in peaks} == {t["label"]}
    [css] = {tuple(s.css for s in b.stylesheets) for _f, b, _t in doc.folds}
    assert f"color: {t['page_fg']}" in css[0]
    if theme == "dark":
        dark = {"black", "#000000", "#333333", "#444444"}
        assert not (outline | segs | texts) & dark


def test_bar_hover_follows_the_mouse(tmp_path):
    """Timeline and region-strip tooltips at the pointer, not at the bar's
    centre (off-screen when zoomed into a long bar)."""
    doc = vi.build_static(_strip_meta(tmp_path))
    _events, regions = doc.strips
    for fig in (doc.timeline, regions):
        [hv] = [t for t in fig.tools if type(t).__name__ == "HoverTool"]
        assert hv.point_policy == "follow_mouse"
