"""Exit end lines: each exited process's series ends in a dashed vertical
line from 0 to its last value, on every per-process gauge panel (RSS) and
every cumulative panel, black in the PNG and the theme's ink on the Bokeh
page; not on rate panels. --no-exit-lines (both tools, passed through by
the vLLM example): no line, the series just stops at its last value."""

import os
import subprocess
import sys

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)


def _meta(tmp_path):
    procs = [viz_trace.proc(800, comm="root", discovered=False, rss=4e8),
             viz_trace.proc(801, ppid=800, comm="gone", start_s=1.0, end_s=6.0, rss=3e8)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, disk=True)


@pytest.mark.parametrize("on", [True, False])
def test_exit_lines_static(tmp_path, on):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_meta(tmp_path), exit_lines=on)
    seen = set()
    for p, _s, kind, ax in r.panel_axes:
        ends = [ln for ln in ax.get_lines() if hasattr(ln, "_end_line")]
        gauge_or_cum = kind == "integrated" or p.series_glob == "proc__rss_bytes"
        if not on or not gauge_or_cum:
            assert not ends, (p.title, kind)
            continue
        assert ends and {int(ln._end_line[1]) for ln in ends} == {801}, (p.title, kind)
        for ln in ends:
            assert ln.get_color() == "black" and ln.get_linestyle() == "--"
            x, y = ln.get_xdata(), ln.get_ydata()
            assert x[0] == x[1] and 5.8 <= x[0] <= 6.0 and y[0] == 0 and y[1] > 0
        seen.add(kind)
    if on:
        assert seen == {"metric", "integrated"}                  # RSS and a cumulative panel
    [rss] = [ax for p, _s, _k, ax in r.panel_axes if p.series_glob == "proc__rss_bytes"]
    for ln in rss.get_lines():                                   # the series stops at its last value
        if getattr(ln, "_series_key", (None, None))[1] in ("801", 801):
            ys = [v for v in ln.get_ydata() if v == v]
            assert ys[-1] > 0


@pytest.mark.parametrize("theme,on", [("light", True), ("dark", True), ("light", False)])
def test_exit_lines_bokeh(tmp_path, monkeypatch, theme, on):
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    monkeypatch.setattr(vi, "_THEME", theme)
    doc = vi.build_static(_meta(tmp_path), exit_lines=on)
    kinds = set()
    for p, kind, f in doc.panel_figs:
        segs = [r for r in f.renderers if r.name == "end-lines"]
        if not on or not (kind == "cumulative" or p.series_glob == "proc__rss_bytes"):
            assert not segs, (p.title, kind)
            continue
        [seg] = segs
        assert seg.glyph.line_color == vi._THEMES[theme]["ink"] and seg.glyph.line_dash == "dashed"
        assert len(seg.data_source.data["x0"]) >= 1
        kinds.add(kind)
    if on:
        assert kinds == {"metric", "cumulative"}


@pytest.mark.parametrize("tool", ["visualize_all.py", "visualize_interactive.py"])
def test_cli_knob(tool):
    if tool == "visualize_interactive.py":
        pytest.importorskip("bokeh")
    out = subprocess.run([sys.executable, os.path.join(viz_trace.TOOLS, tool), "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    assert "--no-exit-lines" in out
    ex = open(os.path.join(viz_trace.TOOLS, "..", "examples", "vllm_serving_profiling.py")).read()
    assert '["--no-exit-lines"] if args.no_exit_lines' in ex


def test_hover_lists_only_series_with_a_value(tmp_path):
    """The unified hover's rows are built at hover time: a process with no
    value there (not started yet / gone: NaN) gets no row. Every panel;
    the formatter run under node on a NaN and a finite column."""
    import json
    import shutil
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    doc = vi.build_static(_meta(tmp_path))
    for p, _k, f in doc.panel_figs:
        [hv] = [t for t in f.tools if type(t).__name__ == "HoverTool" and isinstance(t.tooltips, str)]
        assert "@_anchor_y{present}" in hv.tooltips, p.title
        fmt = hv.formatters["@_anchor_y"]
        assert "Number.isFinite(v)" in fmt.code
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = ("const f = new Function('special_vars', 'src', 'rows', 'dec', 'unit', %s);"
          "console.log(JSON.stringify(f({index: 1}, {data: {y_0: [1, 2.5], y_1: [NaN, NaN],"
          " y_2: [null, 7]}}, [['y_0', 'a'], ['y_1', 'gone'], ['y_2', 'c']], 3, ' %%')));"
          % json.dumps(fmt.code))
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True,
                                    check=True, timeout=60).stdout)
    assert out.split("\n") == ["a: 2.500 %", "c: 7.000 %"]
