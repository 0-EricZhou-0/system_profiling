"""Legends of the interactive visualizer's static page
(tools/visualize_interactive.py): above each panel, never inside or
beside the plot frame, capped at LEGEND_MAX_ENTRIES plus "+k more";
the plot frame keeps its height whatever the legend's."""

import pytest

pytest.importorskip("bokeh")

import viz_trace  # noqa: E402,F401  (puts tools/ on sys.path)
import panel_legend  # noqa: E402
import visualize_interactive  # noqa: E402

N = panel_legend.LEGEND_MAX_ENTRIES


def _doc(tmp_path, n):
    procs = [viz_trace.proc(500 + i, ppid=500, comm=f"c{i}", cpu=1.0 + i, discovered=i > 0)
             for i in range(n)]
    return visualize_interactive.build_static(viz_trace.write_trace(str(tmp_path / "t"), procs))


def _labels(fig):
    [legend] = fig.legend
    return [it.label.value for it in legend.items]


def test_legend_above_every_panel(tmp_path):
    doc = _doc(tmp_path, 3)
    assert doc.panel_figs
    for panel, _kind, fig in doc.panel_figs:
        legends = [r for r in fig.above if type(r).__name__ == "Legend"]
        assert len(legends) == 1, panel.title
        assert not [r for r in fig.center + fig.right if type(r).__name__ == "Legend"]
        assert fig.frame_height == visualize_interactive._FRAME_HEIGHT


def test_legend_capped_with_more_entry(tmp_path):
    n = N + 7
    doc = _doc(tmp_path, n)
    [cpu] = [f for p, k, f in doc.panel_figs if k == "metric" and p.series_glob.startswith("proc__cycles")]
    labels = _labels(cpu)
    assert len(labels) == N + 1 and labels[-1] == f"+{n - N} more", labels
    listed = {int(lab.split("PID ")[1].split(",")[0].rstrip(")]")) for lab in labels[:-1]}
    assert listed == {500 + i for i in range(n - N, n)}
    [legend] = cpu.legend
    assert len(legend.items[-1].renderers) == n - N
    colors = [it.renderers[0].glyph.line_color for it in legend.items[:-1]]
    assert len(set(colors)) == N, colors
