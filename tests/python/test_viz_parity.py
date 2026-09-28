"""The Bokeh static page draws what the static PNG draws: one colour per
process on the whole page (panels and process timeline), the same as
the PNG's; in a panel with several metrics per process, a line style
per metric and a legend of processes plus line styles (not every
pair); discovered processes labelled with the parent they were found
under."""

import pytest

pytest.importorskip("bokeh")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import viz_trace  # noqa: E402
import visualize_all  # noqa: E402
import visualize_interactive  # noqa: E402

LAYOUT = """
panels { title: "CPU" series_glob: "proc__cycles_*" }
panels { title: "Two metrics" series_glob: "proc__[cr]*" }
"""


def _trace(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False, cpu=80),
             viz_trace.proc(11, ppid=10, comm="a", start_s=1.0, cpu=40),
             viz_trace.proc(12, ppid=11, comm="b", start_s=2.0, end_s=6.0, cpu=20)]
    meta = viz_trace.write_trace(str(tmp_path / "t"), procs)
    layout = tmp_path / "layout.pbtxt"
    layout.write_text(LAYOUT)
    return meta, str(layout)


def test_one_colour_per_process_everywhere(tmp_path):
    meta, layout = _trace(tmp_path)
    doc = visualize_interactive.build_static(meta, panel_layout=layout)
    png = visualize_all.build_figure(meta, panel_layout=layout)
    [cpu] = [f for p, k, f in doc.panel_figs if p.title == "CPU"]
    [cpu_ax] = [ax for p, _s, _k, ax in png.panel_axes if p.title == "CPU"]
    bokeh_color = {int(it.label.value.split("PID ")[1].split(",")[0].rstrip(")]")):
                   it.renderers[0].glyph.line_color for it in cpu.legend[0].items}
    png_color = {int(t.get_text().split("PID ")[1].split(",")[0].rstrip(")]")): h.get_color()
                 for t, h in zip(cpu_ax.get_legend().get_texts(),
                                 cpu_ax.get_legend().legend_handles)}
    assert bokeh_color == png_color and len(set(bokeh_color.values())) == 3
    # the timeline's bars have their process's colour
    bars = {}
    for r in doc.timeline.renderers:
        d = r.data_source.data
        if "pid" in d:
            bars.update(zip(d["pid"], d["color"]))
    assert bars == bokeh_color


def test_metric_line_styles_and_compact_legend(tmp_path):
    meta, layout = _trace(tmp_path)
    doc = visualize_interactive.build_static(meta, panel_layout=layout)
    [two] = [f for p, k, f in doc.panel_figs if p.title == "Two metrics"]
    labels = [it.label.value for it in two.legend[0].items]
    # 3 processes + 2 line styles, not 3 x 2 pairs
    assert len(labels) == 5, labels
    assert labels[-2:] == ["CPU (sum)", "Rss Bytes"]
    assert "(PID 11, child of 10)" in labels[1] and "child of" not in labels[0]
    dashes = {it.label.value: it.renderers[-1].glyph.line_dash for it in two.legend[0].items[-2:]}
    assert dashes["CPU (sum)"] != dashes["Rss Bytes"]
    assert {it.renderers[-1].glyph.line_color for it in two.legend[0].items[-2:]} == {"black"}
    # every drawn series: its process's colour, its metric's style
    styles = {(r.glyph.line_color, tuple(r.glyph.line_dash) if isinstance(r.glyph.line_dash, list)
               else r.glyph.line_dash)
              for r in two.renderers if type(r.glyph).__name__ == "Line"
              and len(r.data_source.data.get("x", [])) > 2 and r.glyph.line_alpha != 0}
    assert len(styles) == 6, styles                    # 3 colours x 2 styles
