"""The disk bandwidth panel's cumulative companion
(PANEL_AGGREGATION_INTEGRATE_SUM): read and write summed over all
devices, then integrated — two lines, one hue, read solid / write
dashed, each with its run total. Both renderers."""

import pytest

import viz_trace  # noqa: F401
import units  # noqa: E402

M = 1024.0 ** 2
DEVICES = {"nvme0n1": (4 * M, 1 * M), "nvme1n1": (2 * M, 3 * M)}   # B/s, constant
DUR = 10.0
# the trace spans DUR s: read 6 MiB/s x 10 s, write 4 MiB/s x 10 s
WANT = {"Read Bytes": units.fmt_bytes(6 * M * DUR), "Write Bytes": units.fmt_bytes(4 * M * DUR)}


def _meta(tmp_path):
    return viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 duration_s=DUR, disk=True, devices=DEVICES)


def _check(labels, styles, colors):
    assert len(labels) == 2, labels
    for lab in labels:
        name = "Read Bytes" if "Read" in lab else "Write Bytes"
        assert "[all 2 devices]" in lab and lab.endswith("= " + WANT[name]), lab
    assert len(set(colors)) == 1 and len({str(s) for s in styles}) == 2


def test_disk_sum_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_meta(tmp_path))
    [ax] = [ax for p, _s, k, ax in r.panel_axes if k == "integrated_sum"]
    assert ax.get_title(loc="left").endswith("(cumulative, all 2 devices)")
    leg = ax.get_legend()
    _check([t.get_text() for t in leg.get_texts()],
           [h.get_linestyle() for h in leg.legend_handles],
           [h.get_color() for h in leg.legend_handles])
    assert [ln.get_linestyle() for ln in ax.get_lines() if hasattr(ln, "_series_key")] \
        in (["-", "--"], ["--", "-"])


def test_disk_sum_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_meta(tmp_path))
    [f] = [f for p, k, f in doc.panel_figs if k == "cumulative" and p.series_glob.startswith("disk__")]
    items = [it for lg in f.legend for it in lg.items]
    _check([it.label.value for it in items],
           [it.renderers[-1].glyph.line_dash for it in items],
           [it.renderers[-1].glyph.line_color for it in items])
    assert any("all 2 devices" in b.label for _f, b, _t in doc.folds)
