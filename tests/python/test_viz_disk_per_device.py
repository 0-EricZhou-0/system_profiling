"""The disk bandwidth panel's cumulative companion is per device, never
summed across devices: one colour per device (the same as in the rate
panel), read solid / write dashed, each device's own bytes so far.
There is no summing aggregation at all. Both renderers."""

import pytest

import viz_trace  # noqa: F401
import panel_legend  # noqa: E402
import panels_pb2  # noqa: E402

M = 1024.0 ** 2
DEVICES = {"nvme0n1": (4 * M, 1 * M), "nvme1n1": (2 * M, 3 * M)}   # B/s, constant
DUR = 10.0
KEY = ["Read Bytes (sum)", "Write Bytes (sum)"]


def _meta(tmp_path):
    return viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 duration_s=DUR, disk=True, devices=DEVICES)


def test_no_summing_aggregation():
    assert "PANEL_AGGREGATION_INTEGRATE_SUM" not in panels_pb2.PanelAggregation.keys()
    assert not hasattr(panel_legend, "aggregate")


def _check(lines, entries):
    """lines: (fqn, device) -> (colour, dash, last value in MiB);
    entries: legend labels, device entries first."""
    assert sorted(lines) == sorted((f, d) for f in viz_trace.DEV_FQNS for d in DEVICES)
    read, write = viz_trace.DEV_FQNS
    for d, (rd, wr) in DEVICES.items():
        assert lines[(read, d)][0] == lines[(write, d)][0]            # one colour per device
        assert lines[(read, d)][1] != lines[(write, d)][1]            # read / write by style
        assert lines[(read, d)][2] == pytest.approx(rd * DUR / M, rel=0.02)   # its own bytes
        assert lines[(write, d)][2] == pytest.approx(wr * DUR / M, rel=0.02)
    assert lines[(read, "nvme0n1")][0] != lines[(read, "nvme1n1")][0]
    assert sorted(entries) == sorted(list(DEVICES) + KEY), entries


def test_disk_cumulative_per_device_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_meta(tmp_path))
    disk = {k: ax for p, _s, k, ax in r.panel_axes if p.series_glob.startswith("disk__")}
    assert set(disk) == {"metric", "integrated"}
    ax = disk["integrated"]
    assert ax.get_title(loc="left").endswith("(cumulative)")
    assert ax.get_ylabel() == "MiB"
    lines = {ln._series_key: (ln.get_color(), ln.get_linestyle(), float(ln.get_ydata()[-1]))
             for ln in ax.get_lines() if hasattr(ln, "_series_key")}
    _check(lines, [t.get_text() for t in ax.get_legend().get_texts() if t.get_text()])
    rate = {ln._series_key: ln.get_color() for ln in disk["metric"].get_lines()
            if hasattr(ln, "_series_key")}
    assert rate == {k: v[0] for k, v in lines.items()}              # same colours as the rates


def test_disk_cumulative_per_device_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_meta(tmp_path))
    disk = {k: f for p, k, f in doc.panel_figs if p.series_glob.startswith("disk__")}
    assert set(disk) == {"metric", "cumulative"}
    f = disk["cumulative"]
    [title] = [t for fig, _b, t in doc.folds if fig is f]      # the fold header carries it
    assert title.endswith("(cumulative)") and f.yaxis[0].axis_label == "MiB"
    items = [it for lg in f.legend for it in lg.items]
    read, write = viz_trace.DEV_FQNS
    lines = {}
    for it in items:
        if it.label.value in DEVICES:
            rd, wr = it.renderers[:2]                   # the device's lines, then its swatch
            for fqn, rr in ((read, rd), (write, wr)):
                lines[(fqn, it.label.value)] = (rr.glyph.line_color, str(rr.glyph.line_dash),
                                                float(rr.data_source.data["y"][-1]))
    _check(lines, [it.label.value for it in items])
    rate = {it.label.value: it.renderers[0].glyph.line_color
            for lg in disk["metric"].legend for it in lg.items if it.label.value in DEVICES}
    assert rate == {d: lines[(read, d)][0] for d in DEVICES}
