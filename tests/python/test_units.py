"""One byte-unit rule (tools/units.py) for every bytes and bytes/s axis,
the run totals and the write-rate footer: the unit comes from the largest
value plotted, with a 2x threshold (>= 2 TiB -> TiB, >= 2 GiB -> GiB,
>= 2 MiB -> MiB, >= 2 KiB -> KiB, else B), not from the panel's ceiling."""

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)
import units  # noqa: E402

K, M, G, T = 1024.0, 1024.0 ** 2, 1024.0 ** 3, 1024.0 ** 4


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("value,unit", [
    (0, "B"), (2 * K - 1, "B"), (2 * K, "KiB"),
    (2 * M - 1, "KiB"), (2 * M, "MiB"),
    (2 * G - 1, "MiB"), (2 * G, "GiB"),
    (2 * T - 1, "GiB"), (2 * T, "TiB"),
])
def test_thresholds(value, unit, rate):
    div, label = units.byte_unit(value, rate=rate)
    assert label == unit + ("/s" if rate else "")
    assert div == {"B": 1, "KiB": K, "MiB": M, "GiB": G, "TiB": T}[unit]


def test_values_and_empty():
    assert units.fmt_bytes(1.5 * G) == "1,536 MiB"          # not 1.5 GiB
    assert units.fmt_bytes(3 * G, rate=True) == "3 GiB/s"
    assert units.byte_unit(None) == (1.0, "B") and units.byte_unit(0, rate=True)[1] == "B/s"
    assert units.largest([]) == 0.0 and units.largest([[1.0, float("nan"), 5.0]]) == 5.0


PCIE = "pcie__read_bytes.sum.per_second"


def _trace(tmp_path, bps):
    return viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 gpu_fqns=[PCIE], gpu_values=[bps])


def test_axis_unit_from_the_data_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    for bps, want in ((3 * G, "GiB/s"), (1.5 * G, "MiB/s"), (3 * K, "KiB/s")):
        r = visualize_all.build_figure(_trace(tmp_path / str(bps), bps))
        [ax] = [ax for p, _s, k, ax in r.panel_axes if p.series_glob.startswith("pcie") and k == "metric"]
        assert ax.get_ylabel() == want, (bps, ax.get_ylabel())


def test_axis_unit_from_the_data_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    for bps, want in ((3 * G, "GiB/s"), (1.5 * G, "MiB/s"), (3 * K, "KiB/s")):
        doc = visualize_interactive.build_static(_trace(tmp_path / str(bps), bps))
        [f] = [f for p, k, f in doc.panel_figs if p.series_glob.startswith("pcie") and k == "metric"]
        assert f.yaxis[0].axis_label == want, (bps, f.yaxis[0].axis_label)


@pytest.mark.parametrize("factor,value,unit", [
    (1.0, K - 1, "B"), (1.0, K, "KiB"), (1.0, G - 1, "MiB"), (1.0, G, "GiB"),
    (2.0, 2 * G - 1, "MiB"), (2.0, 2 * G, "GiB"),
    (4.0, 4 * K - 1, "B"), (4.0, 4 * K, "KiB"), (4.0, 4 * G - 1, "MiB"), (4.0, 4 * G, "GiB"),
])
def test_factor_thresholds(factor, value, unit):
    assert units.byte_unit(value, factor=factor)[1] == unit
    assert units.byte_unit(value, rate=True, factor=factor)[1] == unit + "/s"


def test_factor_below_one_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        units.set_scale_factor(0.5)
    assert units.scale_factor() == units.DEFAULT_SCALE_FACTOR


@pytest.mark.parametrize("tool", ["visualize_all.py", "visualize_interactive.py"])
def test_cli_rejects_factor_below_one(tool):
    import os
    import subprocess
    import sys
    if tool == "visualize_interactive.py":
        pytest.importorskip("bokeh")
    p = subprocess.run([sys.executable, os.path.join(viz_trace.TOOLS, tool), "none.pb",
                        "--unit-scale-factor", "0.5"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "--unit-scale-factor" in p.stderr and ">= 1" in p.stderr, p.stderr


def test_factor_changes_the_axis_both_renderers(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("bokeh")
    import visualize_all
    import visualize_interactive
    try:
        for bps, factor, want in ((3 * G, 4.0, "MiB/s"), (1.5 * G, 1.0, "GiB/s")):
            meta = _trace(tmp_path / f"{bps}-{factor}", bps)
            r = visualize_all.build_figure(meta, unit_scale_factor=factor)
            [ax] = [ax for p, _s, k, ax in r.panel_axes if p.series_glob.startswith("pcie") and k == "metric"]
            doc = visualize_interactive.build_static(meta, unit_scale_factor=factor)
            [f] = [f for p, k, f in doc.panel_figs if p.series_glob.startswith("pcie") and k == "metric"]
            assert ax.get_ylabel() == f.yaxis[0].axis_label == want, (bps, factor)
    finally:
        units.set_scale_factor(units.DEFAULT_SCALE_FACTOR)
