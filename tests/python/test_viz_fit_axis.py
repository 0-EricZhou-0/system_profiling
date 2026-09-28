"""--fit-axis-to-data (opt-in; default off = the axis reaches the Peak):
a ceiling more than units.OFFSCALE_FACTOR x the largest plotted value is
written as "Peak: <value> (off-scale)" and the axis fits the data; a
ceiling near the data is drawn as before. Both renderers."""

import pytest

import viz_trace  # noqa: F401
import units  # noqa: E402

G, K = 1024.0 ** 3, 1024.0
PCIE = "pcie__read_bytes.sum.per_second"
SM = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"


def _meta(tmp_path):
    # PCIe: 3 KiB/s under a 60 GiB/s ceiling (off-scale); SM: 50 % under 100 % (not)
    return viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 gpu_fqns=[PCIE, SM], gpu_values=[3 * K, 50.0], pcie_peak=60 * G)


def _axes(r):
    return {p.series_glob[:4]: ax for p, _s, k, ax in r.panel_axes if k == "metric"}


def test_static(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import visualize_all
    meta = _meta(tmp_path)
    off = _axes(visualize_all.build_figure(meta))
    on = _axes(visualize_all.build_figure(meta, fit_axis_to_data=True))
    # default: stretched to the Peak, drawn
    assert off["pcie"].get_ylim()[1] > 1e6 and any(t.get_text().startswith("Peak:")
                                                   and "off-scale" not in t.get_text()
                                                   for t in off["pcie"].texts)
    # on: fits the data (3 KiB/s -> KiB/s axis, top ~3.3), Peak written off-scale, no line
    top = on["pcie"].get_ylim()[1]
    assert 3.0 < top < 4.0, top
    assert any(t.get_text().endswith("(off-scale)") for t in on["pcie"].texts)
    assert not [ln for ln in on["pcie"].get_lines() if ln.get_linestyle() == ":"]
    # SM: its ceiling is near the data, unchanged by the flag
    assert on["sm__"].get_ylim() == off["sm__"].get_ylim()
    assert not any("off-scale" in t.get_text() for t in on["sm__"].texts)


def test_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive as vi
    meta = _meta(tmp_path)
    try:
        for fit in (False, True):
            doc = vi.build_static(meta, fit_axis_to_data=fit)
            figs = {p.series_glob[:4]: f for p, k, f in doc.panel_figs if k == "metric"}
            pcie, sm = figs["pcie"], figs["sm__"]
            peakish = lambda f, t: [r for r in f.center if type(r).__name__ == t
                                    and r.name not in ("stop-line", "stop-note")]   # not the stop
            labels = [r.text for r in peakish(pcie, "Label")]
            spans = peakish(pcie, "Span")
            if fit:
                assert labels and labels[0].endswith("(off-scale)") and not spans
            else:
                assert spans and not any("off-scale" in l for l in labels)
            assert len(peakish(sm, "Span")) == 1
    finally:
        vi.build_static(meta)                  # back to the default for later tests


def test_factor_documented():
    assert units.OFFSCALE_FACTOR == 5.0
    assert units.off_scale(60 * G, 3 * K) and not units.off_scale(100.0, 50.0)
    assert not units.off_scale(None, 1.0) and not units.off_scale(0.0, 1.0)
