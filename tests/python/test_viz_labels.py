"""Legend labels of the visualizers (tools/). Per-series legends use the
short counter name; a process's `cycles_active` is its CPU (% of one
core), so its legend must say CPU, not the GPU term "Active Cycles".
A metric that is a statistic over its entity's instances (the FQN's
rollup) says which: "(avg)", "(max)", "(sum)". The dotted line at a
panel's ceiling is labelled "Peak:", not "Max:" (it is not the data's
maximum)."""

import os
import sys

import pytest

from google.protobuf import text_format

import metric_catalog_pb2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import metric_layout  # noqa: E402

CATALOG = os.path.join(os.path.dirname(__file__), "..", "..", "lib", "data", "metric_catalog.pbtxt")


def descriptor(fqn):
    cat = text_format.Parse(open(CATALOG).read(), metric_catalog_pb2.MetricCatalog())
    [d] = [m for m in cat.metrics if m.fqn == fqn]
    return d


def test_process_cpu_legend_reads_cpu():
    d = descriptor("proc__cycles_active.sum.per_second")
    series = [metric_layout.ResolvedSeries(d.fqn, d.scope, pid, d) for pid in (101, 202)]
    assert series[0].label_short == "CPU"
    labels = metric_layout.disambiguate_short_labels(series)
    assert set(labels.values()) == {"CPU (sum)"}, labels


def test_gpu_active_cycles_unchanged():
    d = metric_catalog_pb2.MetricDescriptor(fqn="sm__cycles_active.avg", entity="sm",
                                            counter="cycles_active", rollup="avg")
    assert metric_layout.ResolvedSeries(d.fqn, d.scope, 0, d).label_short == "Active Cycles"


def _gpu(fqn):
    return metric_layout.ResolvedSeries(fqn, metric_catalog_pb2.SCOPE_GPU, 0,
                                        metric_layout.synthesize_descriptor(fqn))


def test_statistic_in_every_label():
    avg = _gpu("sm__cycles_active.avg.pct_of_peak_sustained_elapsed")
    mx = _gpu("sm__cycles_active.max.pct_of_peak_sustained_elapsed")
    # alone in its panel, the statistic is still named
    assert metric_layout.disambiguate_short_labels([avg]) == {(avg.fqn, 0): "Active Cycles (avg)"}
    labels = metric_layout.disambiguate_short_labels([avg, mx])
    assert labels == {(avg.fqn, 0): "Active Cycles (avg)", (mx.fqn, 0): "Active Cycles (max)"}
    rd = _gpu("dram__read_throughput.avg.pct_of_peak_sustained_elapsed")
    wr = _gpu("dram__write_throughput.avg.pct_of_peak_sustained_elapsed")
    assert set(metric_layout.disambiguate_short_labels([rd, wr]).values()) == \
        {"Read Throughput (avg)", "Write Throughput (avg)"}
    # entity collision keeps the statistic
    rx = _gpu("nvlrx__bytes.sum.per_second")
    tx = _gpu("nvltx__bytes.sum.per_second")
    assert set(metric_layout.disambiguate_short_labels([rx, tx]).values()) == \
        {"NVLink RX Bytes (sum)", "NVLink TX Bytes (sum)"}


def test_no_statistic_without_rollup():
    d = descriptor("mem__used_bytes")
    s = metric_layout.ResolvedSeries(d.fqn, d.scope, None, d)
    assert metric_layout.disambiguate_short_labels([s]) == {(d.fqn, None): "Used Bytes"}


def test_peak_line_is_labelled_peak(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import viz_trace
    import visualize_all
    meta = viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
                                           "sm__cycles_active.max.pct_of_peak_sustained_elapsed"])
    r = visualize_all.build_figure(meta)
    [(sm_ax, sm_legend)] = [(ax, [t.get_text() for t in ax.get_legend().get_texts()])
                            for p, _s, _k, ax in r.panel_axes if p.series_glob.startswith("sm__")]
    assert sm_legend == ["Active Cycles (avg)", "Active Cycles (max)"]
    texts = [t.get_text() for t in sm_ax.texts]
    assert "Peak: 100 %" in texts and not any(t.startswith("Max") for t in texts), texts


def test_peak_line_is_labelled_peak_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import viz_trace
    import visualize_interactive
    meta = viz_trace.write_trace(str(tmp_path / "t"), [viz_trace.proc(10, discovered=False)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])
    doc = visualize_interactive.build_static(meta)
    [fig] = [f for p, _k, f in doc.panel_figs if p.series_glob.startswith("sm__")]
    labels = [r.text for r in fig.center if type(r).__name__ == "Label"]
    assert labels == ["Peak: 100 %"], labels
    assert [it.label.value for it in fig.legend[0].items] == ["Active Cycles (avg)"]
