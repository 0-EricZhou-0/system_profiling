"""Legend labels of the visualizers (tools/). Per-series legends use the
short counter name; a process's `cycles_active` is its CPU (% of one
core), so its legend must say CPU, not the GPU term "Active Cycles"."""

import os
import sys

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
    assert set(labels.values()) == {"CPU"}, labels


def test_gpu_active_cycles_unchanged():
    d = metric_catalog_pb2.MetricDescriptor(fqn="sm__cycles_active.avg", entity="sm",
                                            counter="cycles_active", rollup="avg")
    assert metric_layout.ResolvedSeries(d.fqn, d.scope, 0, d).label_short == "Active Cycles"
