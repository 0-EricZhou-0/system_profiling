"""Legends of the static visualizer (tools/visualize_all.py) sit above
their panel, outside the axes, under the panel title and clear of the
panel above; with many series they list the LEGEND_MAX_ENTRIES most
active plus one "+k more" entry."""

import re

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import viz_trace  # noqa: E402  (puts tools/ on sys.path)
import panel_legend  # noqa: E402
import visualize_all  # noqa: E402

N = panel_legend.LEGEND_MAX_ENTRIES


def _render(tmp_path, procs, **kw):
    meta = viz_trace.write_trace(str(tmp_path / "trace"), procs, **kw)
    r = visualize_all.build_figure(meta)
    r.fig.canvas.draw()
    return r


def _cpu_panel(r):
    [(ax)] = [ax for p, _s, kind, ax in r.panel_axes
              if kind == "metric" and p.series_glob.startswith("proc__cycles")]
    return ax


@pytest.mark.parametrize("n_procs", [4, N + 15])
def test_legend_above_axes_under_title_clear_of_panel_above(tmp_path, n_procs):
    procs = [viz_trace.proc(100 + i, ppid=100, comm=f"worker-process-{i}", cpu=10.0 * (i + 1),
                            discovered=i > 0) for i in range(n_procs)]
    r = _render(tmp_path, procs, gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])
    renderer = r.fig.canvas.get_renderer()
    fig_w = r.fig.bbox.width
    axes = [ax for _p, _s, _k, ax in r.panel_axes]
    assert len(axes) >= 4
    prev = None
    for ax in axes:
        leg = ax.get_legend()
        assert leg is not None, ax.get_title(loc="left")
        lb = leg.get_window_extent(renderer)
        ab = ax.get_window_extent(renderer)
        tbb = ax._left_title.get_window_extent(renderer)
        name = ax.get_title(loc="left")
        assert lb.y0 >= ab.y1, (name, lb, ab)            # above the axes, not inside
        assert tbb.y0 >= lb.y1 - 0.5, (name, tbb, lb)    # under the title
        assert lb.x0 >= 0 and lb.x1 <= fig_w, (name, lb)  # within the figure width
        if prev is not None:                            # clear of the panel above
            pb = prev.get_tightbbox(renderer)
            assert tbb.y1 <= pb.y0, (name, tbb, pb)
        prev = ax


def test_legend_caps_entries_and_counts_the_rest(tmp_path):
    n = N + 15
    # cpu grows with i: the N most active are the last N
    procs = [viz_trace.proc(1000 + i, ppid=1000, comm=f"c{i}", cpu=1.0 + i,
                            discovered=i > 0) for i in range(n)]
    r = _render(tmp_path, procs)
    ax = _cpu_panel(r)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(texts) == N + 1, texts
    assert texts[-1] == f"+{n - N} more"
    listed = {int(t.split("PID ")[1].split(",")[0].rstrip(")]")) for t in texts[:-1]}
    assert listed == {1000 + i for i in range(n - N, n)}
    # the unlisted are drawn, in the "more" colour
    greys = [ln for ln in ax.get_lines() if ln.get_color() == panel_legend.OTHER_COLOR]
    assert len(greys) == n - N


def test_listed_colours_are_distinct(tmp_path):
    # Process colours follow the CPU ranking (rank r gets colour r mod N).
    # The memory panel lists ranks 0 and N, which share a colour: the
    # panel must give one of them another.
    n = N + 5
    big = {n - 1, n - 1 - N}                   # CPU ranks 0 and N
    procs = [viz_trace.proc(2000 + i, ppid=2000, comm=f"c{i}", cpu=1.0 + i,
                            rss=5e9 if i in big else 1e8 + i, discovered=i > 0)
             for i in range(n)]
    r = _render(tmp_path, procs)
    for _p, _s, _k, ax in r.panel_axes:
        handles = ax.get_legend().legend_handles
        colors = [h.get_color() for h in handles if h.get_linestyle() == "-"]
        assert len(colors) == len(set(colors)), (ax.get_title(loc="left"), colors)


def test_few_series_no_cap(tmp_path):
    procs = [viz_trace.proc(300 + i, ppid=300, comm=f"s{i}", discovered=i > 0) for i in range(3)]
    r = _render(tmp_path, procs)
    texts = [t.get_text() for t in _cpu_panel(r).get_legend().get_texts()]
    assert len(texts) == 3 and not any("more" in t for t in texts), texts


def test_cap_unit():
    shown, hidden = panel_legend.cap([("a", 1), ("b", 5), ("c", 3)], limit=2)
    assert shown == ["b", "c"] and hidden == ["a"]
    shown, hidden = panel_legend.cap([("a", 1), ("b", 5)], limit=2)
    assert shown == ["a", "b"] and hidden == []
    # missing values (NaN) count as nothing, not as NaN
    import numpy as np
    ts = np.array([0, 10**9, 2 * 10**9], dtype=np.uint64)
    assert panel_legend.activity(ts, np.array([2.0, float("nan"), 2.0])) == 2.0


def test_smaller_series_drawn_over_larger(tmp_path):
    """The mean over the SMs is drawn over the busiest SM (max >= avg
    everywhere), not hidden under it."""
    avg, mx = ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
               "sm__cycles_active.max.pct_of_peak_sustained_elapsed")
    r = _render(tmp_path, [viz_trace.proc(10, discovered=False)],
                gpu_fqns=[avg, mx], gpu_values=[40.0, 90.0])
    [ax] = [ax for p, _s, _k, ax in r.panel_axes if p.series_glob.startswith("sm__")]
    z = {ln._series_key[0]: ln.get_zorder() for ln in ax.get_lines()
         if getattr(ln, "_series_key", None)}
    assert z[avg] > z[mx], z


def _io_trace(tmp_path):
    procs = [viz_trace.proc(700, comm="root", discovered=False, cpu=50),
             viz_trace.proc(701, ppid=700, comm="worker", cpu=20)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, disk=True)


KEY = ["IO rchar (sum)", "IO wchar (sum)"]


def test_style_key_on_its_own_first_row_and_process_only(tmp_path):
    """A per-process panel with several metrics: the line-style key alone
    on the legend's first row, the processes under it, each entry naming
    the process only (a cumulative panel's too: no metric, no totals)."""
    meta = _io_trace(tmp_path)
    r = visualize_all.build_figure(meta)
    r.fig.canvas.draw()
    rend = r.fig.canvas.get_renderer()
    io = [(k, ax) for p, _s, k, ax in r.panel_axes if p.series_glob == "proc__io_?char.*"]
    assert {k for k, _ax in io} == {"metric", "integrated"}
    for kind, ax in io:
        texts = [t for t in ax.get_legend().get_texts() if t.get_text()]
        key = [t for t in texts if t.get_text() in KEY]
        procs = [t for t in texts if t.get_text() not in KEY]
        assert [t.get_text() for t in key] == KEY and len(procs) == 2
        ky = {round(t.get_window_extent(rend).y0) for t in key}
        assert len(ky) == 1                                   # one row
        assert min(ky) > max(t.get_window_extent(rend).y1 for t in procs)   # above the processes
        for t in procs:
            assert "PID" in t.get_text() and "rchar" not in t.get_text() \
                and "wchar" not in t.get_text(), t.get_text()
            assert t.get_text().endswith(")"), t.get_text()      # "... (PID n, ...)"


def test_style_key_own_legend_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_io_trace(tmp_path))
    io = [(k, f) for p, k, f in doc.panel_figs if p.series_glob == "proc__io_?char.*"]
    assert {k for k, _f in io} == {"metric", "cumulative"}
    for kind, f in io:
        legends = [r for r in f.above if type(r).__name__ == "Legend"]
        assert len(legends) == 2
        procs, key = legends                       # the key added last: stacked on top
        assert [it.label.value for it in key.items] == KEY and key.ncols == len(KEY)
        labels = [it.label.value for it in procs.items]
        assert len(labels) == 2 and all("PID" in l and "rchar" not in l and "wchar" not in l
                                        for l in labels), labels


def test_exited_process_ends_in_a_dashed_line(tmp_path):
    """Per-process gauge (RSS): an exited process's end, dashed, from 0
    up to its last value; a process still alive at the end has none."""
    procs = [viz_trace.proc(800, comm="root", discovered=False, rss=4e8),
             viz_trace.proc(801, ppid=800, comm="gone", start_s=1.0, end_s=6.0, rss=3e8)]
    meta = viz_trace.write_trace(str(tmp_path / "t"), procs)
    r = visualize_all.build_figure(meta)
    [ax] = [ax for p, _s, k, ax in r.panel_axes if p.series_glob == "proc__rss_bytes"]
    ends = [ln for ln in ax.get_lines() if hasattr(ln, "_end_line")]
    assert [ln._end_line[1] for ln in ends] == [801]
    x, y = ends[0].get_xdata(), ends[0].get_ydata()
    assert x[0] == x[1] and 5.8 <= x[0] <= 6.0 and y[0] == 0 and y[1] > 0
    assert ends[0].get_linestyle() == "--"
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(meta)
    [f] = [f for p, k, f in doc.panel_figs if p.series_glob == "proc__rss_bytes"]
    [seg] = [rr for rr in f.renderers if rr.name == "end-lines"]
    assert len(seg.data_source.data["x0"]) == 1 and seg.glyph.line_dash == "dashed"


# A run total in a label: "= 1.5 MiB", ": ── 3 KiB", "0.2556 GiB".
_TOTAL = re.compile(r"(=|\u2500\u2500|\u254c\u254c|\d\s*(B|KiB|MiB|GiB|TiB)\b)")


def test_cumulative_legends_name_only_both_renderers(tmp_path):
    """Every cumulative companion's legend entries name the series or
    process only, no run totals (the values are on the axis). Both
    renderers, every cumulative panel of the default layout."""
    procs = [viz_trace.proc(900 + i, ppid=900 if i else 0, comm=f"p{i}", discovered=bool(i),
                            cpu=10 + i) for i in range(N + 3)]    # with a +k more entry
    meta = viz_trace.write_trace(str(tmp_path / "t"), procs, disk=True,
                                 devices={"nvme0n1": (4 << 20, 1 << 20), "sda": (1 << 20, 0)})
    r = visualize_all.build_figure(meta)
    static = [(p.title, [t.get_text() for t in ax.get_legend().get_texts()])
              for p, _s, k, ax in r.panel_axes if k.startswith("integrated")]
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(meta)
    bokeh = [(p.title, [it.label.value for lg in f.legend for it in lg.items])
             for p, k, f in doc.panel_figs if k == "cumulative"]
    assert len(static) == len(bokeh) >= 2, (static, bokeh)      # per-process I/O, disk
    assert any(any(l.startswith("+") for l in labels) for _t, labels in static)
    for title, labels in static + bokeh:
        assert labels, title
        for label in labels:
            assert not _TOTAL.search(label), (title, label)
