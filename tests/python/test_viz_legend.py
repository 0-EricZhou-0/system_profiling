"""Legends of the static visualizer (tools/visualize_all.py) sit above
their panel, outside the axes, under the panel title and clear of the
panel above; with many series they list the LEGEND_MAX_ENTRIES most
active plus one "+k more" entry."""

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
