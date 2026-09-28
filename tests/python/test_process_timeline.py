"""The process timeline (tools/process_timeline.py): lane packing uses
exactly as many lanes as the most processes alive at one instant, with
no two bars of a lane overlapping; every process with a tracked parent
has one fork link from that parent's lane to its own at its start; the
timeline sits right under the Region strip in both renderers."""

import random

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)
import process_timeline as pt  # noqa: E402
from metric_projector import ProcessRecord  # noqa: E402

S = 1_000_000_000


def _table(rows):
    """rows: (pid, ppid, start_s, end_s or None, discovered)"""
    return {(pid, int(s * S)): ProcessRecord(pid=pid, parent_pid=ppid, discovered=disc,
                                              comm=f"p{pid}", start_time_ns=int(s * S),
                                              end_time_ns=int(e * S) if e is not None else 0,
                                              removed=e is not None)
            for pid, ppid, s, e, disc in rows}


def _max_overlap(procs):
    ev = sorted([(p.start_ns, 1) for p in procs] + [(p.end_ns, -1) for p in procs],
                key=lambda x: (x[0], x[1]))          # an end before a start at the same instant
    cur = best = 0
    for _t, d in ev:
        cur += d
        best = max(best, cur)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_lane_count_is_max_overlap_and_lanes_do_not_overlap(seed):
    rng = random.Random(seed)
    rows = [(1, 0, 0.0, None, False)]
    for pid in range(2, 202):
        s = rng.uniform(0, 100)
        rows.append((pid, rng.choice([1] + list(range(2, pid))), s, s + rng.expovariate(0.2),
                     True))
    procs = pt.build(_table(rows), 0, 120 * S)
    lanes, n = pt.pack_lanes(procs)
    assert n == _max_overlap(procs)
    by_lane = {}
    for p in procs:
        by_lane.setdefault(lanes[p.key], []).append(p)
    for ps in by_lane.values():
        ps.sort(key=lambda p: p.start_ns)
        for a, b in zip(ps, ps[1:]):
            assert a.end_ns <= b.start_ns, (a, b)


def test_child_goes_next_to_its_parent():
    # Lanes 0-5 busy until 5 s; 3 and 4 free from then on. The child of
    # the process in lane 5 goes to lane 4 (next to it), not lane 3 (the
    # first free lane).
    rows = [(1, 0, 0, None, False), (2, 1, 1, 50, True), (3, 1, 1.5, 50, True),
            (4, 1, 2, 5, True), (5, 1, 2, 5, True), (6, 1, 2.5, 50, True), (7, 6, 6, 10, True)]
    procs = pt.build(_table(rows), 0, 100 * S)
    lanes, n = pt.pack_lanes(procs)
    key = {p.pid: p.key for p in procs}
    assert n == 6 and lanes[key[6]] == 5
    assert lanes[key[7]] == 4


def test_fork_links_one_per_tracked_child():
    rows = [(1, 0, 0, None, False), (2, 1, 1, 50, True), (3, 2, 2, 5, True),
            (4, 99, 3, 8, True)]                    # 99 not tracked: an orphan
    procs = pt.build(_table(rows), 0, 100 * S)
    lanes, _n = pt.pack_lanes(procs)
    links = pt.fork_links(procs, lanes)
    key = {p.pid: p.key for p in procs}
    kind = {p.pid: p.kind for p in procs}
    assert kind == {1: pt.ROOT, 2: pt.DISCOVERED, 3: pt.DISCOVERED, 4: pt.ORPHAN}
    assert sorted(lk.child for lk in links) == sorted([key[2], key[3]])
    for lk in links:
        child = next(p for p in procs if p.key == lk.child)
        assert lk.parent == child.parent
        assert lk.t_ns == child.start_ns
        assert (lk.parent_lane, lk.child_lane) == (lanes[lk.parent], lanes[lk.child])


def test_reused_pid_links_to_the_parent_alive_at_the_fork():
    # PID 7 is used twice; the child forked at 30 s belongs to the second.
    rows = [(1, 0, 0, None, False), (7, 1, 1, 10, True), (7, 1, 20, 60, True), (9, 7, 30, 40, True)]
    procs = pt.build(_table(rows), 0, 100 * S)
    child = next(p for p in procs if p.pid == 9)
    assert child.parent == (7, 20 * S)


def test_ends():
    rows = [(1, 0, 0, None, False)]
    table = _table(rows)
    table[(2, 5 * S)] = ProcessRecord(pid=2, parent_pid=1, discovered=True, comm="x",
                                     start_time_ns=5 * S, removed=True)   # removed by request
    table[(3, 0)] = ProcessRecord(pid=3, parent_pid=1, discovered=True, comm="y")  # no start
    procs = {p.pid: p for p in pt.build(table, 0, 100 * S, first_sample_ns={3: 7 * S},
                                        last_sample_ns={2: 9 * S})}
    assert procs[1].alive and procs[1].end_ns == 100 * S
    assert not procs[2].alive and procs[2].end_ns == 9 * S
    assert procs[3].start_ns == 7 * S


def _trace(tmp_path):
    procs = [viz_trace.proc(100, comm="vllm", discovered=False),
             viz_trace.proc(101, ppid=100, comm="python3", start_s=1.0,
                            comms=[(2.0, "VLLM::Engine")]),
             viz_trace.proc(102, ppid=101, comm="cc", start_s=3.0, end_s=4.0),
             viz_trace.proc(103, ppid=55, comm="sh", start_s=3.5, end_s=5.0)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, regions=[("load", 1.0, 8.0)])


def test_timeline_under_region_static(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_trace(tmp_path))
    ax = r.process_ax
    assert ax is not None
    region = r.region_ax.get_position()
    tl = ax.get_position()
    first = r.panel_axes[0][3].get_position()
    assert first.y1 < tl.y0 and tl.y1 < region.y0     # region, then timeline, then panels
    assert ax.get_xlim() == r.region_ax.get_xlim() == r.panel_axes[0][3].get_xlim()
    assert len(ax.patches) == 4                        # one bar per process
    assert r.timeline.n_lanes == 4                     # all four alive at 3.5 s
    assert len(r.timeline.links) == 2                  # 101 <- 100, 102 <- 101; 103 is an orphan


def test_every_axis_has_the_same_frame_static(tmp_path):
    """A time is at the same x on every row of the PNG: region strip,
    timeline and every panel share the plot frame's left and right edge
    (in pixels, after drawing)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_trace(tmp_path))
    r.fig.canvas.draw()
    axes = [r.region_ax, r.process_ax] + [ax for *_x, ax in r.panel_axes]
    edges = {(round(b.x0, 1), round(b.x1, 1)) for b in (ax.get_window_extent() for ax in axes)}
    assert len(edges) == 1, edges

def test_timeline_under_region_bokeh(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_trace(tmp_path))
    tl = doc.timeline
    assert tl is not None
    band = doc.band.children                            # the sticky band: pinned
    assert tl in doc.timeline_block.children
    assert band.index(doc.timeline_block) == band.index(doc.strips[-1]) + 1   # right under Region
    kids = doc.root.children
    assert doc.panel_figs[0][2] in kids[kids.index(doc.band) + 1].children    # then the first panel
    assert tl.x_range is doc.panel_figs[0][2].x_range
    hover = [t for t in tl.tools if type(t).__name__ == "HoverTool"]
    names = []
    for r in hover[0].renderers:
        names += r.data_source.data["name"]
    assert "python3 -> VLLM::Engine" in names


# ---- labels: every process, none overlapping --------------------------------

def _burst(n=120):
    """A root plus n short processes starting within 3 s of a 60 s trace."""
    rows = [(1, 0, 0, None, False)]
    rng = random.Random(7)
    for pid in range(2, 2 + n):
        s = 20 + rng.uniform(0, 3)
        rows.append((pid, 1, s, s + rng.uniform(0.05, 1.5), True))
    return rows


def test_every_process_labelled_without_overlap():
    procs = pt.build(_table(_burst()), 0, 60 * S)
    lanes, n = pt.pack_lanes(procs)
    width = 1400.0                                      # points
    tw = lambda t: len(t) * 3.6                         # ~6.5 pt text
    labels, rows = pt.place_labels(procs, lanes, 0, 60 * S, width, tw)   # default spacing
    assert {lab.key for lab in labels} == {p.key for p in procs}
    out = [lab for lab in labels if lab.row >= 0]
    assert out and 1 <= rows <= pt.LABEL_MAX_ROWS
    per_s = width / 60
    for r in range(rows):
        iv = sorted((lab.x_s * per_s - tw(lab.text) / 2, lab.x_s * per_s + tw(lab.text) / 2)
                    for lab in out if lab.row == r)
        assert all(a[1] + 8.0 <= b[0] + 1e-6 for a, b in zip(iv, iv[1:])), r   # >= 8 pt apart
        assert iv[0][0] >= -1e-6 and iv[-1][1] <= width + 1e-6
    for lab in labels:                                   # inside labels fit their bars
        if lab.row < 0:
            p = next(q for q in procs if q.key == lab.key)
            assert tw(lab.text) + 8.0 < (p.end_ns - p.start_ns) / 1e9 * per_s
    # the first label row clear of the lowest lane by more than half a lane
    assert pt.label_row_y(n, 0) - pt.LABEL_ROW / 2 - n >= 0.5


def _dense_trace(tmp_path, n=60):
    procs = [viz_trace.proc(100, comm="root", discovered=False)]
    rng = random.Random(3)
    for i in range(n):
        s = 3 + rng.uniform(0, 1.5)
        procs.append(viz_trace.proc(200 + i, ppid=100, comm=f"cc1plus{i % 7}", start_s=s,
                                    end_s=s + rng.uniform(0.1, 1.0)))
    return viz_trace.write_trace(str(tmp_path / "t"), procs, duration_s=10.0,
                                 regions=[("load", 1.0, 8.0)])


def test_timeline_labels_static_no_overlap(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import visualize_all
    r = visualize_all.build_figure(_dense_trace(tmp_path))
    r.fig.canvas.draw()
    rend = r.fig.canvas.get_renderer()
    ax = r.process_ax
    texts = [t for t in ax.texts if hasattr(t, "_process_key")]
    assert {t._process_key for t in texts} == {p.key for p in r.timeline.procs}
    boxes = [t.get_window_extent(rend) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (texts[i].get_text(), texts[j].get_text())
    bars = [p.get_window_extent(rend) for p in ax.patches]
    outside = [(t, b) for t, b in zip(texts, boxes)
               if next(lab for lab in r.timeline.labels if lab.key == t._process_key).row >= 0]
    assert outside
    for t, b in outside:
        assert not any(b.overlaps(bb) for bb in bars), t.get_text()


def test_timeline_labels_bokeh_every_process(tmp_path):
    pytest.importorskip("bokeh")
    import visualize_interactive
    doc = visualize_interactive.build_static(_dense_trace(tmp_path))
    shown = []
    for rr in doc.timeline.renderers:
        if type(rr.glyph).__name__ == "Text":
            shown += list(rr.data_source.data["text"])
    assert len(shown) == 61
    assert all(any(s.startswith(p) for p in ("root", "cc1plus")) for s in shown)
