"""The write-rate footer as a table (tools/write_rate.py), the same in
both renderers: a header row (Probe, Sampling, Estimated, Measured,
Samples), one row per probe and a Total; the Sampling column from the
session metadata's probe rates, "—" where it does not apply; headers and
probe names left-aligned, numbers right-aligned."""

import re

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)
import write_rate  # noqa: E402

K = 1024.0
ROWS = [("GPU", 1000, 66.41 * K, 52.81 * K, 169776), ("System", 100, 16.31 * K, 28.1 * K, 44314),
        ("Disk", 100, 26.66 * K, 38.75 * K, 185940), ("Events", None, 0.0, 13.2, 10)]


def _columns(header: str) -> list[int]:
    return [m.start() for m in re.finditer(r"\S+", header)]


def test_text_table_alignment():
    lines = write_rate.text(ROWS).split("\n")
    assert lines[0] == write_rate.TITLE
    header, body = lines[1], lines[2:]
    assert header.split() == write_rate.HEADERS
    starts = _columns(header)                           # where each header begins
    width = max(len(l) for l in lines[1:])
    assert [r.split()[0] for r in body] == ["GPU", "System", "Disk", "Events", "Total"]
    for r in body:
        assert r[0] != " "                              # probe names left-aligned
    for i in range(1, len(starts)):
        end = starts[i + 1] - 2 if i + 1 < len(starts) else width   # the column's right edge
        rights = {len(r[:end].rstrip()) for r in body}
        assert rights == {end}, (write_rate.HEADERS[i], rights, end)   # numbers right-aligned
        lefts = {len(r[:end]) - len(r[:end].rstrip()[::-1].split("  ")[0][::-1]) for r in body}
        assert min(lefts) >= starts[i]                  # a number starts at or after its header
    cells = write_rate.cells(ROWS)
    assert cells[0][1] == "1000 Hz" and cells[3][1] == "—" and cells[3][2] == "—"
    assert cells[-1] == ["Total", "—", write_rate.units.fmt_bytes(109.38 * K, rate=True),
                         write_rate.units.fmt_bytes((52.81 + 28.1 + 38.75) * K + 13.2, rate=True),
                         str(169776 + 44314 + 185940 + 10)]


def test_html_table_alignment():
    h = write_rate.html(ROWS, "#888")
    ths = re.findall(r"<th style=\"text-align:(\w+)", h)
    assert ths == ["left"] * 5                                   # every header left
    rows = re.findall(r"<tr>(.*?)</tr>", h)[1:]
    assert len(rows) == 5
    for r in rows:
        aligns = re.findall(r"<td style=\"text-align:(\w+)", r)
        assert aligns == ["left", "right", "right", "right", "right"]   # name left, numbers right
    texts = [re.findall(r">([^<]*)</td>", r) for r in rows]
    assert texts == write_rate.cells(ROWS)


def _meta(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False)]
    return viz_trace.write_trace(str(tmp_path / "t"), procs, disk=True, regions=[("load", 1.0, 8.0)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])


def test_both_renderers_same_table(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("bokeh")
    import visualize_all
    import visualize_interactive
    meta = _meta(tmp_path)
    r = visualize_all.build_figure(meta)
    [txt] = [t.get_text() for ax in r.fig.axes for t in ax.texts
             if t.get_text().startswith(write_rate.TITLE)]
    lines = txt.split("\n")
    assert lines[1].split() == write_rate.HEADERS
    png = [re.split(r"\s{2,}", l.strip()) for l in lines[2:]]
    html = visualize_interactive.static_page(meta)
    [table] = re.findall(r'<div class="cupti-write-rate".*?</table>', html)
    bk = [re.findall(r">([^<]*)</td>", row) for row in re.findall(r"<tr>(.*?)</tr>", table)[1:]]
    assert png == bk
    names = [row[0] for row in bk]
    assert names[-1] == "Total" and "System" in names and "Events" in names
    by = {row[0]: row for row in bk}
    assert by["System"][1] == f"{viz_trace.HZ} Hz"               # from the session metadata
    assert by["Events"][1] == "—" and by["Events"][2] == "—" and by["Total"][1] == "—"
