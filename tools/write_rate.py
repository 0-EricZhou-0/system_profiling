"""The write-rate footer of both visualizers, as a table: one row per
probe and a Total, columns Probe, Sampling, Estimated, Measured, Samples.

Rows come in as (probe, sampling_hz, estimated_Bps, measured_Bps,
samples); sampling_hz is the probe's configured rate from the trace's
session metadata (SessionMetadata.probes[].sampling_frequency_hz), None
or 0 where it does not apply (Events), and an estimated rate of 0 means
none (Events: user-driven). Every header and the probe names are
left-aligned, every number right-aligned; "—" where a cell does not
apply. text() is the PNG's monospace block, html() the Bokeh page's
table: the same cells.
"""

from __future__ import annotations

import html as _html

import units

TITLE = "Write rate — estimated vs measured (file_size / trace_duration)"
HEADERS = ["Probe", "Sampling", "Estimated", "Measured", "Samples"]
DASH = "—"


def cells(rows: list[tuple]) -> list[list[str]]:
    """The table's body, the Total row last: one list of cell strings per
    row, in HEADERS order."""
    out = []
    total_est = total_meas = 0.0
    total_n = 0
    for name, hz, est, meas, n in rows:
        out.append([name, f"{int(hz)} Hz" if hz else DASH,
                    units.fmt_bytes(est, rate=True) if est > 0 else DASH,
                    units.fmt_bytes(meas, rate=True), str(int(n))])
        total_est += est
        total_meas += meas
        total_n += n
    out.append(["Total", DASH, units.fmt_bytes(total_est, rate=True),
                units.fmt_bytes(total_meas, rate=True), str(total_n)])
    return out


def text(rows: list[tuple]) -> str:
    """Title, header row, body: columns padded to their widest cell, two
    spaces apart; headers and probe names left-aligned, numbers
    (and "—") right-aligned."""
    body = cells(rows)
    widths = [max(len(r[i]) for r in [HEADERS] + body) for i in range(len(HEADERS))]
    line = lambda r, left: "  ".join(c.ljust(w) if (i == 0 or left) else c.rjust(w)
                                      for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    return "\n".join([TITLE, line(HEADERS, True)] + [line(r, False) for r in body])


def html(rows: list[tuple], color: str) -> str:
    """The same table as HTML (monospace; headers left, numbers right)."""
    esc = _html.escape
    td = "padding:0 0 0 18px;"
    head = "".join(f'<th style="text-align:left;{td if i else "padding:0;"}font-weight:bold">'
                   f"{esc(h)}</th>" for i, h in enumerate(HEADERS))
    body = "".join(
        "<tr>" + "".join(
            f'<td style="text-align:{"left" if i == 0 else "right"};{td if i else "padding:0;"}">'
            f"{esc(c)}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in cells(rows))
    return (f'<div class="cupti-write-rate" style="margin:12px 0 12px 12px;font-family:monospace;'
            f'font-size:11px;line-height:1.35;color:{color};">'
            f"<div>{esc(TITLE)}</div>"
            f'<table style="border-collapse:collapse;font:inherit;color:inherit">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")
