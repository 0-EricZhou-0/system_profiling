"""visualize_interactive.py serves on 127.0.0.1 by default, in static
and live mode: the page shows the whole trace, so it is not offered to
every interface unless asked (--host 0.0.0.0)."""

import sys

import pytest

import viz_trace

pytest.importorskip("bokeh")
import visualize_interactive as vi  # noqa: E402


class _Stop(Exception):
    pass


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["visualize_interactive.py", *argv])
    return vi.main()


@pytest.mark.parametrize("extra,host", [([], "127.0.0.1"), (["--host", "0.0.0.0"], "0.0.0.0")])
def test_static_mode_host(tmp_path, monkeypatch, extra, host):
    meta = viz_trace.write_trace(str(tmp_path / "trace"), [viz_trace.proc(10)])
    served = []
    monkeypatch.setattr(vi, "_serve", lambda path, h, port: served.append((h, port)))
    assert _main(monkeypatch, meta, "-o", str(tmp_path / "p.html"), *extra) == 0
    assert served == [(host, 8000)]


@pytest.mark.parametrize("extra,host", [([], "127.0.0.1"), (["--host", "0.0.0.0"], "0.0.0.0")])
def test_live_mode_host(tmp_path, monkeypatch, extra, host):
    meta = viz_trace.write_trace(str(tmp_path / "trace"), [viz_trace.proc(10)])
    # Live mode refuses a layout with cumulative (INTEGRATE) panels, which
    # the default layout has: use it without them.
    layout = tmp_path / "layout.pbtxt"
    layout.write_text("\n".join(l for l in open(vi._HERE.parent / "configs" / "visualizer_panels.pbtxt")
                                if "PANEL_AGGREGATION_INTEGRATE" not in l))
    seen = []

    def fake_server(apps, **kw):
        seen.append(kw["address"])
        raise _Stop

    monkeypatch.setattr(vi, "Server", fake_server)
    with pytest.raises(_Stop):
        _main(monkeypatch, "--live", meta, "--panel-layout", str(layout), *extra)
    assert seen == [host]
