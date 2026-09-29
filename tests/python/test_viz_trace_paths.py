"""The readers (visualize_all.py, visualize_interactive.py, live_tail.py)
find a trace's probe files through one helper (tools/trace_paths.py): a
recorded path is relative to session_metadata.pb's directory, first; an
old trace's absolute path that no longer exists falls back to the file's
name next to session_metadata.pb."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import session_metadata_pb2
import viz_trace
import trace_paths

PROBE_FILES = {"system_metrics.pb", "disk_metrics.pb", "events.pb", "gpu_metrics.pb"}


def _trace(tmp_path):
    procs = [viz_trace.proc(10, comm="root", discovered=False)]
    return viz_trace.write_trace(str(tmp_path / "trace"), procs, disk=True,
                                 regions=[("load", 1.0, 8.0)], events=[("ready", 2.0)],
                                 gpu_fqns=["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"])


def _rewrite_probe_paths(meta_path, fn):
    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString(open(meta_path, "rb").read())
    for p in meta.probes:
        p.output_file = fn(p.output_file)
    open(meta_path, "wb").write(meta.SerializeToString())
    return meta


def _old_absolute(meta_path):
    """A v0.2.0 trace recorded on another machine: absolute paths into a
    directory that does not exist here."""
    return _rewrite_probe_paths(meta_path, lambda f: "/nonexistent/machine_a/run/" + os.path.basename(f))


def test_old_absolute_paths_fall_back_to_the_metadata_directory(tmp_path):
    meta_path = _trace(tmp_path)
    meta = _old_absolute(meta_path)
    got = {Path(trace_paths.resolve_probe_path(meta_path, p.output_file)) for p in meta.probes}
    assert got == {Path(meta_path).parent / f for f in PROBE_FILES}


def test_existing_absolute_path_is_used(tmp_path):
    """An old trace read where it was written: its absolute path exists and wins."""
    meta_path = _trace(tmp_path)
    other = tmp_path / "elsewhere" / "system_metrics.pb"
    other.parent.mkdir()
    other.write_bytes(b"")
    assert trace_paths.resolve_probe_path(meta_path, str(other)) == other


def test_relative_path_resolves_against_the_metadata_directory_first(tmp_path, monkeypatch):
    """A same-named file in the working directory does not shadow the trace's own."""
    meta_path = _trace(tmp_path)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "system_metrics.pb").write_bytes(b"")
    monkeypatch.chdir(cwd)
    assert trace_paths.resolve_probe_path(meta_path, "system_metrics.pb") == \
        Path(meta_path).parent / "system_metrics.pb"


def test_not_yet_written_file_resolves_to_the_metadata_directory(tmp_path, monkeypatch):
    """live_tail resolves before the probes flush: a relative path names
    the metadata directory's file even when nothing exists yet."""
    monkeypatch.chdir(tmp_path)
    assert trace_paths.resolve_probe_path(tmp_path / "t" / "session_metadata.pb", "gpu_metrics.pb") == \
        tmp_path / "t" / "gpu_metrics.pb"


@pytest.mark.parametrize("tool,out,extra", [
    ("visualize_all.py", "out.png", []),
    ("visualize_interactive.py", "out.html", ["--no-serve"]),
])
def test_old_absolute_trace_renders(tmp_path, tool, out, extra):
    meta_path = _trace(tmp_path)
    _old_absolute(meta_path)
    out = tmp_path / out
    p = subprocess.run([sys.executable, os.path.join(viz_trace.TOOLS, tool), meta_path, "-o", str(out)] + extra,
                       capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log
    assert "not found" not in log, log
    assert out.stat().st_size > 0


def test_live_tail_follows_old_absolute_paths(tmp_path):
    pytest.importorskip("bokeh")
    import live_tail
    import metric_catalog
    import metric_layout

    meta_path = _trace(tmp_path)
    meta = _old_absolute(meta_path)
    layout = metric_layout.load_panel_layout(os.path.join(viz_trace.REPO, "configs", "visualizer_panels.pbtxt"))
    coord = live_tail.LiveCoordinator(metric_catalog.load_catalog_from_session_metadata(meta), layout,
                                      Path(meta_path), meta, log=lambda _m: None,
                                      series_factory=None, panel_factory=None, t0_ns=None)
    paths = {Path(t.path) for _k, t in coord.tails}
    assert paths == {Path(meta_path).parent / f for f in ("system_metrics.pb", "disk_metrics.pb",
                                                          "gpu_metrics.pb")}
