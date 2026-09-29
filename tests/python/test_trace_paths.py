"""A trace directory is self-contained: session_metadata.pb records every
probe file relative to its own directory (in both probe modes, with an
absolute or a relative output_dir, and with files in subdirectories), so
the directory still renders after it is moved to another place."""

import os
import shutil
import subprocess
import sys

import pytest

import cupti_profiler as cp
import session_metadata_pb2
from tracing_helpers import MODES

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(REPO, "tools")


def _config(out_dir, mode, sys_file="system_metrics.pb", disk_file="disk_metrics.pb",
            events_file="events.pb", meta_file=""):
    procs = [{"pid": 0, "alias": "self"}]
    return {
        "output_dir": out_dir,
        "session_metadata_file": meta_file,
        "gpu": {"enabled": False},
        "system": {"enabled": True, "sampling_frequency_hz": 50, "flush_interval_ms": 100,
                   "output_file": sys_file, "processes": procs, "mode": MODES[mode]},
        "disk": {"enabled": True, "sampling_frequency_hz": 20, "flush_interval_ms": 100,
                 "output_file": disk_file, "processes": procs, "mode": MODES[mode]},
        "events": {"enabled": True, "flush_interval_ms": 100, "output_file": events_file},
    }


def _run(cfg):
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, cfg)
    suite.start()
    gen = suite.get_event_profiler().get_generic_tracker()
    rid = gen.begin_region("work")
    sum(i * i for i in range(2_000_000))
    gen.end_region(rid)
    gen.mark_event("done")
    suite.stop()


def _probe_files(meta_path):
    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString(open(meta_path, "rb").read())
    return [p.output_file for p in meta.probes]


@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_absolute_output_dir_records_file_names(tmp_path, mode):
    out = tmp_path / "run"
    _run(_config(str(out), mode))
    files = _probe_files(out / "session_metadata.pb")
    assert sorted(files) == ["disk_metrics.pb", "events.pb", "system_metrics.pb"]


def test_relative_output_dir_records_file_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(_config("nested/run", "legacy"))
    files = _probe_files(tmp_path / "nested" / "run" / "session_metadata.pb")
    assert sorted(files) == ["disk_metrics.pb", "events.pb", "system_metrics.pb"]


@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_subdirectories_resolve_from_the_metadata_directory(tmp_path, mode):
    """session_metadata.pb in meta/, probe files in probes/: each recorded
    path, joined onto meta/, is the file the probe (or the sidecar) wrote."""
    out = tmp_path / "run"
    for d in ("meta", "probes"):
        (out / d).mkdir(parents=True)
    _run(_config(str(out), mode, sys_file="probes/system_metrics.pb",
                 disk_file="probes/disk_metrics.pb", events_file="probes/events.pb",
                 meta_file="meta/session_metadata.pb"))
    meta_dir = out / "meta"
    files = _probe_files(meta_dir / "session_metadata.pb")
    assert sorted(files) == ["../probes/disk_metrics.pb", "../probes/events.pb",
                             "../probes/system_metrics.pb"]
    for f in files:
        assert (meta_dir / f).is_file(), f


@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_moved_trace_directory_renders(tmp_path, mode):
    """Record into an absolute output_dir, move the whole directory away
    (the original path no longer exists), and render it with both
    visualizers: every probe file is found."""
    src = tmp_path / "machine_a" / "run"
    _run(_config(str(src), mode))
    dst = tmp_path / "machine_b" / "copied_run"
    dst.parent.mkdir()
    shutil.move(str(src), str(dst))
    assert not src.exists()
    meta = str(dst / "session_metadata.pb")

    renders = [
        ("visualize_all.py", tmp_path / "out.png", []),
        ("visualize_interactive.py", tmp_path / "out.html", ["--no-serve"]),
    ]
    for tool, out, extra in renders:
        p = subprocess.run([sys.executable, os.path.join(TOOLS, tool), meta, "-o", str(out)] + extra,
                           capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
        log = p.stdout + p.stderr
        assert p.returncode == 0, f"{tool}:\n{log}"
        assert "not found" not in log, f"{tool}:\n{log}"
        assert out.stat().st_size > 0, tool
