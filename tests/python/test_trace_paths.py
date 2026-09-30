"""A trace directory is self-contained: every file is directly in
output_dir (a configured name with a "/" is rejected), session_metadata.pb
records each probe file relative to its own directory (in both probe
modes, with an absolute or a relative output_dir), so the directory still
renders after it is moved to another place."""

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


BAD_NAME = "must be a plain file name: subdirectories are not allowed; use output_dir"
SAME_NAME = "every file of a trace needs its own name"


@pytest.mark.parametrize("kw,message", [
    ({"sys_file": "probes/system_metrics.pb"}, "system.output_file " + '"probes/system_metrics.pb" ' + BAD_NAME),
    ({"disk_file": "sub/disk_metrics.pb"}, "disk.output_file " + '"sub/disk_metrics.pb" ' + BAD_NAME),
    ({"events_file": "../events.pb"}, "events.output_file " + '"../events.pb" ' + BAD_NAME),
    ({"meta_file": "meta/session_metadata.pb"}, "session_metadata_file " + '"meta/session_metadata.pb" ' + BAD_NAME),
    ({"meta_file": "/tmp/session_metadata.pb"}, "session_metadata_file " + '"/tmp/session_metadata.pb" ' + BAD_NAME),
    ({"sys_file": ".."}, "system.output_file " + '".." ' + BAD_NAME),
    ({"disk_file": "system_metrics.pb"}, 'disk.output_file and system.output_file are both "system_metrics.pb": ' + SAME_NAME),
    ({"events_file": "session_metadata.pb"}, 'session_metadata_file and events.output_file are both "session_metadata.pb": ' + SAME_NAME),
    ({"sys_file": "", "disk_file": "system_metrics.pb"}, 'disk.output_file and system.output_file are both "system_metrics.pb": ' + SAME_NAME),
], ids=["probes/", "sub/", "../", "meta/", "absolute", "dotdot", "two-probes", "probe-and-metadata", "default-and-explicit"])
def test_bad_file_names_are_rejected(tmp_path, kw, message):
    """Every file of a trace is directly in output_dir, with its own name:
    a name with a "/", or one used twice, fails Configure() with
    InvalidConfig and a message naming the fields, and nothing is written."""
    out = tmp_path / "run"
    code = (
        "import sys, json, cupti_profiler as cp\n"
        "s = cp.ProfilerSuite()\n"
        "try:\n"
        "    cp.configure_suite(s, json.loads(sys.argv[1]))\n"
        "except Exception as e:\n"
        "    print('RAISED', e)\n"
        "    sys.exit(3)\n"
        "s.start(); s.stop()\n"
    )
    import json
    p = subprocess.run([sys.executable, "-c", code, json.dumps(_config(str(out), "legacy", **kw))],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 3, p.stdout + p.stderr
    assert "InvalidConfig" in p.stdout, p.stdout + p.stderr
    assert message in p.stderr, p.stderr
    assert not (out / "session_metadata.pb").exists()


def test_unset_file_names_get_defaults(tmp_path):
    """An enabled probe with no output_file writes its default file, and
    the metadata names it: no empty name reaches session_metadata.pb."""
    out = tmp_path / "run"
    cfg = _config(str(out), "legacy")
    cfg["gpu"] = {"enabled": True, "sampling_frequency_hz": 100,
                  "metrics": ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"]}
    for k in ("system", "disk", "events"):
        del cfg[k]["output_file"]
    _run(cfg)
    files = _probe_files(out / "session_metadata.pb")
    assert sorted(files) == ["disk_metrics.pb", "events.pb", "gpu_metrics.pb", "system_metrics.pb"]
    for f in files:
        assert (out / f).is_file(), f


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
