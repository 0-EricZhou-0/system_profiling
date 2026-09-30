"""Every trace records the version that wrote it (SessionMetadata.producer:
the package version from pyproject.toml, through the build, into the
library), and every reader prints ONE warning when that is not its own
version, or is missing, then renders anyway: there are no compatibility
code paths."""

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import session_metadata_pb2
import viz_trace
import trace_paths

WARNING = "may not render correctly"


def _pyproject_version():
    with open(os.path.join(viz_trace.REPO, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _trace(tmp_path, producer):
    """A synthetic trace; producer: None (no field, as v0.2.0 wrote) or
    (version, git_commit)."""
    meta_path = viz_trace.write_trace(str(tmp_path / "trace"), [viz_trace.proc(10, comm="root")],
                                      regions=[("load", 1.0, 8.0)])
    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString(open(meta_path, "rb").read())
    if producer is None:
        meta.ClearField("producer")
    else:
        meta.producer.version, meta.producer.git_commit = producer
    open(meta_path, "wb").write(meta.SerializeToString())
    return meta_path


def test_reader_version_is_the_pyproject_version():
    assert trace_paths.reader_version() == _pyproject_version()


@pytest.mark.parametrize("tool,out,extra", [
    ("visualize_all.py", "out.png", []),
    ("visualize_interactive.py", "out.html", ["--no-serve"]),
])
@pytest.mark.parametrize("producer", ["same", "older", "missing"])
def test_reader_warns_once_on_another_version_and_renders(tmp_path, tool, out, extra, producer):
    mine = _pyproject_version()
    meta_path = _trace(tmp_path, {"same": (mine, "0123456789ab"),
                                  "older": ("0.0.7", "0123456789ab"),
                                  "missing": None}[producer])
    out = tmp_path / out
    p = subprocess.run([sys.executable, os.path.join(viz_trace.TOOLS, tool), meta_path, "-o", str(out)] + extra,
                       capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.stat().st_size > 0            # rendered anyway
    warnings = [l for l in p.stderr.splitlines() if WARNING in l]
    if producer == "same":
        assert warnings == [], p.stderr
        return
    assert len(warnings) == 1, p.stderr
    [w] = warnings
    assert f"read by {tool} from cupti-profiler {mine}" in w, w
    if producer == "older":
        assert "written by cupti-profiler 0.0.7 (commit 0123456789ab)" in w, w
    else:
        assert "written by an unrecorded version" in w, w


def test_live_tail_warns_once_across_rereads(tmp_path, capsys):
    """Live readers re-read session_metadata.pb; the warning is printed once."""
    pytest.importorskip("bokeh")
    import live_tail

    meta_path = Path(_trace(tmp_path, ("0.0.7", "")))
    for _ in range(3):
        meta = live_tail.wait_for_metadata(meta_path, 5.0, lambda _m: None)
        assert meta.producer.version == "0.0.7"
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1, err
    assert "written by cupti-profiler 0.0.7 and is being read by live_tail.py" in err, err


# ---- the library side (needs the build) -----------------------------------

def test_package_version_is_the_pyproject_version():
    cp = pytest.importorskip("cupti_profiler")
    assert cp.__version__ == _pyproject_version()
    # 12 hex digits (+ "-dirty"), or "" when built outside a git checkout.
    assert re.fullmatch(r"([0-9a-f]{12}(-dirty)?)?", cp.__git_commit__), cp.__git_commit__


@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_suite_records_its_version(tmp_path, mode):
    cp = pytest.importorskip("cupti_profiler")
    from tracing_helpers import MODES
    procs = [{"pid": 0, "alias": "self"}]
    suite = cp.ProfilerSuite()
    cp.configure_suite(suite, {
        "output_dir": str(tmp_path),
        "gpu": {"enabled": False},
        "system": {"enabled": True, "sampling_frequency_hz": 50, "flush_interval_ms": 100,
                   "processes": procs, "mode": MODES[mode]},
        "disk": {"enabled": False},
        "events": {"enabled": False},
    })
    suite.start()
    suite.stop()
    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString((tmp_path / "session_metadata.pb").read_bytes())
    assert meta.HasField("producer")
    assert meta.producer.name == "cupti-profiler"
    assert meta.producer.version == cp.__version__ == _pyproject_version()
    assert meta.producer.git_commit == cp.__git_commit__
    assert trace_paths.warn_on_version_mismatch(meta, tmp_path / "session_metadata.pb", "test") is False
