"""Viewing a trace needs no build: requirements-viz.txt plus
tools/gen_protos.py (grpc_tools.protoc, proto/ -> generated/proto/).
Without the generated modules the visualizers say what to run instead of
an ImportError traceback."""

import os
import re
import shutil
import subprocess
import sys

import pytest

import viz_trace

REPO = viz_trace.REPO


def _checkout(tmp_path):
    """The files a viewer needs from a fresh clone, without generated/."""
    root = tmp_path / "clone"
    for d in ("tools", "proto", "configs"):
        shutil.copytree(os.path.join(REPO, d), root / d,
                        ignore=shutil.ignore_patterns("__pycache__", "src"))
    for f in ("pyproject.toml", "requirements-viz.txt"):
        shutil.copy(os.path.join(REPO, f), root / f)
    assert not (root / "generated").exists()
    return root


def _run(root, *args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          cwd=str(root), timeout=300)


@pytest.mark.parametrize("tool,out,extra", [
    ("visualize_all.py", "out.png", []),
    ("visualize_interactive.py", "out.html", ["--no-serve"]),
    ("visualize_single.py", "out.png", []),
])
def test_missing_generated_modules_say_what_to_run(tmp_path, tool, out, extra):
    root = _checkout(tmp_path)
    meta = viz_trace.write_trace(str(tmp_path / "trace"), [viz_trace.proc(10)])
    p = _run(root, f"tools/{tool}", meta, "-o", str(tmp_path / out), *extra)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "Traceback" not in p.stderr, p.stderr
    assert f"{tool} cannot start" in p.stderr, p.stderr
    assert "python tools/gen_protos.py" in p.stderr, p.stderr


@pytest.mark.parametrize("tool,out,extra", [
    ("visualize_all.py", "out.png", []),
    ("visualize_interactive.py", "out.html", ["--no-serve"]),
])
def test_gen_protos_then_render(tmp_path, tool, out, extra):
    pytest.importorskip("grpc_tools")
    root = _checkout(tmp_path)
    p = _run(root, "tools/gen_protos.py")
    assert p.returncode == 0, p.stdout + p.stderr
    protos = sorted(f[:-len(".proto")] for f in os.listdir(root / "proto"))
    assert sorted(f[:-len("_pb2.py")] for f in os.listdir(root / "generated" / "proto")
                  if f.endswith("_pb2.py")) == protos
    meta = viz_trace.write_trace(str(tmp_path / "trace"), [viz_trace.proc(10)],
                                 regions=[("load", 1.0, 8.0)])
    p = _run(root, f"tools/{tool}", meta, "-o", str(tmp_path / out), *extra)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / out).stat().st_size > 0
    assert "warning" not in p.stderr.lower(), p.stderr   # same version, fresh modules


def test_missing_package_says_what_to_install(capsys):
    import gen_protos
    with pytest.raises(SystemExit) as e:
        gen_protos.require_viewer_modules("some_tool.py", ["numpy", "no_such_package_xyz"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "missing Python packages: no_such_package_xyz" in err, err
    assert "pip install -r" in err and "requirements-viz.txt" in err, err


def test_requirements_viz_lists_the_viewer_packages():
    lines = [l.strip() for l in open(os.path.join(REPO, "requirements-viz.txt"))
             if l.strip() and not l.startswith("#")]
    names = {re.split(r"[<>=!~ ]", l, maxsplit=1)[0] for l in lines}
    assert names == {"numpy", "matplotlib", "bokeh", "protobuf", "grpcio-tools"}, lines
    assert "bokeh>=3.9" in lines, lines
