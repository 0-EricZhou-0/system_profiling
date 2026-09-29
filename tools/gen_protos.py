#!/usr/bin/env python3
"""Generate the Python protobuf modules the visualizers read traces with.

    python tools/gen_protos.py

Runs grpc_tools.protoc over proto/*.proto into generated/proto/, the
directory every tool puts on sys.path. The CMake build does the same;
this is for viewing traces without a build (no nvcc, no CUDA): install
requirements-viz.txt, run this, then run the visualizers.

Also the check every visualizer runs before it imports anything else:
missing packages or generated modules print what to run instead of an
ImportError traceback.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO / "proto"
OUT_DIR = REPO / "generated" / "proto"


def _protos() -> list[Path]:
    return sorted(PROTO_DIR.glob("*.proto"))


def generate() -> int:
    """Run grpc_tools.protoc; returns its exit code."""
    if importlib.util.find_spec("grpc_tools") is None:
        print("gen_protos.py: grpc_tools is not installed in this Python "
              f"({sys.executable}). Install the viewer requirements:\n"
              f"    pip install -r {_rel(REPO / 'requirements-viz.txt')}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "grpc_tools.protoc", f"-I{PROTO_DIR}",
           f"--python_out={OUT_DIR}", *map(str, _protos())]
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"generated {len(_protos())} modules into {_rel(OUT_DIR)}")
    return rc


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def require_viewer_modules(tool: str, packages: list[str]) -> None:
    """Exit with instructions when `tool` cannot run: a package in
    `packages` (import names), protobuf, or a generated *_pb2 module is
    missing. Warns (and continues) when a generated module is older than
    its .proto."""
    missing = [p for p in ["google.protobuf", *packages] if _find(p) is None]
    missing_pb2 = [p for p in _protos() if not (OUT_DIR / f"{p.stem}_pb2.py").exists()]
    if missing or missing_pb2:
        lines = [f"{tool} cannot start:"]
        if missing:
            names = ["protobuf" if m == "google.protobuf" else m for m in missing]
            lines += [f"  missing Python packages: {', '.join(names)} (in {sys.executable})",
                      f"    pip install -r {_rel(REPO / 'requirements-viz.txt')}"]
        if missing_pb2:
            lines += [f"  missing generated protobuf modules in {_rel(OUT_DIR)}/ "
                      f"({len(missing_pb2)} of {len(_protos())})",
                      f"    python {_rel(REPO / 'tools' / 'gen_protos.py')}"]
        lines.append("  (no build needed: see docs/tools/README.md, \"Viewing without a build\")")
        print("\n".join(lines), file=sys.stderr)
        sys.exit(2)
    stale = [p.name for p in _protos()
             if p.stat().st_mtime > (OUT_DIR / f"{p.stem}_pb2.py").stat().st_mtime]
    if stale:
        print(f"warning: {tool}: generated protobuf modules are older than {', '.join(stale)}; "
              f"regenerate them: python {_rel(REPO / 'tools' / 'gen_protos.py')}", file=sys.stderr)


def _find(name: str):
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:   # the parent package is missing
        return None


if __name__ == "__main__":
    sys.exit(generate())
