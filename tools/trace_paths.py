"""What every reader (visualize_all.py, visualize_interactive.py,
live_tail.py) needs from session_metadata.pb besides the data: where a
probe file named in it is, and whether the version that wrote the trace
is the one reading it.

session_metadata.pb records each probe file (ActiveProbe.output_file)
relative to its own directory, so a trace directory can be copied or
moved. Traces written by v0.2.0 and earlier recorded the path the probe
opened instead: absolute when output_dir was, else relative to the
writer's working directory. When such a path does not exist here, the
file is looked up by name next to session_metadata.pb.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def resolve_probe_path(metadata_path: str | Path, recorded: str) -> Path:
    """The probe file `recorded` of the trace whose session_metadata.pb is
    `metadata_path`. When no candidate exists yet (a live trace whose
    probe has not flushed), the path the recording names."""
    meta_dir = Path(metadata_path).parent
    p = Path(recorded)
    if p.is_absolute():
        cands = [p, meta_dir / p.name]
    else:
        cands = [meta_dir / p, Path.cwd() / p, meta_dir / p.name]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def reader_version() -> str | None:
    """The version of these tools: the package version in the
    pyproject.toml of the checkout they are in (the library's version
    comes from the same line). None when it cannot be read."""
    try:
        text = _PYPROJECT.read_text()
    except OSError:
        return None
    m = re.search(r'^\[project\][^\[]*?^version = "([^"]+)"', text, re.M | re.S)
    return m.group(1) if m else None


_warned: set[str] = set()


def warn_on_version_mismatch(meta, metadata_path: str | Path, tool: str) -> bool:
    """Print one warning to stderr when the trace whose session_metadata.pb
    is `meta` (read from `metadata_path`) was not written by this version,
    or does not say which version wrote it. There are no compatibility
    code paths: the trace is rendered anyway, and may not render
    correctly. Warns once per trace per process (live readers re-read the
    file). Returns whether it warned."""
    reader = reader_version()
    p = meta.producer if meta.HasField("producer") else None
    if p is not None and p.version and p.version == reader:
        return False
    key = str(Path(metadata_path).resolve())
    if key in _warned:
        return False
    _warned.add(key)
    if p is None or not p.version:
        writer = "an unrecorded version of cupti-profiler (v0.2.0 or earlier; no producer in session_metadata.pb)"
    else:
        writer = f"{p.name or 'cupti-profiler'} {p.version}"
        if p.git_commit:
            writer += f" (commit {p.git_commit})"
    reading = f"cupti-profiler {reader}" if reader else "cupti-profiler of an unknown version"
    print(f"warning: {metadata_path}: this trace was written by {writer} and is being "
          f"read by {tool} from {reading}; it may not render correctly.",
          file=sys.stderr, flush=True)
    return True
