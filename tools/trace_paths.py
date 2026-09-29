"""Where a probe file named in session_metadata.pb is, for every reader
(visualize_all.py, visualize_interactive.py, live_tail.py).

session_metadata.pb records each probe file (ActiveProbe.output_file)
relative to its own directory, so a trace directory can be copied or
moved. Traces written by v0.2.0 and earlier recorded the path the probe
opened instead: absolute when output_dir was, else relative to the
writer's working directory. When such a path does not exist here, the
file is looked up by name next to session_metadata.pb.
"""

from __future__ import annotations

from pathlib import Path


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
