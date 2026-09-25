"""Startup situation report: written to session_metadata.pb, one line
per check with its consequence, and one line per tracked root that
classifies it as spawned (a descendant of this process) or attached.
"""

import os
import subprocess
import sys
import time

import pytest

import session_metadata_pb2
from tracing_helpers import running_suite

MODES = ["legacy", "sidecar"]

REQUIRED = [
    "observer",
    "/proc/<pid>/task/<tid>/children",
    "pidfd_open (syscall)",
    "descendant tracking",
    "yama ptrace_scope",
    "observer effective capabilities",
    "secure-exec (AT_SECURE) of the observer",
    "CUDA forward-compat on LD_LIBRARY_PATH",
    "child subreaper (this process)",
    "taskstats per-PID query",
]


def _fstype(path):
    """Independent read of the filesystem type holding `path`."""
    path, best = os.path.realpath(path), None
    for line in open("/proc/self/mountinfo"):
        pre, post = line.split(" - ")
        mnt = pre.split()[4]
        if path == mnt or path.startswith(mnt.rstrip("/") + "/"):
            if best is None or len(mnt) > len(best[0]):
                best = (mnt, post.split()[0])
    return best[1]


@pytest.mark.parametrize("mode", MODES)
def test_situation_report(tmp_path, mode):
    attached = os.getppid()   # our parent: never our descendant
    spawned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with running_suite(tmp_path, mode,
                           discovery={"enabled": True, "scan_interval_ms": 100},
                           processes=[(attached, "parent")]) as suite:
            suite.add_tracked_process(spawned.pid, "child")   # mid-run
            time.sleep(0.2)
    finally:
        spawned.kill()
        spawned.wait()

    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString((tmp_path / "session_metadata.pb").read_bytes())
    lines = {c.check: c for c in meta.situation}
    for line in meta.situation:
        print(f"{'!' if line.degraded else ' '} {line.check}: {line.observed} -> {line.consequence}")

    missing = [c for c in REQUIRED if c not in lines]
    assert not missing, f"missing checks: {missing}"
    for c in meta.situation:
        assert c.observed and c.consequence, f"line without a consequence: {c}"

    # Kernel 5.15 on the compute nodes has both.
    assert lines["pidfd_open (syscall)"].observed == "available"
    assert lines["/proc/<pid>/task/<tid>/children"].observed == "present"
    assert lines["descendant tracking"].observed.startswith("on by default, recursive, every 100 ms")
    assert lines["child subreaper (this process)"].observed == "not set"
    # A per-PID taskstats query needs CAP_NET_ADMIN (EPERM otherwise).
    cap_eff = int(next(l for l in open("/proc/self/status")
                       if l.startswith("CapEff")).split()[1], 16)
    if not cap_eff >> 12 & 1:
        assert lines["taskstats per-PID query"].observed.startswith("Operation not permitted"), \
            lines["taskstats per-PID query"]

    # Observer placement, and the filesystem of the observer's binary.
    obs = lines["observer"].observed
    if mode == "sidecar":
        assert obs.startswith("sidecar pid "), obs
        binary = obs.split("(", 1)[1].rstrip(")")
    else:
        assert obs == f"in-process (pid {os.getpid()})", obs
        binary = os.path.realpath("/proc/self/exe")
    fs = lines[f"filesystem of {binary}"]
    fstype = _fstype(binary)
    assert fs.observed.startswith(fstype + " on "), (fs.observed, fstype)
    if fstype.startswith("nfs"):
        # The cluster's /home is NFSv4: it cannot hold file capabilities.
        assert fs.degraded and "network filesystem" in fs.consequence, fs

    # Per-root classification.
    child = lines[f"target {spawned.pid}"]
    parent = lines[f"target {attached}"]
    assert "spawned by this process" in child.observed, child
    assert "ATTACHED" in parent.observed, parent
    assert "(same)" in child.observed and "io readable" in child.observed, child
