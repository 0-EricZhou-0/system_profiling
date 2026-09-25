"""SIDECAR mode: the sidecar process's lifecycle and its control pipes.

The sidecar is found as the child of this process whose comm starts with
"cupti-profiler" (the binary is cupti-profiler-sidecar; comm keeps 15
characters).
"""

import os
import subprocess

from tracing_helpers import running_suite


def sidecar_pids(parent=None):
    parent = parent or os.getpid()
    out = []
    for tid in os.listdir(f"/proc/{parent}/task"):
        try:
            kids = open(f"/proc/{parent}/task/{tid}/children").read().split()
        except OSError:
            continue
        for k in kids:
            try:
                comm = open(f"/proc/{k}/comm").read().strip()
            except OSError:
                continue
            if comm.startswith("cupti-profiler"):
                out.append(int(k))
    return out


def pipe_fds(pid):
    """fd -> 'pipe:[inode]' for every pipe the process holds."""
    out = {}
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("pipe:"):
            out[int(fd)] = link
    return out


def test_sidecar_pipes_not_inherited(tmp_path):
    # A program this process execs after Configure() (close_fds=False, as
    # with os.posix_spawn or a C++ host's fork+exec) must not inherit the
    # host's ends of the sidecar's pipes: an inherited write end of the
    # control pipe keeps it open after the host exits, so the sidecar
    # never sees EOF.
    with running_suite(tmp_path, "sidecar"):
        [sc] = sidecar_pids()
        control = {v for fd, v in pipe_fds(sc).items() if fd in (3, 4, 5)}
        child = subprocess.Popen(["sleep", "30"], close_fds=False)
        try:
            inherited = control & set(pipe_fds(child.pid).values())
        finally:
            child.kill()
            child.wait()
    assert len(control) == 3, f"sidecar fds 3/4/5 should be its three pipes: {pipe_fds(sc)}"
    assert not inherited, f"an exec'd child of the host holds the sidecar's pipes: {inherited}"
