"""Python wrapper around the cupti_profiler C++ suite.

Re-exports the pybind11 extension classes and adds:
  - configure_suite(suite, dict): build a ProfilerSuiteConfig from a Python
    dict and push it into the suite (skips the .pbtxt round-trip).
  - CudaStream: context manager around create/destroy_cuda_stream.
  - adopt_orphans(): opt-in subreaper helper for launchers.
"""

from . import _native
from ._native import (
    ProfilerSuite,
    GpuProfiler,
    SystemProfiler,
    DiskProfiler,
    EventProfiler,
    EventTracker,
    Domain,
    GpuProfilerConfig,
    SystemProfilerConfig,
    DiskProfilerConfig,
    EventProfilerConfig,
    create_cuda_stream,
    destroy_cuda_stream,
)
from ._stream import CudaStream

__all__ = [
    "ProfilerSuite",
    "GpuProfiler",
    "SystemProfiler",
    "DiskProfiler",
    "EventProfiler",
    "EventTracker",
    "Domain",
    "GpuProfilerConfig",
    "SystemProfilerConfig",
    "DiskProfilerConfig",
    "EventProfilerConfig",
    "create_cuda_stream",
    "destroy_cuda_stream",
    "CudaStream",
    "configure_suite",
    "adopt_orphans",
]


def adopt_orphans() -> None:
    """Make this process (the launcher) a child subreaper. Opt-in.

    Sets ``PR_SET_CHILD_SUBREAPER`` on the calling process, so orphaned
    descendants of the workload it spawns re-parent here instead of to
    init, and lets descendant tracking reap the orphans it saw being
    adopted. The library never does this on its own. Call it once,
    before spawning the workload. Raises ``OSError`` on failure.

    What changes when it is enabled (measured or verified on kernel
    5.15, 2026-09-24):

    ==========================  ==================================  ==========================================
    Aspect                      Without it                          With it
    ==========================  ==================================  ==========================================
    Who is marked               --                                  the calling process (the launcher) only
    Inherited by children?      --                                  **no** -- neither ``fork`` nor ``Popen``
                                                                    children get it, so the workload is never
                                                                    a subreaper (verified)
    Survives launcher execve?   --                                  **yes** (verified): a program the launcher
                                                                    execs is still a subreaper
    Where orphans go            init, or the nearest subreaper      **the launcher** -- their PPid becomes the
                                (``systemd --user``, slurmstepd)    launcher's PID
    Signals to the launcher     none for orphans                    **SIGCHLD for every adopted orphan that
                                                                    exits** -- code that handles SIGCHLD or
                                                                    calls ``waitpid(-1)`` sees children it
                                                                    never started
    Zombies                     reaped by init                      owned by the launcher until reaped. The
                                                                    helper reaps those discovery saw adopted;
                                                                    **orphans adopted before the first scan
                                                                    are not reaped** and stay zombies until
                                                                    the launcher exits (a PID and a
                                                                    process-table slot; no memory or CPU)
    Accounting                  orphans' CPU and storage I/O are    credited to the launcher:
                                credited to init -- invisible       ``getrusage(RUSAGE_CHILDREN)`` and
                                                                    ``os.times()`` child fields **increase**
                                                                    (0.001 s -> 0.501 s for a 0.500 s orphan;
                                                                    32.0 MiB written -> 32.0 MiB folded)
    Process group, session,     --                                  **unchanged** -- Ctrl-C still reaches the
    signal delivery                                                 orphans if they stay in the foreground
                                                                    process group
    Attached (not spawned)      --                                  **no effect** -- only descendants of the
    targets                                                         launcher are covered
    Cost                        --                                  zero at steady state; ~45-85 us per orphan
                                                                    event, on the launcher, never the target
                                                                    (prototype measurement)
    Turning it off              --                                  ``prctl(PR_SET_CHILD_SUBREAPER, 0)``;
                                                                    already-adopted orphans stay adopted
    ==========================  ==================================  ==========================================

    Reaping never uses ``waitpid(-1)``, which would steal the exit status
    of the launcher's own children (``Popen.wait()`` on the workload would
    then see ``ECHILD``, which Python reports as return code 0). An orphan
    is reaped only if discovery found it with a parent other than the
    launcher and, at its exit, its parent is the launcher; it is then
    reaped alone, through its pidfd. Under SIDECAR the sidecar reports
    such exits to the launcher, which reaps them.

    What it does **not** buy: an orphan whose intermediate parent died
    before the first scan shows up as the launcher's child,
    indistinguishable from the launcher's other children, so it is not
    auto-tracked.

    The startup situation report (``session_metadata.pb``) records
    whether this was set, so a trace says which guarantee it was
    collected under.
    """
    _native.enable_child_subreaper()


def _apply_dict_to_message(msg, data: dict) -> None:
    """Generic walker: copy a Python dict onto a protobuf message via reflection.

    Scalars and bytes/strings assign directly. Repeated scalar fields accept
    Python lists. Nested message fields recurse on dict values. Unknown keys
    raise ValueError so typos surface immediately.
    """
    from google.protobuf.descriptor import FieldDescriptor

    fields_by_name = {f.name: f for f in msg.DESCRIPTOR.fields}
    for key, value in data.items():
        if key not in fields_by_name:
            raise ValueError(
                f"Unknown config key {key!r} for message "
                f"{msg.DESCRIPTOR.full_name}; valid keys are "
                f"{sorted(fields_by_name)}"
            )
        field = fields_by_name[key]
        if hasattr(field, "is_repeated"):
            is_repeated = field.is_repeated
        else:
            is_repeated = field.label == FieldDescriptor.LABEL_REPEATED
        if is_repeated:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                # repeated message: list of dicts
                for entry in value:
                    sub = getattr(msg, key).add()
                    _apply_dict_to_message(sub, entry)
            else:
                getattr(msg, key).extend(value)
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            _apply_dict_to_message(getattr(msg, key), value)
        else:
            setattr(msg, key, value)


def configure_suite(suite: ProfilerSuite, config: dict) -> None:
    """Build a ProfilerSuiteConfig from a dict and push it into the suite.

    Equivalent to writing a .pbtxt and calling suite.load_config(path), but
    keeps everything in-process. Calls suite.configure() afterward.

    Example:
        cp.configure_suite(suite, {
            "output_dir": "/tmp/run1",
            "events": {"enabled": True, "flush_interval_ms": 200,
                       "output_file": "events.pb"},
            "gpu":    {"enabled": False},
            "system": {"enabled": False},
            "disk":   {"enabled": False},
        })
    """
    from .proto import profiler_config_pb2  # generated, ships inside the package

    pb = profiler_config_pb2.ProfilerSuiteConfig()
    _apply_dict_to_message(pb, config)
    suite.load_config_from_bytes(pb.SerializeToString())
    suite.configure()
