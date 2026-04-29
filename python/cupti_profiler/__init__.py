"""Python wrapper around the cupti_profiler C++ suite.

Re-exports the pybind11 extension classes and adds:
  - configure_suite(suite, dict): build a ProfilerSuiteConfig from a Python
    dict and push it into the suite (skips the .pbtxt round-trip).
  - CudaStream: context manager around create/destroy_cuda_stream.
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
]


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
