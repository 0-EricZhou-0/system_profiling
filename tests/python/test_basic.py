"""End-to-end smoke test for the pybind11 wrapper.

Drives ProfilerSuite from Python, exercises both the Generic and GPU
EventTracker domains (using a binding-owned CUDA stream so no cupy/torch
is needed), and verifies the contents of events.pb / session_metadata.pb.
"""

import time

import pytest

import cupti_profiler as cp
import events_pb2
import session_metadata_pb2


def _read_event_traces(path):
    """Read a length-delimited events.pb file into a flat dict, mirroring
    the visualizer's merge logic but trimmed down to what this test needs.
    """
    out = {
        "generic_regions": [],
        "generic_events":  [],
        "gpu_regions":     [],
        "gpu_events":      [],
    }
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data):
        length = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            length |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        msg = events_pb2.EventTrace()
        msg.ParseFromString(data[i:i + length])
        i += length
        for buf in msg.buffers:
            if buf.domain == events_pb2.TIME_DOMAIN_GENERIC:
                out["generic_regions"].extend(buf.regions)
                out["generic_events"].extend(buf.events)
            elif buf.domain == events_pb2.TIME_DOMAIN_GPU:
                out["gpu_regions"].extend(buf.regions)
                out["gpu_events"].extend(buf.events)
    return out


def test_generic_and_gpu_regions(tmp_path):
    suite = cp.ProfilerSuite()
    cp.configure_suite(
        suite,
        {
            "output_dir": str(tmp_path),
            "gpu":    {"enabled": False},
            "system": {"enabled": False},
            "disk":   {"enabled": False},
            "events": {
                "enabled": True,
                "flush_interval_ms": 200,
                "output_file": "events.pb",
            },
        },
    )
    suite.start()

    ep = suite.get_event_profiler()
    gen = ep.get_generic_tracker()
    gpu = ep.get_gpu_tracker()

    with cp.CudaStream() as stream:
        gpu.set_stream(stream)

        gen.mark_event("py-test start")

        rid = gen.begin_region("setup")
        time.sleep(0.05)
        gen.end_region(rid)

        for label in ("phase A", "phase B", "phase C"):
            r = gpu.begin_region(label)
            time.sleep(0.02)
            gpu.end_region(r)

        gpu.mark_event("done")

    suite.stop()

    # --- assertions on events.pb ---
    events_path = tmp_path / "events.pb"
    assert events_path.exists(), "events.pb was not produced"
    events = _read_event_traces(events_path)

    names_generic = [r.name for r in events["generic_regions"]]
    names_gpu     = [r.name for r in events["gpu_regions"]]
    assert "setup" in names_generic, names_generic
    assert {"phase A", "phase B", "phase C"}.issubset(set(names_gpu)), names_gpu
    assert any(e.name == "py-test start" for e in events["generic_events"])
    assert any(e.name == "done"          for e in events["gpu_events"])

    # Region durations should be non-trivial (≥ a few ms each).
    for r in events["generic_regions"]:
        if r.name == "setup":
            assert r.end_timestamp_ns - r.start_timestamp_ns > 1_000_000, \
                f"setup region too short: {r.end_timestamp_ns - r.start_timestamp_ns} ns"
    for r in events["gpu_regions"]:
        assert r.end_timestamp_ns >= r.start_timestamp_ns

    # --- assertions on session_metadata.pb ---
    meta_path = tmp_path / "session_metadata.pb"
    assert meta_path.exists(), "session_metadata.pb was not produced"
    meta = session_metadata_pb2.SessionMetadata()
    meta.ParseFromString(meta_path.read_bytes())
    kinds = {p.kind for p in meta.probes}
    assert session_metadata_pb2.PROBE_KIND_EVENTS in kinds
    # GPU/System/Disk were disabled, so only events should be listed.
    assert session_metadata_pb2.PROBE_KIND_GPU    not in kinds
    assert session_metadata_pb2.PROBE_KIND_SYSTEM not in kinds
    assert session_metadata_pb2.PROBE_KIND_DISK   not in kinds
    assert meta.hostname
    assert meta.start_iso8601
