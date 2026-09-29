"""A fatal CUDA/CUPTI error exits the process (intended), with one
unmistakable stderr line naming the library, the failing call, the error
string and code, the source location and the PID — findable inside a
host application's log.
"""

import json
import re
import subprocess
import sys

import pytest

HOST = """
import json, sys
import cupti_profiler as cp
suite = cp.ProfilerSuite()
cp.configure_suite(suite, json.loads(sys.argv[1]))
print("configure returned", flush=True)
"""

LINE = re.compile(r"^\[cupti-profiler\] FATAL: (?P<call>.+) failed: (?P<what>.+) \((?P<code>-?\d+)\)"
                  r" at (?P<file>[\w.]+):(?P<line>\d+), pid (?P<pid>\d+) — exiting$")


def _gpu_config(tmp_path, **gpu):
    g = {"enabled": True, "sampling_frequency_hz": 1000, "max_samples": 1000,
         "hw_buffer_size": 64 << 20, "output_file": "gpu_metrics.pb",
         "metrics": ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"]}
    g.update(gpu)
    return {"output_dir": str(tmp_path), "gpu": g, "system": {"enabled": False},
            "disk": {"enabled": False}, "events": {"enabled": False}}


@pytest.mark.parametrize("case", ["driver_bad_device", "cupti_bad_metric"])
def test_fatal_error_message(tmp_path, case):
    if case == "driver_bad_device":
        cfg = _gpu_config(tmp_path, device_indices=[4095])
    else:
        cfg = _gpu_config(tmp_path, metrics=["sm__no_such_counter.avg.pct_of_peak_sustained_elapsed"])
    p = subprocess.Popen([sys.executable, "-c", HOST, json.dumps(cfg)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=120)
    print(err)
    assert p.returncode == 1, f"rc={p.returncode}; stdout={out!r}"
    assert "configure returned" not in out
    fatal = [l for l in err.splitlines() if "FATAL" in l]
    assert len(fatal) == 1, f"expected exactly one FATAL line, got {fatal}"
    m = LINE.match(fatal[0])
    assert m, f"FATAL line not in the documented format: {fatal[0]!r}"
    assert int(m["pid"]) == p.pid
    assert m["what"] != "unknown error" and m["file"].endswith(".cpp") and "/" not in m["file"]
    if case == "driver_bad_device":
        assert m["call"].startswith("cuDeviceGet(")
        assert m["code"] == "101"   # CUDA_ERROR_INVALID_DEVICE
        assert m["what"] == "invalid device ordinal"
    else:
        assert m["call"].startswith("cupti") or "CreateConfigImage" in m["call"], m["call"]
        assert m["code"] != "0"
