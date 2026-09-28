"""A process that exits normally without calling stop() still gets
complete traces: the Python package's atexit hook (or, for a bare C
exit(), the library's own) stops and flushes everything, and one
"[cupti-profiler] warning:" line says stop() was not called. A suite
destroyed while running is stopped the same way. Needs a GPU with PM
sampling.
"""

import json

import pytest

from gpu_helpers import full_config, last_times_steady, run_child

MODES = {"legacy": 1, "sidecar": 2}

END = """
suite.get_event_profiler().get_generic_tracker().mark_event("before")
time.sleep(3.5)
print(json.dumps({"t_end": time.monotonic_ns()}), flush=True)
"""


def _check(tmp_path, out, err, rc, when):
    assert rc == 0, err[-2000:]
    t_end = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])["t_end"]
    w = [l for l in err.splitlines() if "stop() was not called" in l]
    assert w == [f"[cupti-profiler] warning: stop() was not called; the ProfilerSuite was "
                 f"stopped {when} and its traces flushed"], w
    last = last_times_steady(tmp_path)
    assert None not in last.values(), last
    assert last["gpu"] > t_end - 1_200_000_000 and last["system"] > t_end - 200_000_000, (last, t_end)
    assert (tmp_path / "events.pb").stat().st_size > 0


@pytest.mark.parametrize("mode", MODES)
def test_script_ends_without_stop(tmp_path, mode):
    rc, out, err = run_child(full_config(tmp_path, MODES[mode]), END)
    _check(tmp_path, out, err, rc, "at process exit")


@pytest.mark.parametrize("mode", MODES)
def test_c_exit_without_stop(tmp_path, mode):
    """libc exit() directly: no Python finalization, only C atexit handlers."""
    body = END + "import ctypes, sys; sys.stdout.flush(); ctypes.CDLL(None).exit(0)\n"
    rc, out, err = run_child(full_config(tmp_path, MODES[mode]), body)
    _check(tmp_path, out, err, rc, "at process exit")


def test_destroyed_without_stop(tmp_path):
    body = END + "del suite\nimport gc; gc.collect()\nprint('deleted', flush=True)\n"
    rc, out, err = run_child(full_config(tmp_path, 1), body)
    assert "deleted" in out
    _check(tmp_path, out, err, rc, "when it was destroyed")
