"""On a catchable fatal signal the profiler stops and flushes everything
buffered (final GPU decode, every probe, the sidecar), then the signal
takes its course: the handler that was there before still runs (Python's
SIGINT -> KeyboardInterrupt), or the default action does, so the exit
status is unchanged. Needs a GPU with PM sampling.
"""

import json
import signal

import pytest

from gpu_helpers import full_config, last_times_steady, run_child

MODES = {"legacy": 1, "sidecar": 2}

# Flush interval 5 s, signal at 3.5 s: nothing was flushed before it.
SIGNAL_SELF = """
suite.get_event_profiler().get_generic_tracker().mark_event("before")
time.sleep(3.5)
print(json.dumps({"t_sig": time.monotonic_ns()}), flush=True)
os.kill(os.getpid(), %d)
time.sleep(10)
print("survived", flush=True)
"""


def _t_sig(out):
    return json.loads([l for l in out.splitlines() if l.startswith("{")][-1])["t_sig"]


def _assert_flushed_up_to(tmp_path, t_sig):
    last = last_times_steady(tmp_path)
    assert None not in last.values(), f"a trace is empty: {last}"
    # GPU: within one decode interval (1 s); System/Disk: within a tick.
    assert last["gpu"] > t_sig - 1_200_000_000, (last, t_sig)
    assert last["system"] > t_sig - 200_000_000, (last, t_sig)
    assert last["disk"] > t_sig - 200_000_000, (last, t_sig)
    assert (tmp_path / "events.pb").stat().st_size > 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP],
                         ids=["TERM", "INT", "HUP"])
def test_flush_on_signal(tmp_path, mode, sig):
    rc, out, err = run_child(full_config(tmp_path, MODES[mode]), SIGNAL_SELF % sig)
    assert "survived" not in out
    assert rc == -sig, f"rc={rc}\n{err[-2000:]}"      # exit status as without the profiler
    if sig == signal.SIGINT:
        assert "KeyboardInterrupt" in err            # Python's own handler still ran
    assert "stopping the profiler and flushing its traces" in err
    _assert_flushed_up_to(tmp_path, _t_sig(out))


def test_previous_python_handler_runs(tmp_path):
    """A handler installed before the suite starts still runs, after the
    flush: here it prints and exits with status 3."""
    pre = ("signal.signal(signal.SIGTERM, lambda s, f: (print('host handler', flush=True),"
           " os._exit(3)))")
    rc, out, err = run_child(full_config(tmp_path, 1), SIGNAL_SELF % signal.SIGTERM,
                             env={"CHILD_PRE": pre})
    assert rc == 3, f"rc={rc}\n{err[-2000:]}"
    assert "host handler" in out and "survived" not in out
    _assert_flushed_up_to(tmp_path, _t_sig(out))


@pytest.mark.parametrize("mode", MODES)
def test_best_effort_flush_on_segv(tmp_path, mode):
    body = """
suite.get_event_profiler().get_generic_tracker().mark_event("before")
time.sleep(3.5)
print(json.dumps({"t_sig": time.monotonic_ns()}), flush=True)
import ctypes; ctypes.string_at(0)
"""
    rc, out, err = run_child(full_config(tmp_path, MODES[mode]), body)
    assert rc == -signal.SIGSEGV, f"rc={rc}\n{err[-2000:]}"
    _assert_flushed_up_to(tmp_path, _t_sig(out))
