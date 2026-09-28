"""ProfilerSuiteConfig.disable_signal_handlers: default (false) installs
the flush-on-signal handlers at start(); true leaves the process's signal
dispositions exactly as they were. Needs a GPU with PM sampling.
"""

import json
import signal

import pytest

from gpu_helpers import full_config, run_child

# Which signals this process catches (SigCgt), before and after start().
CAUGHT = """
def caught():
    for l in open("/proc/self/status"):
        if l.startswith("SigCgt:"):
            m = int(l.split()[1], 16)
            return [s for s in range(1, 32) if m >> (s - 1) & 1]
print(json.dumps({"running": caught()}), flush=True)
suite.stop()
print(json.dumps({"stopped": caught()}), flush=True)
"""


def _lines(out):
    r = {}
    for l in out.splitlines():
        if l.startswith("{"):
            r.update(json.loads(l))
    return r


@pytest.mark.parametrize("disable", [False, True])
def test_signal_handlers_knob(tmp_path, disable):
    cfg = full_config(tmp_path, 1)
    cfg["disable_signal_handlers"] = disable
    pre = "import json as _j; _c = [l for l in open('/proc/self/status') if l.startswith('SigCgt:')][0]; print(_j.dumps({'before': int(_c.split()[1], 16)}), flush=True)"
    rc, out, err = run_child(cfg, CAUGHT, env={"CHILD_PRE": pre})
    assert rc == 0, err
    r = _lines(out)
    # Standard signals only: glibc installs its own real-time ones
    # (SIGSETXID = 33) once threads exist.
    before = [s for s in range(1, 32) if r["before"] >> (s - 1) & 1]
    ours = {signal.SIGTERM, signal.SIGHUP, signal.SIGSEGV}   # none caught by Python itself
    if disable:
        assert r["running"] == before, "dispositions changed with disable_signal_handlers"
    else:
        assert ours <= set(r["running"]), r["running"]
    assert r["stopped"] == before, "stop() must leave the dispositions as they were"


def test_disabled_handlers_do_not_flush(tmp_path):
    cfg = full_config(tmp_path, 1)
    cfg["disable_signal_handlers"] = True
    rc, out, err = run_child(cfg, "time.sleep(3.5); os.kill(os.getpid(), signal.SIGTERM); time.sleep(10)")
    assert rc == -signal.SIGTERM
    assert "stopping the profiler" not in err
    assert (tmp_path / "system_metrics.pb").stat().st_size == 0   # nothing flushed before 5 s
