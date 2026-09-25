"""Gap A: a reading taken by PID number is kept only if the process is
still alive when its pidfd is checked AFTER the read.

The kernel frees a number only when its process is reaped, after it has
exited; so if the pidfd reports the process alive after the reads, the
number was never reused during them and the data is the process's. A
reading followed by a dead pidfd may belong to the number's next owner
and must be dropped.

A real reuse inside the few microseconds between a read and the check
cannot be arranged, so the test uses a test-only hook
(testing::KillAfterNextRead): right after the probe reads the target's
CPU clock, the hook kills it, waits until it has exited, and marks that
reading as a foreign process's (+1000 s of CPU, i.e. ~10^7 % over one
tick). Polling after the reads discards it; polling before them (the
order the check replaced) lets it through. In-process (LEGACY) probes
only: the hook lives in this process. The sidecar runs the same probe
code.

Roots and discovered processes, with discovery off and on.
"""

import os
import time

import pytest

from cupti_profiler import _native
from tracing_helpers import running_suite, system_frames, tree

ON = {"enabled": True, "scan_interval_ms": 50}
CPU_FQN = "proc__cycles_active.sum.per_second"

BODY = """
c = sleeper(30)
emit(child=c.pid)
sys.stdin.readline()
"""


def gone(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


@pytest.mark.parametrize("kind", ["root", "discovered"])
def test_reading_of_a_process_that_died_is_discarded(tmp_path, kind):
    with tree(BODY) as t:
        child = t.read("child")["child"]
        t.extra_pids.append(child)
        target = t.pid if kind == "root" else child
        disc = ON if kind == "discovered" else None
        with running_suite(tmp_path, "legacy", processes=[(t.pid, "root")],
                           discovery=disc, hz=100, flush_ms=100):
            time.sleep(0.5)                       # tracked, several samples
            _native._testing_kill_after_next_read(target)
            deadline = time.monotonic() + 3
            while not gone(target) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert gone(target), "the hook never fired"
            time.sleep(0.4)
    frames = system_frames(tmp_path)
    col = next(list(s.fqns).index(CPU_FQN) for f in frames for s in f.scope_metric_names
               if CPU_FQN in s.fqns)
    values = [s.values[col] for f in frames for s in f.process_samples if s.pid == target]
    assert len(values) >= 20, f"target barely sampled: {len(values)}"
    foreign = [round(v) for v in values if v > 1000]
    assert not foreign, f"a reading taken after the process died reached the trace: {foreign} %"
    rows = [tp for f in frames for tp in f.tracked_processes if tp.pid == target]
    assert rows[-1].removed and rows[-1].end_time_ns > 0, rows[-1]
    assert rows[-1].discovered == (kind == "discovered")
