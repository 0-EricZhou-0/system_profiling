"""Gap A: a reading taken by PID number is kept only if the process is
still alive when its pidfd is checked AFTER the read.

The kernel frees a number only when its process is reaped, after it has
exited; so if the pidfd reports the process alive after the reads, the
number was never reused during them and the data is the process's. A
reading followed by a dead pidfd may belong to the number's next owner
and must be dropped.

A real reuse inside the few microseconds between a read and the check
cannot be arranged, so the test uses a test-only hook
(testing::KillAfterNextRead): right after the probe reads the target
(System: its CPU clock; Disk: /proc/<pid>/io), the hook kills it, waits
until it has exited, and marks that reading as a foreign process's
(System: +1000 s of CPU, ~10^7 % over one tick; Disk: +1 TB on every
I/O counter). Polling after the reads discards it; polling before them
(the order the check replaced) lets it through.

Both probes, in both modes. In LEGACY the probes run in this process
and the hook is armed directly; in SIDECAR they run in the sidecar,
which arms the same hook at startup from
CUPTI_PROFILER_TEST_KILL_AFTER_READ (inherited from this process) to
fire on its N-th read of the target. Roots and discovered processes.
"""

import os
import time

import pytest

import metric_catalog_pb2
from cupti_profiler import _native
from tracing_helpers import disk_frames, running_suite, system_frames, tree

ON = {"enabled": True, "scan_interval_ms": 50}
HZ = 100
READS_BEFORE_KILL = 50            # ~0.5 s of samples at HZ
FQN = {"system": "proc__cycles_active.sum.per_second",
       "disk": "proc__io_rchar.sum.per_second"}
# A real value stays far below these; the foreign reading is ~10^7 %
# (System) or ~10^14 B/s (Disk).
LIMIT = {"system": 1000.0, "disk": 1e11}

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
@pytest.mark.parametrize("probe", ["system", "disk"])
@pytest.mark.parametrize("mode", ["legacy", "sidecar"])
def test_reading_of_a_process_that_died_is_discarded(tmp_path, monkeypatch, mode, probe, kind):
    with tree(BODY) as t:
        child = t.read("child")["child"]
        t.extra_pids.append(child)
        target = t.pid if kind == "root" else child
        disc = ON if kind == "discovered" else None
        if mode == "sidecar":
            monkeypatch.setenv("CUPTI_PROFILER_TEST_KILL_AFTER_READ",
                               f"{probe}:{target}:{READS_BEFORE_KILL}")
        with running_suite(tmp_path, mode, processes=[(t.pid, "root")], discovery=disc,
                           hz=HZ, disk=(probe == "disk"), disk_hz=HZ, flush_ms=100):
            if mode == "legacy":
                time.sleep(READS_BEFORE_KILL / HZ)   # tracked, several samples
                _native._testing_kill_after_next_read(target, probe)
            deadline = time.monotonic() + 5
            while not gone(target) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert gone(target), "the hook never fired"
            time.sleep(0.4)
    frames = system_frames(tmp_path) if probe == "system" else disk_frames(tmp_path)
    cols = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    col = cols[metric_catalog_pb2.SCOPE_PROCESS].index(FQN[probe])
    values = [s.values[col] for f in frames for s in f.process_samples if s.pid == target]
    assert len(values) >= 20, f"target barely sampled: {len(values)}"
    foreign = [v for v in values if v > LIMIT[probe]]
    assert not foreign, f"a reading taken after the process died reached the trace: {foreign}"
    rows = [tp for f in frames for tp in f.tracked_processes if tp.pid == target]
    assert rows[-1].removed and rows[-1].end_time_ns > 0, rows[-1]
    assert rows[-1].discovered == (kind == "discovered")
