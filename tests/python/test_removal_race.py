"""A RemoveTrackedProcess() that lands while a flush is between its
tracked-process snapshot and its commit must still produce the
removed=true marker (the race fixed in cad56b2).

Driven deterministically with the test-only flush gate
(<cupti_profiler/testing.h>): the flush thread is held after writing a
flush whose snapshot has the PID unmarked, the removal is made, then the
thread commits. Committing every pending entry (the old code) drops the
PID there, and its marker is never written.
"""

import time

from cupti_profiler import _native
from tracing_helpers import running_suite, system_frames, tree


def test_removal_during_flush_keeps_marker(tmp_path):
    with tree("sys.stdin.readline()") as t:
        with running_suite(tmp_path, "legacy", processes=[(t.pid, "victim")],
                           flush_ms=100) as suite:
            time.sleep(0.3)
            _native._testing_arm_flush_gate()
            try:
                assert _native._testing_wait_flush_held(5.0), "no flush reached the gate"
                suite.remove_tracked_process(t.pid)
            finally:
                _native._testing_release_flush_gate()
            time.sleep(0.4)   # several more flushes
    flags = [tp.removed for f in system_frames(tmp_path) for tp in f.tracked_processes
             if tp.pid == t.pid]
    assert flags, "the PID never appeared"
    assert flags.count(True) == 1 and flags[-1], \
        f"removed=true marker lost (flags per flush: {flags})"
