"""Per-PID disk rates are bytes over the ACTUAL interval between samples.

The sample loop sleeps for the nominal period and then does its reads, so
consecutive samples are always somewhat more than 1/hz apart. A rate
computed over the nominal period overstates every byte by that ratio, and
the trace's integral (rate x the interval the samples actually span) no
longer equals the bytes moved.

The writer writes and fsyncs a file, so the syscall-layer and block-layer
counters (wchar, write_bytes) move by the same amount; the test holds
whichever of the two the probe reports.
"""

import os
import time

import pytest

import metric_catalog_pb2
from tracing_helpers import disk_frames, running_suite, tree

MODES = ["legacy", "sidecar"]
MiB = 1 << 20

# On each stdin line (a path): write and fsync 64 MiB there, report the
# process's own counter deltas.
WRITER = """
for line in sys.stdin:
    path = line.strip()
    def io():
        return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}
    before = io()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    chunk = os.urandom(1 << 20)
    for _ in range(64):
        os.write(fd, chunk)
    os.fsync(fd)
    os.close(fd)
    after = io()
    emit(written={k: after[k] - before[k] for k in ("wchar", "write_bytes")})
"""


def fs_type(path):
    best, kind = "", "?"
    for line in open("/proc/mounts"):
        _, mnt, typ = line.split()[:3]
        if path.startswith(mnt) and len(mnt) > len(best):
            best, kind = mnt, typ
    return kind


@pytest.mark.parametrize("mode", MODES)
def test_disk_rate_integrates_to_bytes_written(tmp_path, mode):
    if fs_type(str(tmp_path)) in ("tmpfs", "ramfs", "nfs", "nfs4"):
        pytest.skip(f"{tmp_path} is not a local disk-backed filesystem")
    with tree(WRITER) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "writer")], disk=True,
                           disk_hz=200):
            time.sleep(0.3)
            t.proc.stdin.write(f"{tmp_path / 'data'}\n".encode())
            t.proc.stdin.flush()
            written = t.read("written", 60)["written"]
            time.sleep(0.3)
    assert written["wchar"] == written["write_bytes"] == 64 * MiB, written

    frames = disk_frames(tmp_path)
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    col = fqns[metric_catalog_pb2.SCOPE_PROCESS].index("proc__io_wchar.sum.per_second")
    samples = sorted((s.timestamp_ns, s.values[col]) for f in frames for s in f.process_samples
                     if s.pid == t.pid)
    # The first sample's interval starts at an unseen baseline, but the
    # write starts 0.3 s later, so that sample carries no bytes.
    assert samples and samples[0][1] == 0.0
    traced = sum(rate * (ts - prev_ts) / 1e9
                 for (prev_ts, _), (ts, rate) in zip(samples, samples[1:]))
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    print(f"traced {traced / MiB:.3f} MiB vs written 64 MiB; mean sample gap "
          f"{sum(gaps) / len(gaps) / 1e6:.3f} ms (nominal 5 ms)")
    assert abs(traced - 64 * MiB) <= 0.005 * 64 * MiB
