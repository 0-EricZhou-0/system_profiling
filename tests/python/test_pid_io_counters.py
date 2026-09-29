"""Per-PID I/O: each proc__io_* metric carries the /proc/<pid>/io counter
it is named after.

A tracked process runs one I/O pattern per phase, with idle gaps between
them, and records its own /proc/self/io deltas around each phase — the
kernel's ground truth. Each pattern moves a different subset of the five
counters (see docs/metric-model.md, "Per-PID I/O counters"), so a metric
that carried the wrong counter integrates to the wrong number in at least
one phase:

  write_fsync   write() + fsync          wchar, write_bytes
  read_warm     read(), page cache warm  rchar
  mmap_cold     mmap + touch, cold cache read_bytes
  pipe          write() to a pipe        wchar
  delete_dirty  write(), unlink unsynced wchar, write_bytes, cancelled_write_bytes

Runs on a disk-backed filesystem: tmpfs has no storage layer. The cold
cache is made unprivileged with posix_fadvise(DONTNEED) and checked with
mincore.
"""

import time

import pytest

import metric_catalog_pb2
from test_disk_rates import fs_type
from tracing_helpers import disk_frames, running_suite, tree

MODES = ["legacy", "sidecar"]
MiB = 1 << 20
GAP_S = 0.4

COUNTERS = {
    "rchar":                 "proc__io_rchar.sum.per_second",
    "wchar":                 "proc__io_wchar.sum.per_second",
    "read_bytes":            "proc__io_read_bytes.sum.per_second",
    "write_bytes":           "proc__io_write_bytes.sum.per_second",
    "cancelled_write_bytes": "proc__io_cancelled_write_bytes.sum.per_second",
}

# The pattern each phase exists for: counter -> expected kernel delta in
# MiB (None = anything; the trace is still checked against the kernel).
PREMISE = {
    "write_fsync":  {"wchar": 64, "write_bytes": 64, "rchar": 0, "read_bytes": 0},
    "read_warm":    {"rchar": 64, "read_bytes": 0, "wchar": 0, "write_bytes": 0},
    "mmap_cold":    {"read_bytes": 64, "rchar": 0, "wchar": 0, "write_bytes": 0},
    "pipe":         {"wchar": 8, "write_bytes": 0, "read_bytes": 0},
    "delete_dirty": {"wchar": 32, "write_bytes": 32, "cancelled_write_bytes": 32},
}

BODY = f"""
import ctypes, mmap
MiB = 1 << 20
GAP = {GAP_S!r}
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_long]

def io():
    return {{k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}}

def resident(path):
    fd = os.open(path, os.O_RDONLY)
    n = os.fstat(fd).st_size
    addr = libc.mmap(None, n, 1, 1, fd, 0)   # PROT_READ, MAP_SHARED
    pages = (n + 4095) // 4096
    vec = (ctypes.c_ubyte * pages)()
    libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(n), vec)
    libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(n))
    os.close(fd)
    return sum(v & 1 for v in vec) / pages

def write_file(path, mib, sync):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    for _ in range(mib):
        os.write(fd, chunk)
    if sync:
        os.fsync(fd)
    os.close(fd)

def read_file(path):
    fd = os.open(path, os.O_RDONLY)
    while os.read(fd, MiB):
        pass
    os.close(fd)

def mmap_touch(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        s = 0
        for i in range(0, len(mm), 4096):
            s += mm[i]
        mm.close()

def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def to_pipe(mib):
    for _ in range(mib):
        sink.stdin.write(chunk)
    sink.stdin.flush()

def delete_dirty(path):
    write_file(path, 32, sync=False)
    os.unlink(path)

d = sys.stdin.readline().strip()
a, b = os.path.join(d, "a"), os.path.join(d, "b")
chunk = os.urandom(MiB)
sink = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
phases = []
def phase(name, fn, before=None):
    time.sleep(GAP)
    if before:
        before()
    res = resident(a) if os.path.exists(a) else None
    i0 = io(); t0 = time.monotonic_ns()
    fn()
    t1 = time.monotonic_ns(); i1 = io()
    phases.append(dict(name=name, t0=t0, t1=t1, resident=res,
                       d={{k: i1[k] - i0[k] for k in i1}}))

phase("write_fsync", lambda: write_file(a, 64, sync=True))
phase("read_warm", lambda: read_file(a))
phase("mmap_cold", lambda: mmap_touch(a), before=lambda: drop_cache(a))
phase("pipe", lambda: to_pipe(8))
phase("delete_dirty", lambda: delete_dirty(b))
time.sleep(GAP)
sink.stdin.close(); sink.wait()
emit(phases=phases)
sys.stdin.readline()
"""


def integrate(samples, t0, t1):
    """Bytes a rate series carries for a phase [t0, t1]: every sample whose
    interval ends inside the phase or within half a gap after it. The gaps
    around a phase are idle, so the samples straddling its edges carry
    only the phase's own bytes."""
    total = 0.0
    for (prev_ts, _), (ts, rate) in zip(samples, samples[1:]):
        if t0 < ts <= t1 + GAP_S * 1e9 / 2:
            total += rate * (ts - prev_ts) / 1e9
    return total


@pytest.mark.parametrize("mode", MODES)
def test_pid_io_counters_match_kernel(tmp_path, mode):
    if fs_type(str(tmp_path)) in ("tmpfs", "ramfs", "nfs", "nfs4"):
        pytest.skip(f"{tmp_path} is not a local disk-backed filesystem")
    with tree(BODY) as t:
        with running_suite(tmp_path, mode, processes=[(t.pid, "io")], disk=True,
                           disk_hz=100):
            time.sleep(0.3)
            t.proc.stdin.write(f"{tmp_path}\n".encode())
            t.proc.stdin.flush()
            phases = t.read("phases", 60)["phases"]
            time.sleep(0.3)

    # The premise: each pattern moved the kernel counters as documented.
    by_name = {p["name"]: p for p in phases}
    assert by_name["mmap_cold"]["resident"] == 0.0, "the cold-cache phase was not cold"
    assert by_name["read_warm"]["resident"] == 1.0, "the warm-cache phase was not warm"
    for name, want in PREMISE.items():
        got = by_name[name]["d"]
        for k, mib in want.items():
            assert abs(got[k] - mib * MiB) <= 64 * 1024, (name, k, got[k] / MiB, mib)

    frames = disk_frames(tmp_path)
    fqns = {s.scope: list(s.fqns) for f in frames for s in f.scope_metric_names}
    cols = fqns[metric_catalog_pb2.SCOPE_PROCESS]
    rows = sorted((s.timestamp_ns, list(s.values)) for f in frames for s in f.process_samples
                  if s.pid == t.pid)
    assert len(rows) > 50, f"too few samples: {len(rows)}"

    failures = []
    for p in phases:
        for counter, fqn in COUNTERS.items():
            j = cols.index(fqn)
            traced = integrate([(ts, v[j]) for ts, v in rows], p["t0"], p["t1"])
            kernel = p["d"][counter]
            print(f"{mode} {p['name']:13s} {counter:22s} kernel {kernel / MiB:8.3f} MiB"
                  f"  trace {traced / MiB:8.3f} MiB")
            if abs(traced - kernel) > max(64 * 1024, 0.01 * kernel):
                failures.append((p["name"], fqn, round(kernel / MiB, 3), round(traced / MiB, 3)))
    assert not failures, f"(phase, metric, kernel MiB, trace MiB): {failures}"
