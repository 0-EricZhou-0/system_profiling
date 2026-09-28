"""Synthetic traces for the visualizer tests: a session_metadata.pb, a
system_metrics.pb with a process table and per-process samples, and
optionally events.pb and gpu_metrics.pb, written the way the suite
writes them (length-delimited frames). No library build needed."""

import os
import sys

from google.protobuf import text_format

import disk_metrics_pb2
import events_pb2
import gpu_metrics_pb2
import metric_catalog_pb2
import session_metadata_pb2
import system_metrics_pb2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

CATALOG = os.path.join(REPO, "lib", "data", "metric_catalog.pbtxt")
SYS_FQNS = ["cpu__cycles_busy.avg.pct_of_peak_sustained_elapsed",
            "cpu__cycles_user.avg.pct_of_peak_sustained_elapsed"]
PROC_FQNS = ["proc__cycles_active.sum.per_second", "proc__rss_bytes"]
T0 = 1_000_000_000_000   # trace clock (ns) of the first tick
HZ = 10


def _write_frames(path, frames):
    with open(path, "wb") as f:
        for m in frames:
            b = m.SerializeToString()
            n = len(b)
            while True:                       # varint length prefix
                byte = n & 0x7F
                n >>= 7
                f.write(bytes([byte | (0x80 if n else 0)]))
                if not n:
                    break
            f.write(b)


def proc(pid, ppid=0, comm="p", start_s=0.0, end_s=None, cpu=50.0,
         discovered=True, label="app", comms=(), rss=None):
    """One process of the synthetic table. start_s/end_s: seconds after T0
    (end None = alive at the end); cpu: its constant % of a core; rss: its
    constant RSS in bytes (default grows with cpu); comms: later names as
    (seconds, comm)."""
    return dict(pid=pid, ppid=ppid, comm=comm, start_s=start_s, end_s=end_s, cpu=cpu,
                discovered=discovered, label=label, comms=list(comms),
                rss=1e8 + cpu * 1e6 if rss is None else rss)


IO_FQNS = ["proc__io_rchar.sum.per_second", "proc__io_wchar.sum.per_second"]


DEV_FQNS = ["disk__read_bytes.sum.per_second", "disk__write_bytes.sum.per_second"]


def write_trace(out_dir, procs, duration_s=10.0, regions=(), gpu_fqns=(), gpu_values=None,
                disk=False, devices=None):
    """Write a trace of `procs` (see proc()) over duration_s seconds.
    regions: (name, start_s, end_s). gpu_fqns: GPU metrics to add, each
    a constant: gpu_values[i], else 50. disk: also a Disk trace with each
    process's rchar / wchar rates (cpu x 1 MB/s, cpu x 0.1 MB/s).
    devices: {name: (read B/s, write B/s)}, constant, in that Disk trace.
    Returns the session_metadata.pb path."""
    os.makedirs(out_dir, exist_ok=True)
    catalog = text_format.Parse(open(CATALOG).read(), metric_catalog_pb2.MetricCatalog())
    n_ticks = int(duration_s * HZ) + 1
    ticks = [T0 + int(i * 1e9 / HZ) for i in range(n_ticks)]

    tr = system_metrics_pb2.SystemMetricsTrace()
    tr.header.hostname = "synthetic"
    tr.header.sampling_frequency_hz = HZ
    tr.header.host_cpu_count = 8
    tr.scope_metric_names.add(scope=metric_catalog_pb2.SCOPE_SYSTEM, fqns=SYS_FQNS)
    tr.scope_metric_names.add(scope=metric_catalog_pb2.SCOPE_PROCESS, fqns=PROC_FQNS)
    for p in procs:
        e = tr.tracked_processes.add(pid=p["pid"], parent_pid=p["ppid"],
                                     discovered=p["discovered"], label=p["label"])
        last = p["comms"][-1][1] if p["comms"] else p["comm"]
        e.comm = last
        e.alias = f'{p["label"]}/{last}' if p["discovered"] else p["label"]
        e.start_time_ns = T0 + int(p["start_s"] * 1e9)
        e.comm_history.add(timestamp_ns=e.start_time_ns, comm=p["comm"])
        for t, c in p["comms"]:
            e.comm_history.add(timestamp_ns=T0 + int(t * 1e9), comm=c)
        if p["end_s"] is not None:
            e.removed = True
            e.end_time_ns = T0 + int(p["end_s"] * 1e9)
    for i, ts in enumerate(ticks):
        tr.system_samples.add(timestamp_ns=ts, values=[20.0, 10.0])
        for p in procs:
            t = (ts - T0) / 1e9
            if t < p["start_s"] or (p["end_s"] is not None and t > p["end_s"]):
                continue
            tr.process_samples.add(timestamp_ns=ts, pid=p["pid"],
                                   values=[p["cpu"], p["rss"]])
    _write_frames(os.path.join(out_dir, "system_metrics.pb"), [tr])

    if disk:
        dt = disk_metrics_pb2.DiskMetricsTrace()
        dt.header.CopyFrom(tr.header)
        dt.scope_metric_names.add(scope=metric_catalog_pb2.SCOPE_PROCESS, fqns=IO_FQNS)
        if devices:
            dt.scope_metric_names.add(scope=metric_catalog_pb2.SCOPE_DEVICE, fqns=DEV_FQNS)
            for name, (rd, wr) in devices.items():
                dt.tracked_devices.append(name)
                for ts in ticks:
                    dt.device_samples.add(timestamp_ns=ts, device_name=name, values=[rd, wr])
        for tp in tr.tracked_processes:
            dt.tracked_processes.add().CopyFrom(tp)
        for x in tr.process_samples:
            p = next(q for q in procs if q["pid"] == x.pid)
            dt.process_samples.add(timestamp_ns=x.timestamp_ns, pid=x.pid,
                                   values=[p["cpu"] * 1e6, p["cpu"] * 1e5])
        _write_frames(os.path.join(out_dir, "disk_metrics.pb"), [dt])

    meta = session_metadata_pb2.SessionMetadata(hostname="synthetic",
                                                wall_clock_epoch_ns=1_700_000_000_000_000_000,
                                                start_iso8601="2026-09-28T00:00:00Z")
    meta.catalog.CopyFrom(catalog)
    meta.probes.add(kind=session_metadata_pb2.PROBE_KIND_SYSTEM,
                    output_file="system_metrics.pb", sampling_frequency_hz=HZ)
    if disk:
        meta.probes.add(kind=session_metadata_pb2.PROBE_KIND_DISK,
                        output_file="disk_metrics.pb", sampling_frequency_hz=HZ)

    if regions:
        ev = events_pb2.EventTrace()
        ev.metadata.steady_clock_reference_ns = T0
        buf = ev.buffers.add()
        for name, s, e in regions:
            buf.regions.add(name=name, start_timestamp_ns=T0 + int(s * 1e9),
                            end_timestamp_ns=T0 + int(e * 1e9))
        _write_frames(os.path.join(out_dir, "events.pb"), [ev])
        meta.probes.add(kind=session_metadata_pb2.PROBE_KIND_EVENTS, output_file="events.pb")

    if gpu_fqns:
        g = gpu_metrics_pb2.GPUMetricsTrace()
        g.header.hostname = "synthetic"
        g.header.sampling_frequency_hz = HZ
        g.scope_metric_names.add(scope=metric_catalog_pb2.SCOPE_GPU, fqns=list(gpu_fqns))
        g.tracked_gpus.add(device_index=0, device_name="GPU", chip_name="X")
        for ts in ticks:
            g.samples.add(timestamp_ns=ts, gpu_index=0,
                          values=list(gpu_values or [50.0] * len(gpu_fqns)))
        _write_frames(os.path.join(out_dir, "gpu_metrics.pb"), [g])
        meta.probes.add(kind=session_metadata_pb2.PROBE_KIND_GPU,
                        output_file="gpu_metrics.pb", sampling_frequency_hz=HZ)

    path = os.path.join(out_dir, "session_metadata.pb")
    with open(path, "wb") as f:
        f.write(meta.SerializeToString())
    return path
