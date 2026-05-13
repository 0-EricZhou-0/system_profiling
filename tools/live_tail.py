"""Live-tail support for visualize_interactive.py.

This module is the shared substrate between the static and `--live` paths
of `visualize_interactive.py`:

  * Projection functions (`project_gpu`, `project_system`, `project_disk`,
    `project_events`) walk a parsed `*Trace` proto and return per-panel
    `ColumnDataSource`-shaped dicts. Builders consume the dicts at figure-
    construction time; the live tick re-projects on each poll and assigns
    fresh dicts to the existing CDS objects (full-replace pattern — keeps
    cumulative columns / per-PID alignment correct without delta state).

  * `TraceTail` tracks a single `.pb` file's byte offset across polls,
    parsing only fully-framed delimited messages and leaving the file
    position before any partial trailing bytes for the next read.

  * `LiveState` ties it all together: the four tails + the in-memory
    accumulating traces + the registry of CDS objects to refresh in
    `tick()`.

The module is import-safe: importing it does not pull in Bokeh or any
generated `_pb2` module. Callers are expected to inject the proto types
they need (visualize_interactive.py does this lazily inside its
existing `_load_all` flow).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Strict, offset-aware varint-delimited message reader
# ---------------------------------------------------------------------------

def parse_delimited_strict(data: bytes, proto_class) -> tuple[list, int]:
    """Parse as many fully-framed delimited messages as fit in `data`.

    Returns `(messages, consumed_bytes)`. `consumed_bytes` is the offset
    *past* the last complete message — the caller should retain
    `data[consumed_bytes:]` and concatenate with the next read to pick up
    a torn trailing message.

    Unlike `visualize_all.read_delimited_messages`, this never falls back
    to whole-buffer parsing on a partial frame — that fallback is correct
    for legacy single-message files but would silently corrupt a tailing
    reader.
    """
    msgs = []
    offset = 0
    n = len(data)
    while offset < n:
        # Decode varint length prefix.
        msg_len = 0
        shift = 0
        varint_bytes = 0
        while offset + varint_bytes < n:
            b = data[offset + varint_bytes]
            msg_len |= (b & 0x7F) << shift
            varint_bytes += 1
            shift += 7
            if (b & 0x80) == 0:
                break
        else:
            # Ran out of bytes mid-varint.
            break

        msg_start = offset + varint_bytes
        msg_end = msg_start + msg_len
        if msg_len == 0 or msg_end > n:
            break

        m = proto_class()
        m.ParseFromString(bytes(data[msg_start:msg_end]))
        msgs.append(m)
        offset = msg_end

    return msgs, offset


@dataclass
class TraceTail:
    """Per-file byte-offset tracker.

    Holds at most one partial trailing buffer between polls; calling
    `read_new_messages()` re-issues an `os.open` + `pread` from the
    current offset, prepends any retained tail bytes, and returns
    only the messages that are now fully framed.
    """
    path: str
    proto_class: Any
    offset: int = 0
    _pending: bytes = b""

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def read_new_messages(self) -> list:
        """Read bytes from `self.offset` to EOF, prepend any retained
        partial-trailing bytes from `self._pending`, parse as many complete
        delimited messages as fit, and remember the unparsed tail for next
        time.

        Invariants:
        * `self.offset` is the file position we've already read up to —
          i.e. `f.tell()` after the last successful `read()`. It only
          advances by the number of bytes pulled off disk.
        * `self._pending` is the unparsed *suffix* of the last `buf` we
          examined. Those bytes come from a contiguous tail at the
          high-address end of the read window, so prepending them to the
          next `chunk` reconstructs the original byte sequence with no
          gaps and no overlaps.
        """
        if not os.path.exists(self.path):
            return []
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size <= self.offset and not self._pending:
            return []

        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        # Bookkeeping: file is now positioned at offset+len(chunk). Update
        # immediately — `_pending` tracks unparsed bytes within the window
        # we've already read off disk, so the offset must reflect everything
        # we just pulled regardless of how much the parser consumes.
        self.offset += len(chunk)

        buf = self._pending + chunk
        msgs, consumed = parse_delimited_strict(buf, self.proto_class)
        self._pending = buf[consumed:]
        return msgs


# ---------------------------------------------------------------------------
# Time-domain helpers
# ---------------------------------------------------------------------------

def gpu_to_steady(cupti_ts, cupti_ref, steady_ref):
    return cupti_ts - cupti_ref + steady_ref


def global_t0_from_traces(gpu_trace, sys_trace, disk_trace) -> int:
    """Earliest data point across enabled probes, in steady_clock ns.

    Mirrors `visualize_interactive._global_t0` exactly so live and static
    runs produce the same time origin.
    """
    cands = []
    if gpu_trace and len(gpu_trace.samples) > 1:
        cands.append(gpu_to_steady(gpu_trace.samples[1].start_timestamp_ns,
                                    gpu_trace.cupti_reference_ns,
                                    gpu_trace.steady_clock_reference_ns))
    if sys_trace and len(sys_trace.cpu_system_samples) > 0:
        cands.append(sys_trace.cpu_system_samples[0].timestamp_ns)
    if disk_trace and len(disk_trace.device_samples) > 0:
        cands.append(disk_trace.device_samples[0].timestamp_ns)
    return min(cands) if cands else 0


def global_xmax_ms(gpu_trace, sys_trace, disk_trace, events, t0) -> float:
    cands = []
    if gpu_trace and len(gpu_trace.samples) > 1:
        last = gpu_trace.samples[-1]
        cands.append(gpu_to_steady(last.start_timestamp_ns,
                                    gpu_trace.cupti_reference_ns,
                                    gpu_trace.steady_clock_reference_ns))
    if sys_trace and len(sys_trace.cpu_system_samples) > 0:
        cands.append(sys_trace.cpu_system_samples[-1].timestamp_ns)
    if disk_trace and len(disk_trace.device_samples) > 0:
        cands.append(disk_trace.device_samples[-1].timestamp_ns)
    if events:
        for r in events["generic_regions"]:
            cands.append(r.end_timestamp_ns)
        for e in events["generic_events"]:
            cands.append(e.timestamp_ns)
        m = events["metadata"]
        if m is not None:
            cref, sref = m.cupti_reference_ns, m.steady_clock_reference_ns
            for r in events["gpu_regions"]:
                cands.append(int(gpu_to_steady(r.end_timestamp_ns, cref, sref)))
            for e in events["gpu_events"]:
                cands.append(int(gpu_to_steady(e.timestamp_ns, cref, sref)))
    if not cands:
        return 0.0
    return (max(cands) - t0) / 1e6


# ---------------------------------------------------------------------------
# Projection functions — produce per-CDS column dicts for each panel.
#
# The keys returned in each `data` map are stable identifiers that
# visualize_interactive's builders register against. Live tick replaces
# `cds.data` for the same key on each poll.
# ---------------------------------------------------------------------------

def project_gpu(gpu_trace, t0: int) -> dict[str, Any]:
    """Project a GpuMetricsTrace into per-panel CDS column dicts.

    Returns:
        {
            "data": {
                "gpu_sm_util":  {...} | None,    # None means skip this panel
                "gpu_warps":    {...} | None,
                "gpu_dram":     {...} | None,
                "gpu_pcie":     {...} | None,
                "gpu_pcie_cum": {...} | None,
                "gpu_nvlink":   {...} | None,
            },
            "meta": {
                "device_name": str, "chip_name": str,
                "peak_dram_gibps":   float | None,
                "peak_pcie_bidi_gibps": float | None,
                "peak_nvlink_bidi_gibps": float | None,
                "pcie_cum_unit": str, "pcie_cum_div": int,
            }
        }
    """
    out_data: dict[str, Any] = {
        "gpu_sm_util":  None, "gpu_warps":    None,
        "gpu_dram":     None, "gpu_pcie":     None,
        "gpu_pcie_cum": None, "gpu_nvlink":   None,
    }
    meta: dict[str, Any] = {"device_name": "", "chip_name": "",
                            "peak_dram_gibps": None,
                            "peak_pcie_bidi_gibps": None,
                            "peak_nvlink_bidi_gibps": None,
                            "pcie_cum_unit": "B", "pcie_cum_div": 1}
    if not gpu_trace or len(gpu_trace.samples) <= 1:
        return {"data": out_data, "meta": meta}

    samples = list(gpu_trace.samples)[1:]  # skip init artifact
    cref = gpu_trace.cupti_reference_ns
    sref = gpu_trace.steady_clock_reference_ns
    ts_steady = np.array([s.start_timestamp_ns for s in samples], dtype=np.float64)
    time_ms = (gpu_to_steady(ts_steady, cref, sref) - t0) / 1e6

    metric_names = list(gpu_trace.metric_names)
    idx = {n: i for i, n in enumerate(metric_names)}
    vals = np.zeros((len(samples), len(metric_names)))
    for i, s in enumerate(samples):
        for j, v in enumerate(s.values):
            vals[i, j] = v

    meta["device_name"] = gpu_trace.device_name
    meta["chip_name"]   = gpu_trace.chip_name

    if "sm__cycles_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        d = {"t":   time_ms,
             "avg": np.where(elapsed > 0,
                              vals[:, idx["sm__cycles_active.avg"]] / elapsed * 100, 0)}
        if "sm__cycles_active.max" in idx:
            d["max"] = np.where(elapsed > 0,
                                 vals[:, idx["sm__cycles_active.max"]] / elapsed * 100, 0)
        out_data["gpu_sm_util"] = d

    if "sm__warps_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
        elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
        d = {"t":   time_ms,
             "avg": np.where(elapsed > 0,
                              vals[:, idx["sm__warps_active.avg"]] / elapsed, 0)}
        if "sm__warps_active.max" in idx:
            d["max"] = np.where(elapsed > 0,
                                 vals[:, idx["sm__warps_active.max"]] / elapsed, 0)
        out_data["gpu_warps"] = d

    drum = "dram__read_throughput.avg.pct_of_peak_sustained_elapsed"
    if drum in idx:
        peak_dram_gibps = (gpu_trace.peak_dram_bw_gbps * 1e9 / (1024 ** 3)
                           if gpu_trace.peak_dram_bw_gbps > 0 else None)
        meta["peak_dram_gibps"] = peak_dram_gibps
        scale = (peak_dram_gibps / 100.0) if peak_dram_gibps else 1.0
        d = {"t": time_ms, "rd": vals[:, idx[drum]] * scale}
        wr_key = "dram__write_throughput.avg.pct_of_peak_sustained_elapsed"
        if wr_key in idx:
            d["wr"] = vals[:, idx[wr_key]] * scale
        out_data["gpu_dram"] = d

    if "pcie__read_bytes.sum.per_second" in idx:
        d = {"t": time_ms,
             "rd": vals[:, idx["pcie__read_bytes.sum.per_second"]] / (1024 ** 3)}
        if "pcie__write_bytes.sum.per_second" in idx:
            d["wr"] = vals[:, idx["pcie__write_bytes.sum.per_second"]] / (1024 ** 3)
        out_data["gpu_pcie"] = d
        if gpu_trace.peak_pcie_bw_bytes_per_sec > 0:
            meta["peak_pcie_bidi_gibps"] = (
                gpu_trace.peak_pcie_bw_bytes_per_sec * 2 / (1024 ** 3))

    if "pcie__read_bytes.sum" in idx and "pcie__write_bytes.sum" in idx:
        rd_cum = np.cumsum(vals[:, idx["pcie__read_bytes.sum"]])
        wr_cum = np.cumsum(vals[:, idx["pcie__write_bytes.sum"]])
        max_bytes = float(max(rd_cum[-1] if rd_cum.size else 0,
                               wr_cum[-1] if wr_cum.size else 0))
        for div, unit in [(1024 ** 4, "TiB"), (1024 ** 3, "GiB"),
                          (1024 ** 2, "MiB"), (1024, "KiB"), (1, "B")]:
            if max_bytes >= div:
                break
        meta["pcie_cum_unit"] = unit
        meta["pcie_cum_div"] = div
        out_data["gpu_pcie_cum"] = {"t": time_ms,
                                     "rd": rd_cum / div, "wr": wr_cum / div}

    if "nvlrx__bytes.sum.per_second" in idx and "nvltx__bytes.sum.per_second" in idx:
        rx_gibps = vals[:, idx["nvlrx__bytes.sum.per_second"]] / (1024 ** 3)
        tx_gibps = vals[:, idx["nvltx__bytes.sum.per_second"]] / (1024 ** 3)
        out_data["gpu_nvlink"] = {"t": time_ms, "rx": rx_gibps, "tx": tx_gibps}
        if gpu_trace.peak_nvlink_bw_bytes_per_sec > 0:
            meta["peak_nvlink_bidi_gibps"] = (
                gpu_trace.peak_nvlink_bw_bytes_per_sec * 2 / (1024 ** 3))

    return {"data": out_data, "meta": meta}


def project_system(sys_trace, t0: int) -> dict[str, Any]:
    """Returns per-panel column dicts for system CPU + memory panels."""
    out_data: dict[str, Any] = {
        "sys_cpu_total": None, "sys_cpu_proc": None,
        "sys_mem_total": None, "sys_mem_proc": None,
    }
    meta: dict[str, Any] = {"pids": [], "tracked_processes": [],
                            "total_ram_gib": 0.0,
                            "cpu_proc_max": 0.0, "mem_proc_max_gib": 0.0}
    if not sys_trace:
        return {"data": out_data, "meta": meta}

    meta["tracked_processes"] = list(sys_trace.tracked_processes)

    if len(sys_trace.cpu_system_samples) > 0:
        s = sys_trace.cpu_system_samples
        time_ms = (np.array([x.timestamp_ns for x in s]) - t0) / 1e6
        user = np.array([x.user_pct for x in s])
        sysp = np.array([x.system_pct for x in s])
        iow  = np.array([x.iowait_pct for x in s])
        total = np.array([x.total_utilization_pct for x in s])
        out_data["sys_cpu_total"] = {
            "t": time_ms,
            "user": user, "sys": sysp, "iow": iow, "total": total,
            "y_user_top": user,
            "y_sys_top":  user + sysp,
            "y_iow_top":  user + sysp + iow,
        }

    if len(sys_trace.cpu_process_samples) > 0:
        proc = list(sys_trace.cpu_process_samples)
        pids = sorted(set(x.pid for x in proc))
        meta["pids"] = pids
        per_pid = {pid: [x for x in proc if x.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms = (np.array([x.timestamp_ns for x in base]) - t0) / 1e6
        d: dict[str, Any] = {"t": time_ms}
        max_val = 0.0
        for pid in pids:
            samps = per_pid[pid]
            user = np.array([x.user_pct for x in samps])
            sysp = np.array([x.system_pct for x in samps])
            n = len(time_ms)
            if user.size != n: user = np.resize(user, n)
            if sysp.size != n: sysp = np.resize(sysp, n)
            d[f"pid_{pid}_user"] = user
            d[f"pid_{pid}_sum"]  = user + sysp
            max_val = max(max_val, float((user + sysp).max() if user.size else 0))
        meta["cpu_proc_max"] = max_val
        out_data["sys_cpu_proc"] = d

    if len(sys_trace.memory_system_samples) > 0:
        s = sys_trace.memory_system_samples
        time_ms = (np.array([x.timestamp_ns for x in s]) - t0) / 1e6
        used    = np.array([x.used_bytes    for x in s]) / (1024 ** 3)
        buffers = np.array([x.buffers_bytes for x in s]) / (1024 ** 3)
        cached  = np.array([x.cached_bytes  for x in s]) / (1024 ** 3)
        out_data["sys_mem_total"] = {
            "t": time_ms,
            "used": used, "buffers": buffers, "cached": cached,
            "y_used_top":    used,
            "y_buffers_top": used + buffers,
            "y_cached_top":  used + buffers + cached,
        }
        meta["total_ram_gib"] = (s[0].total_bytes / (1024 ** 3)
                                  if s[0].total_bytes else 0.0)

    if len(sys_trace.memory_process_samples) > 0:
        proc = list(sys_trace.memory_process_samples)
        pids = sorted(set(x.pid for x in proc))
        # cpu and mem track the same set of PIDs, but be defensive.
        if not meta["pids"]:
            meta["pids"] = pids
        per_pid = {pid: [x for x in proc if x.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms = (np.array([x.timestamp_ns for x in base]) - t0) / 1e6
        d = {"t": time_ms}
        max_val = 0.0
        for pid in pids:
            samps = per_pid[pid]
            rss = np.array([x.rss_bytes for x in samps]) / (1024 ** 3)
            n = len(time_ms)
            if rss.size != n: rss = np.resize(rss, n)
            d[f"pid_{pid}_rss"] = rss
            max_val = max(max_val, float(rss.max() if rss.size else 0))
        meta["mem_proc_max_gib"] = max_val
        out_data["sys_mem_proc"] = d

    return {"data": out_data, "meta": meta}


def project_disk(disk_trace, t0: int) -> dict[str, Any]:
    out_data: dict[str, Any] = {"disk_dev_bw": None, "disk_proc_bw": None,
                                 "disk_dev_q": None}
    meta: dict[str, Any] = {"devs": [], "pids": [], "tracked_processes": [],
                            "dev_bw_max_mibps": 0.0,
                            "proc_bw_max_mibps": 0.0,
                            "q_max": 0.0}
    if not disk_trace or len(disk_trace.device_samples) == 0:
        return {"data": out_data, "meta": meta}

    meta["tracked_processes"] = list(disk_trace.tracked_processes)

    by_dev: dict[str, list] = {}
    for s in disk_trace.device_samples:
        by_dev.setdefault(s.device_name, []).append(s)
    devs = list(by_dev.keys())
    meta["devs"] = devs

    base_samps = by_dev[devs[0]]
    time_ms = (np.array([x.timestamp_ns for x in base_samps]) - t0) / 1e6

    bw_data: dict[str, Any] = {"t": time_ms}
    bw_max = 0.0
    for dev in devs:
        samps = by_dev[dev]
        rd = np.array([x.read_bytes_per_sec  for x in samps]) / (1024 ** 2)
        wr = np.array([x.write_bytes_per_sec for x in samps]) / (1024 ** 2)
        n = len(time_ms)
        if rd.size != n: rd = np.resize(rd, n)
        if wr.size != n: wr = np.resize(wr, n)
        bw_data[f"{dev}_rd"] = rd
        bw_data[f"{dev}_wr"] = wr
        bw_max = max(bw_max, float(rd.max() if rd.size else 0),
                              float(wr.max() if wr.size else 0))
    meta["dev_bw_max_mibps"] = bw_max
    out_data["disk_dev_bw"] = bw_data

    if len(disk_trace.process_samples) > 0:
        proc = list(disk_trace.process_samples)
        pids = sorted(set(x.pid for x in proc))
        meta["pids"] = pids
        per_pid = {pid: [x for x in proc if x.pid == pid] for pid in pids}
        base = per_pid[pids[0]]
        time_ms_p = (np.array([x.timestamp_ns for x in base]) - t0) / 1e6
        d: dict[str, Any] = {"t": time_ms_p}
        pmax = 0.0
        for pid in pids:
            samps = per_pid[pid]
            rd = np.array([x.read_bytes_per_sec  for x in samps]) / (1024 ** 2)
            wr = np.array([x.write_bytes_per_sec for x in samps]) / (1024 ** 2)
            n = len(time_ms_p)
            if rd.size != n: rd = np.resize(rd, n)
            if wr.size != n: wr = np.resize(wr, n)
            d[f"pid_{pid}_rd"] = rd
            d[f"pid_{pid}_wr"] = wr
            pmax = max(pmax, float(rd.max() if rd.size else 0),
                              float(wr.max() if wr.size else 0))
        meta["proc_bw_max_mibps"] = pmax
        out_data["disk_proc_bw"] = d

    q_data: dict[str, Any] = {"t": time_ms}
    q_max = 0.0
    for dev in devs:
        samps = by_dev[dev]
        rdq = np.array([x.read_queue_depth  for x in samps], dtype=np.float64)
        wrq = np.array([x.write_queue_depth for x in samps], dtype=np.float64)
        n = len(time_ms)
        if rdq.size != n: rdq = np.resize(rdq, n)
        if wrq.size != n: wrq = np.resize(wrq, n)
        q_data[f"{dev}_rdq"] = rdq
        q_data[f"{dev}_wrq"] = wrq
        q_max = max(q_max, float(rdq.max() if rdq.size else 0),
                            float(wrq.max() if wrq.size else 0))
    meta["q_max"] = q_max
    out_data["disk_dev_q"] = q_data

    return {"data": out_data, "meta": meta}


def project_events(events: dict | None, gpu_trace, t0: int,
                    region_colors: list[str], event_colors: list[str]) -> dict[str, Any]:
    """Project the merged events bundle into the annotation-strip CDS dicts.

    `region_colors` / `event_colors` are forwarded so the projection stays
    deterministic without depending on visualize_interactive's import path.
    """
    out_data = {"annot_regions": None, "annot_events": None}
    if not events:
        return {"data": out_data, "meta": {}}

    regions = []
    instant = []
    for r in events["generic_regions"]:
        regions.append((r.name, r.start_timestamp_ns, r.end_timestamp_ns))
    for e in events["generic_events"]:
        instant.append((e.name, e.timestamp_ns))
    meta_msg = events["metadata"]
    if meta_msg is not None:
        cref, sref = meta_msg.cupti_reference_ns, meta_msg.steady_clock_reference_ns
        for r in events["gpu_regions"]:
            regions.append((r.name,
                             int(gpu_to_steady(r.start_timestamp_ns, cref, sref)),
                             int(gpu_to_steady(r.end_timestamp_ns,   cref, sref))))
        for e in events["gpu_events"]:
            instant.append((e.name,
                             int(gpu_to_steady(e.timestamp_ns, cref, sref))))

    if regions:
        names, lefts, rights, dur_ms, colors = [], [], [], [], []
        for i, (name, s_ns, e_ns) in enumerate(regions):
            names.append(name)
            lefts.append((s_ns - t0) / 1e6)
            rights.append((e_ns - t0) / 1e6)
            dur_ms.append((e_ns - s_ns) / 1e6)
            colors.append(region_colors[i % len(region_colors)])
        out_data["annot_regions"] = dict(name=names, left=lefts, right=rights,
                                          dur_ms=dur_ms, color=colors,
                                          bottom=[-0.4] * len(names),
                                          top=[0.4] * len(names))

    if instant:
        en, ex, ec = [], [], []
        for i, (name, ts) in enumerate(instant):
            en.append(name)
            ex.append((ts - t0) / 1e6)
            ec.append(event_colors[i % len(event_colors)])
        out_data["annot_events"] = dict(name=en, x=ex, color=ec,
                                         y=[0.7] * len(en),
                                         y0=[-1.0] * len(en),
                                         y1=[1.0] * len(en))

    return {"data": out_data, "meta": {}}


# ---------------------------------------------------------------------------
# LiveState — orchestrates tails + accumulating traces + CDS registry
# ---------------------------------------------------------------------------

# Keys understood by visualize_interactive's builders. Live tick uses these
# to dispatch projected dicts onto the matching CDS.
PANEL_KEYS = (
    "annot_regions", "annot_events",
    "gpu_sm_util", "gpu_warps", "gpu_dram", "gpu_pcie",
    "gpu_pcie_cum", "gpu_nvlink",
    "sys_cpu_total", "sys_cpu_proc", "sys_mem_total", "sys_mem_proc",
    "disk_dev_bw", "disk_proc_bw", "disk_dev_q",
)


class LiveState:
    """Owns one TraceTail per active probe + the accumulating in-memory
    proto traces. `initial_load()` populates them from whatever bytes are
    already on disk; `tick()` reads new bytes, extends the in-memory
    traces, and refreshes every registered CDS via full-replace.

    Full-replace (rather than `cds.stream`) is intentional — it keeps
    cumulative columns (PCIe cumulative bytes), per-PID column groupings,
    and varea cumulative-top derivations all consistent on every tick
    without per-panel delta state.
    """

    def __init__(self, metadata_path: str, region_colors, event_colors):
        self.metadata_path = metadata_path
        self.region_colors = region_colors
        self.event_colors  = event_colors
        self.gpu_trace = None
        self.sys_trace = None
        self.disk_trace = None
        self.events = None     # merged events bundle (dict, not a proto)
        self.session_meta = None
        self.t0 = 0
        self.gpu_tail: TraceTail | None = None
        self.sys_tail: TraceTail | None = None
        self.disk_tail: TraceTail | None = None
        self.events_tail: TraceTail | None = None
        # CDS registry keyed by PANEL_KEYS entries; populated by the
        # builders at figure-construction time.
        self.cds: dict[str, Any] = {}
        # Most recent projection result, kept so tick() can also push
        # axis-bound updates if we ever wire those up.
        self.last_meta: dict[str, dict] = {}
        # ---- Numpy caches (populated incrementally) -----------------
        # Each new proto sample is extracted to numpy exactly once, when
        # it's first seen. Subsequent ticks read directly from these
        # arrays, so projection cost is O(numpy vectorize over total)
        # rather than O(walk-every-proto-message) per tick. Without this
        # 30 s of 10 kHz GPU at 1 Hz polling = ~5M Python attribute
        # accesses per tick — the dominant lag.
        self._gpu_ts_ns: np.ndarray = np.zeros(0, dtype=np.float64)
        self._gpu_vals: np.ndarray | None = None    # (N, n_metrics)
        self._gpu_metric_idx: dict[str, int] = {}
        self._sys_cpu_total_cache: dict[str, np.ndarray] = {}    # cpu_system_samples
        self._sys_cpu_proc_cache: dict[int, dict[str, np.ndarray]] = {}  # by pid
        self._sys_mem_total_cache: dict[str, np.ndarray] = {}
        self._sys_mem_proc_cache: dict[int, dict[str, np.ndarray]] = {}
        self._sys_total_bytes: int = 0
        self._disk_dev_cache: dict[str, dict[str, np.ndarray]] = {}    # by device
        self._disk_proc_cache: dict[int, dict[str, np.ndarray]] = {}    # by pid

    # -- one-shot setup ----------------------------------------------------

    def wait_for_metadata(self, timeout_s: float = 30.0) -> bool:
        """Block until session_metadata.pb appears (atomic rename means we
        only ever observe a fully-written file). Returns True if found."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(self.metadata_path):
                return True
            time.sleep(0.1)
        return os.path.exists(self.metadata_path)

    def initial_load(self):
        """Parse session_metadata, locate the per-probe files, install a
        TraceTail on each, and read whatever's already there.
        """
        import session_metadata_pb2

        self.session_meta = session_metadata_pb2.SessionMetadata()
        with open(self.metadata_path, "rb") as f:
            self.session_meta.ParseFromString(f.read())

        meta_dir = os.path.dirname(os.path.abspath(self.metadata_path))
        paths: dict[int, str] = {}
        for p in self.session_meta.probes:
            for cand in (p.output_file,
                         os.path.join(meta_dir, p.output_file),
                         os.path.join(meta_dir, os.path.basename(p.output_file))):
                if os.path.exists(cand):
                    paths[p.kind] = cand
                    break
                # else: probe file may not exist yet — store the most
                # plausible path so tick() can pick it up later.
            else:
                paths[p.kind] = os.path.join(meta_dir,
                                              os.path.basename(p.output_file))

        if session_metadata_pb2.PROBE_KIND_GPU in paths:
            import gpu_metrics_pb2
            self.gpu_tail = TraceTail(paths[session_metadata_pb2.PROBE_KIND_GPU],
                                       gpu_metrics_pb2.GpuMetricsTrace)
        if session_metadata_pb2.PROBE_KIND_SYSTEM in paths:
            import system_metrics_pb2
            self.sys_tail = TraceTail(paths[session_metadata_pb2.PROBE_KIND_SYSTEM],
                                       system_metrics_pb2.SystemMetricsTrace)
        if session_metadata_pb2.PROBE_KIND_DISK in paths:
            import disk_metrics_pb2
            self.disk_tail = TraceTail(paths[session_metadata_pb2.PROBE_KIND_DISK],
                                        disk_metrics_pb2.DiskMetricsTrace)
        if session_metadata_pb2.PROBE_KIND_EVENTS in paths:
            import events_pb2
            self.events_tail = TraceTail(paths[session_metadata_pb2.PROBE_KIND_EVENTS],
                                          events_pb2.EventTrace)

        # Drain whatever's already present before computing t0 and
        # showing the first frame.
        self._drain_into_traces()
        self.t0 = global_t0_from_traces(self.gpu_trace, self.sys_trace,
                                         self.disk_trace)

    # -- per-tick ----------------------------------------------------------

    def _drain_into_traces(self) -> bool:
        """Pull new messages from each tail and merge them into the
        accumulating in-memory traces. Returns True if anything changed.
        """
        changed = False

        if self.gpu_tail is not None:
            new = self.gpu_tail.read_new_messages()
            if new:
                changed = True
                if self.gpu_trace is None:
                    self.gpu_trace = new[0].__class__()
                    self.gpu_trace.CopyFrom(new[0])
                    self.gpu_trace.ClearField("samples")
                # Extract just the new samples to numpy (one-time cost
                # per sample), then append to gpu_trace.samples so any
                # legacy code path that still walks the proto works.
                self._extend_gpu_cache(new)
                for t in new:
                    self.gpu_trace.samples.extend(t.samples)
                # GPU regions used to live on GpuMetricsTrace.regions and
                # were resolved only at Stop(); they now live in events.pb
                # via EventBuffer (TIME_DOMAIN_GPU) and stream periodically
                # like everything else, so there's nothing GPU-specific to
                # do here.

        if self.sys_tail is not None:
            new = self.sys_tail.read_new_messages()
            if new:
                changed = True
                if self.sys_trace is None:
                    self.sys_trace = new[0].__class__()
                    self.sys_trace.CopyFrom(new[0])
                    self.sys_trace.ClearField("cpu_system_samples")
                    self.sys_trace.ClearField("cpu_process_samples")
                    self.sys_trace.ClearField("memory_system_samples")
                    self.sys_trace.ClearField("memory_process_samples")
                self._extend_sys_cache(new)
                for t in new:
                    self.sys_trace.cpu_system_samples.extend(t.cpu_system_samples)
                    self.sys_trace.cpu_process_samples.extend(t.cpu_process_samples)
                    self.sys_trace.memory_system_samples.extend(t.memory_system_samples)
                    self.sys_trace.memory_process_samples.extend(t.memory_process_samples)

        if self.disk_tail is not None:
            new = self.disk_tail.read_new_messages()
            if new:
                changed = True
                if self.disk_trace is None:
                    self.disk_trace = new[0].__class__()
                    self.disk_trace.CopyFrom(new[0])
                    self.disk_trace.ClearField("device_samples")
                    self.disk_trace.ClearField("process_samples")
                self._extend_disk_cache(new)
                for t in new:
                    self.disk_trace.device_samples.extend(t.device_samples)
                    self.disk_trace.process_samples.extend(t.process_samples)

        if self.events_tail is not None:
            new = self.events_tail.read_new_messages()
            if new:
                changed = True
                import events_pb2
                if self.events is None:
                    self.events = {"metadata": None,
                                    "generic_regions": [], "generic_events": [],
                                    "gpu_regions": [], "gpu_events": []}
                for t in new:
                    if t.HasField("metadata") and self.events["metadata"] is None:
                        self.events["metadata"] = t.metadata
                    for buf in t.buffers:
                        if buf.domain == events_pb2.TIME_DOMAIN_GENERIC:
                            self.events["generic_regions"].extend(buf.regions)
                            self.events["generic_events"].extend(buf.events)
                        elif buf.domain == events_pb2.TIME_DOMAIN_GPU:
                            self.events["gpu_regions"].extend(buf.regions)
                            self.events["gpu_events"].extend(buf.events)

        return changed

    # -- numpy-cache extension helpers ------------------------------------

    def _extend_gpu_cache(self, new_msgs):
        """Walk each new GPU message's samples once, extracting timestamps
        and the per-metric values matrix into numpy. Skips the first ever
        sample (CUPTI init artifact) the first time it sees one — same
        rule visualize_all applies via `samples[1:]`."""
        new_samples = []
        for t in new_msgs:
            new_samples.extend(t.samples)
        # Drop the init artifact once. _gpu_metric_idx being empty is the
        # signal that no batch has been ingested yet, so we're seeing the
        # very first sample of the run.
        if not self._gpu_metric_idx and new_samples:
            metric_names = list(self.gpu_trace.metric_names)
            n_metrics = len(metric_names)
            self._gpu_metric_idx = {n: i for i, n in enumerate(metric_names)}
            self._gpu_vals = np.zeros((0, n_metrics), dtype=np.float64)
            new_samples = new_samples[1:]
        if not new_samples:
            return
        n_new = len(new_samples)
        n_metrics = self._gpu_vals.shape[1]
        new_ts = np.fromiter((s.start_timestamp_ns for s in new_samples),
                              dtype=np.float64, count=n_new)
        # One Python loop over new_samples (small per tick), then bulk
        # numpy reshape — same content as the old per-tick full-trace loop.
        flat = np.empty(n_new * n_metrics, dtype=np.float64)
        for i, s in enumerate(new_samples):
            flat[i * n_metrics:(i + 1) * n_metrics] = s.values
        new_vals = flat.reshape(n_new, n_metrics)
        self._gpu_ts_ns = np.concatenate([self._gpu_ts_ns, new_ts])
        self._gpu_vals = np.vstack([self._gpu_vals, new_vals])

    def _extend_sys_cache(self, new_msgs):
        """Extract CPU + memory samples (system-wide and per-PID) from the
        new SystemMetricsTrace messages into numpy arrays."""
        for t in new_msgs:
            if t.cpu_system_samples:
                self._append_cols(self._sys_cpu_total_cache,
                    {"ts": [x.timestamp_ns for x in t.cpu_system_samples],
                     "user": [x.user_pct for x in t.cpu_system_samples],
                     "sys":  [x.system_pct for x in t.cpu_system_samples],
                     "iow":  [x.iowait_pct for x in t.cpu_system_samples],
                     "total":[x.total_utilization_pct for x in t.cpu_system_samples]})
            if t.cpu_process_samples:
                # Group by pid before append so each PID's cache is one column set.
                by_pid: dict[int, list] = {}
                for s in t.cpu_process_samples:
                    by_pid.setdefault(s.pid, []).append(s)
                for pid, samps in by_pid.items():
                    cache = self._sys_cpu_proc_cache.setdefault(pid, {})
                    self._append_cols(cache,
                        {"ts":   [x.timestamp_ns for x in samps],
                         "user": [x.user_pct    for x in samps],
                         "sys":  [x.system_pct  for x in samps]})
            if t.memory_system_samples:
                if not self._sys_total_bytes and t.memory_system_samples[0].total_bytes:
                    self._sys_total_bytes = t.memory_system_samples[0].total_bytes
                self._append_cols(self._sys_mem_total_cache,
                    {"ts":      [x.timestamp_ns   for x in t.memory_system_samples],
                     "used":    [x.used_bytes     for x in t.memory_system_samples],
                     "buffers": [x.buffers_bytes  for x in t.memory_system_samples],
                     "cached":  [x.cached_bytes   for x in t.memory_system_samples]})
            if t.memory_process_samples:
                by_pid_mem: dict[int, list] = {}
                for s in t.memory_process_samples:
                    by_pid_mem.setdefault(s.pid, []).append(s)
                for pid, samps in by_pid_mem.items():
                    cache = self._sys_mem_proc_cache.setdefault(pid, {})
                    self._append_cols(cache,
                        {"ts":  [x.timestamp_ns for x in samps],
                         "rss": [x.rss_bytes    for x in samps]})

    def _extend_disk_cache(self, new_msgs):
        """Extract per-device + per-PID disk samples into numpy arrays."""
        for t in new_msgs:
            if t.device_samples:
                by_dev: dict[str, list] = {}
                for s in t.device_samples:
                    by_dev.setdefault(s.device_name, []).append(s)
                for dev, samps in by_dev.items():
                    cache = self._disk_dev_cache.setdefault(dev, {})
                    self._append_cols(cache,
                        {"ts":  [x.timestamp_ns        for x in samps],
                         "rd":  [x.read_bytes_per_sec  for x in samps],
                         "wr":  [x.write_bytes_per_sec for x in samps],
                         "rdq": [x.read_queue_depth    for x in samps],
                         "wrq": [x.write_queue_depth   for x in samps]})
            if t.process_samples:
                by_pid: dict[int, list] = {}
                for s in t.process_samples:
                    by_pid.setdefault(s.pid, []).append(s)
                for pid, samps in by_pid.items():
                    cache = self._disk_proc_cache.setdefault(pid, {})
                    self._append_cols(cache,
                        {"ts": [x.timestamp_ns        for x in samps],
                         "rd": [x.read_bytes_per_sec  for x in samps],
                         "wr": [x.write_bytes_per_sec for x in samps]})

    @staticmethod
    def _append_cols(cache: dict[str, np.ndarray], new: dict[str, list]) -> None:
        for k, vals in new.items():
            arr = np.asarray(vals, dtype=np.float64)
            if k in cache:
                cache[k] = np.concatenate([cache[k], arr])
            else:
                cache[k] = arr

    # -- projection from the numpy caches ---------------------------------

    def project_all(self) -> dict[str, Any]:
        """Re-project every panel from the live numpy caches.

        Reads only from cached arrays (no per-tick proto walks), so the
        per-tick cost is O(numpy ops over total samples) rather than
        O(walk every proto sample). On long live runs this is ~50-100x
        faster than the proto-walking path used by the static loader.
        """
        return {"data": {**self._project_gpu_cached(),
                          **self._project_sys_cached(),
                          **self._project_disk_cached(),
                          **self._project_events_cached()},
                 "meta": {"gpu":  self._gpu_meta(),
                          "sys":  self._sys_meta(),
                          "disk": self._disk_meta()}}

    # ---- GPU --------------------------------------------------------------

    def _gpu_meta(self) -> dict:
        meta: dict = {"device_name": "", "chip_name": "",
                       "peak_dram_gibps": None,
                       "peak_pcie_bidi_gibps": None,
                       "peak_nvlink_bidi_gibps": None,
                       "pcie_cum_unit": "B", "pcie_cum_div": 1}
        if self.gpu_trace is None:
            return meta
        meta["device_name"] = self.gpu_trace.device_name
        meta["chip_name"]   = self.gpu_trace.chip_name
        if self.gpu_trace.peak_dram_bw_gbps > 0:
            meta["peak_dram_gibps"] = self.gpu_trace.peak_dram_bw_gbps * 1e9 / (1024 ** 3)
        if self.gpu_trace.peak_pcie_bw_bytes_per_sec > 0:
            meta["peak_pcie_bidi_gibps"] = (
                self.gpu_trace.peak_pcie_bw_bytes_per_sec * 2 / (1024 ** 3))
        if self.gpu_trace.peak_nvlink_bw_bytes_per_sec > 0:
            meta["peak_nvlink_bidi_gibps"] = (
                self.gpu_trace.peak_nvlink_bw_bytes_per_sec * 2 / (1024 ** 3))
        # Pick the unit for the cumulative-PCIe panel from the current max.
        if self._gpu_vals is not None and self._gpu_vals.shape[0] > 0 \
                and "pcie__read_bytes.sum" in self._gpu_metric_idx:
            rd_sum = float(self._gpu_vals[:, self._gpu_metric_idx["pcie__read_bytes.sum"]].sum())
            wr_sum = (float(self._gpu_vals[:, self._gpu_metric_idx["pcie__write_bytes.sum"]].sum())
                       if "pcie__write_bytes.sum" in self._gpu_metric_idx else 0.0)
            max_bytes = max(rd_sum, wr_sum)
            for div, unit in [(1024 ** 4, "TiB"), (1024 ** 3, "GiB"),
                              (1024 ** 2, "MiB"), (1024, "KiB"), (1, "B")]:
                if max_bytes >= div:
                    meta["pcie_cum_unit"] = unit
                    meta["pcie_cum_div"] = div
                    break
        return meta

    def _project_gpu_cached(self) -> dict:
        keys = ("gpu_sm_util", "gpu_warps", "gpu_dram",
                 "gpu_pcie", "gpu_pcie_cum", "gpu_nvlink")
        out: dict = {k: None for k in keys}
        if self._gpu_vals is None or self._gpu_vals.shape[0] == 0:
            return out
        idx = self._gpu_metric_idx
        vals = self._gpu_vals
        cref = self.gpu_trace.cupti_reference_ns
        sref = self.gpu_trace.steady_clock_reference_ns
        time_ms = (gpu_to_steady(self._gpu_ts_ns, cref, sref) - self.t0) / 1e6

        if "sm__cycles_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
            elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
            d = {"t": time_ms,
                 "avg": np.where(elapsed > 0,
                                  vals[:, idx["sm__cycles_active.avg"]] / elapsed * 100, 0)}
            if "sm__cycles_active.max" in idx:
                d["max"] = np.where(elapsed > 0,
                                     vals[:, idx["sm__cycles_active.max"]] / elapsed * 100, 0)
            out["gpu_sm_util"] = d

        if "sm__warps_active.avg" in idx and "sm__cycles_elapsed.avg" in idx:
            elapsed = vals[:, idx["sm__cycles_elapsed.avg"]]
            d = {"t": time_ms,
                 "avg": np.where(elapsed > 0,
                                  vals[:, idx["sm__warps_active.avg"]] / elapsed, 0)}
            if "sm__warps_active.max" in idx:
                d["max"] = np.where(elapsed > 0,
                                     vals[:, idx["sm__warps_active.max"]] / elapsed, 0)
            out["gpu_warps"] = d

        drum = "dram__read_throughput.avg.pct_of_peak_sustained_elapsed"
        if drum in idx:
            peak = self._gpu_meta().get("peak_dram_gibps")
            scale = (peak / 100.0) if peak else 1.0
            d = {"t": time_ms, "rd": vals[:, idx[drum]] * scale}
            wr_key = "dram__write_throughput.avg.pct_of_peak_sustained_elapsed"
            if wr_key in idx:
                d["wr"] = vals[:, idx[wr_key]] * scale
            out["gpu_dram"] = d

        if "pcie__read_bytes.sum.per_second" in idx:
            d = {"t": time_ms,
                 "rd": vals[:, idx["pcie__read_bytes.sum.per_second"]] / (1024 ** 3)}
            if "pcie__write_bytes.sum.per_second" in idx:
                d["wr"] = vals[:, idx["pcie__write_bytes.sum.per_second"]] / (1024 ** 3)
            out["gpu_pcie"] = d

        if "pcie__read_bytes.sum" in idx and "pcie__write_bytes.sum" in idx:
            div = self._gpu_meta()["pcie_cum_div"]
            rd_cum = np.cumsum(vals[:, idx["pcie__read_bytes.sum"]])
            wr_cum = np.cumsum(vals[:, idx["pcie__write_bytes.sum"]])
            out["gpu_pcie_cum"] = {"t": time_ms,
                                     "rd": rd_cum / div, "wr": wr_cum / div}

        if "nvlrx__bytes.sum.per_second" in idx and "nvltx__bytes.sum.per_second" in idx:
            out["gpu_nvlink"] = {"t":  time_ms,
                                  "rx": vals[:, idx["nvlrx__bytes.sum.per_second"]] / (1024 ** 3),
                                  "tx": vals[:, idx["nvltx__bytes.sum.per_second"]] / (1024 ** 3)}
        return out

    # ---- System -----------------------------------------------------------

    def _sys_meta(self) -> dict:
        pids = sorted(set(list(self._sys_cpu_proc_cache.keys())
                           + list(self._sys_mem_proc_cache.keys())))
        cpu_max = 0.0
        for c in self._sys_cpu_proc_cache.values():
            if "user" in c and "sys" in c and c["user"].size:
                cpu_max = max(cpu_max, float((c["user"] + c["sys"]).max()))
        mem_max_gib = 0.0
        for c in self._sys_mem_proc_cache.values():
            if "rss" in c and c["rss"].size:
                mem_max_gib = max(mem_max_gib, float(c["rss"].max() / (1024 ** 3)))
        return {"pids": pids,
                 "tracked_processes": (list(self.sys_trace.tracked_processes)
                                        if self.sys_trace is not None else []),
                 "total_ram_gib": (self._sys_total_bytes / (1024 ** 3)
                                    if self._sys_total_bytes else 0.0),
                 "cpu_proc_max": cpu_max,
                 "mem_proc_max_gib": mem_max_gib}

    def _project_sys_cached(self) -> dict:
        out: dict = {"sys_cpu_total": None, "sys_cpu_proc": None,
                      "sys_mem_total": None, "sys_mem_proc": None}
        c = self._sys_cpu_total_cache
        if c.get("ts") is not None and c["ts"].size:
            time_ms = (c["ts"] - self.t0) / 1e6
            user, sysp, iow = c["user"], c["sys"], c["iow"]
            out["sys_cpu_total"] = {
                "t": time_ms,
                "user": user, "sys": sysp, "iow": iow, "total": c["total"],
                "y_user_top": user,
                "y_sys_top":  user + sysp,
                "y_iow_top":  user + sysp + iow,
            }

        if self._sys_cpu_proc_cache:
            pids = sorted(self._sys_cpu_proc_cache.keys())
            base = self._sys_cpu_proc_cache[pids[0]]
            time_ms = (base["ts"] - self.t0) / 1e6
            d: dict = {"t": time_ms}
            n = len(time_ms)
            for pid in pids:
                cache = self._sys_cpu_proc_cache[pid]
                user = cache.get("user", np.zeros(n))
                sysp = cache.get("sys",  np.zeros(n))
                if user.size != n: user = np.resize(user, n)
                if sysp.size != n: sysp = np.resize(sysp, n)
                d[f"pid_{pid}_user"] = user
                d[f"pid_{pid}_sum"]  = user + sysp
            out["sys_cpu_proc"] = d

        c = self._sys_mem_total_cache
        if c.get("ts") is not None and c["ts"].size:
            time_ms = (c["ts"] - self.t0) / 1e6
            used    = c["used"]    / (1024 ** 3)
            buffers = c["buffers"] / (1024 ** 3)
            cached  = c["cached"]  / (1024 ** 3)
            out["sys_mem_total"] = {
                "t": time_ms,
                "used": used, "buffers": buffers, "cached": cached,
                "y_used_top":    used,
                "y_buffers_top": used + buffers,
                "y_cached_top":  used + buffers + cached,
            }

        if self._sys_mem_proc_cache:
            pids = sorted(self._sys_mem_proc_cache.keys())
            base = self._sys_mem_proc_cache[pids[0]]
            time_ms = (base["ts"] - self.t0) / 1e6
            d = {"t": time_ms}
            n = len(time_ms)
            for pid in pids:
                rss = self._sys_mem_proc_cache[pid].get("rss", np.zeros(n)) / (1024 ** 3)
                if rss.size != n: rss = np.resize(rss, n)
                d[f"pid_{pid}_rss"] = rss
            out["sys_mem_proc"] = d
        return out

    # ---- Disk -------------------------------------------------------------

    def _disk_meta(self) -> dict:
        devs = list(self._disk_dev_cache.keys())
        pids = sorted(self._disk_proc_cache.keys())
        bw_max = 0.0
        for c in self._disk_dev_cache.values():
            if "rd" in c and c["rd"].size:
                bw_max = max(bw_max, float(c["rd"].max() / (1024 ** 2)),
                                       float(c["wr"].max() / (1024 ** 2)))
        proc_bw_max = 0.0
        for c in self._disk_proc_cache.values():
            if "rd" in c and c["rd"].size:
                proc_bw_max = max(proc_bw_max,
                                    float(c["rd"].max() / (1024 ** 2)),
                                    float(c["wr"].max() / (1024 ** 2)))
        q_max = 0.0
        for c in self._disk_dev_cache.values():
            if "rdq" in c and c["rdq"].size:
                q_max = max(q_max, float(c["rdq"].max()),
                                    float(c["wrq"].max()))
        return {"devs": devs, "pids": pids,
                 "tracked_processes": (list(self.disk_trace.tracked_processes)
                                        if self.disk_trace is not None else []),
                 "dev_bw_max_mibps": bw_max,
                 "proc_bw_max_mibps": proc_bw_max,
                 "q_max": q_max}

    def _project_disk_cached(self) -> dict:
        out: dict = {"disk_dev_bw": None, "disk_proc_bw": None,
                      "disk_dev_q": None}
        if not self._disk_dev_cache:
            return out
        devs = list(self._disk_dev_cache.keys())
        base = self._disk_dev_cache[devs[0]]
        if base.get("ts") is None or not base["ts"].size:
            return out
        time_ms = (base["ts"] - self.t0) / 1e6
        n = len(time_ms)

        bw: dict = {"t": time_ms}
        for dev in devs:
            c = self._disk_dev_cache[dev]
            rd = c.get("rd", np.zeros(n)) / (1024 ** 2)
            wr = c.get("wr", np.zeros(n)) / (1024 ** 2)
            if rd.size != n: rd = np.resize(rd, n)
            if wr.size != n: wr = np.resize(wr, n)
            bw[f"{dev}_rd"] = rd
            bw[f"{dev}_wr"] = wr
        out["disk_dev_bw"] = bw

        if self._disk_proc_cache:
            pids = sorted(self._disk_proc_cache.keys())
            base_p = self._disk_proc_cache[pids[0]]
            time_ms_p = (base_p["ts"] - self.t0) / 1e6
            np_ = len(time_ms_p)
            d: dict = {"t": time_ms_p}
            for pid in pids:
                c = self._disk_proc_cache[pid]
                rd = c.get("rd", np.zeros(np_)) / (1024 ** 2)
                wr = c.get("wr", np.zeros(np_)) / (1024 ** 2)
                if rd.size != np_: rd = np.resize(rd, np_)
                if wr.size != np_: wr = np.resize(wr, np_)
                d[f"pid_{pid}_rd"] = rd
                d[f"pid_{pid}_wr"] = wr
            out["disk_proc_bw"] = d

        q: dict = {"t": time_ms}
        for dev in devs:
            c = self._disk_dev_cache[dev]
            rdq = c.get("rdq", np.zeros(n))
            wrq = c.get("wrq", np.zeros(n))
            if rdq.size != n: rdq = np.resize(rdq, n)
            if wrq.size != n: wrq = np.resize(wrq, n)
            q[f"{dev}_rdq"] = rdq
            q[f"{dev}_wrq"] = wrq
        out["disk_dev_q"] = q
        return out

    # ---- Events -----------------------------------------------------------

    def _project_events_cached(self) -> dict:
        # Events are small (one entry per region/event), so the existing
        # per-tick walk is cheap. Fall back to the standalone projector.
        return project_events(self.events, self.gpu_trace, self.t0,
                                self.region_colors, self.event_colors)["data"]

    def tick(self):
        """Periodic-callback entry point. Idempotent if no new bytes."""
        if not self._drain_into_traces():
            return
        frame = self.project_all()
        for key, cols in frame["data"].items():
            if cols is None:
                continue
            cds = self.cds.get(key)
            if cds is None:
                continue
            cds.data = {k: list(v) if isinstance(v, np.ndarray) else v
                         for k, v in cols.items()}
