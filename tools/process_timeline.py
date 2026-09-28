"""The process timeline, shared by both visualizers: one bar per tracked
process from its start to its end, packed into as few lanes as
possible, with a fork link from each parent to each child.

Input is the trace's process table (TrackedProcessV2, collected by
TraceProjector.process_table): pid, parent_pid (the parent at
registration, never updated on reparenting), kind (listed root or
discovered), comm and its history, start (/proc stat, 10 ms ticks) and
end (first tick the probe saw it gone). Processes only: threads are not
traced.

Lane packing is greedy interval partitioning: processes in start order,
each into a lane that is free at its start, a new lane only when none
is. That uses exactly as many lanes as the most processes alive at one
instant (when a new lane opens, every existing lane is busy at that
start), whichever free lane is picked; the pick prefers the parent's
lane, then the free lane nearest to it, so fork links stay short.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ROOT, DISCOVERED, ORPHAN = "root", "discovered", "orphan"


@dataclass
class TimelineProcess:
    key: tuple                 # (pid, start_time_ns as recorded), unique across PID reuse
    pid: int
    ppid: int
    comm: str                  # its latest name
    comms: list = field(default_factory=list)   # [(ns, comm)], oldest first
    kind: str = DISCOVERED     # ROOT (listed), DISCOVERED, ORPHAN (discovered, parent not tracked)
    start_ns: int = 0          # clipped to the trace start
    end_ns: int = 0            # the trace end while alive
    alive: bool = False        # still running when the trace ended
    started_before: bool = False  # started before the trace
    parent: tuple | None = None   # key of the tracked parent


@dataclass
class ForkLink:
    parent: tuple
    child: tuple
    t_ns: int                  # the child's start
    parent_lane: int
    child_lane: int


def build(process_table: dict, t0_ns: int, t_end_ns: int,
          first_sample_ns: dict | None = None,
          last_sample_ns: dict | None = None) -> list[TimelineProcess]:
    """TimelineProcesses from TraceProjector.process_table, ordered by
    start. A process with no recorded start begins at its first sample
    (else the trace start); one removed without an end time (removed by
    request) ends at its last sample; one still alive ends at t_end_ns."""
    first_sample_ns = first_sample_ns or {}
    last_sample_ns = last_sample_ns or {}
    out = []
    for key, r in process_table.items():
        start = r.start_time_ns or first_sample_ns.get(r.pid, t0_ns)
        if r.end_time_ns:
            end, alive = r.end_time_ns, False
        elif r.removed:
            end, alive = last_sample_ns.get(r.pid, t_end_ns), False
        else:
            end, alive = t_end_ns, True
        before = start < t0_ns
        start = max(start, t0_ns)
        end = max(min(end, t_end_ns), start)
        comms = list(r.comm_history) or [(start, r.comm)]
        out.append(TimelineProcess(key=key, pid=r.pid, ppid=r.parent_pid,
                                   comm=r.comm or comms[-1][1], comms=comms,
                                   kind=DISCOVERED if r.discovered else ROOT,
                                   start_ns=start, end_ns=end, alive=alive,
                                   started_before=before))
    out.sort(key=lambda p: (p.start_ns, p.pid))
    by_pid: dict[int, list[TimelineProcess]] = {}
    for p in out:
        by_pid.setdefault(p.pid, []).append(p)
    for p in out:
        p.parent = _parent_of(p, by_pid.get(p.ppid, []))
        if p.parent is None and p.kind == DISCOVERED:
            p.kind = ORPHAN
    return out


def _parent_of(child: TimelineProcess, candidates: list[TimelineProcess]):
    """The tracked process that forked `child`: of the records of its
    parent PID (a PID number can be reused, one record per process, in
    start order and never alive at once), the latest one started by the
    child's start."""
    before = [c for c in candidates if c.key != child.key and c.start_ns <= child.start_ns]
    return before[-1].key if before else None


def pack_lanes(procs: list[TimelineProcess]) -> tuple[dict, int]:
    """Lane of each process (key -> lane, lane 0 on top) and the lane
    count, which equals the most processes alive at one instant. Two
    bars share a lane only if one ends no later than the other starts."""
    lane_end: list[int] = []
    lanes: dict = {}
    for p in sorted(procs, key=lambda p: (p.start_ns, p.pid)):
        free = [i for i, e in enumerate(lane_end) if e <= p.start_ns]
        if not free:
            lane = len(lane_end)
            lane_end.append(p.end_ns)
        else:
            ref = lanes.get(p.parent) if p.parent is not None else None
            lane = min(free, key=lambda i: (abs(i - ref), i)) if ref is not None else free[0]
            lane_end[lane] = p.end_ns
        lanes[p.key] = lane
    return lanes, len(lane_end)


def fork_links(procs: list[TimelineProcess], lanes: dict) -> list[ForkLink]:
    """One link per process whose parent is tracked: from the parent's
    bar at the child's start to the child's bar."""
    return [ForkLink(parent=p.parent, child=p.key, t_ns=p.start_ns,
                     parent_lane=lanes[p.parent], child_lane=lanes[p.key])
            for p in procs if p.parent is not None and p.parent in lanes]


def name_history(p: TimelineProcess) -> str:
    """'python3.12 -> VLLM::EngineCor' (every name it had, oldest first)."""
    names = [c for i, (_t, c) in enumerate(p.comms) if i == 0 or c != p.comms[i - 1][1]]
    return " -> ".join(names) if names else p.comm
