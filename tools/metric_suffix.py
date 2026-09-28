"""CUPTI PerfWorks FQN suffix table — informational.

The FQN format is `<entity>__<counter>[.<rollup>][.<submetric>]`. CUPTI's
host library (`cuptiProfilerHostEvaluateToGpuValues`) parses the suffix
and applies the transform during metric evaluation; our code only
chooses which FQN to request. This module documents the suffix
vocabulary and provides two helpers used by the Python tooling:

  * `unit_for(counter, submetric)`        -> MetricCatalog Unit enum
  * `label_for(entity, counter, rollup, submetric)` -> panel title

See `docs/cupti-fqn-suffixes.md` for the full description.
"""

from __future__ import annotations

from dataclasses import dataclass

import metric_catalog_pb2 as _mc


# ---------------------------------------------------------------------------
# Rollup names (the segment between `__counter` and `.submetric`)
# ---------------------------------------------------------------------------

ROLLUP_LABELS: dict[str, str] = {
    "sum": "sum",
    "avg": "avg",
    "min": "min",
    "max": "max",
}


# ---------------------------------------------------------------------------
# Entity pretty-names. CUPTI's entity prefixes are terse acronyms; this
# table maps them to display-friendly strings for auto-derived titles.
# Anything not listed falls back to ENTITY.upper().
# ---------------------------------------------------------------------------

ENTITY_PRETTY: dict[str, str] = {
    # GPU entities (CUPTI PerfWorks)
    "sm":     "SM",
    "smsp":   "SM Subpartition",
    "gpc":    "GPC",
    "tpc":    "TPC",
    "dram":   "DRAM",
    "l1tex":  "L1/Tex",
    "lts":    "L2",
    "fbpa":   "FBPA",
    "pcie":   "PCIe",
    "nvltx":  "NVLink TX",
    "nvlrx":  "NVLink RX",
    "gr":     "GR",
    "sys":    "System",
    # Non-GPU entities (this repo's catalog)
    "cpu":    "CPU",
    "mem":    "Memory",
    "proc":   "Process",
    "disk":   "Disk",
}


def pretty_entity(entity: str) -> str:
    return ENTITY_PRETTY.get(entity.lower(), entity.upper())


# Counters whose name means something else for a given entity: a
# process's `cycles_active` is its on-CPU time (% of one core), not GPU
# active cycles. Keyed by (entity, counter).
ENTITY_COUNTER_PRETTY: dict[tuple[str, str], str] = {
    ("proc", "cycles_active"): "CPU",
}


# Words written as acronyms wherever they appear in a counter name.
ACRONYMS: dict[str, str] = {
    "io": "IO", "rss": "RSS", "vms": "VMS", "cpu": "CPU", "gpu": "GPU", "sm": "SM",
    "dram": "DRAM", "pcie": "PCIe", "nvlink": "NVLink", "l1": "L1", "l2": "L2",
    "tlb": "TLB", "numa": "NUMA",
}

# Counters that are a kernel-exported record's fields, named after it:
# `<prefix>_<field>`; the field keeps the kernel's own spelling
# (`io_read_bytes` -> "IO read_bytes": /proc/<pid>/io's read_bytes).
KERNEL_FIELD_PREFIXES: tuple[str, ...] = ("io",)


def pretty_counter(counter: str, entity: str = "") -> str:
    """`warps_active` -> `Active Warps`, `cycles_active` -> `Active Cycles`
    (but `CPU` for entity `proc`), `read_throughput` -> `Read Throughput`.
    Acronyms fully capitalised (ACRONYMS); a kernel field keeps its own
    name after its record's prefix (KERNEL_FIELD_PREFIXES: `io_rchar` ->
    `IO rchar`); an acronym followed only by `bytes` is the acronym
    (`rss_bytes` -> `RSS`: the unit is on the axis)."""
    if not counter:
        return ""
    special = ENTITY_COUNTER_PRETTY.get((entity.lower(), counter))
    if special:
        return special
    head, _, rest = counter.partition("_")
    if head in KERNEL_FIELD_PREFIXES and rest:
        return f"{ACRONYMS.get(head, head)} {rest}"
    if head in ACRONYMS and rest == "bytes":
        return ACRONYMS[head]
    # Word reorderings for the most common counters where the natural
    # English order differs from the underscore order.
    REORDER = {
        "warps_active":    "Active Warps",
        "warps_eligible":  "Eligible Warps",
        "cycles_active":   "Active Cycles",
        "cycles_elapsed":  "Elapsed Cycles",
    }
    if counter in REORDER:
        return REORDER[counter]
    return " ".join(ACRONYMS.get(word, word.capitalize()) for word in counter.split("_"))


# ---------------------------------------------------------------------------
# Submetric table
# ---------------------------------------------------------------------------

_BYTE_COUNTER_FRAGMENTS = ("bytes", "throughput")


@dataclass(frozen=True)
class SuffixSpec:
    """One row of SUBMETRIC_TABLE. `unit` is the resolved MetricCatalog
    Unit; `label_fragment` is appended after the counter name when
    auto-generating a panel title."""
    unit:                 int
    label_fragment:       str
    expects_byte_counter: bool = False


SUBMETRIC_TABLE: dict[str, SuffixSpec] = {
    # Peak-normalization suffixes — value is 0..100 percent of the
    # chip's sustained or burst peak. CUPTI divides the rollup by the
    # peak and by either elapsed or active cycles.
    "pct_of_peak_sustained_elapsed":
        SuffixSpec(_mc.UNIT_PCT, "(% of sustained peak)"),
    "pct_of_peak_sustained_active":
        SuffixSpec(_mc.UNIT_PCT, "(% of sustained peak, active)"),
    "pct_of_peak_sustained_region":
        SuffixSpec(_mc.UNIT_PCT, "(% of sustained peak, region)"),
    "pct_of_peak_sustained_frame":
        SuffixSpec(_mc.UNIT_PCT, "(% of sustained peak, frame)"),
    "pct_of_peak_burst_elapsed":
        SuffixSpec(_mc.UNIT_PCT, "(% of burst peak)"),
    "pct_of_peak_burst_active":
        SuffixSpec(_mc.UNIT_PCT, "(% of burst peak, active)"),
    "pct_of_peak_burst_region":
        SuffixSpec(_mc.UNIT_PCT, "(% of burst peak, region)"),
    "pct_of_peak_burst_frame":
        SuffixSpec(_mc.UNIT_PCT, "(% of burst peak, frame)"),

    # Rate suffixes. `.per_second` resolves to BYTES_PER_SEC for
    # byte-y counters (anything with "bytes" or "throughput" in the
    # counter name) and to HZ otherwise.
    "per_second":
        SuffixSpec(_mc.UNIT_HZ, "/ s", expects_byte_counter=True),
    "per_cycle_active":
        SuffixSpec(_mc.UNIT_COUNT, "/ active cycle"),
    "per_cycle_elapsed":
        SuffixSpec(_mc.UNIT_COUNT, "/ elapsed cycle"),
    "per_cycle_region":
        SuffixSpec(_mc.UNIT_COUNT, "/ region cycle"),
    "per_cycle_frame":
        SuffixSpec(_mc.UNIT_COUNT, "/ frame cycle"),

    # Already-normalized.
    "pct":   SuffixSpec(_mc.UNIT_PCT,   "%"),
    "ratio": SuffixSpec(_mc.UNIT_RATIO, "ratio"),

    # Peak inquiry — returns the hardware peak itself.
    "peak_sustained":          SuffixSpec(_mc.UNIT_COUNT, "peak sustained"),
    "peak_sustained_active":   SuffixSpec(_mc.UNIT_COUNT, "peak / active cycle"),
    "peak_sustained_elapsed":  SuffixSpec(_mc.UNIT_COUNT, "peak / elapsed cycle"),
    "peak_burst":              SuffixSpec(_mc.UNIT_COUNT, "peak burst"),
    "peak_burst_active":       SuffixSpec(_mc.UNIT_COUNT, "peak burst / active cycle"),
    "peak_burst_elapsed":      SuffixSpec(_mc.UNIT_COUNT, "peak burst / elapsed cycle"),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def unit_for(counter: str, submetric: str) -> int:
    """Resolve the MetricCatalog Unit for an FQN's submetric. Used by
    `metric_layout.synthesize_descriptor` when no catalog entry
    declares the unit explicitly."""
    s = (submetric or "").lower()
    spec = SUBMETRIC_TABLE.get(s)
    if spec is None:
        return _mc.UNIT_COUNT
    if spec.expects_byte_counter and s == "per_second":
        is_bytey = any(frag in counter.lower() for frag in _BYTE_COUNTER_FRAGMENTS)
        return _mc.UNIT_BYTES_PER_SEC if is_bytey else _mc.UNIT_HZ
    return spec.unit


def label_for(entity: str, counter: str, rollup: str, submetric: str) -> str:
    """Compose a human-readable panel title from a parsed FQN.

    Examples:
      sm__warps_active.avg.per_cycle_active
        -> "SM Active Warps / active cycle (avg)"
      dram__read_throughput.avg.pct_of_peak_sustained_elapsed
        -> "DRAM Read Throughput (% of sustained peak) (avg)"
      pcie__read_bytes.sum.per_second
        -> "PCIe Read Bytes / s (sum)"
    """
    entity_p  = pretty_entity(entity)
    counter_p = pretty_counter(counter)

    sub_spec  = SUBMETRIC_TABLE.get((submetric or "").lower())
    sub_label = sub_spec.label_fragment if sub_spec else (submetric or "")

    parts: list[str] = [entity_p, counter_p]
    if sub_label:
        # The "/" prefix style (per-cycle, per-second) reads better as
        # a tail; the parenthesized ones too. Keep both inline after
        # the counter.
        parts.append(sub_label)

    title = " ".join(p for p in parts if p)

    if rollup and rollup.lower() in ROLLUP_LABELS:
        # Only add the rollup hint when it isn't `sum` of an obviously
        # aggregate metric — but cheaper to always show it for
        # transparency.
        title += f" ({ROLLUP_LABELS[rollup.lower()]})"

    return title
