// Internal: typed metric descriptor that backs the per-TU descriptor
// arrays in each probe (system_flush_thread.cpp, disk_flush_thread.cpp).
// The same array drives BOTH the emit path (which double goes into
// which ProcessSample.values[i] slot) AND the catalog registry (FQN +
// unit + peak + description seeded into the in-memory MetricCatalog at
// profiler startup), so the two cannot drift out of sync — add a
// descriptor without a matching `.read` extractor and you've removed a
// column from the emitted trace too.
//
// Kept under lib/src/ rather than lib/include/cupti_profiler/ because
// the example .cu files stay on C++17 (CMake 3.22 + NVCC 12.8 don't
// have cuda_std_20 wired up) and we don't want any C++20 header
// transitively included from the public surface.
#pragma once

#include <string_view>
#include <variant>

namespace cupti_profiler {
namespace internal {

// Mirrors of the proto enums (proto/metric_catalog.proto :: MetricType,
// Unit, Scope). Kept as thin C++ enum classes so descriptor literals
// don't need to include the generated proto header; the translator in
// metric_catalog_builtins.cpp does the one-time cast over to the proto
// enums during catalog registration.
//
// The numeric values match the proto on purpose — the translator can
// `static_cast<::MetricType>(d.type)` rather than running a switch.
// RATIO is omitted because no non-GPU probe emits ratios; GPU
// descriptors come from CUPTI's own enumeration path, not this header.
enum class MetricType : int { Counter = 1, Throughput = 3 };
enum class Unit : int {
    Count       = 1,
    Bytes       = 2,
    BytesPerSec = 3,
    Pct         = 4,    // 0-100
    PctOfCore   = 5,    // 0-100*ncpus (per-PID CPU)
    Ratio       = 6,    // 0-1
    Hz          = 7,
    Cycles      = 8,
    Requests    = 9,    // in-flight I/O depth
};
enum class Scope : int { System = 1, Device = 2, Process = 3 };

// Tag types for the Peak variant — captures the proto-level oneof
// {peak_constant | peak_ref | peak_expr} as a sum type.
//   PeakConstant — fixed ceiling baked into the catalog (100 for %).
//   PeakRef      — peak is whatever a different FQN reports at render
//                  time (e.g. mem__rss_bytes peaks at mem__capacity_bytes).
//   PeakExpr     — named expression resolved by metric_catalog.resolve_peak
//                  against the trace's host info (ncpus_x_100,
//                  max_warps_per_sm, peak_pcie_bw_bytes_per_s, …).
//   monostate    — no peak; the panel auto-scales.
struct PeakConstant { double           value; };
struct PeakRef      { std::string_view fqn;   };
struct PeakExpr     { std::string_view name;  };
using  Peak = std::variant<std::monostate, PeakConstant, PeakRef, PeakExpr>;

// One descriptor per metric emitted by a probe. Parameterised on the
// probe's per-tick struct (SystemTick, ProcessTick, DiskDeviceTick,
// DiskProcessTick) so the `.read` extractor is type-checked against
// the actual tick layout — renaming a tick field breaks the build
// right at the descriptor literal that references it.
//
// Fields with `= ...` defaults may be omitted in a designated-init
// literal; fields without a default are mandatory. `read` deliberately
// has no default — forgetting it would silently null-init the function
// pointer and crash at emit time, so we leave the absence as a runtime
// assert in the registry instead (each probe's RegisterBuiltinMetrics
// walks its array on startup).
template <typename Tick>
struct MetricDescriptor {
    std::string_view fqn;
    MetricType       type;
    std::string_view entity;
    std::string_view counter;
    std::string_view rollup      = {};   // "" when there's no rollup suffix
    std::string_view submetric   = {};   // "" when there's no submetric suffix
    Unit             unit;
    Scope            scope;
    Peak             peak        = std::monostate{};
    bool             smoothable  = true;
    std::string_view description = {};
    double         (*read)(const Tick&);
};

} // namespace internal
} // namespace cupti_profiler
