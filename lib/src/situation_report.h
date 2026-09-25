// Internal: the startup situation report — which tracking guarantees
// hold in THIS environment. Computed at ProfilerSuite::Configure(),
// logged once, and written to session_metadata.pb (SituationCheck) so a
// trace always says what it was collected under.
//
// Everything is observed from the observer side: /proc, /sys, prctl on
// ourselves, and — under SIDECAR — the sidecar's /proc/<pid>/status and
// its binary's mount and file capabilities. Nothing touches the targets
// beyond reading their /proc entries.
#pragma once

#include "process_discovery.h"

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

namespace cupti_profiler {
namespace internal {

struct SituationLine {
    std::string check;
    std::string observed;
    std::string consequence;
    bool        degraded = false;
};

struct SituationInputs {
    bool              sidecar    = false;  // observer is the sidecar process
    pid_t             sidecarPid = -1;
    std::string       observerBinary;      // sidecar path, or /proc/self/exe
    DiscoverySettings discovery;
};

/// Environment lines (everything except the per-root lines).
std::vector<SituationLine> ProbeSituation(const SituationInputs& in);

/// Subreaper state of this process right now. Re-probed on every
/// manifest write, since adopt_orphans() may be called after Configure().
SituationLine ProbeSubreaper();

/// One line for a tracked root: same uid, spawned by this process vs
/// attached, /proc/<pid>/io readable.
SituationLine ProbeRoot(uint32_t pid, bool tracksDescendants, uint64_t scanIntervalMs);

/// Print lines to stderr, one per check, each with its consequence.
void LogSituation(const std::string& heading, const std::vector<SituationLine>& lines);

} // namespace internal
} // namespace cupti_profiler
