// cupti-profiler sidecar — external observer for CPU / memory / disk
// probes.
//
// Runs as a child process of the workload (fork+exec'd by
// ProfilerSuite::Configure() when SystemProbeMode::SIDECAR is
// requested). Reads its config + PID list + sync anchor from the
// workload over a pipe, samples /proc or netlink (taskstats) for the
// tracked PIDs, and writes its own .pb files referenced from the
// same session_metadata.pb the workload writes.
//
// This TU is currently a skeleton — later commits in the sidecar
// series wire up the pipe handshake, the netlink TaskstatsClient,
// and the mid-run Add/RemoveTrackedProcess Unix-socket listener.
// Skeleton exists now so the build target, install path, and the
// POST_BUILD CAP_NET_ADMIN check ship independently of the runtime
// work.

#include <iostream>

int main(int /*argc*/, char** /*argv*/) {
    std::cerr << "cupti-profiler sidecar: skeleton only, no probes wired yet.\n";
    return 0;
}
