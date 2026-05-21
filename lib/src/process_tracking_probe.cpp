#include <cupti_profiler/process_tracking_probe.h>

#include <algorithm>

namespace cupti_profiler {

void ProcessTrackingProbe::AddTrackedProcess(uint32_t pid, std::string alias) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    // Idempotent: if the PID is already tracked, just clear any
    // pending_removal flag (a re-Add after Remove un-removes).
    for (auto& e : processes_) {
        if (e.pid == pid) {
            e.pending_removal = false;
            if (!alias.empty()) e.alias = std::move(alias);
            return;
        }
    }
    processes_.push_back({pid, std::move(alias), /*pending_removal=*/false});
}

void ProcessTrackingProbe::RemoveTrackedProcess(uint32_t pid) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    for (auto& e : processes_) {
        if (e.pid == pid) {
            e.pending_removal = true;
            return;
        }
    }
    // PID wasn't tracked — silently no-op (matches Add idempotency).
}

void ProcessTrackingProbe::SetInitialProcesses(std::vector<ProcessEntry> entries) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    processes_ = std::move(entries);
}

std::vector<ProcessTrackingProbe::ProcessEntry>
ProcessTrackingProbe::SnapshotProcesses() const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return processes_;
}

void ProcessTrackingProbe::CommitPendingRemovals() {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    processes_.erase(
        std::remove_if(processes_.begin(), processes_.end(),
                       [](const ProcessEntry& e) { return e.pending_removal; }),
        processes_.end());
}

} // namespace cupti_profiler
