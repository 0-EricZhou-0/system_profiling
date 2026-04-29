#include <cupti_profiler/event_profiler.h>

#include "helper_cupti.h"
#include "event_tracker_internal.h"

#include <cuda_runtime.h>
#include <cupti_target.h>  // CUptiResult

#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" CUptiResult cuptiGetTimestamp(uint64_t *timestamp);

namespace cupti_profiler {

namespace {
uint64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

struct EventTracker::Impl {
    Domain domain;
    std::mutex mutex;

    // Regions whose Begin has been recorded but End has NOT been called yet.
    // Keyed by stable id returned from BeginRegion. Drain() never walks this
    // map — once End is called, the entry moves on to a different bucket.
    struct PendingRegion {
        std::string name;
        // GENERIC: timestamp captured at BeginRegion site.
        uint64_t startNs = 0;
        // GPU: cudaEvent recorded on the stream at BeginRegion.
        cudaEvent_t startEvent = nullptr;
    };
    std::unordered_map<size_t, PendingRegion> pendingRegions;
    size_t nextRegionId = 0;

    // GPU-only: Regions whose End has fired (so both cudaEvents are recorded)
    // but whose cudaEvents may not have completed yet. Drain() walks this list
    // and promotes ready entries to resolvedRegions. Generally short.
    struct AwaitingRegion {
        std::string name;
        cudaEvent_t startEvent = nullptr;
        cudaEvent_t endEvent   = nullptr;
    };
    std::vector<AwaitingRegion> awaitingRegions;

    // GPU-only: instantaneous events awaiting cudaEvent completion.
    struct PendingGpuEvent {
        std::string name;
        cudaEvent_t event = nullptr;
    };
    std::vector<PendingGpuEvent> pendingGpuEvents;

    // Buffers ready for the next flush. GENERIC regions/events land here
    // directly at EndRegion / MarkEvent time; GPU entries arrive here only
    // after Drain() has resolved them.
    std::vector<internal::ResolvedRegion> resolvedRegions;
    std::vector<internal::ResolvedEvent>  resolvedEvents;

    // GPU sync anchor: a cudaEvent recorded at SetStream() paired with the
    // CUPTI timestamp at the same moment.
    cudaStream_t stream    = nullptr;
    cudaEvent_t  refEvent  = nullptr;
    uint64_t     cuptiRefNs = 0;
};

EventTracker::EventTracker(Domain d) : m_impl(std::make_unique<Impl>()) {
    m_impl->domain = d;
}
EventTracker::~EventTracker() {
    if (!m_impl) return;
    if (m_impl->refEvent) cudaEventDestroy(m_impl->refEvent);
    for (auto& kv : m_impl->pendingRegions) {
        if (kv.second.startEvent) cudaEventDestroy(kv.second.startEvent);
    }
    for (auto& ar : m_impl->awaitingRegions) {
        if (ar.startEvent) cudaEventDestroy(ar.startEvent);
        if (ar.endEvent)   cudaEventDestroy(ar.endEvent);
    }
    for (auto& pe : m_impl->pendingGpuEvents) {
        if (pe.event) cudaEventDestroy(pe.event);
    }
}
EventTracker::EventTracker(EventTracker&&) noexcept = default;
EventTracker& EventTracker::operator=(EventTracker&&) noexcept = default;

EventTracker::Domain EventTracker::GetDomain() const { return m_impl->domain; }

void EventTracker::SetStream(void* stream) {
    if (m_impl->domain != Domain::GPU) {
        std::cerr << "[EventTracker] SetStream() only valid for GPU domain\n";
        return;
    }
    m_impl->stream = static_cast<cudaStream_t>(stream);
    RUNTIME_API_CALL(cudaEventCreate(&m_impl->refEvent));
    RUNTIME_API_CALL(cudaEventRecord(m_impl->refEvent, m_impl->stream));
    RUNTIME_API_CALL(cudaEventSynchronize(m_impl->refEvent));
    cuptiGetTimestamp(&m_impl->cuptiRefNs);
}

size_t EventTracker::BeginRegion(const std::string& name) {
    // Record cudaEvent / capture timestamp BEFORE acquiring the mutex —
    // the lock only protects the map insert. CUDA APIs and steady_clock::now
    // are themselves thread-safe.
    Impl::PendingRegion pr;
    pr.name = name;
    if (m_impl->domain == Domain::GPU) {
        RUNTIME_API_CALL(cudaEventCreate(&pr.startEvent));
        RUNTIME_API_CALL(cudaEventRecord(pr.startEvent, m_impl->stream));
    } else {  // GENERIC default
        pr.startNs = SteadyNowNs();
    }
    std::lock_guard<std::mutex> lk(m_impl->mutex);
    size_t id = m_impl->nextRegionId++;
    m_impl->pendingRegions.emplace(id, std::move(pr));
    return id;
}

void EventTracker::EndRegion(size_t idx) {
    // GPU branch: record the end cudaEvent outside the lock, then under the
    // lock erase from pendingRegions and append to awaitingRegions in one go.
    // This means Drain() never has to walk pendingRegions — it only iterates
    // the (typically short) awaitingRegions list.
    if (m_impl->domain == Domain::GPU) {
        cudaEvent_t endEvent = nullptr;
        RUNTIME_API_CALL(cudaEventCreate(&endEvent));
        RUNTIME_API_CALL(cudaEventRecord(endEvent, m_impl->stream));

        std::lock_guard<std::mutex> lk(m_impl->mutex);
        auto it = m_impl->pendingRegions.find(idx);
        if (it == m_impl->pendingRegions.end()) {
            // Unknown id — clean up the just-created event.
            cudaEventDestroy(endEvent);
            return;
        }
        Impl::AwaitingRegion ar;
        ar.name       = std::move(it->second.name);
        ar.startEvent = it->second.startEvent;
        ar.endEvent   = endEvent;
        m_impl->pendingRegions.erase(it);
        m_impl->awaitingRegions.push_back(std::move(ar));
        return;
    }

    // GENERIC default: both timestamps are known immediately, so jump
    // straight to the resolved buffer — no awaiting state needed.
    uint64_t endNs = SteadyNowNs();
    std::lock_guard<std::mutex> lk(m_impl->mutex);
    auto it = m_impl->pendingRegions.find(idx);
    if (it == m_impl->pendingRegions.end()) return;
    m_impl->resolvedRegions.push_back(
        {std::move(it->second.name), it->second.startNs, endNs});
    m_impl->pendingRegions.erase(it);
}

void EventTracker::MarkEvent(const std::string& name) {
    if (m_impl->domain == Domain::GPU) {
        // Record cudaEvent before lock, then append.
        cudaEvent_t ev = nullptr;
        RUNTIME_API_CALL(cudaEventCreate(&ev));
        RUNTIME_API_CALL(cudaEventRecord(ev, m_impl->stream));
        Impl::PendingGpuEvent pe;
        pe.name = name;
        pe.event = ev;
        std::lock_guard<std::mutex> lk(m_impl->mutex);
        m_impl->pendingGpuEvents.push_back(std::move(pe));
        return;
    }
    // GENERIC default: timestamp known now → straight to resolved buffer.
    uint64_t ns = SteadyNowNs();
    std::lock_guard<std::mutex> lk(m_impl->mutex);
    m_impl->resolvedEvents.push_back({name, ns});
}

void EventTracker::Drain(bool forceResolve,
                         std::vector<internal::ResolvedRegion>& outRegions,
                         std::vector<internal::ResolvedEvent>& outEvents) {
    using internal::ResolvedRegion;
    auto& impl = *m_impl;

    // Phase 1: under lock, swap out the resolved buffers AND the GPU work
    // lists. Once the swap is done the lock can be dropped — user threads
    // will write into the now-fresh empty containers, while we resolve the
    // detached snapshots without contention.
    std::vector<Impl::AwaitingRegion> localAwaiting;
    std::vector<Impl::PendingGpuEvent> localPending;
    {
        std::lock_guard<std::mutex> lk(impl.mutex);
        outRegions.swap(impl.resolvedRegions);
        outEvents.swap(impl.resolvedEvents);
        if (impl.domain == Domain::GPU) {
            localAwaiting.swap(impl.awaitingRegions);
            localPending.swap(impl.pendingGpuEvents);
        }
    }

    // GENERIC: nothing more to do — End/Mark already populated the resolved
    // buffers we just swapped out.
    if (impl.domain != Domain::GPU) return;

    // Phase 2: outside the lock, resolve completed cudaEvents from the
    // detached snapshots. Anything that's still not-ready goes into a
    // "leftovers" list that we'll re-insert under the lock at the end.
    std::vector<Impl::AwaitingRegion> leftoverAwaiting;
    std::vector<Impl::PendingGpuEvent> leftoverPending;

    for (auto& ar : localAwaiting) {
        cudaError_t startReady = cudaEventQuery(ar.startEvent);
        cudaError_t endReady   = cudaEventQuery(ar.endEvent);
        bool ready = (startReady == cudaSuccess && endReady == cudaSuccess);
        if (!ready && forceResolve) {
            cudaEventSynchronize(ar.endEvent);
            ready = true;
        }
        if (!ready) {
            leftoverAwaiting.push_back(std::move(ar));
            continue;
        }
        float startMs = 0, endMs = 0;
        cudaEventElapsedTime(&startMs, impl.refEvent, ar.startEvent);
        cudaEventElapsedTime(&endMs,   impl.refEvent, ar.endEvent);
        outRegions.push_back({std::move(ar.name),
                              impl.cuptiRefNs + (uint64_t)(startMs * 1e6),
                              impl.cuptiRefNs + (uint64_t)(endMs   * 1e6)});
        cudaEventDestroy(ar.startEvent);
        cudaEventDestroy(ar.endEvent);
    }

    for (auto& pe : localPending) {
        cudaError_t ready = cudaEventQuery(pe.event);
        if (ready != cudaSuccess && forceResolve) {
            cudaEventSynchronize(pe.event);
            ready = cudaSuccess;
        }
        if (ready != cudaSuccess) {
            leftoverPending.push_back(std::move(pe));
            continue;
        }
        float ms = 0;
        cudaEventElapsedTime(&ms, impl.refEvent, pe.event);
        outEvents.push_back({std::move(pe.name),
                             impl.cuptiRefNs + (uint64_t)(ms * 1e6)});
        cudaEventDestroy(pe.event);
    }

    // Phase 3: re-acquire the lock briefly only if we have not-ready
    // entries to put back. New entries appended by user threads in the
    // meantime are preserved (we splice leftovers in front of them).
    if (leftoverAwaiting.empty() && leftoverPending.empty()) return;

    std::lock_guard<std::mutex> lk(impl.mutex);
    // Splice leftovers in: keep them ahead of any newly-arrived entries so
    // older outstanding regions still drain first on the next call.
    if (!leftoverAwaiting.empty()) {
        leftoverAwaiting.insert(leftoverAwaiting.end(),
                                std::make_move_iterator(impl.awaitingRegions.begin()),
                                std::make_move_iterator(impl.awaitingRegions.end()));
        impl.awaitingRegions = std::move(leftoverAwaiting);
    }
    if (!leftoverPending.empty()) {
        leftoverPending.insert(leftoverPending.end(),
                               std::make_move_iterator(impl.pendingGpuEvents.begin()),
                               std::make_move_iterator(impl.pendingGpuEvents.end()));
        impl.pendingGpuEvents = std::move(leftoverPending);
    }
}

} // namespace cupti_profiler
