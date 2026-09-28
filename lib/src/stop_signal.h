// Internal: a stop flag a thread can wait on. Every periodic probe thread
// waits for its next tick here instead of sleeping, so Stop() wakes it at
// once rather than after the rest of its interval.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace cupti_profiler {
namespace internal {

class StopSignal {
public:
    void Set() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
    }
    void Reset() {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = false;
    }
    bool IsSet() {
        std::lock_guard<std::mutex> lk(m_);
        return stop_;
    }
    /// Wait until `deadline` or Set(). True = stop was requested.
    template <class Clock, class Dur>
    bool WaitUntil(const std::chrono::time_point<Clock, Dur>& deadline) {
        std::unique_lock<std::mutex> lk(m_);
        return cv_.wait_until(lk, deadline, [this] { return stop_; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace internal
} // namespace cupti_profiler
