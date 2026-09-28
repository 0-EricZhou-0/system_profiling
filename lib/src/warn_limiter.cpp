#include "warn_limiter.h"

#include <atomic>
#include <climits>
#include <iostream>

namespace cupti_profiler {
namespace internal {

namespace {
std::atomic<size_t> g_keys{0};
}

size_t WarnLimiterKeysInProcess() { return g_keys.load(); }

bool WarnLimiter::Warn(uint64_t serial, int type, const std::string& text, uint64_t nowNs) {
    std::lock_guard<std::mutex> lk(mu_);
    auto [it, fresh] = keys_.try_emplace({serial, type});
    if (fresh) g_keys.fetch_add(1);
    State& s = it->second;
    s.text = text;
    if (!fresh && nowNs < s.lastNs + kPeriodNs) {
        ++s.suppressed;
        return false;
    }
    std::string line = text;
    if (s.suppressed) line += " (" + std::to_string(s.suppressed) + " suppressed)";
    std::cerr << line << "\n";
    s.lastNs = nowNs;
    s.suppressed = 0;
    return true;
}

void WarnLimiter::Summarize(const State& s, const char* when) {
    if (s.suppressed)
        std::cerr << s.text << " (" << s.suppressed << " suppressed; " << when << ")\n";
}

void WarnLimiter::Remove(uint64_t serial) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = keys_.lower_bound({serial, INT32_MIN});
         it != keys_.end() && it->first.first == serial; ) {
        Summarize(it->second, "no longer tracked");
        it = keys_.erase(it);
        g_keys.fetch_sub(1);
    }
}

void WarnLimiter::Flush() {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& [k, s] : keys_) Summarize(s, "at stop");
    g_keys.fetch_sub(keys_.size());
    keys_.clear();
}

size_t WarnLimiter::Size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return keys_.size();
}

WarnLimiter::~WarnLimiter() { g_keys.fetch_sub(keys_.size()); }

} // namespace internal
} // namespace cupti_profiler
