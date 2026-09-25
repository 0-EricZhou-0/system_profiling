#include <cupti_profiler/testing.h>

#include "testing_hooks.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <unistd.h>

#include "proc_readers.h"

namespace cupti_profiler {

namespace {

std::atomic<bool>       g_armed{false};
std::mutex              g_mu;
std::condition_variable g_cv;
bool                    g_held     = false;
bool                    g_released = false;

std::atomic<uint32_t>   g_killAfterRead{0};

} // namespace

namespace testing {

void ArmFlushGate() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_held = false;
    g_released = false;
    g_armed.store(true);
}

bool WaitFlushHeld(unsigned timeoutMs) {
    std::unique_lock<std::mutex> lk(g_mu);
    return g_cv.wait_for(lk, std::chrono::milliseconds(timeoutMs), [] { return g_held; });
}

void ReleaseFlushGate() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_armed.store(false);
    g_released = true;
    g_cv.notify_all();
}

void KillAfterNextRead(uint32_t pid) { g_killAfterRead.store(pid); }

} // namespace testing

namespace internal {

void PassFlushGate() {
    if (!g_armed.load(std::memory_order_relaxed)) return;
    std::unique_lock<std::mutex> lk(g_mu);
    if (!g_armed.load() || g_held) return;   // one flush only
    g_held = true;
    g_cv.notify_all();
    g_cv.wait(lk, [] { return g_released; });
}

bool PassReadHook(uint32_t pid) {
    if (g_killAfterRead.load(std::memory_order_relaxed) != pid || pid == 0) return false;
    uint32_t expected = pid;
    if (!g_killAfterRead.compare_exchange_strong(expected, 0)) return false;
    int fd = PidfdOpen(pid);
    ::kill(static_cast<pid_t>(pid), SIGKILL);
    if (fd >= 0) {
        struct pollfd p{fd, POLLIN, 0};
        ::poll(&p, 1, 2000);   // exited (a zombie is enough)
        ::close(fd);
    }
    return true;
}

} // namespace internal
} // namespace cupti_profiler
