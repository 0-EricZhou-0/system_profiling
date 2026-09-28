#include <cupti_profiler/testing.h>

#include "testing_hooks.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
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

// Kill-after-read, one per probe: the target PID (0 = disarmed) and how
// many of its reads to let through first. Each is read only by that
// probe's sampling thread.
struct ReadHook {
    std::atomic<uint32_t> pid{0};
    std::atomic<uint32_t> skip{0};
};
ReadHook g_readHook[2];

std::atomic<unsigned> g_flushDelayMs{0};
std::atomic<unsigned> g_backlogPeriodMs{30000};
std::atomic<unsigned> g_decodeStallMs{0};

ReadHook& HookFor(testing::ReadProbe probe) {
    return g_readHook[probe == testing::ReadProbe::Disk ? 1 : 0];
}

void Arm(testing::ReadProbe probe, uint32_t pid, uint32_t skip) {
    auto& h = HookFor(probe);
    h.pid.store(0);
    h.skip.store(skip);
    h.pid.store(pid);
}

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

void SetFlushDelayMs(unsigned ms) { g_flushDelayMs.store(ms); }

void SetBacklogReportPeriodMs(unsigned ms) { g_backlogPeriodMs.store(ms); }

void StallNextDecodeMs(unsigned ms) { g_decodeStallMs.store(ms); }

void ReleaseFlushGate() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_armed.store(false);
    g_released = true;
    g_cv.notify_all();
}

void KillAfterNextRead(uint32_t pid, ReadProbe probe) { Arm(probe, pid, 0); }

bool ArmKillAfterReadFromEnv() {
    const char* env = std::getenv("CUPTI_PROFILER_TEST_KILL_AFTER_READ");
    if (!env || !*env) return false;
    char probe[16] = {};
    unsigned pid = 0, n = 0;
    if (std::sscanf(env, "%15[a-z]:%u:%u", probe, &pid, &n) != 3 || pid == 0 || n == 0 ||
        (std::strcmp(probe, "system") != 0 && std::strcmp(probe, "disk") != 0)) {
        std::fprintf(stderr, "[testing] ignoring malformed CUPTI_PROFILER_TEST_KILL_AFTER_READ=%s\n", env);
        return false;
    }
    Arm(std::strcmp(probe, "disk") == 0 ? ReadProbe::Disk : ReadProbe::System, pid, n - 1);
    std::fprintf(stderr, "[testing] kill-after-read armed: %s probe, pid %u, read %u\n", probe, pid, n);
    return true;
}

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

bool PassReadHook(uint32_t pid, testing::ReadProbe probe) {
    auto& h = HookFor(probe);
    if (h.pid.load(std::memory_order_relaxed) != pid || pid == 0) return false;
    if (h.skip.load() > 0) { h.skip.fetch_sub(1); return false; }
    uint32_t expected = pid;
    if (!h.pid.compare_exchange_strong(expected, 0)) return false;
    int fd = PidfdOpen(pid);
    ::kill(static_cast<pid_t>(pid), SIGKILL);
    if (fd >= 0) {
        struct pollfd p{fd, POLLIN, 0};
        ::poll(&p, 1, 2000);   // exited (a zombie is enough)
        ::close(fd);
    }
    return true;
}

void PassFlushDelay() {
    const unsigned ms = g_flushDelayMs.load(std::memory_order_relaxed);
    if (ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void PassDecodeStall() {
    const unsigned ms = g_decodeStallMs.exchange(0, std::memory_order_relaxed);
    if (ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

unsigned BacklogReportPeriodMs() { return g_backlogPeriodMs.load(std::memory_order_relaxed); }

} // namespace internal

} // namespace cupti_profiler
