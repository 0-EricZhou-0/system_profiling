#include "lifecycle.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace cupti_profiler {
namespace internal {
namespace lifecycle {

// How the handler behaves (the library lives inside programs, such as
// Python or vLLM, that have their own handlers):
//   * It only does async-signal-safe work: it writes the signal number
//     to a pipe and waits (poll) for the flusher thread, a normal thread,
//     to stop everything registered. Then it chains to the handler that
//     was installed before it (Python's SIGINT handler still raises
//     KeyboardInterrupt), or, if there was none, restores the default
//     action and re-raises, so the exit status and any core dump are what
//     they would have been.
//   * Termination signals wait up to kTerminateBoundMs for the flush,
//     crash signals up to kCrashBoundMs (the crashed state may keep the
//     flush from finishing).
//   * A second signal while a flush is running kills the process at once.
//   * A signal that arrives while a Stop() is already running waits for
//     it (bounded) instead of starting another one, unless it interrupted
//     that very Stop(), which cannot finish until the handler returns.
//   * In a forked child it only chains (the flusher is the parent's).
//   * Signals that were ignored (SIG_IGN) stay ignored. SIGKILL and
//     SIGSTOP cannot be caught: up to one flush interval of samples is
//     lost then.
//   * The library's own threads block the termination signals, so the
//     handler always runs on a host thread, never on a probe thread the
//     flush must join.

namespace {

constexpr int kTerminate[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGALRM,
                              SIGPIPE, SIGXCPU, SIGXFSZ, SIGVTALRM, SIGPROF, SIGIO, SIGPWR};
constexpr int kCrash[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGSYS};
constexpr int kTerminateBoundMs = 10000;
constexpr int kCrashBoundMs = 2000;

struct Entry {
    const void* key;
    Order order;
    const char* kind;
    std::function<void()> stop;
};

// Leaked on purpose: must outlive every static destructor and atexit
// handler that could still stop something.
std::mutex& RegistryMutex() { static auto* m = new std::mutex; return *m; }
std::vector<Entry>& Registry() { static auto* v = new std::vector<Entry>; return *v; }

std::atomic<int>   g_running{0};         // registered objects
std::atomic<int>   g_stopping{0};        // Stop()s in progress
std::atomic<pid_t> g_stoppingTid{0};     // thread of the latest one
std::atomic<bool>  g_flushing{false};    // a handler is waiting for a flush
std::atomic<pid_t> g_ownerPid{0};        // the process whose flusher it is
std::atomic<pid_t> g_registryPid{0};     // the process that registered

struct sigaction g_prev[NSIG];
bool             g_ours[NSIG];
std::mutex       g_installMutex;
int              g_installRefs = 0;
int              g_wakeFd[2] = {-1, -1}; // handler -> flusher (signal number)
int              g_doneFd = -1;          // flusher -> handler (eventfd)

bool IsCrash(int sig) {
    for (int s : kCrash) if (s == sig) return true;
    return false;
}

pid_t Tid() { return static_cast<pid_t>(::syscall(SYS_gettid)); }

int64_t NowMs() {
    struct timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

void Say(const char* msg) {
    ssize_t r = ::write(STDERR_FILENO, msg, std::strlen(msg));
    (void)r;
}

[[noreturn]] void RaiseDefault(int sig) {
    struct sigaction dfl{};
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    ::sigaction(sig, &dfl, nullptr);
    sigset_t one;
    sigemptyset(&one);
    sigaddset(&one, sig);
    ::sigprocmask(SIG_UNBLOCK, &one, nullptr);
    ::raise(sig);
    ::_exit(128 + sig);   // not reached for any signal handled here
}

void Chain(int sig, siginfo_t* si, void* uc) {
    const struct sigaction& p = g_prev[sig];
    if (p.sa_flags & SA_SIGINFO) {
        if (p.sa_sigaction) p.sa_sigaction(sig, si, uc);
        return;
    }
    if (p.sa_handler == SIG_IGN) return;
    if (p.sa_handler == SIG_DFL) RaiseDefault(sig);
    p.sa_handler(sig);
}

// Wait until the flusher says it is done, up to `boundMs`.
void WaitDone(int boundMs) {
    const int64_t deadline = NowMs() + boundMs;
    for (;;) {
        const int64_t left = deadline - NowMs();
        if (left <= 0) {
            Say("[cupti-profiler] flush on signal did not finish in time; traces may be incomplete\n");
            return;
        }
        struct pollfd p{g_doneFd, POLLIN, 0};
        const int r = ::poll(&p, 1, static_cast<int>(left));
        if (r > 0) return;
        if (r < 0 && errno != EINTR) return;
    }
}

// Another thread is in Stop(): wait for it to end, up to `boundMs`.
void WaitStopEnds(int boundMs) {
    const int64_t deadline = NowMs() + boundMs;
    while (g_stopping.load() > 0 && NowMs() < deadline) {
        struct timespec ms{0, 1000000};
        ::nanosleep(&ms, nullptr);
    }
}

extern "C" void OnSignal(int sig, siginfo_t* si, void* uc) {
    const int savedErrno = errno;
    // A forked child inherits the handler and the registry but not the
    // threads; the pipe would reach the parent's flusher.
    if (::getpid() != g_ownerPid.load()) {
        Chain(sig, si, uc);
        errno = savedErrno;
        return;
    }
    if (g_flushing.exchange(true)) {
        Say("[cupti-profiler] another signal during the flush: exiting now\n");
        RaiseDefault(sig);
    }
    if (g_running.load() > 0) {
        const int bound = IsCrash(sig) ? kCrashBoundMs : kTerminateBoundMs;
        if (g_stopping.load() > 0) {
            if (g_stoppingTid.load() != Tid()) WaitStopEnds(bound);
        } else if (g_wakeFd[1] >= 0) {
            uint64_t stale;
            while (::read(g_doneFd, &stale, sizeof(stale)) > 0) {}
            const unsigned char b = static_cast<unsigned char>(sig);
            if (::write(g_wakeFd[1], &b, 1) == 1) WaitDone(bound);
        }
    }
    g_flushing.store(false);
    Chain(sig, si, uc);
    errno = savedErrno;
}

void FlusherLoop() {
    BlockSignalsInThisThread();
    ::pthread_setname_np(::pthread_self(), "cupti-sigflush");
    for (;;) {
        unsigned char sig = 0;
        const ssize_t r = ::read(g_wakeFd[0], &sig, 1);
        if (r < 0 && errno == EINTR) continue;
        if (r != 1) return;
        // snprintf + write: no stdio lock, which the interrupted thread
        // may hold.
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "[cupti-profiler] %s: stopping the profiler and flushing its traces\n",
                      ::strsignal(sig));
        Say(msg);
        StopAll();
        const uint64_t one = 1;
        ssize_t w = ::write(g_doneFd, &one, sizeof(one));
        (void)w;
    }
}

bool StartFlusher() {
    if (g_doneFd >= 0) return true;
    if (::pipe2(g_wakeFd, O_CLOEXEC) != 0) return false;
    g_doneFd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (g_doneFd < 0) {
        ::close(g_wakeFd[0]); ::close(g_wakeFd[1]);
        g_wakeFd[0] = g_wakeFd[1] = -1;
        return false;
    }
    g_ownerPid.store(::getpid());
    std::thread(FlusherLoop).detach();
    return true;
}

} // namespace

void OnExit() { StopAtExit(); }

void Register(const void* key, Order order, const char* kind, std::function<void()> stop) {
    // Exit hook, once. Registered at the first Start(), after the CUDA
    // runtime registered its own teardown (Configure() initialized it),
    // so it runs before that (atexit is LIFO); the registry is leaked,
    // so it outlives every static destructor.
    static std::once_flag hook;
    std::call_once(hook, [] {
        RegistryMutex();
        g_registryPid.store(::getpid());
        std::atexit(OnExit);
    });
    std::lock_guard<std::mutex> lk(RegistryMutex());
    Registry().push_back({key, order, kind, std::move(stop)});
    g_running.store(static_cast<int>(Registry().size()));
}

void Unregister(const void* key) {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    auto& r = Registry();
    for (auto it = r.begin(); it != r.end(); ++it) {
        if (it->key == key) { r.erase(it); break; }
    }
    g_running.store(static_cast<int>(r.size()));
}

std::vector<std::string> StopAll() {
    std::vector<std::string> stopped;
    for (;;) {
        Entry e;
        {
            std::lock_guard<std::mutex> lk(RegistryMutex());
            auto& r = Registry();
            if (r.empty()) break;
            auto pick = r.begin();
            for (auto it = r.begin(); it != r.end(); ++it)
                if (it->order < pick->order) pick = it;
            e = *pick;
        }
        e.stop();
        Unregister(e.key);   // in case stop() did not
        stopped.push_back(e.kind);
    }
    return stopped;
}

void WarnNotStopped(const char* kind, const char* when) {
    std::fprintf(stderr, "[cupti-profiler] warning: stop() was not called; the %s was stopped %s "
                         "and its traces flushed\n", kind, when);
}

void StopAtExit() {
    // A forked child that exits normally must not join threads it does
    // not have.
    if (::getpid() != g_registryPid.load()) return;
    for (const auto& kind : StopAll()) WarnNotStopped(kind.c_str(), "at process exit");
}

StopScope::StopScope() {
    g_stopping.fetch_add(1);
    g_stoppingTid.store(Tid());
}

StopScope::~StopScope() {
    if (g_stopping.fetch_sub(1) == 1) g_stoppingTid.store(0);
}

void InstallSignalHandlers() {
    std::lock_guard<std::mutex> lk(g_installMutex);
    if (g_installRefs++ > 0) return;
    if (!StartFlusher()) {
        std::fprintf(stderr, "[cupti-profiler] warning: cannot start the signal flusher (%s); "
                             "traces are not flushed on signals\n", std::strerror(errno));
        return;
    }
    auto install = [](int sig) {
        struct sigaction cur{};
        if (::sigaction(sig, nullptr, &cur) != 0) return;
        if (!(cur.sa_flags & SA_SIGINFO) && cur.sa_handler == SIG_IGN) return;   // stays ignored
        if ((cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == OnSignal) return;
        g_prev[sig] = cur;
        struct sigaction ours{};
        ours.sa_sigaction = OnSignal;
        sigemptyset(&ours.sa_mask);
        ours.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK | (cur.sa_flags & SA_RESTART);
        if (::sigaction(sig, &ours, nullptr) == 0) g_ours[sig] = true;
    };
    for (int s : kTerminate) install(s);
    for (int s : kCrash) install(s);
}

void RemoveSignalHandlers() {
    std::lock_guard<std::mutex> lk(g_installMutex);
    if (g_installRefs == 0 || --g_installRefs > 0) return;
    for (int sig = 1; sig < NSIG; ++sig) {
        if (!g_ours[sig]) continue;
        struct sigaction cur{};
        // Someone installed a handler over ours: leave theirs.
        if (::sigaction(sig, nullptr, &cur) == 0 &&
            (cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == OnSignal) {
            ::sigaction(sig, &g_prev[sig], nullptr);
        }
        g_ours[sig] = false;
    }
}

void BlockSignalsInThisThread() {
    sigset_t set;
    sigemptyset(&set);
    for (int s : kTerminate) sigaddset(&set, s);
    ::pthread_sigmask(SIG_BLOCK, &set, nullptr);
}

} // namespace lifecycle
} // namespace internal
} // namespace cupti_profiler
