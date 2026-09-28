// Internal: what is running in this process, and stopping it when the
// process is about to die on a signal.
//
// Every suite and probe registers itself at Start() and unregisters in
// Stop(). A ProfilerSuite started with signal handlers installed
// (ProfilerSuiteConfig.disable_signal_handlers = false, the default)
// makes a catchable fatal signal stop everything registered — final GPU
// decode, every probe's buffered samples, the sidecar's final flush,
// trace files closed — before the signal takes its course. See
// lifecycle.cpp for the rules the handler follows.
#pragma once

#include <cupti_profiler/profiler_error.h>   // CUPTI_PROFILER_API

#include <functional>
#include <string>
#include <vector>

namespace cupti_profiler {
namespace internal {
namespace lifecycle {

enum class Order { Suite = 0, Probe = 1 };   // stopped in this order

/// `stop` must end with Unregister(key). `kind` names the object in
/// messages ("ProfilerSuite", "GpuProfiler", ...).
void Register(const void* key, Order order, const char* kind, std::function<void()> stop);
void Unregister(const void* key);

/// Stop everything registered, suites first (a suite stops its own
/// probes). Returns the kinds stopped, in order.
std::vector<std::string> StopAll();

/// Process exit with something still running (stop() never called):
/// stop it all and print one "[cupti-profiler] warning:" line per object
/// stopped. Run by a std::atexit handler (registered at the first
/// Register) and by the Python package's atexit hook, which comes first.
/// Does nothing in a forked child.
void StopAtExit();

/// "[cupti-profiler] warning: stop() was not called; the <kind> was
/// stopped <when> and its traces flushed".
void WarnNotStopped(const char* kind, const char* when);

/// Marks a Stop() in progress on this thread: a signal that arrives then
/// does not ask for another flush (it could not start until this one
/// ends) but waits for this one when it runs on another thread.
class StopScope {
public:
    StopScope();
    ~StopScope();
    StopScope(const StopScope&) = delete;
    StopScope& operator=(const StopScope&) = delete;
};

/// Install the handlers (refcounted: one Install per Remove). The first
/// call starts the flusher thread. Exported: the sidecar installs them
/// too.
CUPTI_PROFILER_API void InstallSignalHandlers();
/// Put back each previous disposition whose handler is still ours, once
/// the last installer has removed its handlers.
CUPTI_PROFILER_API void RemoveSignalHandlers();

/// Block the asynchronous termination signals in the calling thread, so
/// the kernel delivers them to one of the host's threads, never to a
/// probe thread that a flush has to join. Every library thread calls it
/// first thing.
void BlockSignalsInThisThread();

} // namespace lifecycle
} // namespace internal
} // namespace cupti_profiler
