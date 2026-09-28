#include "decode_thread.h"
#include "lifecycle.h"
#include "testing_hooks.h"

#include <pthread.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

namespace cupti_profiler {
namespace internal {

namespace {

// A pass stops at END_OF_RECORDS. This only bounds a CUPTI that never
// says so; every step consumes hardware-buffer records.
constexpr int kMaxDecodeStepsPerPass = 1 << 16;

// Drains the hardware buffer into a counter-data image, and has a worker
// thread evaluate every completed sample and re-initialize the image.
//
// Two images: the decode thread decodes into one while the worker
// evaluates and re-initializes the other, so a pass (or a drain step
// after a full image) never waits for evaluation. The CUPTI 13.3 docs do
// not say whether calls on one sampler object (DecodeData, GetSampleInfo,
// CounterDataImageInitialize) or its host object (EvaluateToGpuValues)
// may run concurrently, so every CUPTI call of a device is serialized
// (cupti_, held per call: a decode waits for at most one evaluation).
// Both images have the same size: a GetCounterDataSize query with another
// max_samples makes later image initializations fail.
//
// What CUPTI 13.3 does (measured on H100, 2026-09-28; docs/cupti-hardware-findings.md, 4):
//   * A decode that stops at END_OF_RECORDS (or OTHER) got everything
//     the hardware buffer held. Samples are complete and contiguous
//     (each one starts where the previous one ended) across passes;
//     a sample still being written stays in the buffer.
//   * A decode that stops at COUNTER_DATA_FULL left records behind.
//     Decoding again catches the buffer up, but part of what those
//     decodes return is out of order or without timestamps, and the
//     samples beyond the image's capacity are gone. The next passes are
//     clean again. Those samples are dropped and counted; the image is
//     sized so a pass fits (max_samples).
//   * A hardware-buffer overflow makes this and every later decode fail
//     with CUPTI_ERROR_OUT_OF_MEMORY and return nothing, until the
//     sampler is disabled and enabled again (Stop + Start is not enough).
//     The decoder does that (~115 ms); the samples of the stall and of
//     the restart are lost and counted.
class Decoder {
public:
    Decoder(std::array<std::vector<uint8_t>, 2>& images, const std::vector<const char*>& metrics,
            CuptiPmSampling& target, CuptiProfilerHost& host,
            DecodeTarget device, DecodeStats& stats)
        : images_(images), metrics_(metrics), target_(target), host_(host),
          device_(device), stats_(stats) {
        worker_ = std::thread(&Decoder::Work, this);
    }

    // One pass: decode until the hardware buffer is drained, handing
    // each decoded image to the worker. `final`: sampling has stopped.
    CUptiResult Pass(bool final = false) {
        for (int step = 0; step < kMaxDecodeStepsPerPass; ++step) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !busy_[active_] || workerResult_ != CUPTI_SUCCESS; });
                if (workerResult_ != CUPTI_SUCCESS) return workerResult_;
            }
            CuptiPmSampling::DecodeOutcome o;
            {
                std::lock_guard<std::mutex> g(cupti_);
                o = target_.DecodeData(images_[active_]);
            }
            stats_.decodeCalls.fetch_add(1, std::memory_order_relaxed);
            const bool oom = o.result == CUPTI_ERROR_OUT_OF_MEMORY;
            if (oom || o.overflow) {
                stats_.hwBufferOverflows.fetch_add(1, std::memory_order_relaxed);
                if (oom) return final ? CUPTI_SUCCESS : Restart();
            }
            if (o.result != CUPTI_SUCCESS) return o.result;

            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_[active_] = true;
                queue_.push_back(active_);
            }
            cv_.notify_all();
            active_ ^= 1;
            if (o.overflow && !final) return Restart();

            if (o.stopReason != CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL)
                return CUPTI_SUCCESS;
            stats_.counterDataFull.fetch_add(1, std::memory_order_relaxed);
            // One line per pass that overflowed the image, at most one per
            // 30 s (an undersized max_samples overflows every pass); the
            // summary at stop has the count.
            const auto now = std::chrono::steady_clock::now();
            if (step == 0 && (fullWarnings_ == 0 || now - lastFullWarning_ >= std::chrono::seconds(30))) {
                ++fullWarnings_;
                lastFullWarning_ = now;
                std::fprintf(stderr,
                    "[cupti-profiler] warning: GPU %u: the counter-data image (max_samples) filled up "
                    "in a decode pass (the decode thread fell behind, or max_samples is too small); "
                    "the samples beyond it are lost (counted)\n", device_.gpuIndex);
            }
        }
        return CUPTI_SUCCESS;
    }

    // A pass is about to start `sinceLastNs` after the previous one.
    // When the samples waiting by now fill more than 3/4 of an image,
    // the decode thread is falling behind: say so before an image
    // overflows (one line, at most one per 30 s; counted).
    void CheckBehind(uint64_t sinceLastNs) {
        const uint64_t period = device_.samplingIntervalNs;
        if (period == 0 || device_.maxSamples == 0) return;
        const uint64_t pending = sinceLastNs / period;
        if (pending * 4 <= device_.maxSamples * 3) return;
        stats_.latePasses.fetch_add(1, std::memory_order_relaxed);
        const auto now = std::chrono::steady_clock::now();
        if (lateWarnings_ != 0 && now - lastLateWarning_ < std::chrono::seconds(30)) return;
        ++lateWarnings_;
        lastLateWarning_ = now;
        std::fprintf(stderr,
            "[cupti-profiler] warning: GPU %u: decode pass %.1f s after the previous one "
            "(%llu samples waiting, %.0f%% of the counter-data image): the decode thread is falling "
            "behind; a longer delay loses samples\n", device_.gpuIndex, sinceLastNs / 1e9,
            static_cast<unsigned long long>(pending), 100.0 * pending / device_.maxSamples);
    }

    // Stop sampling (the decode thread owns the sampler: a restart must
    // not race Stop()).
    CUptiResult StopSampling() {
        std::lock_guard<std::mutex> g(cupti_);
        return target_.Stop();
    }

    // After the last pass: the worker evaluates what is queued, then ends.
    CUptiResult Finish() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        return workerResult_;
    }

    ~Decoder() { Finish(); }

private:
    // After a hardware-buffer overflow: disable and re-enable the sampler
    // (a new PM sampling object, same config, fresh images), once the
    // worker is done with the images.
    CUptiResult Restart() {
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return (queue_.empty() && !busy_[0] && !busy_[1]) ||
                                      workerResult_ != CUPTI_SUCCESS; });
            if (workerResult_ != CUPTI_SUCCESS) return workerResult_;
        }
        {
            std::lock_guard<std::mutex> g(cupti_);
            target_.Stop();   // an overflowed sampler may refuse; it is disabled next
            if (auto r = target_.DisablePmSampling(); r != CUPTI_SUCCESS) return r;
            if (auto r = target_.EnablePmSampling(device_.deviceIndex); r != CUPTI_SUCCESS) return r;
            auto& cfg = const_cast<std::vector<uint8_t>&>(*device_.configImage);
            if (auto r = target_.SetConfig(cfg, device_.hwBufferSize, device_.samplingIntervalNs);
                r != CUPTI_SUCCESS) return r;
            for (auto& img : images_) {
                if (auto r = target_.CreateCounterDataImage(device_.maxSamples, metrics_, img);
                    r != CUPTI_SUCCESS) return r;
            }
            if (auto r = target_.Start(); r != CUPTI_SUCCESS) return r;
        }
        active_ = 0;
        stats_.samplerRestarts.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[cupti-profiler] warning: GPU %u: the PM sampling hardware buffer overflowed (the decode "
            "thread fell too far behind, or hw_buffer_size is too small); sampler re-enabled in "
            "%.0f ms, the samples since the last decode are lost (counted)\n",
            device_.gpuIndex,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        return CUPTI_SUCCESS;
    }

    void Work() {
        char name[16];
        lifecycle::BlockSignalsInThisThread();
        std::snprintf(name, sizeof(name), "cupti-eval%u", device_.gpuIndex);
        ::pthread_setname_np(::pthread_self(), name);
        for (;;) {
            int idx;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !queue_.empty() || quit_; });
                if (queue_.empty()) return;
                idx = queue_.front();
                queue_.pop_front();
            }
            const CUptiResult r = Evaluate(images_[idx]);
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_[idx] = false;
                if (r != CUPTI_SUCCESS && workerResult_ == CUPTI_SUCCESS) workerResult_ = r;
            }
            cv_.notify_all();
        }
    }

    // Evaluate every completed sample of the image, then re-initialize it.
    CUptiResult Evaluate(std::vector<uint8_t>& image) {
        CUpti_PmSampling_GetCounterDataInfo_Params info = {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        info.pCounterDataImage = image.data();
        info.counterDataImageSize = image.size();
        {
            std::lock_guard<std::mutex> g(cupti_);
            if (auto r = cuptiPmSamplingGetCounterDataInfo(&info); r != CUPTI_SUCCESS) return r;
        }
        for (size_t i = 0; i < info.numCompletedSamples; ++i) {
            SamplerRange sr;
            {
                std::lock_guard<std::mutex> g(cupti_);
                if (auto r = host_.EvaluateCounterData(target_.GetPmSamplerObject(), i, metrics_, image, sr);
                    r != CUPTI_SUCCESS) return r;
            }
            Keep(std::move(sr));
        }
        std::lock_guard<std::mutex> g(cupti_);
        return target_.ResetCounterDataImage(image);
    }

    // Drop empty and invalid samples; count the ones missing between two
    // kept ones. Worker thread only.
    void Keep(SamplerRange&& sr) {
        // Zero length, at a real time: CUPTI 13.3 returns one or two,
        // stamped at the start of sampling, in the decode after
        // cuptiPmSamplingStop. They cover no time; not a loss.
        if (sr.startTimestamp == sr.endTimestamp && sr.startTimestamp != 0) {
            stats_.emptySamples.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if ((sr.startTimestamp == 0 && sr.endTimestamp == 0) ||
            (lastEndNs_ != 0 && sr.endTimestamp <= lastEndNs_)) {
            stats_.invalidSamples.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const uint64_t period = device_.samplingIntervalNs;
        // Longer than a sampling interval: CUPTI's first sample after a
        // (re)start covers everything since the sampler last delivered
        // one (~0.3 s at the start of a run; the whole stall after a
        // re-enable). Its values average over that window; drop it, and
        // count the window as lost (except before the first sample).
        if (period > 0 && sr.endTimestamp - sr.startTimestamp > period * 3 / 2) {
            stats_.stretchedSamples.fetch_add(1, std::memory_order_relaxed);
            if (lastEndNs_ != 0)
                stats_.samplesLost.fetch_add((sr.endTimestamp - lastEndNs_ + period / 2) / period,
                                             std::memory_order_relaxed);
            lastEndNs_ = sr.endTimestamp;
            return;
        }
        if (lastEndNs_ != 0 && period > 0 && sr.startTimestamp > lastEndNs_) {
            const uint64_t missing = (sr.startTimestamp - lastEndNs_ + period / 2) / period;
            if (missing > 0) {
                stats_.samplesLost.fetch_add(missing, std::memory_order_relaxed);
                if (!warnedLost_) {
                    warnedLost_ = true;
                    std::fprintf(stderr,
                        "[cupti-profiler] warning: GPU %u: %llu sample(s) missing between two decoded "
                        "samples (%.3f ms gap)\n", device_.gpuIndex,
                        static_cast<unsigned long long>(missing),
                        (sr.startTimestamp - lastEndNs_) / 1e6);
                }
            }
        }
        lastEndNs_ = sr.endTimestamp;
        stats_.samples.fetch_add(1, std::memory_order_relaxed);
        host_.PushSample(std::move(sr));
    }

    std::array<std::vector<uint8_t>, 2>& images_;
    const std::vector<const char*>& metrics_;
    CuptiPmSampling& target_;
    CuptiProfilerHost& host_;
    DecodeTarget device_;
    DecodeStats& stats_;

    std::mutex cupti_;            // every CUPTI call of this device

    std::mutex mu_;               // guards the hand-off state below
    std::condition_variable cv_;
    std::deque<int> queue_;       // decoded images, oldest first
    bool busy_[2] = {false, false};
    bool quit_ = false;
    CUptiResult workerResult_ = CUPTI_SUCCESS;

    int active_ = 0;              // decode thread only
    uint64_t lastEndNs_ = 0;      // worker only
    uint64_t fullWarnings_ = 0;
    uint64_t lateWarnings_ = 0;
    std::chrono::steady_clock::time_point lastLateWarning_{};
    std::chrono::steady_clock::time_point lastFullWarning_{};
    bool warnedLost_ = false;
    std::thread worker_;
};

} // namespace

void DecodeThreadFunc(std::array<std::vector<uint8_t>, 2>& counterDataImages,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      DecodeTarget device,
                      DecodeStats& stats,
                      StopSignal& stop,
                      CUptiResult& result)
{
    char name[16];
    lifecycle::BlockSignalsInThisThread();
    std::snprintf(name, sizeof(name), "cupti-decode%u", device.gpuIndex);
    ::pthread_setname_np(::pthread_self(), name);

    Decoder decoder(counterDataImages, metricsList, target, host, device, stats);
    // A pass every decode interval, on a fixed schedule; Stop() cuts the
    // wait short.
    const auto interval = std::chrono::milliseconds(device.decodeIntervalMs);
    auto next = std::chrono::steady_clock::now() + interval;
    auto lastPass = std::chrono::steady_clock::now();
    result = CUPTI_SUCCESS;
    while (!stop.WaitUntil(next)) {
        PassDecodeStall();   // test-only; see <cupti_profiler/testing.h>
        const auto passStart = std::chrono::steady_clock::now();
        decoder.CheckBehind(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(passStart - lastPass).count()));
        lastPass = passStart;
        result = decoder.Pass();
        if (result != CUPTI_SUCCESS) break;
        next += interval;
        const auto now = std::chrono::steady_clock::now();
        if (next < now) next = now + interval;   // overran: don't burst
    }
    // Stop sampling, then decode what is left.
    const CUptiResult stopped = decoder.StopSampling();
    if (result == CUPTI_SUCCESS) result = stopped;
    if (result == CUPTI_SUCCESS) result = decoder.Pass(/*final=*/true);
    const CUptiResult worker = decoder.Finish();
    if (result == CUPTI_SUCCESS) result = worker;
}

void ReportDecodeSummary(uint32_t gpuIndex, const DecodeStats& stats) {
    const auto full     = stats.counterDataFull.load();
    const auto invalid  = stats.invalidSamples.load();
    const auto lost     = stats.samplesLost.load();
    const auto overflow = stats.hwBufferOverflows.load();
    const auto restarts = stats.samplerRestarts.load();
    const auto late     = stats.latePasses.load();
    if (full == 0 && invalid == 0 && lost == 0 && overflow == 0 && late == 0) return;
    std::fprintf(stderr,
        "[cupti-profiler] warning: GPU %u decode summary: %llu samples kept, %llu missing, "
        "%llu invalid dropped; counter-data image full %llu time(s), hardware buffer "
        "overflow %llu time(s), sampler re-enabled %llu time(s), late decode passes %llu\n", gpuIndex,
        static_cast<unsigned long long>(stats.samples.load()),
        static_cast<unsigned long long>(lost), static_cast<unsigned long long>(invalid),
        static_cast<unsigned long long>(full), static_cast<unsigned long long>(overflow),
        static_cast<unsigned long long>(restarts), static_cast<unsigned long long>(late));
}

} // namespace internal
} // namespace cupti_profiler
