#include "decode_thread.h"

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
// What CUPTI 13.3 does (measured on H100, sidecar-impl phase 7):
//   * A decode that stops at END_OF_RECORDS (or OTHER) got everything
//     the hardware buffer held. Samples are complete and contiguous
//     (each one starts where the previous one ended) across passes;
//     a sample still being written stays in the buffer.
//   * A decode that stops at COUNTER_DATA_FULL left records behind.
//     Decoding again catches the buffer up, but the samples of those next
//     decodes come back with both timestamps 0 and the real samples in
//     between are gone, whatever image is used. The image must be large
//     enough for a whole pass (max_samples). Those samples are dropped
//     and counted.
//   * A hardware-buffer overflow makes this and every later decode
//     fail with CUPTI_ERROR_OUT_OF_MEMORY and return nothing.
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
    // each decoded image to the worker.
    CUptiResult Pass() {
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
                if (!warnedOverflow_) {
                    warnedOverflow_ = true;
                    std::fprintf(stderr,
                        "[cupti-profiler] warning: GPU %u: the PM sampling hardware buffer overflowed; "
                        "CUPTI returns no samples from now on. Raise hw_buffer_size, or lower "
                        "decode_interval_ms or the sampling rate\n", device_.gpuIndex);
                }
                if (oom) return CUPTI_SUCCESS;   // nothing decoded; image unchanged
            }
            if (o.result != CUPTI_SUCCESS) return o.result;

            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_[active_] = true;
                queue_.push_back(active_);
            }
            cv_.notify_all();
            active_ ^= 1;

            if (o.stopReason != CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL)
                return CUPTI_SUCCESS;
            stats_.counterDataFull.fetch_add(1, std::memory_order_relaxed);
            if (!warnedFull_) {
                warnedFull_ = true;
                std::fprintf(stderr,
                    "[cupti-profiler] warning: GPU %u: the counter-data image (max_samples) filled up "
                    "within one decode pass; CUPTI loses samples after that. Raise max_samples "
                    "(0 = sized for the decode interval)\n", device_.gpuIndex);
            }
        }
        return CUPTI_SUCCESS;
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
    void Work() {
        char name[16];
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
    bool warnedOverflow_ = false;
    bool warnedFull_ = false;
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
    std::snprintf(name, sizeof(name), "cupti-decode%u", device.gpuIndex);
    ::pthread_setname_np(::pthread_self(), name);

    Decoder decoder(counterDataImages, metricsList, target, host, device, stats);
    // A pass every decode interval, on a fixed schedule; Stop() cuts the
    // wait short.
    const auto interval = std::chrono::milliseconds(device.decodeIntervalMs);
    auto next = std::chrono::steady_clock::now() + interval;
    result = CUPTI_SUCCESS;
    while (!stop.WaitUntil(next)) {
        result = decoder.Pass();
        if (result != CUPTI_SUCCESS) break;
        next += interval;
        const auto now = std::chrono::steady_clock::now();
        if (next < now) next = now + interval;   // overran: don't burst
    }
    // Final drain (sampling has been stopped).
    if (result == CUPTI_SUCCESS) result = decoder.Pass();
    const CUptiResult worker = decoder.Finish();
    if (result == CUPTI_SUCCESS) result = worker;
}

void ReportDecodeSummary(uint32_t gpuIndex, const DecodeStats& stats) {
    const auto full     = stats.counterDataFull.load();
    const auto invalid  = stats.invalidSamples.load();
    const auto lost     = stats.samplesLost.load();
    const auto overflow = stats.hwBufferOverflows.load();
    if (full == 0 && invalid == 0 && lost == 0 && overflow == 0) return;
    std::fprintf(stderr,
        "[cupti-profiler] warning: GPU %u decode summary: %llu samples kept, %llu missing, "
        "%llu invalid dropped; counter-data image full %llu time(s), hardware buffer "
        "overflow %llu time(s)\n", gpuIndex,
        static_cast<unsigned long long>(stats.samples.load()),
        static_cast<unsigned long long>(lost), static_cast<unsigned long long>(invalid),
        static_cast<unsigned long long>(full), static_cast<unsigned long long>(overflow));
}

} // namespace internal
} // namespace cupti_profiler
