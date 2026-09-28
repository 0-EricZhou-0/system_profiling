#include "decode_thread.h"

#include <chrono>
#include <cstdio>

namespace cupti_profiler {
namespace internal {

namespace {

// A pass stops at END_OF_RECORDS. This only bounds a CUPTI that never
// says so; every step consumes hardware-buffer records.
constexpr int kMaxDecodeStepsPerPass = 1 << 16;

// Drains the hardware buffer into the counter-data image, evaluates
// every completed sample and re-initializes the image.
//
// What CUPTI 13.3 does (measured on H100, sidecar-impl phase 7):
//   * A decode that stops at END_OF_RECORDS (or OTHER) got everything
//     the hardware buffer held. Samples are complete and contiguous
//     (each one starts where the previous one ended) across passes;
//     a sample still being written stays in the buffer.
//   * A decode that stops at COUNTER_DATA_FULL left records behind.
//     Decoding again (after re-initializing the image) catches the
//     buffer up, but the samples of those next decodes come back with
//     both timestamps 0 and the real samples in between are gone. No
//     choice of image avoids it; the image must be large enough for a
//     whole pass (max_samples). Those samples are dropped and counted.
//   * A hardware-buffer overflow makes this and every later decode
//     fail with CUPTI_ERROR_OUT_OF_MEMORY and return nothing.
class Decoder {
public:
    Decoder(std::vector<uint8_t>& image, const std::vector<const char*>& metrics,
            CuptiPmSampling& target, CuptiProfilerHost& host,
            DecodeTarget device, DecodeStats& stats)
        : image_(image), metrics_(metrics), target_(target), host_(host),
          device_(device), stats_(stats) {}

    CUptiResult Pass() {
        for (int step = 0; step < kMaxDecodeStepsPerPass; ++step) {
            const auto o = target_.DecodeData(image_);
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
                if (oom) return CUPTI_SUCCESS;
            }
            if (o.result != CUPTI_SUCCESS) return o.result;

            CUpti_PmSampling_GetCounterDataInfo_Params info = {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
            info.pCounterDataImage = image_.data();
            info.counterDataImageSize = image_.size();
            if (auto r = cuptiPmSamplingGetCounterDataInfo(&info); r != CUPTI_SUCCESS) return r;
            for (size_t i = 0; i < info.numCompletedSamples; ++i) {
                SamplerRange sr;
                if (auto r = host_.EvaluateCounterData(target_.GetPmSamplerObject(), i, metrics_, image_, sr);
                    r != CUPTI_SUCCESS) return r;
                Keep(std::move(sr));
            }
            if (auto r = target_.ResetCounterDataImage(image_); r != CUPTI_SUCCESS) return r;

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

private:
    // Drop empty and invalid samples; count the ones missing between two
    // kept ones.
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

    std::vector<uint8_t>& image_;
    const std::vector<const char*>& metrics_;
    CuptiPmSampling& target_;
    CuptiProfilerHost& host_;
    DecodeTarget device_;
    DecodeStats& stats_;
    uint64_t lastEndNs_ = 0;
    bool warnedOverflow_ = false;
    bool warnedFull_ = false;
    bool warnedLost_ = false;
};

} // namespace

void DecodeThreadFunc(std::vector<uint8_t>& counterDataImage,
                      const std::vector<const char*>& metricsList,
                      CuptiPmSampling& target,
                      CuptiProfilerHost& host,
                      DecodeTarget device,
                      DecodeStats& stats,
                      StopSignal& stop,
                      CUptiResult& result)
{
    Decoder decoder(counterDataImage, metricsList, target, host, device, stats);
    // A pass every decode interval, on a fixed schedule; Stop() cuts the
    // wait short.
    const auto interval = std::chrono::milliseconds(device.decodeIntervalMs);
    auto next = std::chrono::steady_clock::now() + interval;
    while (!stop.WaitUntil(next)) {
        result = decoder.Pass();
        if (result != CUPTI_SUCCESS) return;
        next += interval;
        const auto now = std::chrono::steady_clock::now();
        if (next < now) next = now + interval;   // overran: don't burst
    }
    // Final drain (sampling has been stopped).
    result = decoder.Pass();
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
