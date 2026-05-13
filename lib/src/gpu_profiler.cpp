#include <cupti_profiler/gpu_profiler.h>

#include "cupti_pm_sampling.h"
#include "profiler_host_internal.h"
#include "decode_thread.h"
#include "flush_thread.h"
#include "helper_cupti.h"

#include "gpu_metrics.pb.h"
#include "metric_sample.pb.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvml.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

// Declared here to avoid pulling in cupti_activity.h
extern "C" CUptiResult cuptiGetTimestamp(uint64_t *timestamp);

namespace cupti_profiler {
namespace {

// PCIe per-generation spec parameters: raw line rate (Gb/s/lane) and
// line-coding ratio (useful bits / transmitted bits). Only the underlying
// spec values are kept here — per-lane bandwidth is derived.
struct PcieGenSpec {
    double gtPerSec;
    double codingRatio;
};
constexpr PcieGenSpec PCIE_GEN_SPECS[] = {
    { 0.0,   0.0          }, // [0] sentinel
    { 2.5,   8.0 / 10.0   }, // Gen1: 8b/10b
    { 5.0,   8.0 / 10.0   }, // Gen2: 8b/10b
    { 8.0,   128.0 / 130.0 }, // Gen3: 128b/130b
    {16.0,   128.0 / 130.0 }, // Gen4: 128b/130b
    {32.0,   128.0 / 130.0 }, // Gen5: 128b/130b
    {64.0,   242.0 / 256.0 }, // Gen6: PAM-4 + FEC (effective payload ratio)
};

// Per-lane bytes/sec for a given PCIe generation, derived from spec.
double PcieLaneBytesPerSec(unsigned int gen) {
    constexpr unsigned int N = sizeof(PCIE_GEN_SPECS) / sizeof(PCIE_GEN_SPECS[0]);
    if (gen < 1 || gen >= N) return 0.0;
    const auto& s = PCIE_GEN_SPECS[gen];
    return s.gtPerSec * 1e9 * s.codingRatio / 8.0;
}

// NVLink per-generation spec parameters: per-sub-lane bit rate (Gb/s) and
// number of sub-lanes per link. Per-link-per-direction bandwidth is derived.
// (NVLink Gen3+ uses PAM-4 modulation; the laneGbps figure is the effective
// payload bit rate per sub-lane.)
struct NvLinkGenSpec {
    double laneGbps;
    unsigned int lanesPerLink;
};
constexpr NvLinkGenSpec NVLINK_GEN_SPECS[] = {
    {  0.0, 0 }, // [0] sentinel
    { 20.0, 8 }, // NVLink 1 (P100):  8 sub-lanes × 20 Gb/s NRZ
    { 25.0, 8 }, // NVLink 2 (V100):  8 sub-lanes × 25 Gb/s NRZ
    { 50.0, 4 }, // NVLink 3 (A100):  4 sub-lanes × 50 Gb/s PAM-4
    {100.0, 2 }, // NVLink 4 (H100):  2 sub-lanes × 100 Gb/s PAM-4
};

// Per-link bytes/sec/direction for a given NVLink generation.
// Note: NVML's `nvmlDeviceGetNvLinkVersion` does not always match the marketing
// generation. Observed example: H100 NVL hardware (NVLink Gen 4) reports NVML
// version 6. When the reported version exceeds the known table, we fall back
// to the highest known spec and emit a warning.
double NvLinkLinkBytesPerSec(unsigned int gen) {
    constexpr unsigned int N = sizeof(NVLINK_GEN_SPECS) / sizeof(NVLINK_GEN_SPECS[0]);
    if (gen < 1) return 0.0;
    if (gen >= N) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[GPU] NVML reported NVLink version " << gen
                      << ", beyond known spec table (1-" << (N - 1)
                      << "). Falling back to NVLink " << (N - 1) << " spec.\n";
            warned = true;
        }
        gen = N - 1;
    }
    const auto& s = NVLINK_GEN_SPECS[gen];
    return s.laneGbps * 1e9 * s.lanesPerLink / 8.0;
}

} // namespace

class GpuProfiler::Impl {
public:
    ProfilerConfig config;

    // Host info (captured at Configure() — written into TraceHeader)
    std::string hostname;
    uint32_t hostCpuCount = 0;

    // Per-device state. One DeviceState per index in
    // config.deviceIndices (or just {0} if empty). All devices share
    // the same metric set + sampling frequency; each owns its own
    // CUPTI PM Sampling session + decode thread.
    struct DeviceState {
        int                          deviceIndex = 0;
        internal::CuptiPmSampling    target;
        internal::CuptiProfilerHost  host;
        std::vector<uint8_t>         configImage;
        std::vector<uint8_t>         counterDataImage;
        std::string                  deviceName;
        std::string                  chipName;
        double                       peakDramBwGbps = 0.0;
        double                       peakDramBwBytesPerSec = 0.0;
        double                       peakPcieBwBytesPerSec = 0.0;
        double                       peakNvlinkBwBytesPerSec = 0.0;
        std::thread                  decodeThread;
        std::atomic<bool>            stopDecode{false};
        CUptiResult                  decodeResult = CUPTI_SUCCESS;
    };
    // unique_ptr because DeviceState holds an std::atomic which is
    // non-movable — std::vector resize would otherwise invalidate.
    std::vector<std::unique_ptr<DeviceState>> devices;

    // Shared across all devices (same metric set).
    std::vector<const char*> metricsCstr;

    // Single flush thread, one output file.
    std::thread flushThread;
    std::atomic<bool> stopFlush{false};
    std::ofstream outFile;
    std::mutex outMutex;

    internal::PendingFlushStats flushStatsPending;
    std::mutex flushStatsMutex;

    // Sync anchor (recorded at Start())
    uint64_t steadyClockRefNs = 0;
    uint64_t cuptiRefNs = 0;
    uint64_t wallClockEpochNs = 0;

    bool configured = false;
    bool running = false;
};

GpuProfiler::GpuProfiler() : m_impl(std::make_unique<Impl>()) {}
GpuProfiler::~GpuProfiler() {
    if (m_impl && m_impl->running) {
        Stop();
    }
}
GpuProfiler::GpuProfiler(GpuProfiler&&) noexcept = default;
GpuProfiler& GpuProfiler::operator=(GpuProfiler&&) noexcept = default;

void GpuProfiler::Configure(const ProfilerConfig& config) {
    m_impl->config = config;

    // Capture host context for the TraceHeader.
    char hostbuf[256] = {0};
    gethostname(hostbuf, sizeof(hostbuf));
    m_impl->hostname = hostbuf;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    m_impl->hostCpuCount = (nproc > 0) ? static_cast<uint32_t>(nproc) : 0;

    DRIVER_API_CALL(cuInit(0));

    // Build the shared metric list once.
    m_impl->metricsCstr.clear();
    for (const auto& m : config.metrics) {
        m_impl->metricsCstr.push_back(m.c_str());
    }

    // Resolve device list — empty defaults to {0}.
    std::vector<int> indices = config.deviceIndices;
    if (indices.empty()) indices.push_back(0);

    std::cout << "Sampling frequency: " << config.samplingFrequencyHz << " Hz\n";
    std::cout << "Metrics: " << config.metrics.size() << "\n";
    for (const auto& m : config.metrics) std::cout << "  " << m << "\n";
    std::cout << "Devices: " << indices.size() << "\n";

    m_impl->devices.clear();
    m_impl->devices.reserve(indices.size());
    for (int idx : indices) {
        auto state = std::make_unique<Impl::DeviceState>();
        Impl::DeviceState& d = *state;
        d.deviceIndex = idx;

        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, idx));
        {
            CUpti_Profiler_DeviceSupported_Params p =
                {CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
            p.cuDevice = cuDevice;
            p.api = CUPTI_PROFILER_PM_SAMPLING;
            CUPTI_API_CALL(cuptiProfilerDeviceSupported(&p));
            if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
                std::cerr << "PM sampling not supported on device " << idx << "\n";
                exit(1);
            }
        }

        char devName[256];
        DRIVER_API_CALL(cuDeviceGetName(devName, sizeof(devName), cuDevice));
        d.deviceName = devName;
        std::cout << "Device " << idx << ": " << d.deviceName << "\n";

        std::vector<uint8_t> counterAvailImage;
        internal::CuptiPmSampling::GetChipName(idx, d.chipName);
        internal::CuptiPmSampling::GetCounterAvailabilityImage(idx, counterAvailImage);
        std::cout << "  Chip: " << d.chipName << "\n";

        d.host.SetUp(d.chipName, counterAvailImage);
        CUPTI_API_CALL(d.host.CreateConfigImage(m_impl->metricsCstr, d.configImage));

        d.target.SetUp(idx);
        CUPTI_API_CALL(d.target.EnablePmSampling(idx));
        uint64_t intervalNs = static_cast<uint64_t>(1e9 / config.samplingFrequencyHz);
        CUPTI_API_CALL(d.target.SetConfig(d.configImage, config.hwBufferSize, intervalNs));
        CUPTI_API_CALL(d.target.CreateCounterDataImage(
            config.maxSamples, m_impl->metricsCstr, d.counterDataImage));

        cudaDeviceProp prop;
        RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, idx));
        d.peakDramBwGbps        = (double)prop.memoryClockRate * 1e3
                                  * (prop.memoryBusWidth / 8) * 2 / 1e9;
        d.peakDramBwBytesPerSec = d.peakDramBwGbps * 1e9;
        std::cout << "  Peak DRAM BW: " << std::fixed << std::setprecision(1)
                  << d.peakDramBwGbps << " GB/s\n";

        // PCIe + NVLink peaks via NVML — best-effort, no fatal on failure.
        if (nvmlInit_v2() == NVML_SUCCESS) {
            nvmlDevice_t dev;
            if (nvmlDeviceGetHandleByIndex_v2(idx, &dev) == NVML_SUCCESS) {
                unsigned int gen = 0, width = 0;
                if (nvmlDeviceGetMaxPcieLinkGeneration(dev, &gen) == NVML_SUCCESS &&
                    nvmlDeviceGetMaxPcieLinkWidth(dev, &width) == NVML_SUCCESS) {
                    d.peakPcieBwBytesPerSec = PcieLaneBytesPerSec(gen) * width;
                    double gibps = d.peakPcieBwBytesPerSec / (1024.0 * 1024.0 * 1024.0);
                    std::cout << "  Peak PCIe BW: " << std::fixed << std::setprecision(2)
                              << gibps << " GiB/s per direction (Gen"
                              << gen << " x" << width << ")\n";
                }
                unsigned int activeLinks = 0;
                unsigned int linkVersion = 0;
                double totalBytesPerSec  = 0.0;
                for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
                    nvmlEnableState_t isActive;
                    nvmlReturn_t sr = nvmlDeviceGetNvLinkState(dev, link, &isActive);
                    if (sr == NVML_ERROR_INVALID_ARGUMENT ||
                        sr == NVML_ERROR_NOT_SUPPORTED) break;
                    if (sr != NVML_SUCCESS) continue;
                    if (isActive != NVML_FEATURE_ENABLED) continue;
                    unsigned int v = 0;
                    if (nvmlDeviceGetNvLinkVersion(dev, link, &v) != NVML_SUCCESS) continue;
                    linkVersion = v;
                    totalBytesPerSec += NvLinkLinkBytesPerSec(v);
                    ++activeLinks;
                }
                if (activeLinks > 0) {
                    d.peakNvlinkBwBytesPerSec = totalBytesPerSec;
                    double gibps = totalBytesPerSec / (1024.0 * 1024.0 * 1024.0);
                    std::cout << "  Peak NVLink BW: " << std::fixed << std::setprecision(2)
                              << gibps << " GiB/s per direction (NVLink"
                              << linkVersion << " × " << activeLinks
                              << " active links)\n";
                }
            }
            nvmlShutdown();
        } else {
            std::cerr << "[GPU] NVML init failed for device " << idx
                      << "; skipping peak PCIe/NVLink BW query.\n";
        }

        m_impl->devices.push_back(std::move(state));
    }

    m_impl->configured = true;
}

void GpuProfiler::Start() {
    if (!m_impl->configured) {
        std::cerr << "GpuProfiler::Start() called before Configure()\n";
        exit(1);
    }

    // Open output file if configured
    if (!m_impl->config.outputFile.empty()) {
        m_impl->outFile.open(m_impl->config.outputFile, std::ios::binary | std::ios::trunc);
        if (!m_impl->outFile) {
            std::cerr << "Failed to open output file: " << m_impl->config.outputFile << "\n";
            exit(1);
        }
    }

    // Record sync anchor: all clocks at the same moment
    cuptiGetTimestamp(&m_impl->cuptiRefNs);
    m_impl->steadyClockRefNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_impl->wallClockEpochNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Launch one decode thread per device.
    for (auto& d : m_impl->devices) {
        d->stopDecode   = false;
        d->decodeResult = CUPTI_SUCCESS;
        d->decodeThread = std::thread(internal::DecodeThreadFunc,
                                       std::ref(d->counterDataImage),
                                       std::ref(m_impl->metricsCstr),
                                       std::ref(d->target),
                                       std::ref(d->host),
                                       std::ref(d->stopDecode),
                                       std::ref(d->decodeResult));
    }

    // Launch one flush thread that pulls from every device.
    m_impl->stopFlush = false;
    if (m_impl->config.flushIntervalMs > 0 && m_impl->outFile.is_open()) {
        std::cout << "Periodic flush every " << m_impl->config.flushIntervalMs << " ms\n";
        std::vector<internal::DeviceDrainSlot> slots;
        slots.reserve(m_impl->devices.size());
        for (auto& d : m_impl->devices) {
            internal::DeviceDrainSlot s;
            s.gpu_index                  = static_cast<uint32_t>(d->deviceIndex);
            s.device_name                = d->deviceName;
            s.chip_name                  = d->chipName;
            s.peak_dram_bw_bytes_per_s   = &d->peakDramBwBytesPerSec;
            s.peak_pcie_bw_bytes_per_s   = &d->peakPcieBwBytesPerSec;
            s.peak_nvlink_bw_bytes_per_s = &d->peakNvlinkBwBytesPerSec;
            s.host                       = &d->host;
            slots.push_back(std::move(s));
        }
        m_impl->flushThread = std::thread(internal::FlushThreadFunc,
                                           std::move(slots),
                                           std::ref(m_impl->outFile),
                                           std::ref(m_impl->outMutex),
                                           std::cref(m_impl->hostname),
                                           m_impl->config.samplingFrequencyHz,
                                           m_impl->hostCpuCount,
                                           std::cref(m_impl->metricsCstr),
                                           std::ref(m_impl->stopFlush),
                                           m_impl->config.flushIntervalMs,
                                           m_impl->steadyClockRefNs,
                                           m_impl->cuptiRefNs,
                                           m_impl->wallClockEpochNs,
                                           std::ref(m_impl->flushStatsPending),
                                           std::ref(m_impl->flushStatsMutex));
    }

    // Start PM sampling on every device.
    for (auto& d : m_impl->devices) {
        CUPTI_API_CALL(d->target.Start());
    }
    m_impl->running = true;

    std::cout << "\n=== PM sampling started at "
              << m_impl->config.samplingFrequencyHz << " Hz across "
              << m_impl->devices.size() << " device(s) ===\n\n";
}

void GpuProfiler::Stop() {
    if (!m_impl->running) return;

    // Stop sampling on every device.
    for (auto& d : m_impl->devices) {
        CUPTI_API_CALL(d->target.Stop());
    }

    // Join decode threads.
    for (auto& d : m_impl->devices) {
        d->stopDecode = true;
        if (d->decodeThread.joinable()) d->decodeThread.join();
    }

    // Join flush thread.
    m_impl->stopFlush = true;
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    for (auto& d : m_impl->devices) {
        if (d->decodeResult != CUPTI_SUCCESS) {
            const char* errstr;
            cuptiGetResultString(d->decodeResult, &errstr);
            std::cerr << "Decode thread error on device " << d->deviceIndex << ": "
                      << errstr << "\n";
        }
    }

    // Write the final residual flush — one trace covering every device's
    // remaining samples.
    if (m_impl->outFile.is_open()) {
        std::vector<internal::GpuDevicePayload> payloads;
        payloads.reserve(m_impl->devices.size());
        size_t totalRemaining = 0;
        for (auto& d : m_impl->devices) {
            internal::GpuDevicePayload p;
            p.gpu_index                  = static_cast<uint32_t>(d->deviceIndex);
            p.device_name                = d->deviceName;
            p.chip_name                  = d->chipName;
            p.peak_dram_bw_bytes_per_s   = d->peakDramBwBytesPerSec;
            p.peak_pcie_bw_bytes_per_s   = d->peakPcieBwBytesPerSec;
            p.peak_nvlink_bw_bytes_per_s = d->peakNvlinkBwBytesPerSec;
            p.samples                    = d->host.DrainSamples();
            totalRemaining              += p.samples.size();
            payloads.push_back(std::move(p));
        }
        std::cout << "Remaining samples after flush: " << totalRemaining << "\n";

        if (totalRemaining > 0 || m_impl->flushStatsPending.valid) {
            GPUMetricsTrace finalTrace = internal::BuildTrace(
                m_impl->hostname, m_impl->config.samplingFrequencyHz, m_impl->hostCpuCount,
                m_impl->metricsCstr, payloads,
                m_impl->steadyClockRefNs, m_impl->cuptiRefNs, m_impl->wallClockEpochNs);
            if (m_impl->flushStatsPending.valid) {
                auto* fs = finalTrace.add_flush_stats();
                fs->set_flush_byte_size(m_impl->flushStatsPending.bytesWritten);
                fs->set_flush_interval_ns(m_impl->flushStatsPending.intervalNs);
                m_impl->flushStatsPending.valid = false;
            }
            std::lock_guard<std::mutex> lock(m_impl->outMutex);
            internal::WriteDelimitedToSized(finalTrace, m_impl->outFile);
            m_impl->outFile.flush();
        }
        m_impl->outFile.close();
        std::cout << "Wrote trace to " << m_impl->config.outputFile << "\n";
    }

    // Cleanup CUPTI per device.
    for (auto& d : m_impl->devices) {
        CUPTI_API_CALL(d->target.DisablePmSampling());
        d->target.TearDown();
        d->host.TearDown();
    }

    m_impl->running = false;
}

std::vector<SamplerRange> GpuProfiler::DrainSamples() {
    // Single-device convenience: returns the first configured device's
    // samples. Multi-device readers should consume the on-disk trace.
    if (m_impl->devices.empty()) return {};
    return m_impl->devices.front()->host.DrainSamples();
}

std::string GpuProfiler::GetDeviceName() const {
    return m_impl->devices.empty() ? std::string{} : m_impl->devices.front()->deviceName;
}
std::string GpuProfiler::GetChipName() const {
    return m_impl->devices.empty() ? std::string{} : m_impl->devices.front()->chipName;
}
double GpuProfiler::GetPeakDramBwGbps() const {
    return m_impl->devices.empty() ? 0.0 : m_impl->devices.front()->peakDramBwGbps;
}

} // namespace cupti_profiler
