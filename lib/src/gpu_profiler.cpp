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

    // Internal state
    internal::CuptiPmSampling target;
    internal::CuptiProfilerHost host;

    std::vector<const char*> metricsCstr;       // const char* view into config.metrics
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;

    // Host info (captured at Configure() — written into TraceHeader)
    std::string hostname;
    uint32_t hostCpuCount = 0;

    // Device info
    std::string deviceName;
    std::string chipName;
    // Peaks. peakDramBwGbps is kept for the public GetPeakDramBwGbps()
    // API; the bytes/sec version is what the new GPUDeviceInfo wire
    // format consumes.
    double peakDramBwGbps = 0.0;
    double peakDramBwBytesPerSec = 0.0;
    double peakPcieBwBytesPerSec = 0.0;    // theoretical max per direction
    double peakNvlinkBwBytesPerSec = 0.0;  // sum across active NVLink links, per direction

    // Threads
    std::thread decodeThread;
    std::thread flushThread;
    std::atomic<bool> stopDecode{false};
    std::atomic<bool> stopFlush{false};
    CUptiResult decodeResult = CUPTI_SUCCESS;

    // Output file
    std::ofstream outFile;
    std::mutex outMutex;

    // Per-flush write accounting
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

    // Check device support
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, config.deviceIndex));
    {
        CUpti_Profiler_DeviceSupported_Params p = {CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
        p.cuDevice = cuDevice;
        p.api = CUPTI_PROFILER_PM_SAMPLING;
        CUPTI_API_CALL(cuptiProfilerDeviceSupported(&p));
        if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
            std::cerr << "PM sampling not supported on this device\n";
            exit(1);
        }
    }

    // Get device name
    char devName[256];
    DRIVER_API_CALL(cuDeviceGetName(devName, sizeof(devName), cuDevice));
    m_impl->deviceName = devName;
    std::cout << "Device: " << m_impl->deviceName << " (index " << config.deviceIndex << ")\n";

    // Get chip name and counter availability
    std::vector<uint8_t> counterAvailImage;
    internal::CuptiPmSampling::GetChipName(config.deviceIndex, m_impl->chipName);
    internal::CuptiPmSampling::GetCounterAvailabilityImage(config.deviceIndex, counterAvailImage);
    std::cout << "Chip: " << m_impl->chipName << "\n";

    // Build const char* metric list
    m_impl->metricsCstr.clear();
    for (const auto& m : config.metrics) {
        m_impl->metricsCstr.push_back(m.c_str());
    }

    std::cout << "Sampling frequency: " << config.samplingFrequencyHz << " Hz\n";
    std::cout << "Metrics: " << config.metrics.size() << "\n";
    for (const auto& m : config.metrics) std::cout << "  " << m << "\n";

    // Set up profiler host and create config image
    m_impl->host.SetUp(m_impl->chipName, counterAvailImage);
    CUPTI_API_CALL(m_impl->host.CreateConfigImage(m_impl->metricsCstr, m_impl->configImage));

    // Set up PM sampling target
    m_impl->target.SetUp(config.deviceIndex);
    CUPTI_API_CALL(m_impl->target.EnablePmSampling(config.deviceIndex));
    // CUPTI expects interval in nanoseconds; convert from frequency in Hz
    uint64_t intervalNs = static_cast<uint64_t>(1e9 / config.samplingFrequencyHz);
    CUPTI_API_CALL(m_impl->target.SetConfig(m_impl->configImage, config.hwBufferSize, intervalNs));

    // Create counter data image
    CUPTI_API_CALL(m_impl->target.CreateCounterDataImage(config.maxSamples, m_impl->metricsCstr, m_impl->counterDataImage));

    // Query peak DRAM bandwidth
    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, config.deviceIndex));
    m_impl->peakDramBwGbps = (double)prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    // Bytes/sec form for the new GPUDeviceInfo proto field.
    m_impl->peakDramBwBytesPerSec = m_impl->peakDramBwGbps * 1e9;
    std::cout << "Peak DRAM BW: " << std::fixed << std::setprecision(1)
              << m_impl->peakDramBwGbps << " GB/s"
              << " (memClk=" << prop.memoryClockRate / 1000 << " MHz"
              << ", busWidth=" << prop.memoryBusWidth << " bits)\n";

    // Query peak PCIe and NVLink bandwidth via NVML.
    {
        nvmlReturn_t r = nvmlInit_v2();
        if (r == NVML_SUCCESS) {
            nvmlDevice_t dev;
            if (nvmlDeviceGetHandleByIndex_v2(config.deviceIndex, &dev) == NVML_SUCCESS) {
                // PCIe: max link gen × max link width × per-gen lane bw
                unsigned int gen = 0, width = 0;
                if (nvmlDeviceGetMaxPcieLinkGeneration(dev, &gen) == NVML_SUCCESS &&
                    nvmlDeviceGetMaxPcieLinkWidth(dev, &width) == NVML_SUCCESS) {
                    m_impl->peakPcieBwBytesPerSec = PcieLaneBytesPerSec(gen) * width;
                    double gibps = m_impl->peakPcieBwBytesPerSec / (1024.0 * 1024.0 * 1024.0);
                    std::cout << "Peak PCIe BW: " << std::fixed << std::setprecision(2)
                              << gibps << " GiB/s per direction"
                              << " (Gen" << gen << " x" << width << ")\n";
                }

                // NVLink: iterate links, sum active per-version per-link bw.
                // NVML_NVLINK_MAX_LINKS bounds the iteration; non-existent links
                // return INVALID_ARG which we treat as end-of-iteration.
                unsigned int activeLinks = 0;
                unsigned int linkVersion = 0;
                double totalBytesPerSec = 0.0;
                for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
                    nvmlEnableState_t isActive;
                    nvmlReturn_t sr = nvmlDeviceGetNvLinkState(dev, link, &isActive);
                    if (sr == NVML_ERROR_INVALID_ARGUMENT ||
                        sr == NVML_ERROR_NOT_SUPPORTED) break;
                    if (sr != NVML_SUCCESS) continue;
                    if (isActive != NVML_FEATURE_ENABLED) continue;

                    unsigned int v = 0;
                    if (nvmlDeviceGetNvLinkVersion(dev, link, &v) != NVML_SUCCESS) continue;
                    linkVersion = v;  // assume homogeneous; record the last
                    totalBytesPerSec += NvLinkLinkBytesPerSec(v);
                    ++activeLinks;
                }
                if (activeLinks > 0) {
                    m_impl->peakNvlinkBwBytesPerSec = totalBytesPerSec;
                    double gibps = totalBytesPerSec / (1024.0 * 1024.0 * 1024.0);
                    std::cout << "Peak NVLink BW: " << std::fixed << std::setprecision(2)
                              << gibps << " GiB/s per direction"
                              << " (NVLink" << linkVersion << " × "
                              << activeLinks << " active links)\n";
                } else {
                    std::cout << "Peak NVLink BW: no active links detected\n";
                }
            }
            nvmlShutdown();
        } else {
            std::cerr << "[GPU] NVML init failed; skipping peak PCIe/NVLink BW query.\n";
        }
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

    // Launch decode thread
    m_impl->stopDecode = false;
    m_impl->decodeResult = CUPTI_SUCCESS;
    m_impl->decodeThread = std::thread(internal::DecodeThreadFunc,
                                        std::ref(m_impl->counterDataImage),
                                        std::ref(m_impl->metricsCstr),
                                        std::ref(m_impl->target),
                                        std::ref(m_impl->host),
                                        std::ref(m_impl->stopDecode),
                                        std::ref(m_impl->decodeResult));

    // Launch flush thread if configured
    m_impl->stopFlush = false;
    if (m_impl->config.flushIntervalMs > 0 && m_impl->outFile.is_open()) {
        std::cout << "Periodic flush every " << m_impl->config.flushIntervalMs << " ms\n";
        m_impl->flushThread = std::thread(internal::FlushThreadFunc,
                                           std::ref(m_impl->host),
                                           std::ref(m_impl->outFile),
                                           std::ref(m_impl->outMutex),
                                           std::cref(m_impl->hostname),
                                           m_impl->config.samplingFrequencyHz,
                                           m_impl->hostCpuCount,
                                           std::cref(m_impl->deviceName),
                                           std::cref(m_impl->chipName),
                                           std::cref(m_impl->metricsCstr),
                                           &m_impl->peakDramBwBytesPerSec,
                                           &m_impl->peakPcieBwBytesPerSec,
                                           &m_impl->peakNvlinkBwBytesPerSec,
                                           std::ref(m_impl->stopFlush),
                                           m_impl->config.flushIntervalMs,
                                           m_impl->steadyClockRefNs,
                                           m_impl->cuptiRefNs,
                                           m_impl->wallClockEpochNs,
                                           std::ref(m_impl->flushStatsPending),
                                           std::ref(m_impl->flushStatsMutex));
    }

    // Start PM sampling
    CUPTI_API_CALL(m_impl->target.Start());
    m_impl->running = true;

    std::cout << "\n=== PM sampling started at "
              << m_impl->config.samplingFrequencyHz << " Hz ===\n\n";
}

void GpuProfiler::Stop() {
    if (!m_impl->running) return;

    // Stop sampling
    CUPTI_API_CALL(m_impl->target.Stop());

    // Join decode thread
    m_impl->stopDecode = true;
    if (m_impl->decodeThread.joinable()) m_impl->decodeThread.join();

    // Join flush thread
    m_impl->stopFlush = true;
    if (m_impl->flushThread.joinable()) m_impl->flushThread.join();

    if (m_impl->decodeResult != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(m_impl->decodeResult, &errstr);
        std::cerr << "Decode thread error: " << errstr << "\n";
    }

    // Write remaining samples to file
    if (m_impl->outFile.is_open()) {
        auto remaining = m_impl->host.DrainSamples();
        std::cout << "Remaining samples after flush: " << remaining.size() << "\n";
        if (!remaining.empty() || m_impl->flushStatsPending.valid) {
            GPUMetricsTrace finalTrace = internal::BuildTrace(
                m_impl->hostname, m_impl->config.samplingFrequencyHz, m_impl->hostCpuCount,
                m_impl->deviceName, m_impl->chipName,
                m_impl->metricsCstr, remaining,
                m_impl->peakDramBwBytesPerSec, m_impl->peakPcieBwBytesPerSec,
                m_impl->peakNvlinkBwBytesPerSec,
                m_impl->steadyClockRefNs, m_impl->cuptiRefNs, m_impl->wallClockEpochNs);
            // Attach any pending flush stats from the last background flush cycle.
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

    // Cleanup CUPTI
    CUPTI_API_CALL(m_impl->target.DisablePmSampling());
    m_impl->target.TearDown();
    m_impl->host.TearDown();

    m_impl->running = false;
}

std::vector<SamplerRange> GpuProfiler::DrainSamples() {
    return m_impl->host.DrainSamples();
}

std::string GpuProfiler::GetDeviceName() const { return m_impl->deviceName; }
std::string GpuProfiler::GetChipName() const { return m_impl->chipName; }
double GpuProfiler::GetPeakDramBwGbps() const { return m_impl->peakDramBwGbps; }

} // namespace cupti_profiler
