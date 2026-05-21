// Example: Profile cuBLAS GEMM + vectorAdd workload using libcupti_profiler.
//
// Demonstrates:
//   - Configuring and starting GpuProfiler
//   - Annotating GPU workload phases via EventProfiler::GetGpuTracker()
//   - Automatic protobuf output with periodic flush

#include <cupti_profiler/gpu_profiler.h>
#include <cupti_profiler/event_profiler.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#define RUNTIME_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(err) << "\n";              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Simple vector-add kernel: C[i] = A[i] + B[i]
// Purely memory-bound — 2 reads + 1 write per element, no arithmetic intensity.
__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          size_t N) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void RunGemmWorkload(cupti_profiler::EventTracker& gpuTracker, int deviceIndex) {
    RUNTIME_CHECK(cudaSetDevice(deviceIndex));

    cudaStream_t computeStream;
    RUNTIME_CHECK(cudaStreamCreate(&computeStream));
    gpuTracker.SetStream(static_cast<void*>(computeStream));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, computeStream);

    struct Phase { int N; int iterations; const char* label; };
    Phase phases[] = {
        {  512,  200, "warmup 512"},
        { 1024,  150, "ramp-up 1024"},
        { 2048,  100, "medium 2048"},
        { 4096,   50, "peak 4096"},
        { 2048,  100, "ramp-down 2048"},
        { 1024,  150, "cool-down 1024"},
        {  512,  200, "idle 512"},
    };

    int maxN = 0;
    for (const auto& phase : phases)
        maxN = std::max(maxN, phase.N);
    size_t maxGemmBytes = (size_t)maxN * maxN * sizeof(float);

    const size_t vecN = 256 * 1024 * 1024;
    const size_t vecBytes = vecN * sizeof(float);
    size_t maxBytes = std::max(maxGemmBytes, vecBytes);

    float *dA, *dB, *dC;
    RUNTIME_CHECK(cudaMalloc(&dA, maxBytes));
    RUNTIME_CHECK(cudaMalloc(&dB, maxBytes));
    RUNTIME_CHECK(cudaMalloc(&dC, maxBytes));
    RUNTIME_CHECK(cudaMemsetAsync(dA, 0x3f, maxBytes, computeStream));
    RUNTIME_CHECK(cudaMemsetAsync(dB, 0x3f, maxBytes, computeStream));

    float alpha = 1.0f, beta = 0.0f;

    for (const auto& phase : phases) {
        int N = phase.N;
        std::cout << "  Phase: " << phase.label << " (" << phase.iterations << " iterations)\n";

        size_t regionIdx = gpuTracker.BeginRegion(phase.label);
        for (int i = 0; i < phase.iterations; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
        }
        gpuTracker.EndRegion(regionIdx);
    }

    // Memory-bound phase
    {
        const int vecIters = 200;
        std::cout << "  Phase: memcpy vecAdd (" << vecIters << " iterations, "
                  << vecBytes / (1024 * 1024) << " MB/iter)\n";
        int threads = 256;
        int blocks = (vecN + threads - 1) / threads;
        size_t regionIdx = gpuTracker.BeginRegion("vecAdd (mem-bound)");
        for (int i = 0; i < vecIters; ++i) {
            vectorAdd<<<blocks, threads, 0, computeStream>>>(dA, dB, dC, vecN);
        }
        gpuTracker.EndRegion(regionIdx);
    }

    RUNTIME_CHECK(cudaStreamSynchronize(computeStream));

    RUNTIME_CHECK(cudaFree(dA));
    RUNTIME_CHECK(cudaFree(dB));
    RUNTIME_CHECK(cudaFree(dC));
    cublasDestroy(handle);
    RUNTIME_CHECK(cudaStreamDestroy(computeStream));
}

int main(int argc, char* argv[]) {
    int deviceIndex = 0;
    uint64_t samplingFrequencyHz = 10000; // 10 kHz
    std::string outputFile = "gpu_metrics.pb";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--device") && i + 1 < argc)
            deviceIndex = std::stoi(argv[++i]);
        else if ((arg == "-f" || arg == "--frequency") && i + 1 < argc)
            samplingFrequencyHz = std::stoull(argv[++i]);
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
            outputFile = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [-d device] [-f frequency_hz] [-o output.pb]\n"
                      << "  -d  Device index (default: 0)\n"
                      << "  -f  Sampling frequency in Hz (default: 10000 = 10kHz)\n"
                      << "  -o  Output protobuf file (default: gpu_metrics.pb)\n";
            return 0;
        }
    }

    cupti_profiler::ProfilerConfig config;
    config.deviceIndices = {deviceIndex};
    config.samplingFrequencyHz = samplingFrequencyHz;
    config.outputFile = outputFile;
    config.metrics = {
        "sm__cycles_active.avg",
        "sm__cycles_active.max",
        "sm__cycles_elapsed.avg",
        "sm__warps_active.avg",
        "sm__warps_active.max",
        "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__read_throughput.max.pct_of_peak_sustained_elapsed",
        "dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__write_throughput.max.pct_of_peak_sustained_elapsed",
    };

    cupti_profiler::GpuProfiler profiler;
    profiler.Configure(config);
    profiler.Start();

    cupti_profiler::EventProfiler events;
    cupti_profiler::EventProfilerConfig eventCfg;
    eventCfg.outputFile = "events.pb";
    eventCfg.flushIntervalMs = 1000;
    events.Configure(eventCfg);
    events.Start();

    auto t0 = std::chrono::high_resolution_clock::now();
    RunGemmWorkload(events.GetGpuTracker(), config.deviceIndices.empty() ? 0 : config.deviceIndices.front());
    auto t1 = std::chrono::high_resolution_clock::now();

    double walltime = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nWorkload completed in " << std::fixed << std::setprecision(2)
              << walltime << " seconds\n";

    events.Stop();
    profiler.Stop();
    return 0;
}
