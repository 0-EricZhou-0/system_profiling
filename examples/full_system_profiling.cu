// Example: Full-system profiling with GPU + CPU/Mem + Disk.
// Uses ProfilerSuite with a .pbtxt config file.

#include <cupti_profiler/profiler_suite.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>

// Resolve this source file's directory at compile time via __FILE__ so the
// default config path is independent of the binary's working directory.
// Sibling layout: examples/full_system_profiling.cu → ../configs/example.pbtxt
#define SOURCE_FILE_DIR (std::filesystem::path(__FILE__).parent_path())
#define DEFAULT_CONFIG_PATH \
    (SOURCE_FILE_DIR / ".." / "configs" / "example.pbtxt").lexically_normal().string()

#define RUNTIME_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(err) << "\n";              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          size_t N) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void RunGemmWorkload(cupti_profiler::EventTracker& gpuTracker,
                     cupti_profiler::EventTracker& genericTracker,
                     int deviceIndex) {
    RUNTIME_CHECK(cudaSetDevice(deviceIndex));

    cudaStream_t computeStream;
    RUNTIME_CHECK(cudaStreamCreate(&computeStream));
    gpuTracker.SetStream(static_cast<void*>(computeStream));

    // Demonstrate Generic-domain region wrapping host-side workload setup.
    size_t setupRegion = genericTracker.BeginRegion("workload setup");

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

    const size_t vecN = 64 * 1024 * 1024;
    const size_t vecBytes = vecN * sizeof(float);
    size_t maxBytes = std::max(maxGemmBytes, vecBytes);

    float *dA, *dB, *dC;
    RUNTIME_CHECK(cudaMalloc(&dA, maxBytes));
    RUNTIME_CHECK(cudaMalloc(&dB, maxBytes));
    RUNTIME_CHECK(cudaMalloc(&dC, maxBytes));
    RUNTIME_CHECK(cudaMemsetAsync(dA, 0x3f, maxBytes, computeStream));
    RUNTIME_CHECK(cudaMemsetAsync(dB, 0x3f, maxBytes, computeStream));

    float alpha = 1.0f, beta = 0.0f;

    genericTracker.EndRegion(setupRegion);
    genericTracker.MarkEvent("workload begin");

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
    std::string configPath = DEFAULT_CONFIG_PATH;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc)
            configPath = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [-c config.pbtxt]\n"
                      << "  -c  Config file path (default: " << DEFAULT_CONFIG_PATH << ")\n";
            return 0;
        }
    }

    cupti_profiler::ProfilerSuite suite;
    suite.LoadConfig(configPath);
    if (auto err = suite.Configure(); err != cupti_profiler::ProfilerError::Ok) {
        std::cerr << "Configure failed: " << cupti_profiler::ToString(err) << "\n";
        return 1;
    }
    suite.Start();
    suite.GetEventProfiler().GetGenericTracker().MarkEvent("suite start");

    auto t0 = std::chrono::high_resolution_clock::now();
    RunGemmWorkload(suite.GetEventProfiler().GetGpuTracker(),
                    suite.GetEventProfiler().GetGenericTracker(),
                    0);
    auto t1 = std::chrono::high_resolution_clock::now();

    double walltime = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nWorkload completed in " << std::fixed << std::setprecision(2)
              << walltime << " seconds\n";

    suite.Stop();
    return 0;
}
