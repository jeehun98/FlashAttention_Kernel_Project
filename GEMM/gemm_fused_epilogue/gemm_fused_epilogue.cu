// gemm_fused_epilogue.cu
//  - 4.2 shared memory tiled GEMM 기반
//  - Fused epilogue: bias + ReLU + residual
//      out[row, col] = ReLU( (A*B)[row, col] + bias[col] ) + residual[row, col]

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(cmd)                                                         \
    do {                                                                        \
        cudaError_t e = (cmd);                                                  \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e));                                     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// =========================================
// 설정
// =========================================
constexpr int DO_VERIFY   = 1;   // CPU reference와 비교 (지금은 켜두자)
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int TILE = 32;         // blockDim.x = blockDim.y = TILE

// =========================================
// activation (ReLU)
// =========================================
__device__ __forceinline__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// =========================================
// 4.2 스타일 Shared Memory Tiled GEMM + Fused Epilogue
//  C = ReLU(A*B + bias) + residual
// =========================================
__global__ void gemm_tiled_fused_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        const float* __restrict__ bias,      // [N]
                                        const float* __restrict__ residual,  // [M, N]
                                        float* __restrict__ C,               // [M, N]
                                        int M, int N, int K)
{
    int row = blockIdx.y * TILE + threadIdx.y; // global row index
    int col = blockIdx.x * TILE + threadIdx.x; // global col index

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        // Load A tile: A[row, kA]
        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile: B[kB, col]
        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // tile 내 GEMM
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Fused epilogue
    if (row < M && col < N) {
        int idx = row * N + col;

        float val = acc;

        // 1) + bias[col]
        if (bias != nullptr) {
            val += bias[col];
        }

        // 2) activation (ReLU)
        val = relu(val);

        // 3) + residual[row, col]
        if (residual != nullptr) {
            val += residual[idx];
        }

        C[idx] = val;
    }
}

// =========================================
// CPU reference: C = ReLU(A*B + bias) + residual
// (float 기준, 검증용)
// =========================================
void cpu_gemm_fused_ref(const std::vector<float>& A,
                        const std::vector<float>& B,
                        const std::vector<float>& bias,
                        const std::vector<float>& residual,
                        std::vector<float>& C,
                        int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            float val = acc + bias[j];
            val = val > 0.0f ? val : 0.0f; // ReLU
            val += residual[i * N + j];
            C[i * N + j] = val;
        }
    }
}

// =========================================
// main
// =========================================
int main()
{
    printf("Fused GEMM (Tiled + bias + ReLU + residual): M=%d, N=%d, K=%d\n", M, N, K);

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;
    const size_t sizeBias = static_cast<size_t>(N);

    // Host 메모리
    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<float> hBias(sizeBias);
    std::vector<float> hResidual(sizeC);
    std::vector<float> hC(sizeC, 0.0f);
    std::vector<float> hC_ref(sizeC, 0.0f);

    // 난수 초기화
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);
    for (size_t i = 0; i < sizeBias; ++i) hBias[i] = dist(rng);
    for (size_t i = 0; i < sizeC; ++i) hResidual[i] = dist(rng);

#if DO_VERIFY
    printf("Running CPU fused GEMM reference...\n");
    cpu_gemm_fused_ref(hA, hB, hBias, hResidual, hC_ref, M, N, K);
#endif

    // Device 메모리
    float *dA = nullptr, *dB = nullptr;
    float *dBias = nullptr, *dResidual = nullptr, *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA,      sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB,      sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBias,   sizeBias * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dResidual, sizeC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC,      sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA,      hA.data(),      sizeA   * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,      hB.data(),      sizeB   * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBias,   hBias.data(),   sizeBias* sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dResidual, hResidual.data(), sizeC* sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up
    gemm_tiled_fused_kernel<<<grid, block>>>(dA, dB, dBias, dResidual, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 타이밍 측정
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    gemm_tiled_fused_kernel<<<grid, block>>>(dA, dB, dBias, dResidual, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // GEMM 연산량 기준 GFLOPS (epilogue 연산은 포함 안 함)
    double flops  = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS (GEMM only): %.3f\n", ms, gflops);

#if DO_VERIFY
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU fused ref: %.6e\n", max_abs_diff);
#endif

    // 정리
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dBias));
    CHECK_CUDA(cudaFree(dResidual));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
/*
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"      -o gemm_fused_epilogue.exe gemm_fused_epilogue.cu

.\gemm_fused_epilogue.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:gemm_tiled_fused_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     --section WarpStateStats     .\gemm_fused_epilogue.exe


*/