// gemm_4_8_batched.cu
// 4.8 Batched / Strided GEMM (Shared Memory Tiled)
// A: [B, M, K], B: [B, K, N], C: [B, M, N], row-major

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

// batch, M, N, K (FlashAttention에서 B*H로 확장 가능)
constexpr int BATCH = 8;
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;

constexpr int TILE = 32;  // blockDim.x = blockDim.y = TILE

// =========================================
// Shared Memory Tiled Batched GEMM kernel
// C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
// =========================================
__global__ void batched_tiled_gemm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int batch, int M, int N, int K)
{
    int b   = blockIdx.z;                  // batch index
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (b >= batch) return;

    // batch별 base offset
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    const float* Ab = A + b * strideA;
    const float* Bb = B + b * strideB;
    float*       Cb = C + b * strideC;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        // load A[b, row, kA]
        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = Ab[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load B[b, kB, col]
        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = Bb[kB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

// =========================================
// CPU reference: batched GEMM (float)
// C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
// =========================================
void cpu_batched_gemm_ref(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          int batch, int M, int N, int K)
{
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    for (int b = 0; b < batch; ++b) {
        const float* Ab = A.data() + b * strideA;
        const float* Bb = B.data() + b * strideB;
        float*       Cb = C.data() + b * strideC;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += Ab[i * K + k] * Bb[k * N + j];
                }
                Cb[i * N + j] = acc;
            }
        }
    }
}

// =========================================
// main
// =========================================
int main()
{
    printf("Batched GEMM (Shared Tiled): BATCH=%d, M=%d, N=%d, K=%d\n",
           BATCH, M, N, K);

    const size_t sizeA = static_cast<size_t>(BATCH) * M * K;
    const size_t sizeB = static_cast<size_t>(BATCH) * K * N;
    const size_t sizeC = static_cast<size_t>(BATCH) * M * N;

    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<float> hC(sizeC, 0.0f);
    std::vector<float> hC_ref(sizeC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

    printf("Running CPU batched GEMM reference...\n");
    cpu_batched_gemm_ref(hA, hB, hC_ref, BATCH, M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE,
              BATCH);

    printf("Grid = (%d, %d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, grid.z, block.x, block.y);

    // Warm-up
    batched_tiled_gemm_kernel<<<grid, block>>>(
        dA, dB, dC, BATCH, M, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    batched_tiled_gemm_kernel<<<grid, block>>>(
        dA, dB, dC, BATCH, M, N, K
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // 전체 FLOPs (모든 batch 포함): 2 * B * M * N * K
    double flops  = 2.0 * static_cast<double>(BATCH) * M * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS (total): %.3f\n", ms, gflops);

    // 검증
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU batched ref: %.6e\n", max_abs_diff);

    // 정리
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
/*
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"      -o gemm_4_8_batched.exe gemm_4_8_batched.cu

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:batched_tiled_gemm_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\gemm_4_8_batched.exe

.\gemm_4_8_batched.exe

*/