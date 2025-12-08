// gemm_naive_tiled.cu
// 4.1.1 Naive GEMM + 4.2 Shared Memory Tiled GEMM

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
constexpr int DO_VERIFY = 0;      // CPU reference 검증 (원하면 1로)
constexpr int USE_TILED = 1;      // 0: naive, 1: shared memory tiled

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// naive 버전에서 쓰던 block size
constexpr int BLOCK_X_NAIVE = 16;
constexpr int BLOCK_Y_NAIVE = 16;

// tiled 버전 tile 사이즈 (blockDim.x=blockDim.y=32)
constexpr int TILE = 32;

// =========================================
// 4.1.1 Naive GEMM kernel
// C[M, N] = A[M, K] * B[K, N]
// =========================================
__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }

    C[row * N + col] = acc;
}

// =========================================
// 4.2 Shared Memory Tiled GEMM kernel
//  - block당 TILE x TILE C 타일 계산
//  - A, B 를 TILE x TILE 단위로 shared에 올린 후 재사용
//  - A, B 모두 coalesced load 보장
// =========================================
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    // block이 담당하는 C 타일의 시작 좌표
    int row = blockIdx.y * TILE + threadIdx.y; // global row index
    int col = blockIdx.x * TILE + threadIdx.x; // global col index

    // shared memory tile
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    // K 방향으로 TILE씩 잘라서 누적
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + threadIdx.x; // A에서 읽을 k index
        int kB = t * TILE + threadIdx.y; // B에서 읽을 k index

        // --- A tile 로드 (row, t tile) ---
        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // --- B tile 로드 (t tile, col) ---
        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // --- tile 내에서 K 축 합산 ---
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 결과 저장
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =========================================
// CPU reference GEMM (검증용)
// =========================================
void cpu_gemm_ref(const std::vector<float>& A,
                  const std::vector<float>& B,
                  std::vector<float>& C,
                  int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

int main(int argc, char** argv)
{
    printf("GEMM: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Kernel mode: %s\n", USE_TILED ? "Shared Memory Tiled (4.2)" : "Naive (4.1.1)");

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    // Host data
    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<float> hC(sizeC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

#if DO_VERIFY
    std::vector<float> hC_ref(sizeC, 0.0f);
    printf("Running CPU reference GEMM...\n");
    cpu_gemm_ref(hA, hB, hC_ref, M, N, K);
#endif

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block, grid;

    if constexpr (USE_TILED) {
        block = dim3(TILE, TILE);
        grid  = dim3((N + TILE - 1) / TILE,
                     (M + TILE - 1) / TILE);
    } else {
        block = dim3(BLOCK_X_NAIVE, BLOCK_Y_NAIVE);
        grid  = dim3((N + BLOCK_X_NAIVE - 1) / BLOCK_X_NAIVE,
                     (M + BLOCK_Y_NAIVE - 1) / BLOCK_Y_NAIVE);
    }

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up
    if constexpr (USE_TILED) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else {
        naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    if constexpr (USE_TILED) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else {
        naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);

#if DO_VERIFY
    // CPU reference와 비교
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU ref: %.6e\n", max_abs_diff);
#endif

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
/*
nvcc -O3 -std=c++17 -lineinfo -Xcompiler="/EHsc /MD" -o gemm_4_1_4_2.exe gemm_naive_tiled.cu
.\gemm_4_1_4_2.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:tiled_gemm_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\gemm_4_1_4_2.exe

*/