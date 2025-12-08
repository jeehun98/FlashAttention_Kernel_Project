// gemm_4_1_4_3.cu
// 4.1.1 Naive GEMM
// 4.2 Shared Memory Tiled GEMM
// 4.3 cp.async Pipelined Tiled GEMM

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

constexpr int DO_VERIFY   = 0;   // CPU reference 검증 필요하면 1
// 0: naive, 1: shared tiled, 2: cp.async tiled
constexpr int KERNEL_MODE = 2;

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// naive 버전 block size
constexpr int BLOCK_X_NAIVE = 16;
constexpr int BLOCK_Y_NAIVE = 16;

// tiled / cp.async 버전 tile size
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
// block당 TILE x TILE C 타일 계산
// A, B 를 TILE x TILE 단위로 shared에 올려 재사용
// =========================================
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        // A tile load
        if (row < M && kA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // B tile load
        if (kB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
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
        C[row * N + col] = acc;
    }
}

// =========================================
// 4.3 cp.async helper (4B 단위)
// =========================================
__device__ __forceinline__ void cp_async_4B(float* dst_smem, const float* src_gmem, bool pred)
{
#if __CUDA_ARCH__ >= 800
    if (pred) {
        // shared/global 주소를 cp.async용으로 변환
        unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
        unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src_gmem);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n" ::
            "r"(smem_addr), "l"(gmem_addr)
        );
    }
#else
    // sm_80 미만에선 cp.async 미지원 → fall back
    if (pred) {
        *dst_smem = *src_gmem;
    }
#endif

    // 범위 밖인 경우는 그냥 0으로 채움
    if (!pred) {
        *dst_smem = 0.0f;
    }
}

__device__ __forceinline__ void cp_async_commit_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// =========================================
// 4.3 cp.async Pipelined Tiled GEMM
//  - double-buffered shared tile
//  - tile t를 계산하는 동안 tile t+1을 cp.async로 prefetch
// =========================================
__global__ void cp_async_tiled_gemm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // double-buffered shared tiles
    __shared__ float As[2][TILE][TILE];
    __shared__ float Bs[2][TILE][TILE];

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    int stage = 0;

    auto load_tile = [&](int t, int stage_idx) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        float* smemA = &As[stage_idx][threadIdx.y][threadIdx.x];
        float* smemB = &Bs[stage_idx][threadIdx.y][threadIdx.x];

        bool validA = (row < M && kA < K);
        bool validB = (kB < K && col < N);

        const float* gA = validA ? &A[row * K + kA] : nullptr;
        const float* gB = validB ? &B[kB * N + col] : nullptr;

        cp_async_4B(smemA, gA, validA);
        cp_async_4B(smemB, gB, validB);
    };

    if (num_tiles == 0) return;

    // --- 첫 번째 tile prefetch ---
    load_tile(0, stage);
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        int next_stage = stage ^ 1;

        // 다음 tile prefetch
        if (t + 1 < num_tiles) {
            load_tile(t + 1, next_stage);
            cp_async_commit_group();
        }

        // 현재 stage tile 계산
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a = As[stage][threadIdx.y][k];
            float b = Bs[stage][k][threadIdx.x];
            acc += a * b;
        }

        if (t + 1 < num_tiles) {
            cp_async_wait_group_0();
        }
        __syncthreads();

        stage = next_stage;
    }

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

// =========================================
// main
// =========================================
int main(int argc, char** argv)
{
    printf("GEMM: M=%d, N=%d, K=%d\n", M, N, K);
    if      (KERNEL_MODE == 0) printf("Kernel mode: 4.1.1 Naive\n");
    else if (KERNEL_MODE == 1) printf("Kernel mode: 4.2 Shared Memory Tiled\n");
    else if (KERNEL_MODE == 2) printf("Kernel mode: 4.3 cp.async Tiled\n");
    else                       printf("Kernel mode: UNKNOWN\n");

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

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

    if (KERNEL_MODE == 0) {
        block = dim3(BLOCK_X_NAIVE, BLOCK_Y_NAIVE);
        grid  = dim3((N + BLOCK_X_NAIVE - 1) / BLOCK_X_NAIVE,
                     (M + BLOCK_Y_NAIVE - 1) / BLOCK_Y_NAIVE);
    } else {
        block = dim3(TILE, TILE);
        grid  = dim3((N + TILE - 1) / TILE,
                     (M + TILE - 1) / TILE);
    }

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up
    if (KERNEL_MODE == 0) {
        naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else if (KERNEL_MODE == 1) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else if (KERNEL_MODE == 2) {
        cp_async_tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    if (KERNEL_MODE == 0) {
        naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else if (KERNEL_MODE == 1) {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else if (KERNEL_MODE == 2) {
        cp_async_tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops  = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);

#if DO_VERIFY
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
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"      -o gemm_4_1_4_3.exe gemm_4_1_4_3.cu

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:cp_async_tiled_gemm_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\gemm_4_1_4_3.exe


*/