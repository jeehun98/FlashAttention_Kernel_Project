// qk_tiled_5_1_1.cu
// 5.1.1 - QK^T Stage1-only Shared Memory Tiled Kernel (학습용 / NCU용)

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// ---- 타일 크기 설정 (GEMM 4.x 스타일과 비슷하게) ----
constexpr int BLOCK_M = 32;   // 쿼리 방향 타일 크기  (rows of Q / Scores)
constexpr int BLOCK_N = 32;   // 키   방향 타일 크기  (cols of K / Scores)
constexpr int BLOCK_K = 32;   // head_dim 방향 타일 (inner dim D)

// Q, K, Scores: 모두 float, row-major
// Q, K: [BH, N, D]
// Scores: [BH, N, N] = Q * K^T
__global__ void QK_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ Scores,
    int BH, int N, int D,
    float scale  // 보통 1/sqrt(D)
)
{
    // 3D grid:
    //  grid.x = ceil(N / BLOCK_N)  (Scores col 타일)
    //  grid.y = ceil(N / BLOCK_M)  (Scores row 타일)
    //  grid.z = BH                 (batch * head)
    int bh      = blockIdx.z;          // 0..BH-1
    int tile_m  = blockIdx.y;          // row tile index
    int tile_n  = blockIdx.x;          // col tile index

    int row_base = tile_m * BLOCK_M;
    int col_base = tile_n * BLOCK_N;

    // blockDim = (BLOCK_N, BLOCK_M) = (32,32) 가정
    int local_row = threadIdx.y;       // 0..BLOCK_M-1
    int local_col = threadIdx.x;       // 0..BLOCK_N-1

    int row = row_base + local_row;    // global row index in [0, N)
    int col = col_base + local_col;    // global col index in [0, N)

    if (bh >= BH) return;

    // shared memory tile
    __shared__ float Q_tile[BLOCK_M][BLOCK_K];  // [row, k]
    __shared__ float K_tile[BLOCK_N][BLOCK_K];  // [col, k], row-major 저장

    float acc = 0.0f;

    // K dimension(D)을 BLOCK_K로 쪼개서 tiling
    for (int k0 = 0; k0 < D; k0 += BLOCK_K) {
        int k = k0 + local_col;   // BLOCK_K 범위에서 column index 역할
        int k2 = k0 + local_row;  // K_tile 채우는 용도

        // Q_tile load: [BLOCK_M, BLOCK_K]
        if (row < N && k < D) {
            Q_tile[local_row][local_col] =
                Q[(bh * N + row) * D + k];
        } else {
            Q_tile[local_row][local_col] = 0.0f;
        }

        // K_tile load: [BLOCK_N, BLOCK_K]
        //   K[row=col_global, col=k]
        if (col < N && k2 < D) {
            K_tile[local_col][local_row] =
                K[(bh * N + col) * D + k2];
        } else {
            K_tile[local_col][local_row] = 0.0f;
        }

        __syncthreads();

        // 이 타일에서의 partial dot-product
        if (row < N && col < N) {
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                acc += Q_tile[local_row][kk] * K_tile[local_col][kk];
            }
        }

        __syncthreads();
    }

    // scale = 1/sqrt(D)
    if (row < N && col < N) {
        float val = acc * scale;
        Scores[(bh * N + row) * N + col] = val;
    }
}

// ---- CPU Reference: Scores = Q * K^T ----
void qk_cpu_ref(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    std::vector<float>& Scores,
    int BH, int N, int D,
    float scale
) {
    for (int b = 0; b < BH; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < D; ++k) {
                    float qv = Q[(b * N + i) * D + k];
                    float kv = K[(b * N + j) * D + k];
                    acc += static_cast<double>(qv) * kv;
                }
                acc *= scale;
                Scores[(b * N + i) * N + j] = static_cast<float>(acc);
            }
        }
    }
}

int main() {
    // ---- 기본 파라미터 설정 ----
    int BH = 4;     // B * H (예: B=1, H=4 / B=2, H=2 등)
    int N  = 512;   // sequence length
    int D  = 64;    // head dim

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    printf("QK^T Tiled Kernel (5.1.1)\n");
    printf("BH=%d, N=%d, D=%d\n", BH, N, D);
    printf("BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);

    size_t size_Q = static_cast<size_t>(BH) * N * D;
    size_t size_K = static_cast<size_t>(BH) * N * D;
    size_t size_S = static_cast<size_t>(BH) * N * N;

    // ---- Host 메모리 할당 & 초기화 ----
    std::vector<float> h_Q(size_Q);
    std::vector<float> h_K(size_K);
    std::vector<float> h_S(size_S);
    std::vector<float> h_S_ref(size_S);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < size_Q; ++i) h_Q[i] = dist(rng);
    for (size_t i = 0; i < size_K; ++i) h_K[i] = dist(rng);

    // ---- Device 메모리 할당 ----
    float *d_Q = nullptr, *d_K = nullptr, *d_S = nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, size_Q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, size_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, size_S * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), size_Q * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), size_K * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Kernel 설정 ----
    dim3 block(BLOCK_N, BLOCK_M);  // (32,32)
    dim3 grid(
        (N + BLOCK_N - 1) / BLOCK_N,
        (N + BLOCK_M - 1) / BLOCK_M,
        BH
    );

    printf("Grid = (%d, %d, %d), Block = (%d, %d, %d)\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    // ---- Warm-up ----
    QK_tiled_kernel<<<grid, block>>>(d_Q, d_K, d_S, BH, N, D, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 시간 측정 ----
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    QK_tiled_kernel<<<grid, block>>>(d_Q, d_K, d_S, BH, N, D, scale);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, size_S * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- CPU Reference & 검증 ----
    printf("Running CPU reference QK^T...\n");
    qk_cpu_ref(h_Q, h_K, h_S_ref, BH, N, D, scale);

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < size_S; ++i) {
        double diff = std::fabs(static_cast<double>(h_S[i]) - h_S_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    // ---- GFLOPS 계산 ----
    // QK^T: per BH, cost ~ 2 * N * N * D FLOPs
    double flop = 2.0 * static_cast<double>(BH) * N * N * D;
    double gflops = flop / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);
    printf("Max abs diff vs CPU: %.6e\n", max_abs_diff);

    // ---- 정리 ----
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_S));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 qk_tiled_5_1_1.cu -o qk_tiled_5_1_1.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:QK_tiled_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\qk_tiled_5_1_1.exe

*/