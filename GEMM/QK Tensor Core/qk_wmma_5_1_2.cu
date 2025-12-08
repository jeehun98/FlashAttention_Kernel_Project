// qk_wmma_5_1_2.cu
// 5.1.x - QK^T using Tensor Cores (WMMA), Stage1-only

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
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

using namespace nvcuda;

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Q, K: [BH, N, D] (half)
// Scores: [BH, N, N] (float)
// 조건: N % 16 == 0, D % 16 == 0
__global__ void QK_wmma_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    float* __restrict__ Scores,
    int BH, int N, int D,
    float scale    // 1/sqrt(D)
)
{
    int bh     = blockIdx.z;                        // 0..BH-1
    int tile_m = blockIdx.y * WMMA_M;               // row tile start in [0, N)
    int tile_n = blockIdx.x * WMMA_N;               // col tile start in [0, N)

    if (bh >= BH || tile_m >= N || tile_n >= N) return;

    // 한 block = single warp 가정
    int lane_id = threadIdx.x;
    if (lane_id >= 32) return;  // blockDim.x = 32 전제

    // WMMA fragment 선언
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // BH 차원마다 Q/K는 연속된 [N, D] 덩어리
    const __half* Q_base = Q + static_cast<size_t>(bh) * N * D;
    const __half* K_base = K + static_cast<size_t>(bh) * N * D;
    float* S_base        = Scores + static_cast<size_t>(bh) * N * N;

    // GEMM: C = Q * K^T
    //  A: [N, D] row-major      (M = N, K = D)
    //  B: [D, N] col-major      (K = D, N = N)
    //  C: [N, N] row-major

    for (int k0 = 0; k0 < D; k0 += WMMA_K) {
        // A tile: (tile_m, k0)
        const __half* A_tile = Q_base + static_cast<size_t>(tile_m) * D + k0;
        int lda = D;

        // B tile: (k0, tile_n)
        // K는 [N, D] row-major: K[bh, n, k] at idx = (bh*N + n)*D + k
        // 이를 (D, N) col-major로 재해석:
        //   row = k, col = n → ptr = K_base + k + n*D
        const __half* B_tile = K_base + k0 + static_cast<size_t>(tile_n) * D;
        int ldb = D;  // col-major에서 leading dim = #rows = D

        wmma::load_matrix_sync(a_frag, A_tile, lda);
        wmma::load_matrix_sync(b_frag, B_tile, ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // scale 적용 (1/sqrt(D))
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }

    // 결과 저장: C tile (tile_m, tile_n)
    float* C_tile = S_base + static_cast<size_t>(tile_m) * N + tile_n;
    int ldc = N;

    wmma::store_matrix_sync(C_tile, c_frag, ldc, wmma::mem_row_major);
}

// ---- CPU Reference: Scores = Q * K^T (float32) ----
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
    // ----- 파라미터 설정 -----
    int BH = 4;     // B * H
    int N  = 512;   // seq_len (16의 배수)
    int D  = 64;    // head_dim (16의 배수)

    if ((N % WMMA_M) != 0 || (D % WMMA_K) != 0) {
        printf("N, D must be multiples of 16. Got N=%d, D=%d\n", N, D);
        return 0;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    printf("QK^T WMMA Kernel (5.1.x)\n");
    printf("BH=%d, N=%d, D=%d\n", BH, N, D);
    printf("WMMA_M=%d, WMMA_N=%d, WMMA_K=%d\n", WMMA_M, WMMA_N, WMMA_K);

    size_t size_Q = static_cast<size_t>(BH) * N * D;
    size_t size_K = static_cast<size_t>(BH) * N * D;
    size_t size_S = static_cast<size_t>(BH) * N * N;

    // ----- Host 메모리 -----
    std::vector<float> h_Q_f(size_Q);
    std::vector<float> h_K_f(size_K);
    std::vector<__half> h_Q_h(size_Q);
    std::vector<__half> h_K_h(size_K);
    std::vector<float> h_S(size_S);
    std::vector<float> h_S_ref(size_S);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < size_Q; ++i) {
        float v = dist(rng);
        h_Q_f[i] = v;
        h_Q_h[i] = __float2half(v);
    }
    for (size_t i = 0; i < size_K; ++i) {
        float v = dist(rng);
        h_K_f[i] = v;
        h_K_h[i] = __float2half(v);
    }

    // ----- Device 메모리 -----
    __half *d_Q = nullptr, *d_K = nullptr;
    float  *d_S = nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, size_Q * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_K, size_K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_S, size_S * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q_h.data(), size_Q * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K_h.data(), size_K * sizeof(__half), cudaMemcpyHostToDevice));

    // ----- Kernel 설정 -----
    dim3 block(32, 1, 1);  // 1 warp per block
    dim3 grid(
        (N + WMMA_N - 1) / WMMA_N,
        (N + WMMA_M - 1) / WMMA_M,
        BH
    );

    printf("Grid = (%d, %d, %d), Block = (%d, %d, %d)\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    // Warm-up
    QK_wmma_kernel<<<grid, block>>>(d_Q, d_K, d_S, BH, N, D, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ----- 시간 측정 -----
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    QK_wmma_kernel<<<grid, block>>>(d_Q, d_K, d_S, BH, N, D, scale);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, size_S * sizeof(float), cudaMemcpyDeviceToHost));

    // ----- CPU Reference -----
    printf("Running CPU reference QK^T...\n");
    qk_cpu_ref(h_Q_f, h_K_f, h_S_ref, BH, N, D, scale);

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < size_S; ++i) {
        double diff = std::fabs(static_cast<double>(h_S[i]) - h_S_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    // ----- GFLOPS -----
    double flop = 2.0 * static_cast<double>(BH) * N * N * D;
    double gflops = flop / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);
    printf("Max abs diff vs CPU: %.6e\n", max_abs_diff);

    // ----- 정리 -----
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_S));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 qk_wmma_5_1_2.cu -o qk_wmma_5_1_2.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:QK_wmma_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\qk_wmma_5_1_2.exe

.\qk_wmma_5_1_2.exe

*/