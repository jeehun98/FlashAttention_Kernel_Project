// gemm_4_9_tensorcore_batched.cu
// 4.9 Tensor Core Batched GEMM (WMMA 16x16x16, half->float)
// A: [BATCH, M, K], B: [BATCH, K, N], C: [BATCH, M, N] (row-major)

#include <cuda_runtime.h>
#include <mma.h>        // WMMA
#include <cuda_fp16.h>  // half
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

using namespace nvcuda;

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
// 설정 (FlashAttention용으로 B*H로 확장 가능)
// =========================================
constexpr int BATCH = 8;
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;

// WMMA 16x16x16 제약
static_assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "M,N,K must be multiple of 16");

// =========================================
// Batched Tensor Core GEMM 커널
// 각 block(=warp)이 (b, tile_row, tile_col)의 16x16 C 타일 1개를 계산
// C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
// =========================================
__global__ void wmma_batched_gemm_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         float* __restrict__ C,
                                         int batch, int M, int N, int K)
{
#if __CUDA_ARCH__ < 700
    return; // Tensor Core 미지원 아키텍처
#else
    int b        = blockIdx.z; // batch index
    int tile_row = blockIdx.y; // [0, M/16)
    int tile_col = blockIdx.x; // [0, N/16)

    if (b >= batch) return;

    int row_base = tile_row * 16;
    int col_base = tile_col * 16;

    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    const half* Ab = A + b * strideA;
    const half* Bb = B + b * strideB;
    float*      Cb = C + b * strideC;

    // WMMA fragment
    wmma::fragment<wmma::matrix_a,    16, 16, 16, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 16, half,  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                   c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        const half* tileA = Ab + row_base * K + k;   // A[b, row_base, k]
        const half* tileB = Bb + k * N       + col_base; // B[b, k, col_base]

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과를 바로 글로벌 C에 저장 (row-major)
    float* tileC = Cb + row_base * N + col_base;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
#endif
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
    printf("Tensor Core Batched GEMM (WMMA): BATCH=%d, M=%d, N=%d, K=%d\n",
           BATCH, M, N, K);

    const size_t sizeA = static_cast<size_t>(BATCH) * M * K;
    const size_t sizeB = static_cast<size_t>(BATCH) * K * N;
    const size_t sizeC = static_cast<size_t>(BATCH) * M * N;

    // Host: float로 생성 → A,B만 half로 캐스팅
    std::vector<float> hA_f(sizeA);
    std::vector<float> hB_f(sizeB);
    std::vector<float> hC_f(sizeC, 0.0f);
    std::vector<float> hC_ref(sizeC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i) hA_f[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB_f[i] = dist(rng);

    printf("Running CPU batched GEMM reference (float)...\n");
    cpu_batched_gemm_ref(hA_f, hB_f, hC_ref, BATCH, M, N, K);

    // half 버전 A,B
    std::vector<half> hA(sizeA);
    std::vector<half> hB(sizeB);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = __float2half(hA_f[i]);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = __float2half(hB_f[i]);

    // Device 메모리
    half  *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(32, 1, 1);  // warp 1개
    dim3 grid(N / 16, M / 16, BATCH);

    printf("Grid = (%d, %d, %d), Block = (%d, %d, %d)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Warm-up
    wmma_batched_gemm_kernel<<<grid, block>>>(dA, dB, dC, BATCH, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 타이밍
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    wmma_batched_gemm_kernel<<<grid, block>>>(dA, dB, dC, BATCH, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_f.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // 전체 batched GEMM FLOPs: 2 * B * M * N * K
    double flops_total  = 2.0 * static_cast<double>(BATCH) * M * N * K;
    double gflops_total = flops_total / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS (total GEMM): %.3f\n", ms, gflops_total);

    // CPU float batched GEMM 기준으로 정확도 확인 (FP16 입력이라 오차 존재)
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC_f[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU batched float ref: %.6e\n", max_abs_diff);

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
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"      -o gemm_4_9_tensorcore_batched.exe gemm_4_9_tensorcore_batched.cu

.\gemm_4_9_tensorcore_batched.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:wmma_batched_gemm_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\gemm_4_9_tensorcore_batched.exe

*/