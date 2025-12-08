// tensorcore_gemm.cu
// 4.4 Tensor Core GEMM (WMMA, 16x16x16, half->float)

#include <cuda_runtime.h>
#include <mma.h>       // WMMA
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
// 설정
// =========================================

// M, N, K 는 모두 16의 배수여야 함 (16x16x16 WMMA tile 기준)
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// =========================================
// Tensor Core GEMM kernel (WMMA)
// C[M, N] = A[M, K] * B[K, N]
// A, B: half(row-major), C: float(row-major)
// block당 warp 1개, warp당 C 16x16 타일 1개
// =========================================
__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K)
{
#if __CUDA_ARCH__ < 700
    // Tensor Core 미지원 아키텍처일 경우 그냥 리턴
    return;
#else
    // 한 블록 = 한 warp = 16x16 타일 1개
    int tile_row = blockIdx.y; // [0, M/16)
    int tile_col = blockIdx.x; // [0, N/16)

    // C 타일의 좌상단 글로벌 인덱스
    int row = tile_row * 16;
    int col = tile_col * 16;

    // WMMA fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;

    // 누적값 0으로 초기화
    wmma::fill_fragment(c_frag, 0.0f);

    // K 축을 16단위로 잘라서 누적
    for (int k = 0; k < K; k += 16) {
        // row-major A: A[row, k]
        const half* tile_ptr_A = A + row * K + k;
        // row-major B: B[k, col]
        const half* tile_ptr_B = B + k * N + col;

        // WMMA fragment에 로드
        wmma::load_matrix_sync(a_frag, tile_ptr_A, K);
        wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

        // Tensor Core MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과를 C[row, col] 타일에 저장
    // WMMA는 row-major로 저장 가능
    float* tile_ptr_C = C + row * N + col;
    wmma::store_matrix_sync(tile_ptr_C, c_frag, N, wmma::mem_row_major);
#endif
}

// =========================================
// CPU reference GEMM (float, 검증용)
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
int main()
{
    printf("Tensor Core GEMM (WMMA): M=%d, N=%d, K=%d\n", M, N, K);

    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        printf("ERROR: M, N, K must be multiples of 16 for WMMA.\n");
        return 1;
    }

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    // Host data (float 기준으로 난수 생성 후 half로 캐스팅)
    std::vector<float>  hA_f(sizeA);
    std::vector<float>  hB_f(sizeB);
    std::vector<float>  hC_f(sizeC, 0.0f);
    std::vector<float>  hC_ref(sizeC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA_f[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB_f[i] = dist(rng);

    // CPU reference (float GEMM)
    printf("Running CPU reference GEMM...\n");
    cpu_gemm_ref(hA_f, hB_f, hC_ref, M, N, K);

    // half 버퍼 (A, B용)
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

    dim3 block(32, 1, 1); // warp 1개
    dim3 grid(N / 16, M / 16, 1);

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up
    wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_f.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops  = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);

    // CPU reference와 비교 (FP16 입력이라 오차 약간 있는 것 감안)
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC_f[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU float GEMM: %.6e\n", max_abs_diff);

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
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"     -o tensorcore_gemm.exe tensorcore_gemm.cu

.\tensorcore_gemm.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:wmma_gemm_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\tensorcore_gemm.exe

*/