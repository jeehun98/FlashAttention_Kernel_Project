// tensorcore_gemm_fused_epilogue.cu
// Tensor Core GEMM (WMMA, 16x16x16, half->float) + Fused Epilogue
//   out[row, col] = ReLU( (A*B)[row, col] + bias[col] ) + residual[row, col]

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
// 설정
// =========================================
constexpr int DO_VERIFY = 1;   // CPU fused ref와 비교 여부
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// M, N, K는 16의 배수여야 함 (WMMA 16x16x16 용)
static_assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
              "M, N, K must be multiples of 16");

// =========================================
// activation (ReLU)
// =========================================
__device__ __forceinline__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// =========================================
// Tensor Core GEMM + Fused Epilogue kernel
// A: half[M,K], row-major
// B: half[K,N], row-major
// bias: float[N]
// residual: float[M,N]
// C: float[M,N]
// =========================================
__global__ void wmma_gemm_fused_kernel(const half* __restrict__ A,
                                       const half* __restrict__ B,
                                       const float* __restrict__ bias,
                                       const float* __restrict__ residual,
                                       float* __restrict__ C,
                                       int M, int N, int K)
{
#if __CUDA_ARCH__ < 700
    // Tensor Core 미지원 아키텍처일 경우 그냥 리턴
    return;
#else
    // 한 블록 = 한 warp = C 16x16 타일 1개
    int tile_row = blockIdx.y; // [0, M/16)
    int tile_col = blockIdx.x; // [0, N/16)

    int row_base = tile_row * 16;
    int col_base = tile_col * 16;

    // WMMA fragment
    wmma::fragment<wmma::matrix_a,    16, 16, 16, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 16, half,  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                    c_frag;

    // 누적값 0으로 초기화
    wmma::fill_fragment(c_frag, 0.0f);

    // K 축을 16 단위로 잘라서 Tensor Core MMA 반복
    for (int k = 0; k < K; k += 16) {
        const half* tile_ptr_A = A + row_base * K + k;   // A[row_base, k]
        const half* tile_ptr_B = B + k * N + col_base;   // B[k, col_base]

        wmma::load_matrix_sync(a_frag, tile_ptr_A, K);
        wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과를 우선 shared memory에 저장 (warp 전체가 공유)
    __shared__ float shmemC[16 * 16]; // row-major

    // WMMA store → shared memory
    wmma::store_matrix_sync(shmemC, c_frag, 16, wmma::mem_row_major);

    __syncthreads();

    // 각 thread가 tile 내 여러 원소를 처리
    int lane_id = threadIdx.x & 31;  // warp 내 lane id (0..31)

    for (int idx = lane_id; idx < 16 * 16; idx += 32) {
        int r = idx / 16;  // tile 내 row [0..15]
        int c = idx % 16;  // tile 내 col [0..15]

        int global_row = row_base + r;
        int global_col = col_base + c;

        if (global_row < M && global_col < N) {
            float val = shmemC[r * 16 + c];

            // 1) + bias[col]
            if (bias != nullptr) {
                val += bias[global_col];
            }

            // 2) ReLU
            val = relu(val);

            // 3) + residual[row, col]
            if (residual != nullptr) {
                int gidx = global_row * N + global_col;
                val += residual[gidx];
                C[gidx] = val;
            } else {
                C[global_row * N + global_col] = val;
            }
        }
    }
#endif
}

// =========================================
// CPU reference: C = ReLU(A*B + bias) + residual
// (A,B, residual, C: float 기준, 검증용)
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
    printf("Tensor Core Fused GEMM (WMMA + bias + ReLU + residual)\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);

    const size_t sizeA    = static_cast<size_t>(M) * K;
    const size_t sizeB    = static_cast<size_t>(K) * N;
    const size_t sizeC    = static_cast<size_t>(M) * N;
    const size_t sizeBias = static_cast<size_t>(N);

    // Host 데이터 (float 생성 후 half 캐스팅)
    std::vector<float> hA_f(sizeA);
    std::vector<float> hB_f(sizeB);
    std::vector<float> hBias(sizeBias);
    std::vector<float> hResidual(sizeC);
    std::vector<float> hC_f(sizeC, 0.0f);
    std::vector<float> hC_ref(sizeC, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i) hA_f[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB_f[i] = dist(rng);
    for (size_t i = 0; i < sizeBias; ++i) hBias[i] = dist(rng);
    for (size_t i = 0; i < sizeC; ++i) hResidual[i] = dist(rng);

#if DO_VERIFY
    printf("Running CPU fused GEMM reference (float)...\n");
    cpu_gemm_fused_ref(hA_f, hB_f, hBias, hResidual, hC_ref, M, N, K);
#endif

    // A, B를 half로 캐스팅
    std::vector<half> hA(sizeA);
    std::vector<half> hB(sizeB);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = __float2half(hA_f[i]);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = __float2half(hB_f[i]);

    // Device 메모리
    half  *dA = nullptr, *dB = nullptr;
    float *dBias = nullptr, *dResidual = nullptr, *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA,      sizeA    * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB,      sizeB    * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dBias,   sizeBias * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dResidual, sizeC  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC,      sizeC    * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(),      sizeA    * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(),      sizeB    * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBias, hBias.data(),sizeBias * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dResidual, hResidual.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(32, 1, 1);          // warp 1개
    dim3 grid(N / 16, M / 16, 1);  // tile 당 16x16

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up
    wmma_gemm_fused_kernel<<<grid, block>>>(
        dA, dB, dBias, dResidual, dC, M, N, K
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 타이밍
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    wmma_gemm_fused_kernel<<<grid, block>>>(
        dA, dB, dBias, dResidual, dC, M, N, K
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC_f.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops  = 2.0 * static_cast<double>(M) * N * K; // GEMM FLOPs만 기준
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS (GEMM only): %.3f\n", ms, gflops);

#if DO_VERIFY
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC_f[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU fused float ref: %.6e\n", max_abs_diff);
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
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86      -Xcompiler="/EHsc /MT"      -o tensorcore_gemm_fused.exe tensorcore_gemm_fused_epilogue.cu

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:wmma_gemm_fused_kernel     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\tensorcore_gemm_fused.exe

*/