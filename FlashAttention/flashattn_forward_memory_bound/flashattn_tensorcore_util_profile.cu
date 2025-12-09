// flashattn_tensorcore_util_profile.cu
//
// 5.6.3: Tensor Core Utilization 분석용 WMMA GEMM 마이크로 커널
//
// - FP16 x FP16 -> FP32 GEMM을 WMMA로만 구성
// - FlashAttention의 QKᵀ / PV 단계와 동일한 형태의 Tensor Core 사용 패턴
// - softmax / cp.async 등은 전부 제거하고 “순수 WMMA 연산”만 남김
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_tensorcore_util_profile.cu -o flashattn_tensorcore_util.exe
//
// Nsight Compute 예시 (Tensor Core Utilization 확인용):
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_tensorcore_util.exe

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

constexpr int WARP_SIZE      = 32;
constexpr int NUM_WARPS      = 4;
constexpr int BLOCK_THREADS  = WARP_SIZE * NUM_WARPS;

// WMMA tile: 16x16x16
// block당 2 x 2 = 4개의 16x16 타일 계산 → 32x32 output tile
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 16;   // WMMA k dimension (16)

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * WMMA 기반 GEMM kernel
 *
 * C = A * B
 *
 * A: [M, K] (row-major, half)
 * B: [K, N] (row-major, half)
 * C: [M, N] (row-major, float)
 *
 * M, N, K는 모두 16의 배수라고 가정 (32의 배수면 더 좋음).
 *
 * block 당:
 *   - 4 warps (128 threads)
 *   - 32x32 tile의 C 서브 블록 계산
 *   - warp 매핑:
 *       warp 0: rows 0..15,  cols 0..15
 *       warp 1: rows 16..31, cols 0..15
 *       warp 2: rows 0..15,  cols 16..31
 *       warp 3: rows 16..31, cols 16..31
 */
__global__ void wmma_gemm_tensorcore_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;   // 0..3
    int lane_id = tid % WARP_SIZE;
    (void)lane_id;

    int block_m = blockIdx.y;  // row tile index
    int block_n = blockIdx.x;  // col tile index

    int row_start = block_m * BLOCK_M;
    int col_start = block_n * BLOCK_N;

    if (row_start >= M || col_start >= N) return;

    // warp별 16x16 C fragment 매핑
    int warp_m = warp_id / 2; // 0 or 1
    int warp_n = warp_id % 2; // 0 or 1

    int c_row0 = row_start + warp_m * 16;
    int c_col0 = col_start + warp_n * 16;

    // WMMA fragment들
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

    // K dimension을 16씩 잘라서 accumulate
    for (int k0 = 0; k0 < K; k0 += 16) {
        const half* a_ptr = A + c_row0 * K + k0;
        const half* b_ptr = B + (k0 * N + c_col0);

        wmma::load_matrix_sync(A_frag, a_ptr, K);
        wmma::load_matrix_sync(B_frag, b_ptr, N);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    // 결과를 C에 저장
    float* c_ptr = C + c_row0 * N + c_col0;
    wmma::store_matrix_sync(c_ptr, C_frag, N, wmma::mem_row_major);
}

// =========================
// 5.6.3용 메인: Tensor Core Utilization 프로파일용
// =========================
int main() {
    // Tensor Core를 충분히 돌리기 위해 M, N, K를 크게 설정
    // (전형적인 테스트용: 512 x 512 x 512, 16의 배수)
    int M = 512;
    int N = 512;
    int K = 512;

    size_t num_A = (size_t)M * K;
    size_t num_B = (size_t)K * N;
    size_t num_C = (size_t)M * N;

    size_t bytes_A = num_A * sizeof(half);
    size_t bytes_B = num_B * sizeof(half);
    size_t bytes_C = num_C * sizeof(float);

    std::vector<half>  hA(num_A);
    std::vector<half>  hB(num_B);
    std::vector<float> hC(num_C);

    // deterministic 초기화
    for (size_t i = 0; i < num_A; ++i) {
        float v = (float)(i % 13) * 0.01f;
        hA[i] = __float2half(v);
    }
    for (size_t i = 0; i < num_B; ++i) {
        float v = (float)(i % 17) * 0.01f;
        hB[i] = __float2half(v);
    }

    half  *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes_A));
    CHECK_CUDA(cudaMalloc(&dB, bytes_B));
    CHECK_CUDA(cudaMalloc(&dC, bytes_C));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytes_C));

    // grid / block 설정
    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M), 1);

    printf("Launching WMMA TensorCore GEMM kernel\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("grid=(%d,%d), block=%d\n", grid.x, grid.y, block.x);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        wmma_gemm_tensorcore_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    int iters = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        wmma_gemm_tensorcore_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes_C, cudaMemcpyDeviceToHost));

    printf("C[0,0..7]: ");
    for (int j = 0; j < 8 && j < N; ++j) {
        printf("%f ", hC[j]);
    }
    printf("\n");

    // FLOPs 계산: GEMM = 2 * M * N * K
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops * 1.0e-12) / (avg_ms * 1.0e-3);

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx FLOPs: %.3e\n", flops);
    printf("Approx TFLOPS: %.3f\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_tensorcore_util_profile.cu -o flashattn_tensorcore_util.exe

# 실행
.\flashattn_tensorcore_util.exe

# Nsight Compute (Tensor Core Utilization 분석)
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_tensorcore_util.exe

- 여기서 확인할 메트릭:
  - GPU Speed Of Light Throughput → Tensor Pipe / Compute Throughput
  - Compute Workload Analysis → SM Busy, Executed Ipc Active
  - Instruction Stats → HMMA / MMA 명령 비율
*/
