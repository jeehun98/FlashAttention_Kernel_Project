#include <math_constants.h>
// softmax_block_hybrid.cu
//
// 6.4: Block-wide Hybrid Softmax Micro-Kernel
//
// - 각 block 이 하나의 row 를 처리
// - 여러 warp 가 cooperative reduction(max/sum) 수행
// - warp-level reduce (__shfl_xor_sync) + block-wide 두 단계 reduce
// - N <= MAX_N (여기선 512) 가정
//
// 빌드 예시 (Windows, sm_86):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_block_hybrid.cu -o softmax_block_hybrid.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\softmax_block_hybrid.exe

#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

constexpr int WARP_SIZE     = 32;
constexpr int BLOCK_THREADS = 128;
constexpr int MAX_N         = 512;  // row 최대 길이 (FlashAttention score tile 기준)

// ===========================
// warp-level reduction helper
// ===========================
__inline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    // 32-thread warp 기준
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, v, offset);
        v += other;
    }
    return v;
}

// =========================================================
// Block-wide Hybrid Softmax Kernel (N <= MAX_N, 1 block = 1 row)
// =========================================================
//
// X: [num_rows, N]
// P: [num_rows, N]
// num_rows: 총 row 개수
// N: 각 row 길이 (<= MAX_N)
//
// 알고리즘:
//  1) block-wide max reduce
//  2) exp(x - max) 계산 + shared 에 저장 + block-wide sum reduce
//  3) shared 에 있는 exp 를 sum 으로 normalize 후 P 로 write
//
__global__ void softmax_block_hybrid_kernel(
    const float* __restrict__ X,
    float* __restrict__ P,
    int num_rows,
    int N
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid     = threadIdx.x;
    int lane_id = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    const int num_warps = BLOCK_THREADS / WARP_SIZE;

    const float* row_x = X + (size_t)row * N;
    float*       row_p = P + (size_t)row * N;

    // shared memory: 각 row 의 exp(x - max) 저장
    __shared__ float s_exp[MAX_N];
    __shared__ float warp_max[num_warps];
    __shared__ float warp_sum[num_warps];
    __shared__ float g_max;
    __shared__ float g_sum;

    // -----------------------------
    // 1. block-wide max reduce
    // -----------------------------
    float local_max = -FLT_MAX;

    // strided load: block 내 모든 thread 가 row 전체를 커버
    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float x = row_x[idx];
        local_max = fmaxf(local_max, x);
    }

    // warp-level reduce max
    float w_max = warp_reduce_max(local_max);

    // warp 마다 결과를 shared 에 저장
    if (lane_id == 0) {
        warp_max[warp_id] = w_max;
    }
    __syncthreads();

    // warp0 이 warp_max 들을 다시 reduce 해서 block-wide max 계산
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_max[lane_id] : -FLT_MAX;
        float block_max = warp_reduce_max(val);
        if (lane_id == 0) {
            g_max = block_max;
        }
    }
    __syncthreads();

    float m = g_max; // 모든 thread 에 broadcast 완료

    // -----------------------------
    // 2. exp(x - m) + block-wide sum
    // -----------------------------
    float local_sum = 0.0f;

    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float x = row_x[idx];
        float e = __expf(x - m);
        s_exp[idx] = e;  // 나중에 normalize 에 사용
        local_sum += e;
    }

    // warp-level reduce sum
    float w_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = w_sum;
    }
    __syncthreads();

    // warp0 이 warp_sum 들을 다시 reduce 해서 block-wide sum 계산
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
        float block_sum = warp_reduce_sum(val);
        if (lane_id == 0) {
            g_sum = block_sum;
        }
    }
    __syncthreads();

    float l = g_sum;

    // -----------------------------
    // 3. normalize & write out
    // -----------------------------
    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float e = s_exp[idx];
        float p = (l > 0.0f) ? (e / l) : 0.0f;
        row_p[idx] = p;
    }
}

// =========================
// 6.4 테스트용 main
// =========================
int main() {
    const int NUM_ROWS = 4096;
    const int N        = 512;  // <= MAX_N

    printf("Launching block-wide hybrid softmax kernel\n");
    printf("NUM_ROWS=%d, N=%d\n", NUM_ROWS, N);

    size_t num_elem = (size_t)NUM_ROWS * N;
    size_t bytes    = num_elem * sizeof(float);

    std::vector<float> hX(num_elem);
    std::vector<float> hP(num_elem);

    // 입력 초기화 (단순 패턴)
    for (size_t i = 0; i < num_elem; ++i) {
        hX[i] = 0.001f * (float)(i % 97);  // 약간의 variation
    }

    float* dX = nullptr;
    float* dP = nullptr;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dP, bytes));

    CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dP, 0, bytes));

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(NUM_ROWS, 1, 1);

    printf("grid=(%d,1), block=%d\n", grid.x, block.x);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        softmax_block_hybrid_kernel<<<grid, block>>>(dX, dP, NUM_ROWS, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    int iters = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_block_hybrid_kernel<<<grid, block>>>(dX, dP, NUM_ROWS, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hP.data(), dP, bytes, cudaMemcpyDeviceToHost));

    // 첫 row 일부 출력
    printf("Row[0, 0..7] probs: ");
    for (int j = 0; j < 8 && j < N; ++j) {
        printf("%f ", hP[j]);
    }
    printf("\n");

    // 확률 합 확인
    float sum0 = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum0 += hP[j];
    }
    printf("Row[0] sum(p): %f\n", sum0);

    // 대략적인 FLOPs (softmax per element ~5 FLOPs 가정: max(sum 포함)/exp/div 등)
    double flops = (double)NUM_ROWS * (double)N * 5.0;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx FLOPs: %.3e\n", flops);
    printf("Approx TFLOPS: %.3f\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dP));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_block_hybrid.cu -o softmax_block_hybrid.exe

# 실행
.\softmax_block_hybrid.exe

# Nsight Compute
ncu --set full --launch-skip 10 --launch-count 1 .\softmax_block_hybrid.exe
*/
