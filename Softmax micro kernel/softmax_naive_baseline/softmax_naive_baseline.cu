// softmax_naive_baseline.cu
//
// 6.x: Naive Softmax Microkernel (Block-wide Reduction Baseline)
//
// - 입력: scores [NUM_ROWS, N]
// - 출력: probs  [NUM_ROWS, N]  (각 row 별 softmax)
// - 설계 포인트:
//   * 한 row = 한 block
//   * shared memory에 row 전체를 올려두고
//   * 3-pass: max → sum(exp) → normalize
//   * block-wide reduction + __syncthreads() 다수 사용
//   → 이후 warp-level / streaming softmax와 비교할 baseline 커널.
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_naive_baseline.cu -o softmax_naive_baseline.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\softmax_naive_baseline.exe

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ===== 커널 파라미터 =====
constexpr int BLOCK_THREADS = 128;

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * Naive block-wide softmax kernel (baseline)
 *
 * scores: [NUM_ROWS, N]
 * probs : [NUM_ROWS, N]
 *
 * 한 block이 한 row를 담당하고,
 * shared memory에 row 전체를 올려서
 *  - 1-pass: row max
 *  - 2-pass: exp(x - max), sum
 *  - 3-pass: normalize
 */
__global__ void softmax_naive_baseline_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int num_rows,
    int N
) {
    extern __shared__ float smem[]; // [N] + [BLOCK_THREADS]
    float* row_buf = smem;                // size: N
    float* red_buf = smem + N;            // size: BLOCK_THREADS

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) return;

    // ---- 0. row 로드 (global → shared) ----
    const float* row_in  = scores + (size_t)row * N;
    float*       row_out = probs  + (size_t)row * N;

    // row 전체를 shared에 올리기
    for (int i = tid; i < N; i += blockDim.x) {
        row_buf[i] = row_in[i];
    }
    __syncthreads();

    // ---- 1. row max 구하기 (block-wide reduction) ----
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = row_buf[i];
        local_max = fmaxf(local_max, v);
    }

    red_buf[tid] = local_max;
    __syncthreads();

    // reduction (BLOCK_THREADS → 1)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = red_buf[0];  // row 전체 max

    // ---- 2. exp(x - max), sum 구하기 ----
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = expf(row_buf[i] - row_max);
        row_buf[i] = e;   // exp 결과를 그대로 shared에 덮어씀
        local_sum += e;
    }

    red_buf[tid] = local_sum;
    __syncthreads();

    // sum reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_buf[tid] += red_buf[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = red_buf[0];  // row 전체 exp 합

    // ---- 3. normalize ----
    for (int i = tid; i < N; i += blockDim.x) {
        row_buf[i] = row_buf[i] / row_sum;
    }
    __syncthreads();

    // ---- 4. write-back (shared → global) ----
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = row_buf[i];
    }
}

/**
 * 메인: baseline softmax 마이크로커널 실행 + timing + TFLOPS 근사
 */
int main() {
    // FlashAttention과 맞출 수 있도록 대략적인 scale 맞춤
    // NUM_ROWS ~ BH * N 수준으로 생각하면 됨.
    const int NUM_ROWS = 4096;  // 예: BH * N 같은 느낌
    const int N        = 512;   // sequence length or head dim 방향

    size_t num_elem = (size_t)NUM_ROWS * N;
    size_t bytes    = num_elem * sizeof(float);

    std::vector<float> h_scores(num_elem);
    std::vector<float> h_probs(num_elem);

    // deterministic 초기화
    for (size_t i = 0; i < num_elem; ++i) {
        float v = (float)((i % 13) - 6); // -6 ~ +6 사이
        h_scores[i] = v * 0.1f;
    }

    float* d_scores = nullptr;
    float* d_probs  = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scores, bytes));
    CHECK_CUDA(cudaMalloc(&d_probs,  bytes));

    CHECK_CUDA(cudaMemcpy(d_scores, h_scores.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_probs, 0, bytes));

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(NUM_ROWS, 1, 1);

    // shared memory: row_buf[N] + red_buf[BLOCK_THREADS]
    size_t smem_bytes = (size_t)N * sizeof(float) + BLOCK_THREADS * sizeof(float);

    printf("Launching naive softmax baseline kernel\n");
    printf("NUM_ROWS=%d, N=%d\n", NUM_ROWS, N);
    printf("grid=(%d,1), block=%d, smem=%zu bytes\n",
           grid.x, block.x, smem_bytes);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        softmax_naive_baseline_kernel<<<grid, block, smem_bytes>>>(
            d_scores, d_probs, NUM_ROWS, N
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    int iters = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_naive_baseline_kernel<<<grid, block, smem_bytes>>>(
            d_scores, d_probs, NUM_ROWS, N
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(h_probs.data(), d_probs, bytes, cudaMemcpyDeviceToHost));

    // 결과 일부 출력
    printf("Row[0, 0..7] probs: ");
    for (int i = 0; i < 8; ++i) {
        printf("%f ", h_probs[i]);
    }
    printf("\n");

    // 각 row당 FLOP 수 대략 추정:
    //   - max: N compare   → N FLOPs
    //   - exp: N exp + subtract → ~2N FLOPs (rough)
    //   - sum: N add → N FLOPs
    //   - normalize: N div → N FLOPs
    //   총 ~5N FLOPs / row
    double flops_per_row = 5.0 * (double)N;
    double total_flops    = flops_per_row * (double)NUM_ROWS;

    double sec      = avg_ms * 1e-3;
    double tflops   = total_flops / sec / 1e12;

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx FLOPs: %.3e\n", total_flops);
    printf("Approx TFLOPS: %.3f\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_naive_baseline.cu -o softmax_naive_baseline.exe

# 실행
.\softmax_naive_baseline.exe

# Nsight Compute (baseline softmax NCU 분석)
ncu --set full --launch-skip 10 --launch-count 1 .\softmax_naive_baseline.exe
*/
