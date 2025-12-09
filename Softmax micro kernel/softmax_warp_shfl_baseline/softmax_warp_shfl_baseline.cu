// softmax_warp_shfl_baseline.cu
//
// 6.x: Warp-level Softmax Microkernel (N <= 32용)
//
// - 한 warp가 row 하나를 전담
// - shared memory 없이 warp shuffle만으로 max/sum 수행
// - block당 여러 warp를 배치해서 occupancy 확보
//
// 빌드 예시 (Windows, Ampere):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_warp_shfl_baseline.cu -o softmax_warp_shfl_baseline.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\softmax_warp_shfl_baseline.exe

#include <cstdio>
#include <vector>
#include <random>
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

constexpr int WARP_SIZE      = 32;
constexpr int WARPS_PER_BLOCK = 4;   // block = 4 warps = 128 threads
constexpr int BLOCK_THREADS   = WARP_SIZE * WARPS_PER_BLOCK;

// 간단한 warp-level reduce_max
__inline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    // 32 threads 기준 reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// 간단한 warp-level reduce_sum
__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, v, offset);
        v += other;
    }
    return v;
}

/**
 * softmax_warp_shfl_kernel
 *
 *  - scores: [num_rows, N]
 *  - probs : [num_rows, N]
 *  - N <= 32 가정 (단일 warp로 row 전체 처리)
 *
 *  매핑:
 *    - blockDim = 128 = 4 warps
 *    - gridDim.x * WARPS_PER_BLOCK >= num_rows
 *    - warp 하나가 row 하나 처리
 */
__global__ void softmax_warp_shfl_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int num_rows,
    int N
) {
    int tid       = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;        // 0..WARPS_PER_BLOCK-1
    int lane      = tid % WARP_SIZE;        // 0..31
    int row_id    = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row_id >= num_rows) return;

    // row 시작 주소
    const float* row_in  = scores + (size_t)row_id * N;
    float*       row_out = probs  + (size_t)row_id * N;

    // 1) 입력 로드
    //    lane < N 인 thread만 유효 데이터를 갖도록 하고,
    //    나머지는 -inf로 채워서 max/sum에 영향 없게 처리
    float x = -CUDART_INF_F;
    if (lane < N) {
        x = row_in[lane];
    }

    // 2) warp-level max
    float row_max = warp_reduce_max(x);

    // broadcast: lane 0의 값을 모든 lane이 공유
    row_max = __shfl_sync(0xffffffffu, row_max, 0);

    // 3) exp(x - max) 및 sum
    float ex = 0.0f;
    if (lane < N) {
        ex = expf(x - row_max);
    }

    float row_sum = warp_reduce_sum(ex);
    row_sum = __shfl_sync(0xffffffffu, row_sum, 0);

    // 4) normalize
    float prob = 0.0f;
    if (lane < N) {
        prob = ex / row_sum;
        row_out[lane] = prob;
    }
}

/**
 * 메인: naive baseline(6.2)와 비교 가능한 warp-level softmax 마이크로 커널
 *
 *  - NUM_ROWS: 4096
 *  - N       : 32  (warp-per-row에 맞게 설정)
 */
int main() {
    const int NUM_ROWS = 4096;
    const int N        = 32;   // warp-per-row 설계, N <= 32 가정

    size_t num_elems = (size_t)NUM_ROWS * N;
    size_t bytes     = num_elems * sizeof(float);

    std::vector<float> h_scores(num_elems);
    std::vector<float> h_probs(num_elems);

    // 간단한 deterministic 초기화 ([-0.6, 0.6] 근처)
    for (size_t i = 0; i < num_elems; ++i) {
        float v = (float)((int)(i % 100) - 50) / 80.0f;
        h_scores[i] = v;
    }

    float *d_scores = nullptr, *d_probs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scores, bytes));
    CHECK_CUDA(cudaMalloc(&d_probs, bytes));

    CHECK_CUDA(cudaMemcpy(d_scores, h_scores.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_probs, 0, bytes));

    // grid 설정: warp_per_block = 4 -> block 하나당 4 rows
    int rows_per_block = WARPS_PER_BLOCK;
    int grid_x = (NUM_ROWS + rows_per_block - 1) / rows_per_block;

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(grid_x, 1, 1);

    printf("Launching warp-level softmax kernel (shuffle-based)\n");
    printf("NUM_ROWS=%d, N=%d\n", NUM_ROWS, N);
    printf("grid=(%d,1), block=%d\n", grid.x, block.x);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        softmax_warp_shfl_kernel<<<grid, block>>>(
            d_scores, d_probs, NUM_ROWS, N
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    int iters = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_warp_shfl_kernel<<<grid, block>>>(
            d_scores, d_probs, NUM_ROWS, N
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(h_probs.data(), d_probs, bytes, cudaMemcpyDeviceToHost));

    // Row 0 결과 일부 확인
    printf("Row[0, 0..7] probs: ");
    for (int i = 0; i < 8 && i < N; ++i) {
        printf("%.6f ", h_probs[i]);
    }
    printf("\n");

    // FLOPs 대략 계산
    // softmax per element:
    //   - max pass:    1 compare
    //   - exp pass:    1 sub + 1 exp + 1 add(for sum)
    //   - normalize:   1 div
    //   대략 5~6 FLOPs/elem 로 보고 6 FLOPs/elem 사용
    double flops_per_elem = 6.0;
    double total_flops = flops_per_elem * (double)num_elems;
    double avg_s = avg_ms * 1e-3;
    double tflops = (total_flops / avg_s) / 1e12;

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
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 softmax_warp_shfl_baseline.cu -o softmax_warp_shfl_baseline.exe

# 실행
.\softmax_warp_shfl_baseline.exe

# Nsight Compute
ncu --set full --launch-skip 10 --launch-count 1 .\softmax_warp_shfl_baseline.exe
*/
