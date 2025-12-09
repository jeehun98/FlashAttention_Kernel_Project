// flashattn_fused_softmax_block.cu
// 6.5: Block-wide softmax micro-kernel 통합 FlashAttention forward (단일 커널 버전)

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd)                                                         \
    do {                                                                        \
        cudaError_t e = (cmd);                                                  \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e));                                     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int WARP_SIZE      = 32;
constexpr int BLOCK_THREADS  = 128;
constexpr int MAX_N          = 512;   // N <= 512 가정
constexpr int MAX_D          = 64;    // D <= 64 가정

// === warp-level reduce helpers ===
__inline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    // tree-reduction using XOR shuffle
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  8));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  4));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  2));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  1));
    return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v,  8);
    v += __shfl_xor_sync(mask, v,  4);
    v += __shfl_xor_sync(mask, v,  2);
    v += __shfl_xor_sync(mask, v,  1);
    return v;
}

// ============================================================================
// 6.5 Fused FlashAttention-like Forward Kernel
//   - 입력: Q, K, V  [BH, N, D]
//   - 출력: O        [BH, N, D]
//   - 1 block = 1 (bh, row) 쿼리
//   - block 내부에서
//       1) QK^T (해당 row에 대한 score vector 길이 N) 계산
//       2) block-wide softmax (6.4 hybrid 방식)
//       3) PV (softmax weights * V) 로 output row 계산
//   - 단순화를 위해 GEMM 부분은 naive dot-product 로 구현
// ============================================================================

__global__ void flashattn_fused_softmax_block_kernel(
    const float* __restrict__ Q,   // [BH, N, D]
    const float* __restrict__ K,   // [BH, N, D]
    const float* __restrict__ V,   // [BH, N, D]
    float* __restrict__ O,         // [BH, N, D]
    int N,
    int D,
    int BH,
    float scale                    // 1 / sqrt(D)
) {
    // grid:
    //   blockIdx.x = row index (0..N-1)
    //   blockIdx.y = bh index  (0..BH-1)
    int row = blockIdx.x;
    int bh  = blockIdx.y;
    if (row >= N || bh >= BH) return;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Shared memory layout:
    //   s_q[D]        : query row 벡터
    //   s_scores[N]   : score / exp / softmax 결과 (in-place)
    __shared__ float s_q[MAX_D];
    __shared__ float s_scores[MAX_N];

    // block-wide reduce에 사용할 shared 메모리
    __shared__ float warp_max[BLOCK_THREADS / WARP_SIZE];
    __shared__ float warp_sum[BLOCK_THREADS / WARP_SIZE];
    __shared__ float g_max;
    __shared__ float g_sum;

    const int num_warps = BLOCK_THREADS / WARP_SIZE;

    // base offset for this head
    int head_offset = bh * N * D;

    const float* Q_head = Q + head_offset;
    const float* K_head = K + head_offset;
    const float* V_head = V + head_offset;
    float*       O_head = O + head_offset;

    // ------------------------------------------------------------
    // 1) Q row 로드 (shared)
    // ------------------------------------------------------------
    const float* q_row = Q_head + row * D;

    for (int d = tid; d < D; d += BLOCK_THREADS) {
        s_q[d] = q_row[d];
    }
    __syncthreads();

    // ------------------------------------------------------------
    // 2) score_j = <Q[row,:], K[j,:]> * scale  for j in [0, N)
    //    각 thread가 여러 j에 대해 전체 D를 도는 naive dot-product 수행
    // ------------------------------------------------------------
    for (int j = tid; j < N; j += BLOCK_THREADS) {
        const float* k_j = K_head + j * D;
        float acc = 0.0f;
        // dot(Q[row,:], K[j,:])
        #pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
            if (d < D) {
                acc += s_q[d] * k_j[d];
            }
        }
        s_scores[j] = acc * scale; // scaled dot-product
    }
    __syncthreads();

    // ------------------------------------------------------------
    // 3) Block-wide softmax (6.4 hybrid 패턴)
    //      - g_max = max_j s_scores[j]
    //      - g_sum = sum_j exp(s_scores[j] - g_max)
    //      - s_scores[j] = softmax_j
    // ------------------------------------------------------------

    // 3-1) block-wide max
    float local_max = -FLT_MAX;
    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float x = s_scores[idx];
        local_max = fmaxf(local_max, x);
    }

    float w_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        warp_max[warp_id] = w_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_max[lane_id] : -FLT_MAX;
        float block_max = warp_reduce_max(val);
        if (lane_id == 0) g_max = block_max;
    }
    __syncthreads();

    float m = g_max;

    // 3-2) exp(x - m)를 in-place로 s_scores에 저장 + block-wide sum
    float local_sum = 0.0f;
    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float x = s_scores[idx];
        float e = __expf(x - m);
        s_scores[idx] = e;     // in-place update
        local_sum += e;
    }

    float w_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = w_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
        float block_sum = warp_reduce_sum(val);
        if (lane_id == 0) g_sum = block_sum;
    }
    __syncthreads();

    float l = g_sum;

    // 3-3) normalize: s_scores[j] = softmax_j
    for (int idx = tid; idx < N; idx += BLOCK_THREADS) {
        float e = s_scores[idx];
        float p = (l > 0.0f) ? (e / l) : 0.0f;
        s_scores[idx] = p;
    }
    __syncthreads();

    // ------------------------------------------------------------
    // 4) PV: O[row,:] = sum_j softmax_j * V[j,:]
    //    각 thread가 여러 D-dim을 담당
    // ------------------------------------------------------------
    float* o_row = O_head + row * D;

    for (int d = tid; d < D; d += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            float p_ij = s_scores[j];
            const float* v_j = V_head + j * D;
            acc += p_ij * v_j[d];
        }
        o_row[d] = acc;
    }
}

// ============================================================================
// CPU reference: naive Attention = softmax(QK^T * scale) V
//   Q, K, V, O: [BH, N, D]
// ============================================================================
void flashattn_ref(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int N,
    int D,
    int BH,
    float scale
) {
    auto idx = [N, D](int bh, int n, int d) {
        return (bh * N + n) * D + d;
    };

    std::vector<float> scores(N);

    for (int bh = 0; bh < BH; ++bh) {
        for (int i = 0; i < N; ++i) {
            // Q[i,:]
            // 1) scores_j = <Q[i,:], K[j,:]> * scale
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    sum += Q[idx(bh, i, d)] * K[idx(bh, j, d)];
                }
                scores[j] = sum * scale;
            }

            // 2) softmax over j
            float max_x = -1e30f;
            for (int j = 0; j < N; ++j) max_x = std::max(max_x, scores[j]);
            float sum_e = 0.0f;
            for (int j = 0; j < N; ++j) {
                scores[j] = std::exp(scores[j] - max_x);
                sum_e += scores[j];
            }
            for (int j = 0; j < N; ++j) {
                scores[j] /= sum_e;
            }

            // 3) O[i,:] = scores * V
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < N; ++j) {
                    acc += scores[j] * V[idx(bh, j, d)];
                }
                O[idx(bh, i, d)] = acc;
            }
        }
    }
}

// ============================================================================
// main: 간단 벤치 + 검증
// ============================================================================
int main() {
    const int N  = 512;
    const int D  = 64;
    const int BH = 2;      // 너무 크게 잡으면 ref가 느려지니 적당히
    assert(N <= MAX_N);
    assert(D <= MAX_D);

    float scale = 1.0f / std::sqrt((float)D);

    size_t num_elems = (size_t)BH * N * D;
    size_t bytes = num_elems * sizeof(float);

    std::vector<float> h_Q(num_elems);
    std::vector<float> h_K(num_elems);
    std::vector<float> h_V(num_elems);
    std::vector<float> h_O(num_elems, 0.0f);
    std::vector<float> h_O_ref(num_elems, 0.0f);

    // 랜덤 초기화
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_elems; ++i) {
        h_Q[i] = dist(rng);
        h_K[i] = dist(rng);
        h_V[i] = dist(rng);
    }

    // CPU reference
    printf("Running CPU reference...\n");
    flashattn_ref(h_Q, h_K, h_V, h_O_ref, N, D, BH, scale);

    // Device 메모리 할당
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O, bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_THREADS);
    dim3 grid(N, BH);  // (row, bh)

    // Warm-up
    flashattn_fused_softmax_block_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_O, N, D, BH, scale
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    const int iters = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        flashattn_fused_softmax_block_kernel<<<grid, block>>>(
            d_Q, d_K, d_V, d_O, N, D, BH, scale
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // 간단 검증: 첫 head, 첫 row의 일부 출력 + L2 에러
    auto idx = [N, D](int bh, int n, int d) {
        return (bh * N + n) * D + d;
    };

    printf("O[0,0,0..7] (GPU): ");
    for (int d = 0; d < 8; ++d) {
        printf("%f ", h_O[idx(0, 0, d)]);
    }
    printf("\n");

    printf("O_ref[0,0,0..7] (CPU): ");
    for (int d = 0; d < 8; ++d) {
        printf("%f ", h_O_ref[idx(0, 0, d)]);
    }
    printf("\n");

    // L2 error
    double diff2 = 0.0;
    double ref2  = 0.0;
    for (size_t i = 0; i < num_elems; ++i) {
        double diff = (double)h_O[i] - (double)h_O_ref[i];
        diff2 += diff * diff;
        ref2  += (double)h_O_ref[i] * (double)h_O_ref[i];
    }
    double rel_l2 = std::sqrt(diff2 / (ref2 + 1e-8));
    printf("Relative L2 error: %.6e\n", rel_l2);

    // FLOPs 대략 계산:
    //  per head:
    //    - QK^T: N * N * D * 2 (mul+add)
    //    - softmax: N * (몇십 연산) ~ 무시 가능
    //    - PV: N * N * D * 2
    //  → 대략 4 * N * N * D FLOPs
    double flops_per_head = 4.0 * N * (double)N * D;
    double total_flops     = flops_per_head * BH;
    double t_flops         = (total_flops * 1e-12) / (avg_ms * 1e-3);

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx TFLOPS  : %.3f\n", t_flops);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 flashattn_fused_softmax_block.cu -o flashattn_fused_softmax_block.exe


*/