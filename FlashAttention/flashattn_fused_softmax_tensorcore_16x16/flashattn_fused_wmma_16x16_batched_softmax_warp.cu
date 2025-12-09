// flashattn_fused_wmma_16x16_batched_softmax_warp.cu
// 6.12: Batched Fused QK^T + softmax + PV (WMMA, warp-parallel softmax microkernel)

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int M  = 16;
constexpr int N  = 16;
constexpr int Kd = 16;
constexpr int Dv = 16;

constexpr int NUM_BATCH = 4096;

// ============================
// warp-level helpers
// ============================

__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    // full-warp reduction; lanes 16~31는 -INF를 넣어서 무시
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

// ============================
// softmax microkernel (16x16, warp-parallel)
// scores: [16,16] row-major, input
// probs : [16,16] row-major, output
// ----------------------------
// blockDim.x == 32 (1 warp)를 가정
// 각 row에 대해 lane 0~15가 16개의 column을 나눠서 계산
// ============================

__device__ void softmax_16x16_warp(const float* __restrict__ scores,
                                   float* __restrict__ probs) {
    const int lane = threadIdx.x & 31;
    const float scale = 1.0f / sqrtf(static_cast<float>(Kd));

    for (int row = 0; row < M; ++row) {
        // 1) scale 적용 + row-wise max
        float x = -1e30f;
        if (lane < N) {
            x = scores[row * N + lane] * scale;
        }

        float max_v = warp_allreduce_max(x);

        // 2) exp(x - max) 계산
        float e = 0.0f;
        if (lane < N) {
            e = __expf(x - max_v);
            probs[row * N + lane] = e;
        }

        // 3) sum(exp)
        float sum_exp = warp_allreduce_sum(e);

        // 4) normalize
        if (lane < N) {
            probs[row * N + lane] = probs[row * N + lane] / sum_exp;
        }
    }
}

// ============================
// GPU kernel: batched fused FlashAttention tile
// Q: [B, M, Kd]
// K: [B, Kd, N]
// V: [B, N, Dv]
// O: [B, M, Dv]
// ============================

__global__ void flashattn_fused_wmma_16x16_batched_softmax_warp_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int stride_q,  // M * Kd
    int stride_k,  // Kd * N
    int stride_v,  // N * Dv
    int stride_o   // M * Dv
) {
    const int b = blockIdx.x;
    if (b >= num_batches) return;

    const int lane = threadIdx.x & 31;

    const __half* Q_b = Q + b * stride_q;
    const __half* K_b = K + b * stride_k;
    const __half* V_b = V + b * stride_v;
    float*       O_b  = O + b * stride_o;

    // Shared memory tiles
    __shared__ __half sQ[M * Kd];
    __shared__ __half sK[Kd * N];
    __shared__ __half sV[N * Dv];

    __shared__ float s_scores[M * N];   // QK^T
    __shared__ float s_probs[M * N];    // softmax(P)
    __shared__ __half sP_half[M * N];   // probs -> half
    __shared__ float s_out[M * Dv];     // PV

    // 1) Load Q, K, V to shared
    for (int i = lane; i < M * Kd; i += blockDim.x) {
        sQ[i] = Q_b[i];
    }
    for (int i = lane; i < Kd * N; i += blockDim.x) {
        sK[i] = K_b[i];
    }
    for (int i = lane; i < N * Dv; i += blockDim.x) {
        sV[i] = V_b[i];
    }
    __syncthreads();

    // 2) WMMA QK^T (M=16, N=16, K=16)
    if (lane < 32) {
        wmma::fragment<wmma::matrix_a, M, N, Kd, __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, M, N, Kd, __half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, M, N, Kd, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::load_matrix_sync(q_frag, sQ, Kd);
        wmma::load_matrix_sync(k_frag, sK, Kd);
        wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        wmma::store_matrix_sync(s_scores, acc_frag, N, wmma::mem_row_major);
    }
    __syncthreads();

    // 3) warp-parallel softmax microkernel (16x16)
    softmax_16x16_warp(s_scores, s_probs);
    __syncthreads();

    // 4) probs(float) -> half, shared
    for (int i = lane; i < M * N; i += blockDim.x) {
        sP_half[i] = __float2half(s_probs[i]);
    }
    __syncthreads();

    // 5) WMMA PV (M=16, Dv=16, N=16)
    if (lane < 32) {
        wmma::fragment<wmma::matrix_a, M, Dv, N, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, M, Dv, N, __half, wmma::row_major> v_frag;
        wmma::fragment<wmma::accumulator, M, Dv, N, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::load_matrix_sync(p_frag, sP_half, N);
        wmma::load_matrix_sync(v_frag, sV, Dv);
        wmma::mma_sync(acc_frag, p_frag, v_frag, acc_frag);
        wmma::store_matrix_sync(s_out, acc_frag, Dv, wmma::mem_row_major);
    }
    __syncthreads();

    // 6) Store to global O
    for (int i = lane; i < M * Dv; i += blockDim.x) {
        O_b[i] = s_out[i];
    }
}

// ============================
// CPU reference: fused QK^T + softmax + PV
// ============================

void host_flashattn_fused_ref(
    const std::vector<half>& hQ,
    const std::vector<half>& hK,
    const std::vector<half>& hV,
    std::vector<float>& hO,
    int num_batches
) {
    auto idx_q = [&](int b, int m, int k) {
        return b * (M * Kd) + m * Kd + k;
    };
    auto idx_k = [&](int b, int k, int n) {
        return b * (Kd * N) + k * N + n;
    };
    auto idx_v = [&](int b, int n, int dv) {
        return b * (N * Dv) + n * Dv + dv;
    };
    auto idx_o = [&](int b, int m, int dv) {
        return b * (M * Dv) + m * Dv + dv;
    };

    std::vector<float> scores(M * N);
    std::vector<float> probs(M * N);

    for (int b = 0; b < num_batches; ++b) {
        // 1) QK^T
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < Kd; ++k) {
                    float q = __half2float(hQ[idx_q(b, m, k)]);
                    float k_val = __half2float(hK[idx_k(b, k, n)]);
                    acc += q * k_val;
                }
                scores[m * N + n] = acc;
            }
        }

        // 2) scaling + softmax (row-wise)
        const float scale = 1.0f / std::sqrt(static_cast<float>(Kd));
        for (int m = 0; m < M; ++m) {
            float max_v = -1e30f;
            for (int n = 0; n < N; ++n) {
                float v = scores[m * N + n] * scale;
                scores[m * N + n] = v;
                if (v > max_v) max_v = v;
            }
            float sum_exp = 0.0f;
            for (int n = 0; n < N; ++n) {
                float e = std::exp(scores[m * N + n] - max_v);
                probs[m * N + n] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int n = 0; n < N; ++n) {
                probs[m * N + n] *= inv_sum;
            }
        }

        // 3) PV
        for (int m = 0; m < M; ++m) {
            for (int dv = 0; dv < Dv; ++dv) {
                float acc = 0.0f;
                for (int n = 0; n < N; ++n) {
                    float p = probs[m * N + n];
                    float v = __half2float(hV[idx_v(b, n, dv)]);
                    acc += p * v;
                }
                hO[idx_o(b, m, dv)] = acc;
            }
        }
    }
}

// ============================
// Utility: init data, L2 error, timing
// ============================

float rel_l2_error(const std::vector<float>& a,
                   const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += diff * diff;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(std::sqrt(num / (den + 1e-12)));
}

int main() {
    printf("6.12 Batched Fused QK^T + softmax + PV (WMMA, warp-parallel softmax, M=N=K=Dv=16)\n");

    const int stride_q = M * Kd;
    const int stride_k = Kd * N;
    const int stride_v = N * Dv;
    const int stride_o = M * Dv;

    const size_t size_q = NUM_BATCH * stride_q;
    const size_t size_k = NUM_BATCH * stride_k;
    const size_t size_v = NUM_BATCH * stride_v;
    const size_t size_o = NUM_BATCH * stride_o;

    std::vector<half>  hQ(size_q), hK(size_k), hV(size_v);
    std::vector<float> hO_ref(size_o), hO_gpu(size_o);

    // 랜덤 초기화
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < size_q; ++i) hQ[i] = __float2half(dist(rng));
    for (size_t i = 0; i < size_k; ++i) hK[i] = __float2half(dist(rng));
    for (size_t i = 0; i < size_v; ++i) hV[i] = __float2half(dist(rng));

    // CPU reference
    host_flashattn_fused_ref(hQ, hK, hV, hO_ref, NUM_BATCH);

    // Device 메모리 할당
    __half* dQ = nullptr;
    __half* dK = nullptr;
    __half* dV = nullptr;
    float*  dO = nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, size_q * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, size_k * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, size_v * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, size_o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), size_q * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), size_k * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), size_v * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(NUM_BATCH);

    // Warm-up
    flashattn_fused_wmma_16x16_batched_softmax_warp_kernel<<<grid, block>>>(
        dQ, dK, dV, dO,
        NUM_BATCH,
        stride_q, stride_k, stride_v, stride_o
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    const int NUM_ITERS = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITERS; ++it) {
        flashattn_fused_wmma_16x16_batched_softmax_warp_kernel<<<grid, block>>>(
            dQ, dK, dV, dO,
            NUM_BATCH,
            stride_q, stride_k, stride_v, stride_o
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 결과 가져오기
    CHECK_CUDA(cudaMemcpy(hO_gpu.data(), dO, size_o * sizeof(float), cudaMemcpyDeviceToHost));

    // 정확도
    float rel_err = rel_l2_error(hO_gpu, hO_ref);

    // 일부 출력
    printf("Batch 0, O[0, 0..7] (GPU): ");
    for (int dv = 0; dv < 8; ++dv) {
        printf("%f ", hO_gpu[0 * stride_o + 0 * Dv + dv]);
    }
    printf("\n");

    printf("Batch 0, O_ref[0, 0..7] (CPU): ");
    for (int dv = 0; dv < 8; ++dv) {
        printf("%f ", hO_ref[0 * stride_o + 0 * Dv + dv]);
    }
    printf("\n");

    printf("Relative L2 error over all batches: %.9e\n", rel_err);

    // 성능 추정
    float avg_ms = ms / NUM_ITERS;

    // FLOPs 추정:
    // QK^T: 2 * M * N * Kd
    // PV  : 2 * M * Dv * N
    // softmax: 대략 M * (4 * N) ~ 4 * M * N (대충)
    double flops_per_tile =
        2.0 * M * N * Kd +  // QK^T
        2.0 * M * Dv * N +  // PV
        4.0 * M * N;        // softmax rough
    double total_flops = flops_per_tile * NUM_BATCH;

    double tflops = (total_flops * 1e-12) / (avg_ms * 1e-3);

    printf("NUM_BATCH=%d, M=N=K=Dv=%d\n", NUM_BATCH, M);
    printf("Avg kernel time: %f ms (per launch)\n", avg_ms);
    printf("Approx TFLOPS  : %f\n", tflops);

    // clean up
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -O3 -std=c++17   -arch=sm_86   -o flashattn_fused_wmma_16x16_batched_softmax_warp.exe   flashattn_fused_wmma_16x16_batched_softmax_warp.cu

ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_fused_wmma_16x16_batched_softmax_warp.exe

*/