// 6.11 Batched Fused QK^T + softmax + PV (WMMA, M=N=K=Dv=16)
//   - 각 block이 16x16 타일 하나를 처리
//   - grid.x = NUM_BATCH 로 여러 타일을 동시에 실행
//   - 여전히 "per-tile FlashAttention" (QK^T + scaling + softmax + PV) 구조를 유지

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                           \
    do {                                                                          \
        cudaError_t e = (cmd);                                                    \
        if (e != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(e));                                       \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

constexpr int M  = 16;
constexpr int N  = 16;
constexpr int K  = 16;
constexpr int Dv = 16;

// batched tile 개수
constexpr int NUM_BATCH = 4096;

// --------------------------- CPU Reference ---------------------------

// 한 타일(Q 16x16, K^T 16x16, V 16x16)에 대해
// O = softmax(Q K^T / sqrt(K)) V
void cpu_flashattn_tile_ref(
    const float* q,   // [M x K]
    const float* kt,  // [K x N]
    const float* v,   // [N x Dv]
    float* o          // [M x Dv]
) {
    float scores[M * N];

    // 1) scores = Q K^T  (M x N)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                const float qv = q[i * K + kk];
                const float kv = kt[kk * N + j];
                acc += qv * kv;
            }
            scores[i * N + j] = acc;
        }
    }

    // 2) scaling + softmax(row-wise)
    const float scale = 1.0f / std::sqrt((float)K);
    float probs[M * N];

    for (int i = 0; i < M; ++i) {
        float row_max = -1e30f;
        for (int j = 0; j < N; ++j) {
            float val = scores[i * N + j] * scale;
            row_max = std::max(row_max, val);
            scores[i * N + j] = val;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(scores[i * N + j] - row_max);
            probs[i * N + j] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < N; ++j) {
            probs[i * N + j] *= inv_sum;
        }
    }

    // 3) O = probs * V   (M x Dv)
    for (int i = 0; i < M; ++i) {
        for (int d = 0; d < Dv; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) {
                acc += probs[i * N + j] * v[j * Dv + d];
            }
            o[i * Dv + d] = acc;
        }
    }
}


// --------------------------- GPU Kernel ---------------------------

__global__ void flashattn_fused_wmma_16x16_batched_kernel(
    const __half* __restrict__ Q,    // [NUM_BATCH, M, K]
    const __half* __restrict__ Kt,   // [NUM_BATCH, K, N]
    const __half* __restrict__ V,    // [NUM_BATCH, N, Dv]
    float* __restrict__ O,           // [NUM_BATCH, M, Dv]
    int num_batch,
    int iters
) {
    // 각 block이 하나의 타일(batch 하나)을 처리
    int b = blockIdx.x;
    if (b >= num_batch) return;

    // 타일 stride
    int q_stride  = M * K;
    int kt_stride = K * N;
    int v_stride  = N * Dv;
    int o_stride  = M * Dv;

    const __half* Qb  = Q  + b * q_stride;
    const __half* Ktb = Kt + b * kt_stride;
    const __half* Vb  = V  + b * v_stride;
    float* Ob         = O  + b * o_stride;

    // shared memory
    __shared__ float scores_smem[M * N];       // scaled scores
    __shared__ float probs_smem[M * N];        // softmax 결과 (FP32)
    __shared__ __half probs_half_smem[M * N];  // PV용 FP16

    for (int it = 0; it < iters; ++it) {
        // 1) Q K^T (WMMA, 16x16x16)
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, 16,16,16, float> s_frag;

        wmma::fill_fragment(s_frag, 0.0f);

        // lda = K, ldb = N
        wmma::load_matrix_sync(q_frag,  Qb,  K);
        wmma::load_matrix_sync(k_frag, Ktb, N);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);

        // scores_smem[M x N] 에 저장
        wmma::store_matrix_sync(scores_smem, s_frag, N, wmma::mem_row_major);
        __syncthreads();

        // 2) scaling + row-wise softmax (thread 0에서 수행 - 구조/정확도 검증용)
        float scale = 1.0f / sqrtf((float)K);

        if (threadIdx.x == 0) {
            // scaling
            for (int i = 0; i < M * N; ++i) {
                scores_smem[i] *= scale;
            }

            // row-wise softmax
            for (int i = 0; i < M; ++i) {
                float row_max = -1e30f;
                for (int j = 0; j < N; ++j) {
                    float v = scores_smem[i * N + j];
                    row_max = fmaxf(row_max, v);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    float e = __expf(scores_smem[i * N + j] - row_max);
                    probs_smem[i * N + j] = e;
                    sum_exp += e;
                }

                float inv_sum = 1.0f / sum_exp;
                for (int j = 0; j < N; ++j) {
                    probs_smem[i * N + j] *= inv_sum;
                }
            }

            // FP32 → FP16 cast
            for (int i = 0; i < M * N; ++i) {
                probs_half_smem[i] = __float2half(probs_smem[i]);
            }
        }
        __syncthreads();

        // 3) PV (WMMA, 16x16x16)
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> v_frag;
        wmma::fragment<wmma::accumulator, 16,16,16, float> o_frag;

        wmma::fill_fragment(o_frag, 0.0f);

        // lda = N, ldb = Dv
        wmma::load_matrix_sync(p_frag, probs_half_smem, N);
        wmma::load_matrix_sync(v_frag, Vb,             Dv);

        wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

        // O[M x Dv] 에 저장
        wmma::store_matrix_sync(Ob, o_frag, Dv, wmma::mem_row_major);
        __syncthreads();
    }
}


// --------------------------- Main ---------------------------

int main() {
    printf("6.11 Batched Fused QK^T + softmax + PV (WMMA, M=N=K=Dv=16)\n");

    int num_batch = NUM_BATCH;
    int q_stride  = M * K;
    int kt_stride = K * N;
    int v_stride  = N * Dv;
    int o_stride  = M * Dv;

    size_t q_bytes  = (size_t)num_batch * q_stride  * sizeof(__half);
    size_t kt_bytes = (size_t)num_batch * kt_stride * sizeof(__half);
    size_t v_bytes  = (size_t)num_batch * v_stride  * sizeof(__half);
    size_t o_bytes  = (size_t)num_batch * o_stride  * sizeof(float);

    // Host buffers (FP32 ref용, FP16 변환용)
    std::vector<float> h_Q_f(num_batch * q_stride);
    std::vector<float> h_Kt_f(num_batch * kt_stride);
    std::vector<float> h_V_f(num_batch * v_stride);
    std::vector<float> h_O_ref(num_batch * o_stride);

    std::vector<__half> h_Q(num_batch * q_stride);
    std::vector<__half> h_Kt(num_batch * kt_stride);
    std::vector<__half> h_V(num_batch * v_stride);
    std::vector<float>  h_O(num_batch * o_stride);

    // 랜덤 초기화
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < h_Q_f.size(); ++i)  h_Q_f[i]  = dist(rng);
    for (size_t i = 0; i < h_Kt_f.size(); ++i) h_Kt_f[i] = dist(rng);
    for (size_t i = 0; i < h_V_f.size(); ++i)  h_V_f[i]  = dist(rng);

    // CPU reference (batch 단위)
    for (int b = 0; b < num_batch; ++b) {
        const float* q_b   = h_Q_f.data()   + b * q_stride;
        const float* kt_b  = h_Kt_f.data()  + b * kt_stride;
        const float* v_b   = h_V_f.data()   + b * v_stride;
        float*       o_b   = h_O_ref.data() + b * o_stride;

        cpu_flashattn_tile_ref(q_b, kt_b, v_b, o_b);
    }

    // FP32 → FP16
    for (size_t i = 0; i < h_Q_f.size(); ++i)  h_Q[i]  = __float2half(h_Q_f[i]);
    for (size_t i = 0; i < h_Kt_f.size(); ++i) h_Kt[i] = __float2half(h_Kt_f[i]);
    for (size_t i = 0; i < h_V_f.size(); ++i)  h_V[i]  = __float2half(h_V_f[i]);

    // Device alloc
    __half* d_Q  = nullptr;
    __half* d_Kt = nullptr;
    __half* d_V  = nullptr;
    float*  d_O  = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q,  q_bytes));
    CHECK_CUDA(cudaMalloc(&d_Kt, kt_bytes));
    CHECK_CUDA(cudaMalloc(&d_V,  v_bytes));
    CHECK_CUDA(cudaMalloc(&d_O,  o_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q,  h_Q.data(),  q_bytes,  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Kt, h_Kt.data(), kt_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V,  h_V.data(),  v_bytes,  cudaMemcpyHostToDevice));

    // Launch 설정
    dim3 block(32, 1, 1);           // 1 warp per tile
    dim3 grid(num_batch, 1, 1);    // batch당 block 1개

    int iters = 1000;

    // Warm-up
    flashattn_fused_wmma_16x16_batched_kernel<<<grid, block>>>(d_Q, d_Kt, d_V, d_O,
                                                                num_batch, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    flashattn_fused_wmma_16x16_batched_kernel<<<grid, block>>>(d_Q, d_Kt, d_V, d_O,
                                                                num_batch, iters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters; // iteration당 평균 시간

    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, o_bytes, cudaMemcpyDeviceToHost));

    // 정확도 체크: 배치 0, row 0 앞 8개 비교
    printf("Batch 0, O[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O[0 * o_stride + j]);
    }
    printf("\n");

    printf("Batch 0, O_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O_ref[0 * o_stride + j]);
    }
    printf("\n");

    // 전체 relative L2 error
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < h_O.size(); ++i) {
        double diff = (double)h_O[i] - (double)h_O_ref[i];
        num += diff * diff;
        den += (double)h_O_ref[i] * (double)h_O_ref[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));
    printf("Relative L2 error over all batches: %.9e\n", rel_l2);

    // FLOPs 계산 (대략적인 추정)
    //   QK^T: 2 * M * N * K
    //   PV:   2 * M * N * Dv
    //   softmax + scaling: ~5 * M * N
    double flops_per_tile =
        2.0 * M * N * K +
        2.0 * M * N * Dv +
        5.0 * M * N;

    double total_flops = flops_per_tile * (double)num_batch;
    double tflops = (total_flops / (ms * 1e-3)) / 1e12;

    printf("NUM_BATCH=%d, M=N=K=Dv=%d\n", num_batch, M);
    printf("Avg kernel time: %.6f ms (per launch)\n", ms);
    printf("Approx TFLOPS  : %.6f\n", tflops);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_Kt));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -O3 -std=c++17   -arch=sm_86   -o flashattn_fused_wmma_16x16_batched.exe   flashattn_fused_wmma_16x16_batched.cu

ncu --set full .\flashattn_fused_wmma_16x16_batched.exe

*/