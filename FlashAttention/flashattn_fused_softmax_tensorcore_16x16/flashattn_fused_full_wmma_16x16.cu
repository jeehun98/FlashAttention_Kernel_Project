// flashattn_fused_full_wmma_16x16.cu
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t e = (call);                                                  \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int M = 16;  // rows of Q, rows of O, rows of P
constexpr int N = 16;  // cols of K, cols of scores/P, rows of V
constexpr int K = 16;  // head dimension d
constexpr int Dv = 16; // value dimension (cols of V, cols of O)

// ---------------- CPU reference: full attention tile ----------------

void cpu_attention_ref(const std::vector<float>& Q,
                       const std::vector<float>& Kt, // K^T, shape [K x N]
                       const std::vector<float>& V,  // shape [N x Dv]
                       std::vector<float>& O_ref)    // shape [M x Dv]
{
    std::vector<float> scores(M * N);
    std::vector<float> probs(M * N);

    // 1) scores = Q * K^T  (Q: [M x K], Kt: [K x N])
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float q = Q[i * K + k];
                float k_val = Kt[k * N + j];
                acc += q * k_val;
            }
            scores[i * N + j] = acc;
        }
    }

    // 2) scaling
    float scale = 1.0f / std::sqrt(static_cast<float>(K));
    for (int i = 0; i < M * N; ++i) {
        scores[i] *= scale;
    }

    // 3) row-wise softmax → probs
    for (int i = 0; i < M; ++i) {
        float row_max = -1e30f;
        for (int j = 0; j < N; ++j) {
            row_max = std::max(row_max, scores[i * N + j]);
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

    // 4) O_ref = probs * V  (probs: [M x N], V: [N x Dv])
    for (int i = 0; i < M; ++i) {
        for (int d = 0; d < Dv; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) {
                float p = probs[i * N + j];
                float v = V[j * Dv + d];
                acc += p * v;
            }
            O_ref[i * Dv + d] = acc;
        }
    }
}

// ---------------- Device kernel: fused WMMA tile ----------------

__global__ void flashattn_fused_wmma_16x16_kernel(const __half* __restrict__ Q,
                                                  const __half* __restrict__ Kt, // [K x N]
                                                  const __half* __restrict__ V,  // [N x Dv]
                                                  float* __restrict__ O,         // [M x Dv]
                                                  int iters)
{
    if (threadIdx.x >= 32 || blockIdx.x != 0) return;

    __shared__ float scores_smem[M * N];      // QK^T scaled
    __shared__ float probs_smem[M * N];       // softmax result
    __shared__ __half probs_half_smem[M * N]; // casted to half for WMMA PV

    for (int it = 0; it < iters; ++it) {
        // ----- 1) QK^T with WMMA -----
        wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> s_frag;

        wmma::fill_fragment(s_frag, 0.0f);

        // Q: [M x K], row-major, lda = K
        wmma::load_matrix_sync(q_frag, Q, K);
        // Kt: [K x N], row-major, ldb = N (this is K^T)
        wmma::load_matrix_sync(k_frag, Kt, N);

        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);

        // Store scores to shared memory as row-major [M x N]
        wmma::store_matrix_sync(scores_smem, s_frag, N, wmma::mem_row_major);
        __syncthreads();

        // ----- 2) Scaling + row-wise softmax in FP32 -----
        float scale = 1.0f / sqrtf(static_cast<float>(K));

        if (threadIdx.x == 0) {
            // scale
            for (int i = 0; i < M * N; ++i) {
                scores_smem[i] *= scale;
            }
            // softmax per row
            for (int i = 0; i < M; ++i) {
                float row_max = -1e30f;
                for (int j = 0; j < N; ++j) {
                    row_max = fmaxf(row_max, scores_smem[i * N + j]);
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
            // cast probs to half
            for (int i = 0; i < M * N; ++i) {
                probs_half_smem[i] = __float2half(probs_smem[i]);
            }
        }
        __syncthreads();

        // ----- 3) PV with WMMA -----
        wmma::fragment<wmma::matrix_a, M, Dv, N, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, M, Dv, N, __half, wmma::row_major> v_frag;
        wmma::fragment<wmma::accumulator, M, Dv, N, float> o_frag;

        wmma::fill_fragment(o_frag, 0.0f);

        // probs_half: [M x N], lda = N
        wmma::load_matrix_sync(p_frag, probs_half_smem, N);
        // V: [N x Dv], ldb = Dv
        wmma::load_matrix_sync(v_frag, V, Dv);

        wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

        // store O (global) as [M x Dv], row-major, leading dim Dv
        wmma::store_matrix_sync(O, o_frag, Dv, wmma::mem_row_major);

        __syncthreads();
    }
}

// ---------------- Utility: relative L2 error ----------------

float relative_l2_error(const std::vector<float>& ref,
                        const std::vector<float>& out)
{
    double num = 0.0;
    double den = 0.0;
    int n = static_cast<int>(ref.size());
    for (int i = 0; i < n; ++i) {
        double diff = static_cast<double>(ref[i]) - static_cast<double>(out[i]);
        num += diff * diff;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    if (den == 0.0) return 0.0f;
    return static_cast<float>(std::sqrt(num / den));
}

// ---------------- main ----------------

int main()
{
    printf("6.10 Fused QK^T + softmax + PV (WMMA, M=N=K=Dv=16)\n");

    // Host buffers
    std::vector<float> h_Q(M * K);
    std::vector<float> h_Kt(K * N); // store K^T directly as [K x N]
    std::vector<float> h_V(N * Dv);
    std::vector<float> h_O_ref(M * Dv);
    std::vector<float> h_O(M * Dv);

    // Random init
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        h_Q[i] = dist(rng);
    }
    for (int i = 0; i < K * N; ++i) {
        h_Kt[i] = dist(rng); // this is K^T logically
    }
    for (int i = 0; i < N * Dv; ++i) {
        h_V[i] = dist(rng);
    }

    // CPU reference
    cpu_attention_ref(h_Q, h_Kt, h_V, h_O_ref);

    // Convert to half
    std::vector<__half> h_Q_half(M * K);
    std::vector<__half> h_Kt_half(K * N);
    std::vector<__half> h_V_half(N * Dv);

    for (int i = 0; i < M * K; ++i)
        h_Q_half[i] = __float2half(h_Q[i]);
    for (int i = 0; i < K * N; ++i)
        h_Kt_half[i] = __float2half(h_Kt[i]);
    for (int i = 0; i < N * Dv; ++i)
        h_V_half[i] = __float2half(h_V[i]);

    // Device buffers
    __half *d_Q = nullptr, *d_Kt = nullptr, *d_V = nullptr;
    float *d_O = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q,  M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Kt, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V,  N * Dv * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_O,  M * Dv * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q,  h_Q_half.data(),  M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Kt, h_Kt_half.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V,  h_V_half.data(),  N * Dv * sizeof(__half), cudaMemcpyHostToDevice));

    // Kernel launch
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    int iters = 1000;

    // Warmup
    flashattn_fused_wmma_16x16_kernel<<<grid, block>>>(d_Q, d_Kt, d_V, d_O, 1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    flashattn_fused_wmma_16x16_kernel<<<grid, block>>>(d_Q, d_Kt, d_V, d_O, iters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / iters;

    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, M * Dv * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    printf("O[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O[0 * Dv + j]);
    }
    printf("\n");

    printf("O_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O_ref[0 * Dv + j]);
    }
    printf("\n");

    float rel_l2 = relative_l2_error(h_O_ref, h_O);
    printf("Relative L2 error: %e\n", rel_l2);

    // FLOPs: QK^T + PV (softmax는 rough하게 무시)
    double flops = 2.0 * M * K * N + 2.0 * M * N * Dv; // 2 GEMM
    double tflops = flops * 1e-9 / (avg_ms * 1e-3);
    printf("Avg kernel time: %f ms (over %d iters)\n", avg_ms, iters);
    printf("Approx TFLOPS  : %f\n", tflops);

    // Cleanup
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_Kt));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -std=c++17 -O3 -arch=sm_86   flashattn_fused_full_wmma_16x16.cu   -o flashattn_fused_full_wmma_16x16.exe

ncu --set full .\flashattn_fused_full_wmma_16x16.exe

*/