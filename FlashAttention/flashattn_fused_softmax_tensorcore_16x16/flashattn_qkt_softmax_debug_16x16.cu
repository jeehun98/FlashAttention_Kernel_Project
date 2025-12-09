// flashattn_qkt_softmax_debug_16x16.cu
// 6.8 QK^T + scaling + softmax Debug (M=N=K=16)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda.h>
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

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// ===== CPU reference: QK^T + scaling + row-wise stable softmax =====
void qkt_softmax_cpu_ref(const std::vector<float> &Q,
                         const std::vector<float> &K_rowmajor,
                         std::vector<float> &P_out,
                         float scale)
{
    // K_rowmajor: (K x N), row-major
    // We conceptually use K^T, but for CPU ref we just index appropriately.
    std::vector<float> scores(M * N, 0.0f);

    // scores = Q * K^T
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float q = Q[i * K + k];           // Q(i,k)
                float kk = K_rowmajor[k * N + j]; // K(k,j)
                acc += q * kk;
            }
            scores[i * N + j] = acc * scale;
        }
    }

    // row-wise stable softmax
    for (int i = 0; i < M; ++i) {
        float max_v = -1e30f;
        for (int j = 0; j < N; ++j) {
            max_v = std::max(max_v, scores[i * N + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(scores[i * N + j] - max_v);
            scores[i * N + j] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; ++j) {
            P_out[i * N + j] = scores[i * N + j] * inv_sum;
        }
    }
}

// ===== GPU kernel: WMMA QK^T + scaling + row-wise softmax =====
__global__ void qkt_softmax_wmma_16x16_kernel(const __half *Q,
                                              const __half *Kt_colmajor,
                                              float *P,
                                              float scale)
{
    // Single block, 32 threads (1 warp)
    __shared__ float s_scores[M * N];

    // 1) WMMA QK^T into accumulator
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ldA: Q, row-major (M x K), leading dim = K
    wmma::load_matrix_sync(a_frag, Q, K);
    // ldB: K^T, col-major (K x N), leading dim = K
    wmma::load_matrix_sync(b_frag, Kt_colmajor, K);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store scores to shared memory as row-major
    wmma::store_matrix_sync(s_scores, c_frag, N, wmma::mem_row_major);
    __syncthreads();

    // 2) scaling + row-wise stable softmax (debug용: 단순 구현)
    int tid = threadIdx.x;

    // 첫 16개의 thread가 각각 한 row씩 책임지는 구조
    if (tid < M) {
        int row = tid;

        float row_vals[N];

        // scale
        for (int j = 0; j < N; ++j) {
            row_vals[j] = s_scores[row * N + j] * scale;
        }

        // max
        float max_v = -1e30f;
        for (int j = 0; j < N; ++j) {
            max_v = fmaxf(max_v, row_vals[j]);
        }

        // exp + sum
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = __expf(row_vals[j] - max_v);
            row_vals[j] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < N; ++j) {
            s_scores[row * N + j] = row_vals[j] * inv_sum;
        }
    }

    __syncthreads();

    // 3) write back to global
    int idx = threadIdx.x;
    int stride = blockDim.x; // 32

    int total = M * N;
    for (int i = idx; i < total; i += stride) {
        P[i] = s_scores[i];
    }
}

// ===== utility: relative L2 error =====
float relative_l2_error(const std::vector<float> &a,
                        const std::vector<float> &b)
{
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += diff * diff;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (den == 0.0) return std::sqrt(num);
    return static_cast<float>(std::sqrt(num / den));
}

int main()
{
    printf("6.8 QK^T + scaling + softmax Debug (M=N=K=16)\n");

    // ===== Host data =====
    std::vector<float> Q_h(M * K);
    std::vector<float> K_row_h(K * N);
    std::vector<float> P_ref_h(M * N);
    std::vector<float> P_gpu_h(M * N);

    // deterministic random
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        Q_h[i] = dist(gen);
    }
    for (int i = 0; i < K * N; ++i) {
        K_row_h[i] = dist(gen);
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(K));

    // CPU reference
    qkt_softmax_cpu_ref(Q_h, K_row_h, P_ref_h, scale);

    // Prepare K^T col-major for WMMA
    // K_row_h: row-major (K x N) => element (k, j) at k*N + j
    // Kt_col_h: col-major (K x N) => element (k, j) at k + j*K
    std::vector<float> Kt_col_h(K * N);
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            Kt_col_h[k + j * K] = K_row_h[k * N + j];
        }
    }

    // Convert to half
    std::vector<__half> Q_half_h(M * K);
    std::vector<__half> Kt_half_h(K * N);
    for (int i = 0; i < M * K; ++i) {
        Q_half_h[i] = __float2half(Q_h[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        Kt_half_h[i] = __float2half(Kt_col_h[i]);
    }

    // ===== Device memory =====
    __half *Q_d = nullptr;
    __half *Kt_d = nullptr;
    float *P_d = nullptr;

    CHECK_CUDA(cudaMalloc(&Q_d, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&Kt_d, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&P_d, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(Q_d, Q_half_h.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(Kt_d, Kt_half_h.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // ===== Kernel launch =====
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        qkt_softmax_wmma_16x16_kernel<<<grid, block>>>(Q_d, Kt_d, P_d, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    const int iters = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        qkt_softmax_wmma_16x16_kernel<<<grid, block>>>(Q_d, Kt_d, P_d, scale);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    // Copy back
    CHECK_CUDA(cudaMemcpy(P_gpu_h.data(), P_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a few values & error
    printf("P[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", P_gpu_h[0 * N + j]);
    }
    printf("\n");

    printf("P_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", P_ref_h[0 * N + j]);
    }
    printf("\n");

    float rel_l2 = relative_l2_error(P_gpu_h, P_ref_h);
    printf("Relative L2 error: %e\n", rel_l2);

    // FLOPs estimate: 2*M*N*K (GEMM) + ~4*M*N (softmax)
    double flops = 2.0 * M * N * K + 4.0 * M * N;
    double tsec = avg_ms * 1e-3;
    double tflops = (flops * 1e-12) / tsec;
    printf("Avg kernel time: %f ms (over %d iters)\n", avg_ms, iters);
    printf("Approx TFLOPS  : %.6f\n", tflops);

    // Cleanup
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(Kt_d));
    CHECK_CUDA(cudaFree(P_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -std=c++17 -O3 -arch=sm_86   flashattn_qkt_softmax_debug_16x16.cu   -o flashattn_qkt_softmax_debug_16x16.exe

ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_qkt_softmax_debug_16x16.exe

*/