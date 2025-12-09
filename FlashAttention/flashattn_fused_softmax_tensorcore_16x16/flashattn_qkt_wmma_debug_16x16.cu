// flashattn_qkt_wmma_debug_16x16.cu
// 6.7 QK^T WMMA Tile Debug Kernel (M=N=K=16)

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(e));                                 \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// WMMA 설정:
// - A: row-major (M x K)
// - B: col-major (K x N)  => 메모리에 K^T 를 col-major 로 저장
// - C: row-major (M x N)
__global__ void qkt_wmma_16x16_kernel(const __half* __restrict__ Q,
                                      const __half* __restrict__ Kt_colmajor,
                                      float* __restrict__ C) {
    // 단일 타일, 단일 warp 가 처리
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // C = 0 초기화
    wmma::fill_fragment(c_frag, 0.0f);

    // Q: row-major, leading_dim = K
    wmma::load_matrix_sync(a_frag, Q, K);
    // Kt_colmajor: col-major, leading_dim = K
    wmma::load_matrix_sync(b_frag, Kt_colmajor, K);

    // C = Q * K^T
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 결과를 row-major 로 저장
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

// CPU reference: Q (row-major MxK), Kt (col-major KxN) 를 사용하여 C = Q * K^T
void qkt_cpu_ref(const std::vector<float>& Q_h,
                 const std::vector<float>& Kt_col_h,
                 std::vector<float>& C_ref_h) {
    // Q_h: [i, k] = Q[i*K + k]
    // Kt_col_h: col-major (K x N): [k, j] = Kt_col_h[k + j*K]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float q = Q_h[i * K + k];
                float b = Kt_col_h[k + j * K];  // (k, j)
                acc += q * b;
            }
            C_ref_h[i * N + j] = acc;
        }
    }
}

float relative_l2_error(const std::vector<float>& a,
                        const std::vector<float>& b) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = static_cast<double>(a[i]);
        double db = static_cast<double>(b[i]);
        double diff = da - db;
        num += diff * diff;
        den += db * db;
    }
    if (den == 0.0) return std::sqrt(num);
    return std::sqrt(num / den);
}

int main() {
    printf("6.7 QK^T WMMA Tile Debug (M=N=K=16)\n");

    // -----------------------------
    // 1. Host 데이터 준비
    // -----------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q_h(M * K);
    std::vector<float> Kt_col_h(K * N);  // col-major KxN
    std::vector<float> C_ref_h(M * N);
    std::vector<float> C_h(M * N);

    // Q_h: row-major MxK
    for (int i = 0; i < M * K; ++i) {
        Q_h[i] = dist(rng);
    }
    // Kt_col_h: col-major (KxN), 실제로는 "K^T"의 메모리 레이아웃
    // (k, j) 에 random 값 할당
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            Kt_col_h[k + j * K] = dist(rng);
        }
    }

    // CPU reference (float)
    qkt_cpu_ref(Q_h, Kt_col_h, C_ref_h);

    // -----------------------------
    // 2. Device 메모리 할당 & 업로드
    // -----------------------------
    __half *Q_d = nullptr, *Kt_col_d = nullptr;
    float* C_d = nullptr;

    CHECK_CUDA(cudaMalloc(&Q_d, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&Kt_col_d, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&C_d, M * N * sizeof(float)));

    // float -> half 변환 후 업로드
    std::vector<__half> Q_half(M * K);
    std::vector<__half> Kt_half(K * N);
    for (int i = 0; i < M * K; ++i) {
        Q_half[i] = __float2half(Q_h[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        Kt_half[i] = __float2half(Kt_col_h[i]);
    }

    CHECK_CUDA(cudaMemcpy(Q_d, Q_half.data(), M * K * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(Kt_col_d, Kt_half.data(), K * N * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_d, 0, M * N * sizeof(float)));

    // -----------------------------
    // 3. 커널 런치 & 타이밍
    // -----------------------------
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);  // 단일 warp

    // warmup
    qkt_wmma_16x16_kernel<<<grid, block>>>(Q_d, Kt_col_d, C_d);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i) {
        qkt_wmma_16x16_kernel<<<grid, block>>>(Q_d, Kt_col_d, C_d);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(C_h.data(), C_d, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // -----------------------------
    // 4. 결과 비교 & 출력
    // -----------------------------
    float rel_err = relative_l2_error(C_h, C_ref_h);

    printf("C[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", C_h[0 * N + j]);
    }
    printf("\n");

    printf("C_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", C_ref_h[0 * N + j]);
    }
    printf("\n");

    printf("Relative L2 error: %.6e\n", rel_err);

    // FLOPs = 2 * M * N * K
    double flops = 2.0 * M * N * K;
    double t_sec = avg_ms * 1e-3;
    double tflops = flops / t_sec / 1e12;
    printf("Avg kernel time: %.6f ms (over %d iters)\n", avg_ms, iters);
    printf("Approx TFLOPS  : %.6f\n", tflops);

    // cleanup
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(Kt_col_d));
    CHECK_CUDA(cudaFree(C_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 -std=c++17   flashattn_qkt_wmma_debug_16x16.cu   -o flashattn_qkt_wmma_debug_16x16.exe

ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_qkt_wmma_debug_16x16.exe

.\flashattn_qkt_wmma_debug_16x16.exe

*/