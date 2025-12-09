// flashattn_pv_wmma_debug_16x16.cu
// 6.9 PV WMMA Tile Debug (M=N=K=16)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

// GPU kernel: O = P * V
// P: [M x K] (row-major, __half)
// V: [K x N] (row-major, __half)
// O: [M x N] (row-major, float)
__global__ void pv_wmma_16x16_kernel(const __half* __restrict__ P,
                                     const __half* __restrict__ V,
                                     float* __restrict__ O)
{
    // 단일 warp, 단일 16x16 타일만 처리 (디버그용)
    if (threadIdx.x >= 32 || blockIdx.x != 0) return;

    wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // leading dimension = stride = K (16) / N(16) for row-major
    wmma::load_matrix_sync(a_frag, P, K);
    wmma::load_matrix_sync(b_frag, V, N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // row-major로 store
    wmma::store_matrix_sync(O, c_frag, N, wmma::mem_row_major);
}

// CPU reference: O_ref = P * V
void cpu_pv_ref(const std::vector<float>& P,
                const std::vector<float>& V,
                std::vector<float>& O)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += P[i * K + k] * V[k * N + j];
            }
            O[i * N + j] = acc;
        }
    }
}

float rel_l2_error(const std::vector<float>& a,
                   const std::vector<float>& b)
{
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += diff * diff;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (den == 0.0) return std::sqrt(num);
    return std::sqrt(num / den);
}

int main()
{
    printf("6.9 PV WMMA Tile Debug (M=N=K=16)\n");

    // -----------------------------
    // 1. Host 데이터 준비
    // -----------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.0f, 1.0f);

    // P: softmax 이후라고 가정 → 각 row를 확률분포로 normalize
    std::vector<float> h_P(M * K);
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float v = dist_pos(rng);
            h_P[i * K + k] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int k = 0; k < K; ++k) {
            h_P[i * K + k] *= inv;
        }
    }

    // V: 일반 dense value matrix
    std::vector<float> h_V(K * N);
    for (int i = 0; i < K * N; ++i) {
        h_V[i] = dist(rng);
    }

    // GPU용 half 버전
    std::vector<__half> h_P_half(M * K);
    std::vector<__half> h_V_half(K * N);
    for (int i = 0; i < M * K; ++i)
        h_P_half[i] = __float2half(h_P[i]);
    for (int i = 0; i < K * N; ++i)
        h_V_half[i] = __float2half(h_V[i]);

    // CPU reference 출력
    std::vector<float> h_O_ref(M * N, 0.0f);
    cpu_pv_ref(h_P, h_V, h_O_ref);

    // -----------------------------
    // 2. Device 메모리 할당 & 복사
    // -----------------------------
    __half *d_P = nullptr, *d_V = nullptr;
    float* d_O = nullptr;

    CHECK_CUDA(cudaMalloc(&d_P, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_O, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_P, h_P_half.data(), M * K * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V_half.data(), K * N * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // -----------------------------
    // 3. 커널 실행 & 타이밍
    // -----------------------------
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);  // 한 warp

    // warm up
    pv_wmma_16x16_kernel<<<grid, block>>>(d_P, d_V, d_O);
    CHECK_CUDA(cudaDeviceSynchronize());

    int iters = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        pv_wmma_16x16_kernel<<<grid, block>>>(d_P, d_V, d_O);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    // -----------------------------
    // 4. 결과 검증
    // -----------------------------
    std::vector<float> h_O(M * N);
    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float err = rel_l2_error(h_O, h_O_ref);

    printf("O[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O[0 * N + j]);
    }
    printf("\n");

    printf("O_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", h_O_ref[0 * N + j]);
    }
    printf("\n");

    printf("Relative L2 error: %.6e\n", err);

    // FLOPs: M * N * K * 2 (mul + add)
    double flops = static_cast<double>(M) *
                   static_cast<double>(N) *
                   static_cast<double>(K) * 2.0;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("Avg kernel time: %f ms (over %d iters)\n", avg_ms, iters);
    printf("Approx TFLOPS  : %f\n", tflops);

    // -----------------------------
    // 5. 정리
    // -----------------------------
    CHECK_CUDA(cudaFree(d_P));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -std=c++17 -O3 -arch=sm_86   flashattn_pv_wmma_debug_16x16.cu   -o flashattn_pv_wmma_debug_16x16.exe

ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_pv_wmma_debug_16x16.exe


*/