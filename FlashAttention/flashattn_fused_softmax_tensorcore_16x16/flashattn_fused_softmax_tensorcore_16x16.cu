// flashattn_fused_softmax_tensorcore_16x16.cu
// 6.6 Tensor Core 기반 FlashAttention 타일 마이크로커널
//
//  - 단일 타일 (M = N = D = 16)에 대해
//    QK^T -> softmax -> PV 를 하나의 커널 안에서 WMMA(Tensor Core)로 수행
//  - 구조 시연용 마이크로커널 (이걸 기반으로 더 큰 타일/전체 시퀀스로 확장)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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

// ---------------------------------------------------------
// 문제 크기: 16x16 타일 마이크로커널
// ---------------------------------------------------------
constexpr int M = 16;   // #queries in tile
constexpr int N = 16;   // #keys/values in tile
constexpr int D = 16;   // head_dim (K dimension)

// ---------------------------------------------------------
// Device kernel: QK^T + softmax + PV (Tensor Core/WMMA 기반)
// ---------------------------------------------------------
__global__ void flashattn_fused_softmax_tensorcore_kernel(
    const __half* __restrict__ Q,    // [M x D] row-major
    const __half* __restrict__ K_T,  // [D x N] row-major (K^T)
    const __half* __restrict__ V,    // [N x D] row-major
    float* __restrict__ O            // [M x D] row-major (FP32)
) {
    // 이 커널은 단일 타일만 처리한다고 가정 (gridDim = (1,1,1), blockDim = 32)
    // 하나의 warp(32 threads)만 사용
    const int lane_id = threadIdx.x % 32;

    // Shared memory: scores와 softmax 결과를 float로 저장
    __shared__ float scores[M * N];
    __shared__ float probs[M * N];

    // 1) QK^T GEMM (M x D) * (D x N) -> (M x N)
    //    WMMA: 16x16x16 한 타일만 계산하므로 fragment 하나로 충분
    wmma::fragment<wmma::matrix_a, M, N, D, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, D, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, D, float>                c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Q: [M x D], row-major, ld = D
    // K_T: [D x N], row-major, ld = N
    // 한 번의 mma_sync로 전체 16x16 점수를 계산
    wmma::load_matrix_sync(a_frag, Q, D);      // A = Q
    wmma::load_matrix_sync(b_frag, K_T, N);    // B = K^T
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // scores(M x N, row-major)에 저장
    wmma::store_matrix_sync(scores, c_frag, N, wmma::mem_row_major);

    __syncthreads();

    // 2) row-wise softmax(scores) -> probs
    //    M=16, N=16 이라서 warp 내에서 각 thread가 0~1개 row 정도를 담당하는 식으로 처리
    //    (병렬성을 크게 신경 쓰기보단 구조의 명확성에 초점)
    if (lane_id < M) {
        int row = lane_id;
        float* row_scores = &scores[row * N];

        // (1) max 검색
        float maxv = -1e30f;
        for (int j = 0; j < N; ++j) {
            maxv = fmaxf(maxv, row_scores[j]);
        }

        // (2) exp(score - max) 및 sum
        float sum = 0.f;
        for (int j = 0; j < N; ++j) {
            float e = expf(row_scores[j] - maxv);
            row_scores[j] = e;  // scores를 임시 버퍼로 재사용
            sum += e;
        }

        // (3) normalize
        float inv_sum = 1.f / (sum + 1e-9f);
        for (int j = 0; j < N; ++j) {
            probs[row * N + j] = row_scores[j] * inv_sum;
        }
    }

    __syncthreads();

    // 3) PV GEMM (M x N) * (N x D) -> (M x D)
    //    WMMA는 FP16 A/B + FP32 accumulator 이므로
    //    probs(float)을 half로 변환해서 shared memory에 둔다.
    __shared__ __half probs_half[M * N];

    // float -> half 변환 (warp 내 코어스 그레인)
    for (int idx = lane_id; idx < M * N; idx += 32) {
        probs_half[idx] = __float2half(probs[idx]);
    }
    __syncthreads();

    // V: [N x D], row-major
    // Out: [M x D], row-major
    // 다시 WMMA fragment 사용
    wmma::fragment<wmma::matrix_a, M, D, N, __half, wmma::row_major> a_frag2;
    wmma::fragment<wmma::matrix_b, M, D, N, __half, wmma::row_major> b_frag2;
    wmma::fragment<wmma::accumulator, M, D, N, float>                c_frag2;

    wmma::fill_fragment(c_frag2, 0.0f);

    // A = probs_half (M x N)
    // B = V         (N x D)
    // 각각 row-major
    wmma::load_matrix_sync(a_frag2, probs_half, N);  // ld = N
    wmma::load_matrix_sync(b_frag2, V, D);           // ld = D
    wmma::mma_sync(c_frag2, a_frag2, b_frag2, c_frag2);

    // 결과를 O(M x D)에 저장 (row-major)
    if (threadIdx.x == 0) {
        wmma::store_matrix_sync(O, c_frag2, D, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------
// CPU 레퍼런스: QK^T -> softmax -> PV (float 버전)
// ---------------------------------------------------------
void cpu_reference(
    const std::vector<float>& Q,    // [M x D]
    const std::vector<float>& K_T,  // [D x N] (K^T)
    const std::vector<float>& V,    // [N x D]
    std::vector<float>& O_ref       // [M x D]
) {
    // 1) S = Q * K_T -> [M x N]
    std::vector<float> S(M * N, 0.f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < D; ++k) {
                acc += Q[i * D + k] * K_T[k * N + j];
            }
            S[i * N + j] = acc;
        }
    }

    // 2) row-wise softmax
    std::vector<float> P(M * N, 0.f);
    for (int i = 0; i < M; ++i) {
        float maxv = -1e30f;
        for (int j = 0; j < N; ++j) {
            maxv = std::max(maxv, S[i * N + j]);
        }
        float sum = 0.f;
        for (int j = 0; j < N; ++j) {
            float e = std::exp(S[i * N + j] - maxv);
            P[i * N + j] = e;
            sum += e;
        }
        float inv_sum = 1.f / (sum + 1e-9f);
        for (int j = 0; j < N; ++j) {
            P[i * N + j] *= inv_sum;
        }
    }

    // 3) O_ref = P * V -> [M x D]
    std::fill(O_ref.begin(), O_ref.end(), 0.f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < D; ++j) {
            float acc = 0.f;
            for (int k = 0; k < N; ++k) {
                acc += P[i * N + k] * V[k * D + j];
            }
            O_ref[i * D + j] = acc;
        }
    }
}

// ---------------------------------------------------------
// 유틸: float -> half vector 변환
// ---------------------------------------------------------
void float_to_half_vector(const std::vector<float>& src, std::vector<__half>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
}

// ---------------------------------------------------------
// main
// ---------------------------------------------------------
int main() {
    printf("6.6 Tensor Core fused FlashAttention tile (M=N=D=16)\\n");

    // 호스트 버퍼
    std::vector<float> Q_h(M * D);
    std::vector<float> K_T_h(D * N);  // 이미 transpose된 K^T 형태로 직접 생성
    std::vector<float> V_h(N * D);
    std::vector<float> O_ref(M * D);
    std::vector<float> O_h(M * D);

    // 난수 초기화
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * D; ++i) Q_h[i] = dist(rng);
    for (int i = 0; i < D * N; ++i) K_T_h[i] = dist(rng);
    for (int i = 0; i < N * D; ++i) V_h[i]  = dist(rng);

    // CPU reference
    cpu_reference(Q_h, K_T_h, V_h, O_ref);

    // Device 버퍼
    __half* d_Q   = nullptr;
    __half* d_K_T = nullptr;
    __half* d_V   = nullptr;
    float*  d_O   = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q,   M * D * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_K_T, D * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V,   N * D * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_O,   M * D * sizeof(float)));

    // float -> half 변환 후 업로드
    std::vector<__half> Q_half, K_T_half, V_half;
    float_to_half_vector(Q_h,   Q_half);
    float_to_half_vector(K_T_h, K_T_half);
    float_to_half_vector(V_h,   V_half);

    CHECK_CUDA(cudaMemcpy(d_Q,   Q_half.data(),   M * D * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_T, K_T_half.data(), D * N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V,   V_half.data(),   N * D * sizeof(__half), cudaMemcpyHostToDevice));

    // 커널 런치: 단일 warp 커널
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);

    // 워밍업
    flashattn_fused_softmax_tensorcore_kernel<<<grid, block>>>(d_Q, d_K_T, d_V, d_O);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 타이밍 측정
    const int NUM_ITERS = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITERS; ++it) {
        flashattn_fused_softmax_tensorcore_kernel<<<grid, block>>>(d_Q, d_K_T, d_V, d_O);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / NUM_ITERS;

    // 결과 가져오기
    CHECK_CUDA(cudaMemcpy(O_h.data(), d_O, M * D * sizeof(float), cudaMemcpyDeviceToHost));

    // 상대 L2 에러 계산
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < M * D; ++i) {
        double diff = static_cast<double>(O_h[i]) - static_cast<double>(O_ref[i]);
        num += diff * diff;
        den += static_cast<double>(O_ref[i]) * static_cast<double>(O_ref[i]);
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    // FLOPs 대략 계산
    // QK^T: M*N*D*2, softmax: M*(3N), PV: M*D*N*2
    double flops_qk  = static_cast<double>(M) * N * D * 2.0;
    double flops_pv  = static_cast<double>(M) * D * N * 2.0;
    double flops_sm  = static_cast<double>(M) * N * 3.0;
    double total_flops = flops_qk + flops_pv + flops_sm;

    double tflops = (total_flops * 1e-12) / (avg_ms * 1e-3);

    // 일부 값 출력
    printf("O[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", O_h[j]);
    }
    printf("\n");

    printf("O_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", O_ref[j]);
    }
    printf("\n");

    printf("Relative L2 error: %.6e\n", rel_l2);
    printf("Avg kernel time: %.6f ms (over %d iters)\n", avg_ms, NUM_ITERS);
    printf("Approx TFLOPS  : %.6f\n", tflops);

    // 정리
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K_T));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -std=c++17 -O3 -arch=sm_86     flashattn_fused_softmax_tensorcore_16x16.cu     -o flashattn_fused_softmax_tensorcore_16x16.exe     -lineinfo


*/