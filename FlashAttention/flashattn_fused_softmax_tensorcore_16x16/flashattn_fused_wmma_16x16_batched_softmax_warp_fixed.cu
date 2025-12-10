// 6.12 Batched Fused QK^T + softmax + PV (WMMA, warp-parallel softmax, M=N=K=Dv=16, fixed)
//
// - Q, K, V: [NUM_BATCH, 16, 16] half
// - O, O_ref: [NUM_BATCH, 16, 16] float
// - 연산: 각 batch마다
//     S = Q * K         (16x16 x 16x16 -> 16x16, WMMA)
//     P = softmax(S * scale)  (row-wise, warp-parallel)
//     O = P * V         (16x16 x 16x16 -> 16x16, WMMA)
//
// CPU ref 와 Relative L2 error 비교 + kernel 시간 / TFLOPS 출력

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 16;
constexpr int Kdim = 16;
constexpr int Dv = 16;
constexpr int TILE_ELEMS = M * N;

// === Warp-level reductions (N=16에 맞춰 사용, lane>=16은 dummy) ===
__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
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

// === Softmax warp microkernel: 16x16 tile, row-wise ===
// s_scores: [16,16] row-major (float)
// s_probs : [16,16] row-major (float)
__device__ void softmax_warp_16x16(float* s_scores, float* s_probs, float scale) {
    int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int row = 0; row < M; ++row) {
        float x = -INFINITY;

        if (lane < N) {
            float v = s_scores[row * N + lane];
            x = v * scale;
        }

        // row-wise max
        float maxv = warp_allreduce_max(x);

        // exp + sum
        float ex = 0.0f;
        if (lane < N) {
            ex = __expf(x - maxv);
        }
        float sumv = warp_allreduce_sum(ex);

        // normalize
        if (lane < N) {
            float p = ex / (sumv + 1e-6f);
            s_probs[row * N + lane] = p;
        }
    }
}

// === Fused QK + softmax + PV, batched, WMMA + warp-softmax ===
__global__ void flashattn_fused_wmma_16x16_batched_softmax_warp_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif

    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    (void)lane;

    // batch별 베이스 포인터
    const __half* Q_b = Q + batch_id * M * Kdim;
    const __half* K_b = K + batch_id * Kdim * N;
    const __half* V_b = V + batch_id * N * Dv;
    float*       O_b  = O + batch_id * M * Dv;

    __shared__ float  s_scores[TILE_ELEMS];   // QK 결과 (float)
    __shared__ float  s_probs[TILE_ELEMS];    // softmax 결과 (float)
    __shared__ __half s_probs_h[TILE_ELEMS];  // WMMA용 half 버퍼

    // 1) QK (16x16 x 16x16 -> 16x16) : WMMA
    {
        wmma::fragment<wmma::matrix_a, M, N, Kdim, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, Kdim, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, Kdim, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);
        // Q: [M,Kdim] row-major, ld = Kdim
        wmma::load_matrix_sync(a_frag, Q_b, Kdim);
        // K: [Kdim,N] row-major, ld = N
        wmma::load_matrix_sync(b_frag, K_b, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // s_scores: [M,N] row-major, ld = N
        wmma::store_matrix_sync(s_scores, c_frag, N, wmma::mem_row_major);
    }
    __syncthreads();

    // 2) warp-parallel softmax on s_scores -> s_probs (float)
    softmax_warp_16x16(s_scores, s_probs, scale);
    __syncthreads();

    // 2-1) s_probs(float) -> s_probs_h(half)로 캐스팅 (WMMA 입력용)
    for (int idx = threadIdx.x; idx < TILE_ELEMS; idx += blockDim.x) {
        s_probs_h[idx] = __float2half(s_probs[idx]);
    }
    __syncthreads();

    // 3) PV (16x16 x 16x16 -> 16x16) : WMMA
    {
        // A, B 둘 다 half, accumulator 만 float
        wmma::fragment<wmma::matrix_a, M, N, Kdim, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, Kdim, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, Kdim, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        // P: [M,N] row-major, ld = N
        wmma::load_matrix_sync(a_frag, s_probs_h, N);

        // V: [N,Dv] row-major, ld = Dv
        wmma::load_matrix_sync(b_frag, V_b, Dv);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // O_b: [M,Dv] row-major, ld = Dv
        wmma::store_matrix_sync(O_b, c_frag, Dv, wmma::mem_row_major);
    }
}

// === CPU reference: batched QK + softmax + PV, float ===
void flashattn_cpu_ref(
    const std::vector<__half>& hQ,
    const std::vector<__half>& hK,
    const std::vector<__half>& hV,
    std::vector<float>& hO_ref,
    int num_batches,
    float scale
) {
    auto h2f = [](__half x) {
        return __half2float(x);
    };

    for (int b = 0; b < num_batches; ++b) {
        const __half* Q_b = hQ.data() + b * M * Kdim;
        const __half* K_b = hK.data() + b * Kdim * N;
        const __half* V_b = hV.data() + b * N * Dv;
        float* O_b        = hO_ref.data() + b * M * Dv;

        float S[M][N];

        // 1) S = Q * K (M x Kdim) x (Kdim x N)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < Kdim; ++k) {
                    float q  = h2f(Q_b[i * Kdim + k]);
                    float kk = h2f(K_b[k * N + j]);
                    acc += q * kk;
                }
                S[i][j] = acc * scale;
            }
        }

        // 2) P = softmax(S) (row-wise)
        float P[M][N];
        for (int i = 0; i < M; ++i) {
            float maxv = -INFINITY;
            for (int j = 0; j < N; ++j) {
                if (S[i][j] > maxv) maxv = S[i][j];
            }

            float sumv = 0.0f;
            for (int j = 0; j < N; ++j) {
                float e = std::exp(S[i][j] - maxv);
                P[i][j] = e;
                sumv += e;
            }
            float inv = 1.0f / (sumv + 1e-6f);
            for (int j = 0; j < N; ++j) {
                P[i][j] *= inv;
            }
        }

        // 3) O = P * V (M x N) x (N x Dv)
        for (int i = 0; i < M; ++i) {
            for (int d = 0; d < Dv; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < N; ++j) {
                    float v = h2f(V_b[j * Dv + d]);
                    acc += P[i][j] * v;
                }
                O_b[i * Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("6.12 Batched Fused QK^T + softmax + PV (WMMA, warp-parallel softmax, M=N=K=Dv=16, fixed)\n");

    constexpr int NUM_BATCH = 4096;
    const int num_batches = NUM_BATCH;

    float scale = 1.0f / std::sqrt(static_cast<float>(Kdim));

    // 1) Host 메모리 준비 (Q, K, V: half / O_ref: float)
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__half> hQ(num_batches * M * Kdim);
    std::vector<__half> hK(num_batches * Kdim * N);
    std::vector<__half> hV(num_batches * N * Dv);
    std::vector<float>  hO_ref(num_batches * M * Dv);
    std::vector<float>  hO(num_batches * M * Dv);

    for (int i = 0; i < (int)hQ.size(); ++i) {
        hQ[i] = __float2half(dist(rng));
    }
    for (int i = 0; i < (int)hK.size(); ++i) {
        hK[i] = __float2half(dist(rng));
    }
    for (int i = 0; i < (int)hV.size(); ++i) {
        hV[i] = __float2half(dist(rng));
    }

    // 2) CPU ref
    flashattn_cpu_ref(hQ, hK, hV, hO_ref, num_batches, scale);

    // 3) Device 메모리 할당 및 복사
    __half *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    float *dO  = nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // 4) 커널 1회 실행 (정확도 확인)
    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(num_batches, 1, 1);

    flashattn_fused_wmma_16x16_batched_softmax_warp_kernel<<<grid, block>>>(
        dQ, dK, dV, dO, num_batches, scale
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 5) Relative L2 error
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < hO.size(); ++i) {
        double diff = (double)hO[i] - (double)hO_ref[i];
        num += diff * diff;
        den += (double)hO_ref[i] * (double)hO_ref[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    int b0 = 0;
    std::printf("Batch %d, O[0, 0..7] (GPU): ", b0);
    for (int j = 0; j < 8; ++j) {
        std::printf("%f ", hO[b0 * M * Dv + 0 * Dv + j]);
    }
    std::printf("\nBatch %d, O_ref[0, 0..7] (CPU): ", b0);
    for (int j = 0; j < 8; ++j) {
        std::printf("%f ", hO_ref[b0 * M * Dv + 0 * Dv + j]);
    }
    std::printf("\nRelative L2 error over all batches: %.12e\n", rel_l2);

    std::printf("NUM_BATCH=%d, M=N=K=Dv=16\n", num_batches);

    // 6) 성능 측정
    const int NUM_ITERS = 100;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITERS; ++it) {
        flashattn_fused_wmma_16x16_batched_softmax_warp_kernel<<<grid, block>>>(
            dQ, dK, dV, dO, num_batches, scale
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS;  // per launch

    // 대략적 FLOPs 계산:
    // per batch:
    //   QK:   2 * M * N * Kdim
    //   PV:   2 * M * N * Dv
    //   softmax: ~ 5 * M * N (rough)
    double flops_per_batch =
        2.0 * M * N * Kdim +
        2.0 * M * N * Dv +
        5.0 * M * N;
    double total_flops = flops_per_batch * num_batches;
    double sec = ms * 1e-3;
    double tflops = (total_flops / sec) / 1e12;

    std::printf("Avg kernel time: %f ms (per launch)\n", ms);
    std::printf("Approx TFLOPS  : %f\n", tflops);

    // 7) 정리
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

/*
빌드:
nvcc -O3 -std=c++17 -arch=sm_86 \
    -o flashattn_fused_wmma_16x16_batched_softmax_warp_fixed.exe \
    flashattn_fused_wmma_16x16_batched_softmax_warp_fixed.cu

프로파일:
ncu --set full --launch-skip 10 --launch-count 1     .\flashattn_fused_wmma_16x16_batched_softmax_warp_fixed.exe
*/
