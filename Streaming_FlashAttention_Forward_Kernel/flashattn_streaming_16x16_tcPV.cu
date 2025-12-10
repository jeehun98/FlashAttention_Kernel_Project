// 6.xx Streaming Softmax FlashAttention-like (QK^T + EV via WMMA, streaming softmax)
// - M = Kdim = Dv = 16 고정
// - seq_len = N_TILE * NUM_K_TILES (N_TILE = 16)
// - Q: [B, M, Kdim] half          (row-major)
// - K: [B, Kdim, seq_len] half    (row-major, [Kdim, L])
// - V: [B, seq_len, Dv] half      (row-major, [L, Dv])
// - O: [B, M, Dv] float           (row-major)
//
// per batch b, row i:
//   for tile t over key/value (length L):
//      S_t[i, j] = (Q[i,:] · K_t[:,j]) * scale   (j: tile내 column)
//      m_t[i]  = max_j S_t[i,j]
//      e_ij    = exp(S_t[i,j] - m_t[i])
//      l_t[i]  = sum_j e_ij
//      Y_t[i,d] = sum_j e_ij * V_t[j,d]   (EV를 WMMA로)
//      (m, l, y)  <-- streaming softmax 업데이트로 통합
//
// 최종적으로 row i, dim d 에 대해:
//   O[i,d] = y[i,d] / l[i] 가 아니라
//   streaming 업데이트에서 이미 정규화까지 끝난 y를 유지함.
//
// 빌드:
// nvcc -O3 -std=c++17 -arch=sm_86  -o flashattn_streaming_16x16_tcPV.exe    flashattn_streaming_16x16_tcPV.cu
//
// 프로파일:
// ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_tcPV.exe

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M        = 16;   // query rows per batch
constexpr int Kdim     = 16;   // head_dim
constexpr int Dv       = 16;   // value dim
constexpr int N_TILE   = 16;   // key/value tile size (sequence chunk)

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS       = 1e-6f;

// === warp-level reductions ===
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

// === Device kernel: QK^T (WMMA) + streaming softmax + EV (WMMA) ===
__global__ void flashattn_streaming_16x16_kernel_tcPV(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,    // L, multiple of 16
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (threadIdx.x >= WARP_SIZE) return; // 1 warp/block 가정

    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // Q fragment (16x16)
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    wmma::load_matrix_sync(q_frag, Q_b, Kdim);

    // streaming softmax 상태 (row별)
    float m_running[M];
    float l_running[M];
    float y_running[M][Dv];

    if (lane == 0) {
        for (int i = 0; i < M; ++i) {
            m_running[i] = NEG_LARGE;
            l_running[i] = 0.0f;
            for (int d = 0; d < Dv; ++d) {
                y_running[i][d] = 0.0f;
            }
        }
    }

    // shared buffers
    __shared__ float  s_scores[M * N_TILE];   // QK 타일 (float)
    __shared__ __half sE_tile[M * N_TILE];    // exp(shifted) tile (half)
    __shared__ float  sY_tile[M * Dv];        // EV 결과 타일 (float)
    __shared__ float  s_l_tile[M];            // row-wise l_t[i]

    int num_k_tiles = seq_len / N_TILE;

    for (int t = 0; t < num_k_tiles; ++t) {
        int col_start = t * N_TILE;

        // 1) QK_t via WMMA: S_t = Q(16x16) * K_t(16x16)
        {
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            const __half* K_tile_base = K_b + col_start; // [Kdim, seq_len], ld=seq_len
            wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);

            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);
        }
        __syncthreads();

        // 2) row-wise max/exp/sum -> sE_tile (half), s_l_tile (float)
        //    한 warp가 모든 row를 처리
        for (int i = 0; i < M; ++i) {
            float x = NEG_LARGE;
            // S_ij * scale
            if (lane < N_TILE) {
                float s_ij = s_scores[i * N_TILE + lane];
                x = s_ij * scale;
            }
            float maxv = warp_allreduce_max(x);

            float e = 0.0f;
            if (lane < N_TILE) {
                e = __expf(x - maxv);
                sE_tile[i * N_TILE + lane] = __float2half(e);
            }
            float sumv = warp_allreduce_sum(e);

            if (lane == 0) {
                s_l_tile[i] = sumv;
                // tile max는 streaming 업데이트에서 row별 m_t 로 사용
                // m_t[i] = maxv
            }
        }
        __syncthreads();

        // 3) EV via WMMA: Y_t = E_t(16x16) * V_t(16x16)
        {
            // A = E_t: [M,N_TILE] row-major
            wmma::fragment<wmma::matrix_a, M, Dv, N_TILE, __half, wmma::row_major> a_frag_ev;
            // B = V_t: [N_TILE,Dv] row-major
            wmma::fragment<wmma::matrix_b, M, Dv, N_TILE, __half, wmma::row_major> b_frag_ev;
            wmma::fragment<wmma::accumulator, M, Dv, N_TILE, float> c_frag_ev;

            wmma::fill_fragment(c_frag_ev, 0.0f);

            // load E_t
            wmma::load_matrix_sync(a_frag_ev, sE_tile, N_TILE);

            // load V_t: [N_TILE, Dv] from global
            const __half* V_tile_base = V_b + static_cast<size_t>(col_start) * Dv;
            wmma::load_matrix_sync(b_frag_ev, V_tile_base, Dv);

            wmma::mma_sync(c_frag_ev, a_frag_ev, b_frag_ev, c_frag_ev);
            wmma::store_matrix_sync(sY_tile, c_frag_ev, Dv, wmma::mem_row_major);
        }
        __syncthreads();

        // 4) streaming softmax 업데이트 (lane==0에서 row별 수행)
        if (lane == 0) {
            for (int i = 0; i < M; ++i) {
                // tile에서의 row-wise max 다시 계산 (x - maxv 에서 maxv 썼으니 저장 안 했음)
                // 여기서는 s_scores 를 다시 스캔해서 m_t[i]를 구함
                float m_t = NEG_LARGE;
                for (int j = 0; j < N_TILE; ++j) {
                    float v = s_scores[i * N_TILE + j] * scale;
                    if (v > m_t) m_t = v;
                }

                float l_t = s_l_tile[i];
                if (l_t <= 0.0f) continue;

                float m_old = m_running[i];
                float l_old = l_running[i];

                float m_new;
                float alpha, beta;

                if (m_old == NEG_LARGE) {
                    // 첫 타일
                    m_new = m_t;
                    alpha = 0.0f;
                    beta  = 1.0f;
                } else {
                    m_new = fmaxf(m_old, m_t);
                    alpha = __expf(m_old - m_new);
                    beta  = __expf(m_t  - m_new);
                }

                float l_new = l_old * alpha + l_t * beta;
                float inv_l_new = 1.0f / (l_new + EPS);

                float old_scale = (l_old * alpha) * inv_l_new;
                float new_scale = (l_t  * beta) * inv_l_new;

                // Y_t row i: sY_tile[i, d]
                for (int d = 0; d < Dv; ++d) {
                    float y_old = y_running[i][d];
                    float y_t   = sY_tile[i * Dv + d];  // sum_j e_ij * V_jd
                    float y_new = y_old * old_scale + y_t * new_scale;
                    y_running[i][d] = y_new;
                }

                m_running[i] = m_new;
                l_running[i] = l_new;
            }
        }
        __syncthreads();
    } // tile loop

    // 5) 최종 결과 쓰기 (lane0)
    if (lane == 0) {
        for (int i = 0; i < M; ++i) {
            for (int d = 0; d < Dv; ++d) {
                O_b[i * Dv + d] = y_running[i][d];
            }
        }
    }
}

// === CPU reference: full attention (no tiling), float ===
void flashattn_streaming_cpu_ref(
    const std::vector<__half>& hQ,
    const std::vector<__half>& hK,
    const std::vector<__half>& hV,
    std::vector<float>& hO_ref,
    int num_batches,
    int seq_len,
    float scale
) {
    auto h2f = [](__half x) { return __half2float(x); };

    for (int b = 0; b < num_batches; ++b) {
        const __half* Q_b = hQ.data() + static_cast<size_t>(b) * M * Kdim;
        const __half* K_b = hK.data() + static_cast<size_t>(b) * Kdim * seq_len;
        const __half* V_b = hV.data() + static_cast<size_t>(b) * seq_len * Dv;
        float*       O_b  = hO_ref.data() + static_cast<size_t>(b) * M * Dv;

        // S: [M, seq_len]
        std::vector<float> S(M * seq_len);

        // 1) S = Q * K   (M x Kdim) x (Kdim x L)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < Kdim; ++k) {
                    float q  = h2f(Q_b[i * Kdim + k]);
                    float kk = h2f(K_b[k * seq_len + j]);  // [Kdim, L]
                    acc += q * kk;
                }
                S[i * seq_len + j] = acc * scale;
            }
        }

        // 2) softmax row-wise
        std::vector<float> P(M * seq_len);
        for (int i = 0; i < M; ++i) {
            float maxv = NEG_LARGE;
            for (int j = 0; j < seq_len; ++j) {
                float v = S[i * seq_len + j];
                if (v > maxv) maxv = v;
            }
            float sumv = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float e = std::exp(S[i * seq_len + j] - maxv);
                P[i * seq_len + j] = e;
                sumv += e;
            }
            float inv = 1.0f / (sumv + EPS);
            for (int j = 0; j < seq_len; ++j) {
                P[i * seq_len + j] *= inv;
            }
        }

        // 3) O = P * V    (M x L) x (L x Dv)
        for (int i = 0; i < M; ++i) {
            for (int d = 0; d < Dv; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    float pij = P[i * seq_len + j];
                    float v   = h2f(V_b[j * Dv + d]);    // [L, Dv]
                    acc += pij * v;
                }
                O_b[i * Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like (WMMA QK^T + EV, streaming softmax)\n");

    constexpr int NUM_BATCH    = 1024;
    constexpr int NUM_K_TILES  = 8;       // => seq_len = 8 * 16 = 128
    constexpr int SEQ_LEN      = N_TILE * NUM_K_TILES;

    const int num_batches = NUM_BATCH;
    const int seq_len     = SEQ_LEN;

    float scale = 1.0f / std::sqrt(static_cast<float>(Kdim));

    // Host 메모리
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__half> hQ(num_batches * M * Kdim);
    std::vector<__half> hK(num_batches * Kdim * seq_len);
    std::vector<__half> hV(num_batches * seq_len * Dv);
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

    // CPU ref
    flashattn_streaming_cpu_ref(hQ, hK, hV, hO_ref, num_batches, seq_len, scale);

    // Device 메모리
    __half *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    float  *dO = nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(num_batches, 1, 1);

    // 1회 실행 (정확도 체크)
    flashattn_streaming_16x16_kernel_tcPV<<<grid, block>>>(
        dQ, dK, dV, dO,
        num_batches,
        seq_len,
        scale
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Relative L2 error
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
    std::printf("NUM_BATCH=%d, SEQ_LEN=%d, M=Kdim=Dv=16\n", num_batches, seq_len);

    // 성능 측정
    const int NUM_ITERS = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITERS; ++it) {
        flashattn_streaming_16x16_kernel_tcPV<<<grid, block>>>(
            dQ, dK, dV, dO,
            num_batches,
            seq_len,
            scale
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS;

    // FLOPs 대략 계산:
    // per batch:
    //   QK:   2 * M * seq_len * Kdim
    //   EV:   2 * M * seq_len * Dv
    //   softmax: ~ 5 * M * seq_len (rough)
    double flops_per_batch =
        2.0 * M * seq_len * Kdim +
        2.0 * M * seq_len * Dv +
        5.0 * M * seq_len;
    double total_flops = flops_per_batch * num_batches;
    double sec = ms * 1e-3;
    double tflops = (total_flops / sec) / 1e12;

    std::printf("Avg kernel time: %f ms (per launch)\n", ms);
    std::printf("Approx TFLOPS  : %f\n", tflops);

    // 정리
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
