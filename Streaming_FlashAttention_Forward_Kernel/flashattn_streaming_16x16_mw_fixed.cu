// Streaming Softmax FlashAttention-like (WMMA QK^T, streaming softmax + PV, multi-warp, FIXED)
//
// - M = Kdim = Dv = 16
// - seq_len = N_TILE * NUM_K_TILES (N_TILE = 16, 예: L=128)
// - Q: [B, M, Kdim] half          (row-major)
// - K: [B, Kdim, L]   half        (row-major, [Kdim, L])
// - V: [B, L, Dv]     half        (row-major, [L, Dv])
// - O: [B, M, Dv]     float       (row-major)
//
// per batch b, row i:
//   streaming softmax 상태:
//      m[i] = running max
//      l[i] = sum exp(s_ik - m[i])
//      u[i,d] = sum exp(s_ik - m[i]) * V_{k,d}
//   최종:
//      O[i,d] = u[i,d] / l[i]
//
// 빌드:
// nvcc -O3 -std=c++17 -arch=sm_86    -o flashattn_streaming_16x16_mw_fixed.exe    flashattn_streaming_16x16_mw_fixed.cu
//
// 프로파일:
// ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_mw_fixed.exe

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
constexpr int M        = 16;
constexpr int Kdim     = 16;
constexpr int Dv       = 16;
constexpr int N_TILE   = 16;   // tile length in sequence dimension

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS       = 1e-6f;

// === simple warp reductions (모든 32 lane 참여, 필요 시 dummy 값으로 채움) ===
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

// === 커널: QK^T via WMMA + streaming softmax + PV, multi-warp (2 warps) ===
// blockDim.x = 64 (2 warps), 각 batch당 1 block
__global__ void flashattn_streaming_16x16_kernel_mw_fixed(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,    // L (multiple of 16)
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int tid    = threadIdx.x;
    int warpId = tid / WARP_SIZE;      // 0 or 1
    int lane   = tid & (WARP_SIZE - 1);

    // 한 block에서 M=16 row를 2개의 warp가 나눠 가짐
    constexpr int WARPS_PER_BLOCK = 2;
    constexpr int ROWS_PER_WARP   = M / WARPS_PER_BLOCK; // 8

    // layout:
    // Q: [B, M, Kdim]
    // K: [B, Kdim, L]
    // V: [B, L, Dv]
    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // Q tile (16x16) 전체는 warp0만 WMMA로 처리 (Q는 타일 독립)
    __shared__ float s_scores[M * N_TILE]; // [M, N_TILE], row-major

    // streaming 상태 (warp별 row subset만 가짐)
    float m_row[ROWS_PER_WARP];
    float l_row[ROWS_PER_WARP];
    float u_row[ROWS_PER_WARP][Dv];

    // init
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        m_row[r] = NEG_LARGE;
        l_row[r] = 0.0f;
        for (int d = 0; d < Dv; ++d) {
            u_row[r][d] = 0.0f;
        }
    }

    int num_k_tiles = seq_len / N_TILE;

    // --- Q는 한 번만 로드해서 재사용 ---
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    if (warpId == 0) {
        wmma::load_matrix_sync(q_frag, Q_b, Kdim);
    }
    __syncthreads();

    // === 타일 루프 ===
    for (int t = 0; t < num_k_tiles; ++t) {
        int col_start = t * N_TILE;  // key/value global 시작 index in L

        // 1) QK_tile^T via WMMA (warp0만 수행)
        if (warpId == 0) {
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            const __half* K_tile_base = K_b + col_start; // [Kdim, L], ld = L
            wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);

            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);

            // s_scores: [M, N_TILE], ld = N_TILE
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);
        }
        __syncthreads();

        // 2) 각 warp가 담당하는 row subset에 대해 streaming softmax 상태 업데이트
        //    여기서는 수식 정확성을 우선시해서 per-thread (row 단위) 계산으로 구현.
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = warpId * ROWS_PER_WARP + r;   // 0..15

            // 하나의 thread만 row 계산을 담당하게 하고 싶으면 lane == 0 조건을 줄 수 있음
            // 여기서는 lane == 0 만 row를 처리하게 해서 분기 단순화
            if (lane == 0) {
                // (a) 현재 타일에서 이 row의 score들 (scale 적용)
                float s_row[N_TILE];
                float m_t = NEG_LARGE;
                for (int j = 0; j < N_TILE; ++j) {
                    float s = s_scores[row * N_TILE + j] * scale;
                    s_row[j] = s;
                    if (s > m_t) m_t = s;
                }

                // (b) 새로운 기준 max 계산
                float m_prev = m_row[r];
                float m_new;
                float old_scale;
                if (m_prev == NEG_LARGE) {
                    m_new     = m_t;
                    old_scale = 0.0f;   // 이전 기여 없음
                } else {
                    m_new     = fmaxf(m_prev, m_t);
                    old_scale = __expf(m_prev - m_new);
                }

                // 이전 상태를 m_new 기준으로 rescale
                float l_res = l_row[r] * old_scale;
                float u_res[Dv];
                for (int d = 0; d < Dv; ++d) {
                    u_res[d] = u_row[r][d] * old_scale;
                }

                // (c) 이번 타일 기여를 m_new 기준으로 직접 계산
                float l_t = 0.0f;
                float u_t[Dv];
                for (int d = 0; d < Dv; ++d) u_t[d] = 0.0f;

                for (int j = 0; j < N_TILE; ++j) {
                    float e = __expf(s_row[j] - m_new);
                    l_t += e;

                    int kv_index = col_start + j;  // global pos in [0, L)
                    const __half* v_ptr = V_b + kv_index * Dv;
                    for (int d = 0; d < Dv; ++d) {
                        float v = __half2float(v_ptr[d]);
                        u_t[d] += e * v;
                    }
                }

                // (d) running 상태 갱신
                float l_new = l_res + l_t;
                float u_new[Dv];
                for (int d = 0; d < Dv; ++d) {
                    u_new[d] = u_res[d] + u_t[d];
                }

                m_row[r] = m_new;
                l_row[r] = l_new;
                for (int d = 0; d < Dv; ++d) {
                    u_row[r][d] = u_new[d];
                }
            }
        }

        __syncthreads();
    } // tile loop

    // 3) 최종 정규화해서 O_b에 쓰기 (lane 0 만 실행)
    if (lane == 0) {
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = warpId * ROWS_PER_WARP + r;
            float l_val = l_row[r];
            float inv_l = 1.0f / (l_val + EPS);
            for (int d = 0; d < Dv; ++d) {
                O_b[row * Dv + d] = u_row[r][d] * inv_l;
            }
        }
    }
}

// === CPU reference: full attention softmax + PV ===
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
                    float kk = h2f(K_b[k * seq_len + j]);
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
                    float v   = h2f(V_b[j * Dv + d]);
                    acc += pij * v;
                }
                O_b[i * Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like Multi-Warp (WMMA QK^T, streaming softmax + PV) [fixed math]\n");

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

    dim3 block(64, 1, 1);   // 2 warps
    dim3 grid(num_batches, 1, 1);

    // 1회 실행 (정확도 체크)
    flashattn_streaming_16x16_kernel_mw_fixed<<<grid, block>>>(
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
        flashattn_streaming_16x16_kernel_mw_fixed<<<grid, block>>>(
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

    // FLOPs 대략 계산 (이전과 동일한 러프 추정)
    double flops_per_batch =
        2.0 * M * seq_len * Kdim +
        6.0 * M * seq_len * Dv;
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
