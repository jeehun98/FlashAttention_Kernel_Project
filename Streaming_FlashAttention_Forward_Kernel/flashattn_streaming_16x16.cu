// 6.xx Streaming Softmax FlashAttention-like (QK^T via WMMA, streaming softmax + PV)
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
//      e_ij = exp(S_t[i, j] - m_t[i])            (row-wise local max)
//      l_t[i]  = sum_j e_ij
//      y_t[i,d] = sum_j e_ij * V_t[j,d]
//   running 상태 (m, l, y)에 streaming softmax 공식으로 merge:
//
//   m_new = max(m_old, m_t)
//   alpha = exp(m_old - m_new)
//   beta  = exp(m_t   - m_new)
//   l_new = l_old * alpha + l_t * beta
//   y_new = y_old * alpha + y_t * beta
//
// 최종적으로 row i, dim d 에 대해:
//   O[i,d] = y[i,d] / (l[i] + eps)
//
// 빌드:
// nvcc -O3 -std=c++17 -arch=sm_86 \
//   -o flashattn_streaming_16x16.exe \
//   flashattn_streaming_16x16.cu
//
// 프로파일:
// ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16.exe

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

constexpr float NEG_LARGE = -1e30f;  // sentinel for -inf
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

// === fused kernel: QK^T (WMMA) + streaming softmax + PV ===
// Q:  [B, M, Kdim]       half
// K:  [B, Kdim, L]       half
// V:  [B, L, Dv]         half
// O:  [B, M, Dv]         float
__global__ void flashattn_streaming_16x16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,    // L, must be multiple of 16
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int lane = threadIdx.x & (WARP_SIZE - 1);

    // layout:
    // Q: [B, M, Kdim] row-major
    // K: [B, Kdim, L] row-major
    // V: [B, L, Dv]   row-major
    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // Q fragment (matrix_a): [M, Kdim] row-major, ld = Kdim
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    wmma::load_matrix_sync(q_frag, Q_b, Kdim);

    // streaming softmax 상태: row별 running max, sum(exp), sum(exp·V)
    float m_running[M];          // max
    float l_running[M];          // l = Σ exp(x - m)
    float y_running[M][Dv];      // y[d] = Σ exp(x - m) * v_d

    if (lane == 0) {
        for (int i = 0; i < M; ++i) {
            m_running[i] = NEG_LARGE;
            l_running[i] = 0.0f;
            for (int d = 0; d < Dv; ++d) {
                y_running[i][d] = 0.0f;
            }
        }
    }

    // shared: tile scores, V tile
    __shared__ float  s_scores[M * N_TILE];      // [M, N_TILE]
    __shared__ __half sV_tile[N_TILE * Dv];      // [N_TILE, Dv]

    int num_k_tiles = seq_len / N_TILE;

    // === tile loop over key/value dimension ===
    for (int t = 0; t < num_k_tiles; ++t) {
        int col_start = t * N_TILE;  // tile 시작 column index

        // 1) QK_tile^T via WMMA: S_tile = Q[16x16] * K_tile[16x16]
        {
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            // K_b: [Kdim, seq_len] row-major, ld = seq_len
            const __half* K_tile_base = K_b + col_start; // row 0, col=col_start
            wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);

            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);

            // s_scores: [M, N_TILE] row-major
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);
        }

        // 2) V_tile load: [N_TILE, Dv] rows = key positions in this tile
        {
            for (int idx = threadIdx.x; idx < N_TILE * Dv; idx += blockDim.x) {
                int j = idx / Dv;   // 0..N_TILE-1
                int d = idx % Dv;   // 0..Dv-1
                int kv_index = col_start + j; // global key index 0..L-1
                sV_tile[idx] = V_b[kv_index * Dv + d];
            }
        }
        __syncthreads();

        // 3) tile 내 local 통계(m_t, l_t, y_t) 계산 + streaming 업데이트
        if (lane < WARP_SIZE) {
            // lane 0이만 로컬 통계를 저장
            float m_tile_local[M];
            float l_tile_local[M];
            float y_tile_local[M][Dv];

            if (lane == 0) {
                for (int i = 0; i < M; ++i) {
                    m_tile_local[i] = NEG_LARGE;
                    l_tile_local[i] = 0.0f;
                    for (int d = 0; d < Dv; ++d) {
                        y_tile_local[i][d] = 0.0f;
                    }
                }
            }

            // row-wise 처리
            for (int i = 0; i < M; ++i) {
                // 3-1) row-wise max
                float x = NEG_LARGE;
                if (lane < N_TILE) {
                    float s_ij = s_scores[i * N_TILE + lane] * scale;
                    x = s_ij;
                }
                float maxv = warp_allreduce_max(x);

                // 3-2) exp + row-wise sum
                float e = 0.0f;
                if (lane < N_TILE) {
                    e = __expf(x - maxv);
                }
                float sumv = warp_allreduce_sum(e);

                if (lane == 0) {
                    m_tile_local[i] = maxv;
                    l_tile_local[i] = sumv;
                }

                // 3-3) y_t[i,d] = Σ_j e_ij * V_t[j,d]
                for (int d = 0; d < Dv; ++d) {
                    float contrib = 0.0f;
                    if (lane < N_TILE) {
                        __half hv = sV_tile[lane * Dv + d]; // j=lane
                        float  fv = __half2float(hv);
                        contrib = e * fv;
                    }
                    float row_sum_contrib = warp_allreduce_sum(contrib);
                    if (lane == 0) {
                        y_tile_local[i][d] += row_sum_contrib;
                    }
                }
            } // rows

            // 3-4) streaming softmax 업데이트 (lane0)
            if (lane == 0) {
                for (int i = 0; i < M; ++i) {
                    float m_old = m_running[i];
                    float l_old = l_running[i];
                    float m_ti  = m_tile_local[i];
                    float l_ti  = l_tile_local[i];

                    if (l_ti <= 0.0f) {
                        continue; // 이 타일에서 기여 없으면 skip
                    }

                    if (m_old == NEG_LARGE) {
                        // 첫 타일
                        m_running[i] = m_ti;
                        l_running[i] = l_ti;
                        for (int d = 0; d < Dv; ++d) {
                            y_running[i][d] = y_tile_local[i][d];
                        }
                    } else {
                        float m_new = fmaxf(m_old, m_ti);
                        float alpha = __expf(m_old - m_new);
                        float beta  = __expf(m_ti  - m_new);

                        float l_new = l_old * alpha + l_ti * beta;

                        for (int d = 0; d < Dv; ++d) {
                            float y_old = y_running[i][d];
                            float y_t   = y_tile_local[i][d];
                            float y_new = y_old * alpha + y_t * beta;
                            y_running[i][d] = y_new;
                        }

                        m_running[i] = m_new;
                        l_running[i] = l_new;
                    }
                }
            }
        } // if lane < WARP_SIZE

        __syncthreads();
    } // tile loop

    // 4) 최종 결과 쓰기 (정규화: y / l)
    if (lane == 0) {
        for (int i = 0; i < M; ++i) {
            float inv_l = 1.0f / (l_running[i] + EPS);
            for (int d = 0; d < Dv; ++d) {
                O_b[i * Dv + d] = y_running[i][d] * inv_l;
            }
        }
    }
}

// === CPU reference: full attention (no tiling), float ===
// Q: [B, M, Kdim], K: [B, Kdim, L], V: [B, L, Dv]
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
    std::printf("Streaming Softmax FlashAttention-like (WMMA QK^T, streaming softmax + PV)\n");

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
    flashattn_streaming_16x16_kernel<<<grid, block>>>(
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
        flashattn_streaming_16x16_kernel<<<grid, block>>>(
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
    //   softmax/PV: 대충 6 * M * seq_len * Dv 정도로 잡음 (rough)
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

/*
빌드:
nvcc -O3 -std=c++17 -arch=sm_86 \
    -o flashattn_streaming_16x16.exe \
    flashattn_streaming_16x16.cu

프로파일:
ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16.exe
*/
