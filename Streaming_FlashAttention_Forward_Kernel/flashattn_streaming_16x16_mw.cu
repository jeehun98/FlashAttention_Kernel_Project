// 7.3 Multi-Warp Streaming Softmax FlashAttention-like Kernel
// - block당 2 warp (64 threads)
// - warp 0: rows 0..7, warp 1: rows 8..15 담당
// - QK^T: warp 0가 WMMA로 계산 → shared s_scores에 저장 → 두 warp가 같이 사용
// - Streaming softmax + PV: 각 warp가 자기 row range만 처리
// - 최종 O write: warp 전체가 참여해서 coalesced store
//
// 빌드:
// nvcc -O3 -std=c++17 -arch=sm_86 \
//   -o flashattn_streaming_16x16_mw.exe \
//   flashattn_streaming_16x16_mw.cu
//
// 프로파일:
// ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_mw.exe

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

// === Multi-warp fused kernel: QK^T (WMMA) + streaming softmax + PV ===
// Q:  [B, M, Kdim]       half
// K:  [B, Kdim, L]       half
// V:  [B, L, Dv]         half
// O:  [B, M, Dv]         float
__global__ void flashattn_streaming_16x16_kernel_mw(
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

    int tid      = threadIdx.x;
    int lane     = tid & (WARP_SIZE - 1);  // 0..31
    int warp_id  = tid >> 5;               // 0,1,...
    int num_warps = blockDim.x / WARP_SIZE;

    // row partition: warp 0 -> 0..7, warp 1 -> 8..15 (M=16, num_warps=2)
    int rows_per_warp = (M + num_warps - 1) / num_warps; // 8
    int row_start = warp_id * rows_per_warp;
    int row_end   = row_start + rows_per_warp;
    if (row_start >= M) row_start = M;
    if (row_end   >  M) row_end   = M;

    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // Q fragment: [M, Kdim] row-major
    // WMMA는 warp 단위라, warp 0만 사용해서 Q 전체 fragment를 로드
    __shared__ wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> sh_q_frag;
    if (warp_id == 0) {
        wmma::load_matrix_sync(sh_q_frag, Q_b, Kdim);
    }
    __syncthreads();

    // running 상태 (shared)
    __shared__ float sh_m_running[M];      // max
    __shared__ float sh_l_running[M];      // sum of exp
    __shared__ float sh_y_running[M][Dv];  // accumulated y[d]

    // row별 초기화: block 전체 thread로 나눠서 수행
    for (int i = tid; i < M; i += blockDim.x) {
        sh_m_running[i] = NEG_LARGE;
        sh_l_running[i] = 0.0f;
        for (int d = 0; d < Dv; ++d) {
            sh_y_running[i][d] = 0.0f;
        }
    }
    __syncthreads();

    // shared: tile scores, V tile
    __shared__ float  s_scores[M * N_TILE];          // [M, N_TILE]
    __shared__ __half sV_tile[Dv][N_TILE + 1];       // [Dv][N_TILE+1] (padding)

    int num_k_tiles = seq_len / N_TILE;

    // === tile loop over key/value dimension ===
    for (int t = 0; t < num_k_tiles; ++t) {
        int col_start = t * N_TILE;

        // 1) QK_tile^T via WMMA: warp 0만 수행해서 s_scores에 기록
        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            const __half* K_tile_base = K_b + col_start; // [0, col_start]
            wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);
            wmma::mma_sync(c_frag, sh_q_frag, k_frag, c_frag);
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);
        }

        // 2) V tile load: block 전체 thread로 분산
        __syncthreads();
        for (int idx = tid; idx < N_TILE * Dv; idx += blockDim.x) {
            int j = idx / Dv;    // 0..N_TILE-1
            int d = idx % Dv;    // 0..Dv-1
            int kv_index = col_start + j;
            __half hv = V_b[kv_index * Dv + d];
            sV_tile[d][j] = hv;
        }
        __syncthreads();

        // 3) tile 내 row별 local 통계 + streaming update
        //    각 warp는 자기 row range [row_start, row_end)만 처리
        if (lane < WARP_SIZE) {
            for (int i = row_start; i < row_end; ++i) {
                // --- row-wise max ---
                float x = NEG_LARGE;
                if (lane < N_TILE) {
                    float s_ij = s_scores[i * N_TILE + lane] * scale;
                    x = s_ij;
                }
                float maxv = warp_allreduce_max(x);

                // --- exp + row-wise sum ---
                float e = 0.0f;
                if (lane < N_TILE) {
                    e = __expf(x - maxv);
                }
                float sumv = warp_allreduce_sum(e);

                // --- row-wise y_t[d] ---
                float y_t[Dv];
                if (lane == 0) {
                    for (int d = 0; d < Dv; ++d) y_t[d] = 0.0f;
                }

                for (int d = 0; d < Dv; ++d) {
                    float contrib = 0.0f;
                    if (lane < N_TILE) {
                        __half hv = sV_tile[d][lane];
                        float  fv = __half2float(hv);
                        contrib = e * fv;
                    }
                    float row_sum_contrib = warp_allreduce_sum(contrib);
                    if (lane == 0) {
                        y_t[d] += row_sum_contrib;
                    }
                }

                // --- streaming update (랜간 lane0, row i만 갱신) ---
                if (lane == 0) {
                    float m_old = sh_m_running[i];
                    float l_old = sh_l_running[i];
                    float m_t   = maxv;
                    float l_t   = sumv;

                    if (l_t > 0.0f) {
                        if (m_old == NEG_LARGE) {
                            sh_m_running[i] = m_t;
                            sh_l_running[i] = l_t;
                            for (int d = 0; d < Dv; ++d) {
                                sh_y_running[i][d] = y_t[d];
                            }
                        } else {
                            float m_new = fmaxf(m_old, m_t);
                            float alpha = __expf(m_old - m_new);
                            float beta  = __expf(m_t   - m_new);

                            float l_new = l_old * alpha + l_t * beta;

                            for (int d = 0; d < Dv; ++d) {
                                float y_old = sh_y_running[i][d];
                                float y_new = y_old * alpha + y_t[d] * beta;
                                sh_y_running[i][d] = y_new;
                            }

                            sh_m_running[i] = m_new;
                            sh_l_running[i] = l_new;
                        }
                    }
                } // lane == 0
            } // row loop
        } // lane < WARP_SIZE

        __syncthreads();
    } // tile loop

    // 4) 최종 결과 쓰기: 각 warp가 자기 row range를 담당, warp 내에서 coalesced store
    for (int i = row_start; i < row_end; ++i) {
        float inv_l = 1.0f / (sh_l_running[i] + EPS);
        // lane-strided write: Dv=16이므로 lane 0..15가 유효 store
        for (int d = lane; d < Dv; d += WARP_SIZE) {
            if (d < Dv) {
                float val = sh_y_running[i][d] * inv_l;
                O_b[i * Dv + d] = val;
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
    std::printf("Streaming Softmax FlashAttention-like Multi-Warp (WMMA QK^T, streaming softmax + PV)\n");

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

    // multi-warp block: 64 threads (2 warps)
    dim3 block(64, 1, 1);
    dim3 grid(num_batches, 1, 1);

    // 1회 실행 (정확도 체크)
    flashattn_streaming_16x16_kernel_mw<<<grid, block>>>(
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
        flashattn_streaming_16x16_kernel_mw<<<grid, block>>>(
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
    //   softmax/PV: 대충 6 * M * seq_len * Dv 정도 (rough)
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
nvcc -O3 -std=c++17 -arch=sm_86     -o flashattn_streaming_16x16_mw.exe     flashattn_streaming_16x16_mw.cu

프로파일:
ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_mw.exe
*/
