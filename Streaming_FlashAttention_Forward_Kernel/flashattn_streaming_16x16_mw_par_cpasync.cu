// 6.xx Streaming Softmax FlashAttention-like (QK^T via WMMA,
//      multi-warp + V-tile in shared + cp.async double buffering)
//
// - M = Kdim = Dv = 16 고정
// - seq_len = N_TILE * NUM_K_TILES (N_TILE = 16)
// - Q: [B, M, Kdim] half          (row-major)
// - K: [B, Kdim, seq_len] half    (row-major, [Kdim, L])
// - V: [B, seq_len, Dv] half      (row-major, [L, Dv])
// - O: [B, M, Dv] float           (row-major)
//
// 여기 버전은:
//
//  - 블록당 2 warps (64 threads)
//  - WMMA QK^T는 warp0가 담당 (32 threads)
//  - V 타일은 shared에 double-buffer (sV[2][16x16]) + cp.async 로 prefetch
//  - streaming softmax + PV 상태(m, l, u[:])는 전부 스레드 로컬 레지스터에 유지
//    (tid < M, 각 thread가 row 하나씩 담당)
//
// 빌드:
//   nvcc -O3 -std=c++17 -arch=sm_86 \
//        -o flashattn_streaming_16x16_mw_par_cpasync.exe \
//        flashattn_streaming_16x16_mw_par_cpasync.cu

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

constexpr int WARP_SIZE   = 32;
constexpr int M           = 16;   // query rows per batch
constexpr int Kdim        = 16;   // head_dim
constexpr int Dv          = 16;   // value dim
constexpr int N_TILE      = 16;   // key/value tile size (sequence chunk)
constexpr int S_STRIDE    = N_TILE + 1; // s_scores row stride (padding: 17) -> bank conflict 완화

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS       = 1e-6f;

// === warp-level reductions (현재 커널에선 직접 사용 안하지만 남김) ===
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

// === cp.async helpers (global -> shared, 16-byte granularity) ===
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" :: 
        "r"(smem_addr),
        "l"(gmem_ptr),
        "n"(16)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

// QK^T (WMMA) + streaming softmax + PV
// - V 타일을 shared에 double-buffer (sV[2])로 올리고 cp.async로 prefetch
// - QK^T / cp.async는 multi-thread 사용
// - softmax + PV streaming 상태는 tid < M 스레드가 각 row를 전담, 레지스터에 유지
__global__ void flashattn_streaming_16x16_kernel_mw_par_cpasync(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int tid = threadIdx.x;

    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // shared buffers
    __shared__ float  s_scores[M * S_STRIDE];   // 16 x (16+1) tile scores (float, padded stride=17)
    __shared__ __half sV[2][N_TILE * Dv];       // double-buffered V tiles (16 x 16)

    int num_k_tiles = seq_len / N_TILE;

    // === per-thread streaming state (row 단위, 레지스터에 유지) ===
    float m_row = NEG_LARGE;
    float l_row = 0.0f;
    float u_row[Dv];
    #pragma unroll
    for (int d = 0; d < Dv; ++d) {
        u_row[d] = 0.0f;
    }

    // === V tile 0 prefetch ===
    int buf = 0;
    {
        int t0 = 0;
        int col_start = t0 * N_TILE;
        int total_bytes = N_TILE * Dv * sizeof(__half); // 16*16*2 = 512 bytes

    #if __CUDA_ARCH__ >= 800
        for (int byte_off = tid * 16; byte_off < total_bytes; byte_off += blockDim.x * 16) {
            int elem = byte_off / sizeof(__half);
            int j    = elem / Dv;
            int d    = elem % Dv;
            int kv_index = col_start + j;

            char*       smem_ptr = reinterpret_cast<char*>(&sV[buf][0]) + byte_off;
            const char* gmem_ptr = reinterpret_cast<const char*>(
                &V_b[kv_index * Dv + d]
            );
            cp_async_16B(smem_ptr, gmem_ptr);
        }
        cp_async_commit();
        cp_async_wait();
        __syncthreads();
    #else
        for (int idx = tid; idx < N_TILE * Dv; idx += blockDim.x) {
            int j = idx / Dv;
            int d = idx % Dv;
            int kv_index = col_start + j;
            sV[buf][idx] = V_b[kv_index * Dv + d];
        }
        __syncthreads();
    #endif
    }

    // === tile loop ===
    for (int t = 0; t < num_k_tiles; ++t) {
        int col_start = t * N_TILE;

        int next_t   = t + 1;
        int next_buf = buf ^ 1;

        // 1) QK_t^T via WMMA: S_tile = Q[16x16] * K_tile[16x16]
        if (threadIdx.x < WARP_SIZE) {
            // warp 0 전용 WMMA 영역
            wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float>                c_frag;

            wmma::fill_fragment(c_frag, 0.0f);
            // Q: [M, Kdim], row-major, ld = Kdim
            wmma::load_matrix_sync(q_frag, Q_b, Kdim);

            // K_tile: [Kdim, N_TILE], row-major, ld = seq_len
            const __half* K_tile_base = K_b + col_start;
            wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);

            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);

            // padded stride = S_STRIDE(=17)
            wmma::store_matrix_sync(s_scores, c_frag, S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();  // s_scores 준비 완료

        // 2) 다음 V tile prefetch (t+1) – cp.async
        if (next_t < num_k_tiles) {
            int next_col_start = next_t * N_TILE;
            int total_bytes = N_TILE * Dv * sizeof(__half);

        #if __CUDA_ARCH__ >= 800
            for (int byte_off = tid * 16; byte_off < total_bytes; byte_off += blockDim.x * 16) {
                int elem = byte_off / sizeof(__half);
                int j    = elem / Dv;
                int d    = elem % Dv;
                int kv_index = next_col_start + j;

                char*       smem_ptr = reinterpret_cast<char*>(&sV[next_buf][0]) + byte_off;
                const char* gmem_ptr = reinterpret_cast<const char*>(
                    &V_b[kv_index * Dv + d]
                );
                cp_async_16B(smem_ptr, gmem_ptr);
            }
            cp_async_commit();
        #endif
        }

        // 3) streaming softmax + PV (row 단위 병렬화)
        //
        //  - tid < M 인 16개 스레드가 각 row 하나씩 담당
        //  - m_row, l_row, u_row[:]는 전부 레지스터에 유지 (shared 사용 안 함)
        if (tid < M) {
            int row = tid;

            // (a) tile 내 local max m_t 계산
            float m_t = NEG_LARGE;
            for (int j = 0; j < N_TILE; ++j) {
                float x = s_scores[row * S_STRIDE + j] * scale;
                if (x > m_t) m_t = x;
            }

            // (b) e_ij = exp(x_ij - m_t), tile local l_t, u_t 계산
            float l_t = 0.0f;
            float u_t[Dv];
            #pragma unroll
            for (int d = 0; d < Dv; ++d) {
                u_t[d] = 0.0f;
            }

            for (int j = 0; j < N_TILE; ++j) {
                float x = s_scores[row * S_STRIDE + j] * scale;
                float e = __expf(x - m_t);
                l_t += e;

                // V tile from shared: sV[buf][j, d]
                #pragma unroll
                for (int d = 0; d < Dv; ++d) {
                    float v = __half2float(sV[buf][j * Dv + d]);
                    u_t[d] += e * v;
                }
            }

            // (c) streaming 결합 (레지스터 상태 갱신)
            if (m_row == NEG_LARGE) {
                // 첫 tile
                m_row = m_t;
                l_row = l_t;
                #pragma unroll
                for (int d = 0; d < Dv; ++d) {
                    u_row[d] = u_t[d];
                }
            } else {
                float m_new = fmaxf(m_row, m_t);
                float alpha = __expf(m_row - m_new);
                float beta  = __expf(m_t   - m_new);

                float l_new = l_row * alpha + l_t * beta;
                m_row = m_new;
                l_row = l_new;

                #pragma unroll
                for (int d = 0; d < Dv; ++d) {
                    float u_new = u_row[d] * alpha + u_t[d] * beta;
                    u_row[d] = u_new;
                }
            }
        }

        __syncthreads();  // row threads가 V 타일 사용 끝낸 후

        // 4) 다음 tile의 V prefetch 완료 대기 (cp.async)
        if (next_t < num_k_tiles) {
        #if __CUDA_ARCH__ >= 800
            cp_async_wait();
            __syncthreads();
        #else
            int next_col_start = next_t * N_TILE;
            for (int idx = tid; idx < N_TILE * Dv; idx += blockDim.x) {
                int j = idx / Dv;
                int d = idx % Dv;
                int kv_index = next_col_start + j;
                sV[next_buf][idx] = V_b[kv_index * Dv + d];
            }
            __syncthreads();
        #endif
        }

        buf = next_buf;
    }

    // 5) 최종 normalized output: O = u / (l + EPS)
    if (tid < M) {
        int row = tid;
        float inv_l = 1.0f / (l_row + EPS);
        #pragma unroll
        for (int d = 0; d < Dv; ++d) {
            O_b[row * Dv + d] = u_row[d] * inv_l;
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

        std::vector<float> S(M * seq_len);

        // 1) S = Q * K
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

        // 3) O = P * V
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
    std::printf("Streaming Softmax FlashAttention-like Multi-Warp + V-tile + cp.async (WMMA QK^T, streaming softmax + PV)\n");

    constexpr int NUM_BATCH    = 1024;
    constexpr int NUM_K_TILES  = 8;       // => seq_len = 8 * 16 = 128
    constexpr int SEQ_LEN      = N_TILE * NUM_K_TILES;

    const int num_batches = NUM_BATCH;
    const int seq_len     = SEQ_LEN;

    float scale = 1.0f / std::sqrt(static_cast<float>(Kdim));

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

    dim3 block(64, 1, 1);              // 2 warps / block
    dim3 grid(num_batches, 1, 1);      // 1 block per batch

    // 1회 실행 (정확도 체크)
    flashattn_streaming_16x16_kernel_mw_par_cpasync<<<grid, block>>>(
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
        flashattn_streaming_16x16_kernel_mw_par_cpasync<<<grid, block>>>(
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

    // FLOPs 대략 계산
    double flops_per_batch =
        2.0 * M * seq_len * Kdim +      // QK
        6.0 * M * seq_len * Dv;         // softmax/PV rough
    double total_flops = flops_per_batch * num_batches;
    double sec = ms * 1e-3;
    double tflops = (total_flops / sec) / 1e12;

    std::printf("Avg kernel time: %f ms (per launch)\n", ms);
    std::printf("Approx TFLOPS  : %f\n", tflops);

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
