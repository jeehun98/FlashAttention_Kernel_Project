// 8.5 Streaming Softmax FlashAttention-like Multi-Warp (v7: cp.async + ping-pong + warp-level ready flag, no CTA barrier)
// - Target: sm_80+ (tested for sm_86). Uses cp.async but avoids __syncthreads() in the tile loop.
// - Two warps / block (64 threads).
//   * warp0: WMMA QK^T + cp.async prefetch V tiles (producer) + compute rows 0..7
//   * warp1: compute rows 8..15 (consumer)
// - Producer/consumer sync uses shared "ready[2]" flags + __threadfence_block, not __syncthreads.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -o flashattn_streaming_16x16_mw_v7.exe flashattn_streaming_16x16_mw_v7.cu
//
// Profile:
//   ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_mw_v7.exe

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
constexpr int N_TILE   = 16;

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS       = 1e-6f;

// half2 view
constexpr int Dv2 = Dv / 2; // 8

// ============ warp reductions ============
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
// ============ cp.async wrappers (sm80+) ============
// 핵심: shared addr 변환은 PTX cvta를 직접 쓰지 말고 builtin 사용
//      (ptxas operand mismatch 회피)
__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
#if (__CUDA_ARCH__ >= 800)
    // generic -> shared-space u32 address
    unsigned smem_u32 = __cvta_generic_to_shared(smem_ptr);

    // gmem ptr은 64-bit address로 넣기
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_u32), "l"(gmem_ptr)
    );
#else
    *reinterpret_cast<int*>(smem_ptr) = *reinterpret_cast<const int*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait0() {
#if (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
    asm volatile("membar.cta;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
    asm volatile("membar.cta;\n" ::);
#endif
}


// ============ kernel ============
__global__ void flashattn_streaming_16x16_kernel_mw_v7(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,    // multiple of 16
    float scale
) {
#if __CUDA_ARCH__ < 700
    return;
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int tid      = threadIdx.x;
    int lane     = tid & 31;
    int warp_id  = tid >> 5; // 0 or 1

    // rows split across 2 warps
    int row_start = (warp_id == 0) ? 0 : 8;
    int row_end   = (warp_id == 0) ? 8 : 16;

    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // warp0 keeps Q fragment
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    if (warp_id == 0) {
        wmma::load_matrix_sync(q_frag, Q_b, Kdim);
    }

    // running states
    __shared__ float sh_m_running[M];
    __shared__ float sh_l_running[M];
    __shared__ float sh_y_running[M][Dv];

    // init running states (whole block)
    for (int i = tid; i < M; i += blockDim.x) {
        sh_m_running[i] = NEG_LARGE;
        sh_l_running[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < Dv; ++d) sh_y_running[i][d] = 0.0f;
    }
    __syncthreads(); // only once, outside tile loop

    // tile scores produced by warp0 (sync store)
    __shared__ float s_scores[M * N_TILE];

    // V tiles: ping-pong buffers, stored as half2 [Dv2][N_TILE]
    __shared__ __half2 sV2[2][Dv2][N_TILE];

    // producer-ready flags (each buffer corresponds to tile index+1)
    __shared__ int ready[2];

    if (tid < 2) ready[tid] = 0;
    __syncthreads(); // init ready

    int num_k_tiles = seq_len / N_TILE;

    auto wait_ready = [&](int buf, int expect) {
        // spin-wait with volatile load; only lane0 spins, then syncwarp
        if (lane == 0) {
            volatile int* p = ready;
            while (p[buf] < expect) { /* spin */ }
        }
        __syncwarp();
    };

    auto signal_ready = [&](int buf, int val) {
        // ensure shared writes are visible to other warps before flag set
        __threadfence_block();
        if (lane == 0) {
            // atomicExch is overkill but safe for cross-warp visibility
            atomicExch((int*)&ready[buf], val);
        }
        __syncwarp();
    };

    auto prefetch_tile_V = [&](int t, int buf) {
        // warp0 issues cp.async loads for V tile t into sV2[buf]
        if (warp_id != 0) return;

        int col_start = t * N_TILE;
        // total elements: N_TILE * Dv2 (each is half2 -> 4B)
        // distribute across warp lanes
        for (int idx = lane; idx < N_TILE * Dv2; idx += WARP_SIZE) {
            int j  = idx / Dv2; // 0..15
            int d2 = idx % Dv2; // 0..7
            int kv = col_start + j;
            const __half* src_h = V_b + kv * Dv + (2 * d2);
            void* dst = (void*)(&sV2[buf][d2][j]);
            // copy 4 bytes (half2) asynchronously
            cp_async_4B(dst, (const void*)src_h);
        }
        cp_async_commit();
        // NOTE: do not wait here (for overlap). We'll wait later when we need to "publish" this buf.
    };

    auto compute_scores_tile = [&](int t) {
        // warp0 computes QK^T for tile t and writes s_scores
        if (warp_id != 0) return;

        int col_start = t * N_TILE;

        wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
        wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        const __half* K_tile_base = K_b + col_start; // K is [Kdim, L] row-major (leading dim = seq_len)
        wmma::load_matrix_sync(k_frag, K_tile_base, seq_len);
        wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
        wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);
    };

    // ==== prologue: prefetch tile0 V into buf0, and compute scores0 ====
    if (warp_id == 0) {
        compute_scores_tile(0);
        prefetch_tile_V(0, 0);
        // finalize: wait all async copies, then publish ready[0]=1
        cp_async_wait_all();
        signal_ready(0, 1);
    }
    // warp1 waits until tile0 is ready
    if (warp_id == 1) {
        wait_ready(0, 1);
    }
    __syncwarp();

    // ==== main loop ====
    for (int t = 0; t < num_k_tiles; ++t) {
        int cur_buf  = (t & 1);
        int next_buf = (t ^ 1);

        // ensure current buffer is ready for this tile
        wait_ready(cur_buf, t + 1);

        // start prefetch of next tile (overlapped)
        if (t + 1 < num_k_tiles) {
            if (warp_id == 0) {
                // compute next tile scores first (sync), then prefetch V for next tile
                compute_scores_tile(t + 1);
                prefetch_tile_V(t + 1, next_buf);
            }
        }

        // === compute for current tile t (both warps do their row ranges) ===
        for (int i = row_start; i < row_end; ++i) {
            // 1) max over 16 columns (warp-parallel)
            float x = NEG_LARGE;
            if (lane < N_TILE) {
                x = s_scores[i * N_TILE + lane] * scale;
            }
            float maxv = warp_allreduce_max(x);

            // 2) local partial sums for l_t and u_t
            float l_t_partial = 0.0f;
            float u_lo_partial[Dv2];
            float u_hi_partial[Dv2];
            #pragma unroll
            for (int d2 = 0; d2 < Dv2; ++d2) {
                u_lo_partial[d2] = 0.0f;
                u_hi_partial[d2] = 0.0f;
            }

            for (int col = lane; col < N_TILE; col += WARP_SIZE) {
                float s_col = s_scores[i * N_TILE + col] * scale;
                float e = __expf(s_col - maxv);
                l_t_partial += e;

                #pragma unroll
                for (int d2 = 0; d2 < Dv2; ++d2) {
                    __half2 hv2 = sV2[cur_buf][d2][col];
                    float2 fv2  = __half22float2(hv2);
                    u_lo_partial[d2] += e * fv2.x;
                    u_hi_partial[d2] += e * fv2.y;
                }
            }

            float l_t = warp_allreduce_sum(l_t_partial);

            float u_lo[Dv2];
            float u_hi[Dv2];
            #pragma unroll
            for (int d2 = 0; d2 < Dv2; ++d2) {
                u_lo[d2] = warp_allreduce_sum(u_lo_partial[d2]);
                u_hi[d2] = warp_allreduce_sum(u_hi_partial[d2]);
            }

            // 3) streaming update (lane0)
            if (lane == 0 && l_t > 0.0f) {
                float m_old = sh_m_running[i];
                float l_old = sh_l_running[i];
                float m_t   = maxv;

                if (m_old == NEG_LARGE) {
                    sh_m_running[i] = m_t;
                    sh_l_running[i] = l_t;
                    #pragma unroll
                    for (int d2 = 0; d2 < Dv2; ++d2) {
                        sh_y_running[i][2*d2 + 0] = u_lo[d2];
                        sh_y_running[i][2*d2 + 1] = u_hi[d2];
                    }
                } else {
                    float m_new = fmaxf(m_old, m_t);
                    float alpha = __expf(m_old - m_new);
                    float beta  = __expf(m_t   - m_new);
                    float l_new = l_old * alpha + l_t * beta;

                    #pragma unroll
                    for (int d2 = 0; d2 < Dv2; ++d2) {
                        float y0 = sh_y_running[i][2*d2 + 0];
                        float y1 = sh_y_running[i][2*d2 + 1];
                        sh_y_running[i][2*d2 + 0] = y0 * alpha + u_lo[d2] * beta;
                        sh_y_running[i][2*d2 + 1] = y1 * alpha + u_hi[d2] * beta;
                    }

                    sh_m_running[i] = m_new;
                    sh_l_running[i] = l_new;
                }
            }
        }

        // === publish next buffer when prefetch is done ===
        if (t + 1 < num_k_tiles) {
            if (warp_id == 0) {
                // wait until all async copies for next tile are visible, then publish ready
                cp_async_wait_all();
                signal_ready(next_buf, (t + 2)); // next tile expects t+2
            } else {
                // warp1 will just wait_ready at top of next iteration
            }
        }
    }

    // ==== final write ====
    for (int i = row_start; i < row_end; ++i) {
        float inv_l = 1.0f / (sh_l_running[i] + EPS);
        for (int d = lane; d < Dv; d += WARP_SIZE) {
            O_b[i * Dv + d] = sh_y_running[i][d] * inv_l;
        }
    }
}

// ============ CPU reference ============
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

        std::vector<float> P(M * seq_len);
        for (int i = 0; i < M; ++i) {
            float maxv = NEG_LARGE;
            for (int j = 0; j < seq_len; ++j) maxv = fmaxf(maxv, S[i * seq_len + j]);
            float sumv = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float e = std::exp(S[i * seq_len + j] - maxv);
                P[i * seq_len + j] = e;
                sumv += e;
            }
            float inv = 1.0f / (sumv + EPS);
            for (int j = 0; j < seq_len; ++j) P[i * seq_len + j] *= inv;
        }

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
    std::printf("Streaming Softmax FlashAttention-like Multi-Warp (v7: cp.async + ping-pong + ready-flag, no CTA barrier)\n");

    constexpr int NUM_BATCH    = 1024;
    constexpr int NUM_K_TILES  = 8;
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

    for (int i = 0; i < (int)hQ.size(); ++i) hQ[i] = __float2half(dist(rng));
    for (int i = 0; i < (int)hK.size(); ++i) hK[i] = __float2half(dist(rng));
    for (int i = 0; i < (int)hV.size(); ++i) hV[i] = __float2half(dist(rng));

    flashattn_streaming_cpu_ref(hQ, hK, hV, hO_ref, num_batches, seq_len, scale);

    __half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
    float  *dO=nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(64,1,1);
    dim3 grid(num_batches,1,1);

    flashattn_streaming_16x16_kernel_mw_v7<<<grid, block>>>(dQ,dK,dV,dO,num_batches,seq_len,scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double num=0.0, den=0.0;
    for (size_t i=0;i<hO.size();++i) {
        double diff = (double)hO[i] - (double)hO_ref[i];
        num += diff*diff;
        den += (double)hO_ref[i]*(double)hO_ref[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    int b0=0;
    std::printf("Batch %d, O[0, 0..7] (GPU): ", b0);
    for (int j=0;j<8;++j) std::printf("%f ", hO[b0*M*Dv + 0*Dv + j]);
    std::printf("\nBatch %d, O_ref[0, 0..7] (CPU): ", b0);
    for (int j=0;j<8;++j) std::printf("%f ", hO_ref[b0*M*Dv + 0*Dv + j]);
    std::printf("\nRelative L2 error over all batches: %.12e\n", rel_l2);
    std::printf("NUM_BATCH=%d, SEQ_LEN=%d, M=Kdim=Dv=16\n", num_batches, seq_len);

    const int NUM_ITERS=50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it=0; it<NUM_ITERS; ++it) {
        flashattn_streaming_16x16_kernel_mw_v7<<<grid, block>>>(dQ,dK,dV,dO,num_batches,seq_len,scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms=0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS;

    double flops_per_batch =
        2.0 * M * seq_len * Kdim +
        6.0 * M * seq_len * Dv;
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
/*
nvcc -O3 -std=c++17 -arch=sm_86   -o flashattn_streaming_16x16_mw_v7.exe   flashattn_streaming_16x16_mw_v7.cu

ncu --set full --launch-skip 10 --launch-count 1 ./flashattn_streaming_16x16_mw_v7.exe
*/