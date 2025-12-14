// flashattn_streaming_16x32_warp_softmax_v12c.cu
// v12c: warp-full softmax (N_TILE=32), M=16 via 4-row blocks,
//       K uses original layout [Kdim, L] (NO K_T), coalesced cp.async,
//       shared strides aligned to 16B multiples.
//
// - Each batch uses 4 CTAs: row blocks [0..3], [4..7], [8..11], [12..15]
// - Block: 5 warps = 160 threads (warp0 loader + warp1..4 compute rows)
// - K: [B, Kdim, L] half, row-major, ld = L
// - V: [B, L, Dv]   half
// - Q: [B, M, Kdim] half
// - O: [B, M, Dv]   float
//
// Build:
// nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo    -o flashattn_streaming_16x32_warp_softmax_v12c.exe    flashattn_streaming_16x32_warp_softmax_v12c.cu
//
// Profile:
// ncu --set full --launch-skip 10 --launch-count 1    ./flashattn_streaming_16x32_warp_softmax_v12c.exe

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(e));                                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int WARP_SIZE = 32;

// global problem (match your old experiments)
constexpr int M_TOTAL = 16;
constexpr int Kdim    = 16;
constexpr int Dv      = 16;

constexpr int N_TILE  = 32;
constexpr int SEQ_LEN = 128; // multiple of 32

constexpr int ROWS_PER_BLOCK = 4; // compute warps per CTA
static_assert(M_TOTAL % ROWS_PER_BLOCK == 0, "M_TOTAL must be divisible by ROWS_PER_BLOCK");

// Shared padding chosen for 16B-aligned row strides (critical for cp.async 16B stores)
constexpr int PAD_T = 8;  // for shK token dimension: (32 + 8) * 2B = 80B stride
constexpr int PAD_V = 8;  // for shV dv dimension:    (16 + 8) * 2B = 48B stride

constexpr float NEG_INF = -1e30f;
constexpr float EPS     = 1e-6f;

// ---------------- warp reductions ----------------
__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  8));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  4));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  2));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  1));
    return v;
}
__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v,  8);
    v += __shfl_xor_sync(mask, v,  4);
    v += __shfl_xor_sync(mask, v,  2);
    v += __shfl_xor_sync(mask, v,  1);
    return v;
}

// ---------------- cp.async helpers (Ampere+) ----------------
__device__ __forceinline__ void cp_async_cg_16B(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
    unsigned int smem_u32 = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32), "l"(gmem_src) : "memory");
#else
    (void)smem_dst; (void)gmem_src;
#endif
}
__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait_group0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// ---------------- kernel ----------------
// Q: [B, M_TOTAL, Kdim] half
// K: [B, Kdim, L]      half  (row-major, ld=L)  <-- IMPORTANT: original layout
// V: [B, L, Dv]        half
// O: [B, M_TOTAL, Dv]  float
__global__ void flashattn_warp_full_softmax_v12c(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,
    float scale
) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int batch_id = (int)blockIdx.x;
    int row_blk  = (int)blockIdx.y; // group of 4 rows
    if (batch_id >= num_batches) return;

    int row_base = row_blk * ROWS_PER_BLOCK;

    int tid  = threadIdx.x;
    int warp = tid >> 5; // 0..4
    int lane = tid & 31;

    const __half* Q_b = Q + (size_t)batch_id * M_TOTAL * Kdim + row_base * Kdim;
    const __half* K_b = K + (size_t)batch_id * Kdim * seq_len;     // [Kdim, L]
    const __half* V_b = V + (size_t)batch_id * seq_len * Dv;       // [L, Dv]
    float* O_b        = O + (size_t)batch_id * M_TOTAL * Dv + row_base * Dv;

    // Shared:
    // shK[k][token] : convenient for compute (fixed k, lanes read contiguous tokens)
    __shared__ alignas(16) __half shK[Kdim][N_TILE + PAD_T];
    __shared__ alignas(16) __half shV[N_TILE][Dv + PAD_V];

    __shared__ float m_run[ROWS_PER_BLOCK];
    __shared__ float l_run[ROWS_PER_BLOCK];
    __shared__ float y_run[ROWS_PER_BLOCK][Dv];

    if (tid < ROWS_PER_BLOCK) {
        m_run[tid] = NEG_INF;
        l_run[tid] = 0.0f;
        #pragma unroll
        for (int d = 0; d < Dv; ++d) y_run[tid][d] = 0.0f;
    }
    __syncthreads();

    int num_tiles = seq_len / N_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int col_start = t * N_TILE;

        // warp0: coalesced loads
        if (warp == 0) {
            // For K: each k-row has 32 half (=64B). We copy as 4x16B segments.
            // 32 lanes cover 8 rows * 4 segments; each lane loads 2 rows (k and k+8).
            int idx = lane;          // 0..31
            int k0  = idx >> 2;      // 0..7
            int seg = idx & 3;       // 0..3
            int off = seg * 8;       // 8 half = 16B

            int k1 = k0 + 8;         // 8..15

            const __half* gK0 = K_b + k0 * seq_len + (col_start + off);
            const __half* gK1 = K_b + k1 * seq_len + (col_start + off);

            __half* sK0 = &shK[k0][off];
            __half* sK1 = &shK[k1][off];

            cp_async_cg_16B(sK0, gK0);
            cp_async_cg_16B(sK1, gK1);

            // For V: each token has 16 half (=32B). Each lane loads its token in 2x16B.
            const __half* gV0 = V_b + (col_start + lane) * Dv + 0;
            const __half* gV1 = V_b + (col_start + lane) * Dv + 8;
            __half* sV0 = &shV[lane][0];
            __half* sV1 = &shV[lane][8];

            cp_async_cg_16B(sV0, gV0);
            cp_async_cg_16B(sV1, gV1);

            cp_async_commit_group();
        }

        __syncthreads();
        if (warp == 0) cp_async_wait_group0();
        __syncthreads();

        // warp1..warp4: compute one row each
        if (warp >= 1 && warp <= ROWS_PER_BLOCK) {
            int r = warp - 1;

            // QK for this token(lane)
            float qk = 0.0f;
            #pragma unroll
            for (int k = 0; k < Kdim; ++k) {
                float qv = __half2float(Q_b[r * Kdim + k]);
                float kv = __half2float(shK[k][lane]);
                qk += qv * kv;
            }
            qk *= scale;

            float m_t = warp_allreduce_max(qk);
            float e   = __expf(qk - m_t);
            float l_t = warp_allreduce_sum(e);

            float u[Dv];
            #pragma unroll
            for (int d = 0; d < Dv; ++d) {
                float vv = __half2float(shV[lane][d]);
                u[d] = warp_allreduce_sum(e * vv);
            }

            if (lane == 0) {
                float m_old = m_run[r];
                float l_old = l_run[r];

                if (m_old == NEG_INF) {
                    m_run[r] = m_t;
                    l_run[r] = l_t;
                    #pragma unroll
                    for (int d = 0; d < Dv; ++d) y_run[r][d] = u[d];
                } else {
                    float m_new = fmaxf(m_old, m_t);
                    float alpha = __expf(m_old - m_new);
                    float beta  = __expf(m_t   - m_new);

                    float l_new = l_old * alpha + l_t * beta;
                    #pragma unroll
                    for (int d = 0; d < Dv; ++d) {
                        y_run[r][d] = y_run[r][d] * alpha + u[d] * beta;
                    }
                    m_run[r] = m_new;
                    l_run[r] = l_new;
                }
            }
        }

        __syncthreads();
    }

    // write O
    if (warp >= 1 && warp <= ROWS_PER_BLOCK) {
        int r = warp - 1;
        float inv_l = 1.0f / (l_run[r] + EPS);
        for (int d = lane; d < Dv; d += WARP_SIZE) {
            O_b[r * Dv + d] = y_run[r][d] * inv_l;
        }
    }
}

// ---------------- CPU reference ----------------
static void flashattn_cpu_ref(
    const std::vector<__half>& hQ,
    const std::vector<__half>& hK, // [B, Kdim, L]
    const std::vector<__half>& hV, // [B, L, Dv]
    std::vector<float>& hO_ref,
    int num_batches,
    int seq_len,
    float scale
) {
    auto h2f = [](__half x) { return __half2float(x); };

    for (int b = 0; b < num_batches; ++b) {
        const __half* Qb = hQ.data() + (size_t)b * M_TOTAL * Kdim;
        const __half* Kb = hK.data() + (size_t)b * Kdim * seq_len;
        const __half* Vb = hV.data() + (size_t)b * seq_len * Dv;
        float* Ob = hO_ref.data() + (size_t)b * M_TOTAL * Dv;

        std::vector<float> S(M_TOTAL * seq_len);
        for (int i = 0; i < M_TOTAL; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float acc = 0.f;
                for (int k = 0; k < Kdim; ++k) {
                    acc += h2f(Qb[i*Kdim + k]) * h2f(Kb[k*seq_len + j]);
                }
                S[i*seq_len + j] = acc * scale;
            }
        }

        std::vector<float> P(M_TOTAL * seq_len);
        for (int i = 0; i < M_TOTAL; ++i) {
            float maxv = NEG_INF;
            for (int j = 0; j < seq_len; ++j) maxv = std::max(maxv, S[i*seq_len + j]);
            float sumv = 0.f;
            for (int j = 0; j < seq_len; ++j) {
                float e = std::exp(S[i*seq_len + j] - maxv);
                P[i*seq_len + j] = e;
                sumv += e;
            }
            float inv = 1.f / (sumv + EPS);
            for (int j = 0; j < seq_len; ++j) P[i*seq_len + j] *= inv;
        }

        for (int i = 0; i < M_TOTAL; ++i) {
            for (int d = 0; d < Dv; ++d) {
                float acc = 0.f;
                for (int j = 0; j < seq_len; ++j) {
                    acc += P[i*seq_len + j] * h2f(Vb[j*Dv + d]);
                }
                Ob[i*Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like (v12c: NO K_T, coalesced K loads, aligned shared)\n");

    constexpr int NUM_BATCH = 1024;
    const int num_batches = NUM_BATCH;
    const int seq_len = SEQ_LEN;

    float scale = 1.0f / std::sqrt((float)Kdim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__half> hQ((size_t)num_batches * M_TOTAL * Kdim);
    std::vector<__half> hK((size_t)num_batches * Kdim * seq_len);  // [Kdim, L]
    std::vector<__half> hV((size_t)num_batches * seq_len * Dv);    // [L, Dv]
    std::vector<float>  hO_ref((size_t)num_batches * M_TOTAL * Dv);
    std::vector<float>  hO((size_t)num_batches * M_TOTAL * Dv);

    for (size_t i = 0; i < hQ.size(); ++i) hQ[i] = __float2half(dist(rng));
    for (size_t i = 0; i < hK.size(); ++i) hK[i] = __float2half(dist(rng));
    for (size_t i = 0; i < hV.size(); ++i) hV[i] = __float2half(dist(rng));

    flashattn_cpu_ref(hQ, hK, hV, hO_ref, num_batches, seq_len, scale);

    __half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
    float *dO=nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block((1 + ROWS_PER_BLOCK) * WARP_SIZE, 1, 1); // 160
    dim3 grid(num_batches, M_TOTAL / ROWS_PER_BLOCK, 1); // (1024, 4, 1) -> 4096 CTAs

    // correctness
    flashattn_warp_full_softmax_v12c<<<grid, block>>>(dQ, dK, dV, dO, num_batches, seq_len, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double num=0.0, den=0.0;
    for (size_t i=0;i<hO.size();++i){
        double diff = (double)hO[i] - (double)hO_ref[i];
        num += diff*diff;
        den += (double)hO_ref[i]*(double)hO_ref[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    int b0=0;
    std::printf("Batch %d, O[0, 0..7] (GPU): ", b0);
    for(int j=0;j<8;++j) std::printf("%f ", hO[b0*M_TOTAL*Dv + 0*Dv + j]);
    std::printf("\nBatch %d, O_ref[0, 0..7] (CPU): ", b0);
    for(int j=0;j<8;++j) std::printf("%f ", hO_ref[b0*M_TOTAL*Dv + 0*Dv + j]);
    std::printf("\nRelative L2 error: %.12e\n", rel_l2);

    // perf (exe-only)
    const int NUM_ITERS = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(int it=0; it<NUM_ITERS; ++it){
        flashattn_warp_full_softmax_v12c<<<grid, block>>>(dQ, dK, dV, dO, num_batches, seq_len, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms=0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS;

    // FLOPs rough:
    // QK: 2*M*L*K
    // PV: 2*M*L*Dv
    double flops_per_batch = 2.0 * M_TOTAL * seq_len * Kdim + 2.0 * M_TOTAL * seq_len * Dv;
    double total_flops = flops_per_batch * num_batches;
    double sec = ms * 1e-3;
    double tflops = (total_flops / sec) / 1e12;

    std::printf("NUM_BATCH=%d, SEQ_LEN=%d, M_TOTAL=%d, ROWS_PER_BLOCK=%d, N_TILE=%d\n",
                num_batches, seq_len, M_TOTAL, ROWS_PER_BLOCK, N_TILE);
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
