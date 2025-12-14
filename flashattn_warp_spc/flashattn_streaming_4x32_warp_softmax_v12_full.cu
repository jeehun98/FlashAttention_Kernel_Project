// flashattn_streaming_4x32_warp_softmax_v12_full.cu
// v12: warp-full softmax (1 warp = 1 row), N_TILE=32
// - M = 4 rows (4 warps compute, plus 1 loader warp) => block = 5 warps = 160 threads
// - warp0: cp.async loader for K_T and V tiles (32 tokens)
// - warp1..warp4: each warp computes one row: QK + softmax + PV (streaming accumulation)
// - Host transposes K: K [Kdim, L] -> K_T [L, Kdim] per batch
// - Uses cp.async (Ampere+) and __syncthreads (no mbarrier)
//
// Build:
// nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo    -o flashattn_streaming_4x32_warp_softmax_v12.exe    flashattn_streaming_4x32_warp_softmax_v12_full.cu
//
// Profile:
// ncu --set full --launch-skip 10 --launch-count 1    ./flashattn_streaming_4x32_warp_softmax_v12.exe

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

// v12 config
constexpr int M      = 4;   // rows per batch handled by one block
constexpr int Kdim   = 16;
constexpr int Dv     = 16;
constexpr int N_TILE = 32;  // tokens per tile (warp-full softmax)

// padding (16B alignment comfort)
constexpr int PAD_K = 8; // 8 half = 16B
constexpr int PAD_V = 8;

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
// NOTE: smem operand must be 32-bit shared address
__device__ __forceinline__ void cp_async_cg_16B(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
    unsigned int smem_u32 = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_u32), "l"(gmem_src)
        : "memory");
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
// Q:   [B, M, Kdim] half
// K_T: [B, L, Kdim] half (host transposed)
// V:   [B, L, Dv]   half
// O:   [B, M, Dv]   float
__global__ void flashattn_warp_full_softmax_v12(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_T,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,
    float scale
) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int tid  = threadIdx.x;
    int warp = tid >> 5;       // 0..4
    int lane = tid & 31;       // 0..31

    const __half* Q_b  = Q   + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* KT_b = K_T + static_cast<size_t>(batch_id) * seq_len * Kdim;
    const __half* V_b  = V   + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b   = O   + static_cast<size_t>(batch_id) * M * Dv;

    // tile buffers
    __shared__ alignas(16) __half shK[N_TILE][Kdim + PAD_K]; // [token][k]
    __shared__ alignas(16) __half shV[N_TILE][Dv   + PAD_V]; // [token][d]

    // running streaming-softmax states per row
    __shared__ float m_run[M];
    __shared__ float l_run[M];
    __shared__ float y_run[M][Dv];

    // init states
    if (tid < M) {
        m_run[tid] = NEG_INF;
        l_run[tid] = 0.0f;
        #pragma unroll
        for (int d = 0; d < Dv; ++d) y_run[tid][d] = 0.0f;
    }
    __syncthreads();

    int num_tiles = seq_len / N_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int base = t * N_TILE;

        // warp0: cp.async load one tile of K_T and V (32 tokens)
        if (warp == 0) {
            // Each lane copies 16B for K and 16B for V.
            // lane 0..31 -> token n = lane, segment seg = 0/1
            int token = lane; // 0..31
            int seg   = 0;    // we'll do 2 segments using predication on lane halves? better: use lane mapping
            // Use two 16B moves by mapping lane pairs: (token = lane>>1), seg=(lane&1)
            int idx = lane;        // 0..31
            int n   = idx >> 1;    // 0..15
            int s   = idx & 1;     // 0/1
            int off = s * 8;       // 8 half = 16B

            // K: tokens [0..31], but we only cover 16 tokens per group; run twice logically using n and (n+16)
            // We'll load token n and token n+16 by using same lane but different base pointers:
            // First: token0 = n
            int token0 = n;
            const __half* gK0 = KT_b + (base + token0) * Kdim + off;
            __half* sK0 = &shK[token0][off];
            cp_async_cg_16B(sK0, gK0);

            const __half* gV0 = V_b + (base + token0) * Dv + off;
            __half* sV0 = &shV[token0][off];
            cp_async_cg_16B(sV0, gV0);

            // Second: token1 = n + 16
            int token1 = n + 16;
            const __half* gK1 = KT_b + (base + token1) * Kdim + off;
            __half* sK1 = &shK[token1][off];
            cp_async_cg_16B(sK1, gK1);

            const __half* gV1 = V_b + (base + token1) * Dv + off;
            __half* sV1 = &shV[token1][off];
            cp_async_cg_16B(sV1, gV1);

            cp_async_commit_group();
        }

        __syncthreads();
        if (warp == 0) cp_async_wait_group0();
        __syncthreads(); // tile ready

        // warp1..warp4: each warp handles one row (0..3)
        if (warp >= 1 && warp <= M) {
            int row = warp - 1;

            // qk for this token lane (token=lane in [0..31])
            float qk = 0.0f;
            #pragma unroll
            for (int k = 0; k < Kdim; ++k) {
                qk += __half2float(Q_b[row * Kdim + k]) * __half2float(shK[lane][k]);
            }
            qk *= scale;

            // softmax stats within this warp
            float m_t = warp_allreduce_max(qk);
            float e   = __expf(qk - m_t);
            float l_t = warp_allreduce_sum(e);

            // PV partials reduced across warp
            float u[Dv];
            #pragma unroll
            for (int d = 0; d < Dv; ++d) {
                float v = __half2float(shV[lane][d]);
                u[d] = warp_allreduce_sum(e * v);
            }

            // streaming merge (single lane updates shared running state)
            if (lane == 0) {
                float m_old = m_run[row];
                float l_old = l_run[row];

                if (m_old == NEG_INF) {
                    m_run[row] = m_t;
                    l_run[row] = l_t;
                    #pragma unroll
                    for (int d = 0; d < Dv; ++d) y_run[row][d] = u[d];
                } else {
                    float m_new = fmaxf(m_old, m_t);
                    float alpha = __expf(m_old - m_new);
                    float beta  = __expf(m_t   - m_new);

                    float l_new = l_old * alpha + l_t * beta;
                    #pragma unroll
                    for (int d = 0; d < Dv; ++d) {
                        float y_old = y_run[row][d];
                        y_run[row][d] = y_old * alpha + u[d] * beta;
                    }
                    m_run[row] = m_new;
                    l_run[row] = l_new;
                }
            }
        }

        __syncthreads(); // ensure running state visible before next tile
    }

    // write O: warp1..warp4
    if (warp >= 1 && warp <= M) {
        int row = warp - 1;
        float inv_l = 1.0f / (l_run[row] + EPS);
        for (int d = lane; d < Dv; d += WARP_SIZE) {
            O_b[row * Dv + d] = y_run[row][d] * inv_l;
        }
    }
}

// ---------------- CPU reference ----------------
// CPU expects original K layout: [B, Kdim, L]
static void flashattn_cpu_ref(
    const std::vector<__half>& hQ,
    const std::vector<__half>& hK,   // [B, Kdim, L]
    const std::vector<__half>& hV,   // [B, L, Dv]
    std::vector<float>& hO_ref,
    int num_batches,
    int seq_len,
    float scale
) {
    auto h2f = [](__half x) { return __half2float(x); };

    for (int b = 0; b < num_batches; ++b) {
        const __half* Qb = hQ.data() + static_cast<size_t>(b) * M * Kdim;
        const __half* Kb = hK.data() + static_cast<size_t>(b) * Kdim * seq_len;
        const __half* Vb = hV.data() + static_cast<size_t>(b) * seq_len * Dv;
        float* Ob = hO_ref.data() + static_cast<size_t>(b) * M * Dv;

        // scores
        std::vector<float> S(M * seq_len);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float acc = 0.f;
                for (int k = 0; k < Kdim; ++k) {
                    acc += h2f(Qb[i*Kdim + k]) * h2f(Kb[k*seq_len + j]);
                }
                S[i*seq_len + j] = acc * scale;
            }
        }

        // softmax
        std::vector<float> P(M * seq_len);
        for (int i = 0; i < M; ++i) {
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

        // O = P * V
        for (int i = 0; i < M; ++i) {
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

// ---------------- host transpose: K [Kdim, L] -> K_T [L, Kdim] ----------------
static void transpose_K_host(
    const std::vector<__half>& hK,   // [B, Kdim, L]
    std::vector<__half>& hK_T,       // [B, L, Kdim]
    int num_batches,
    int seq_len
) {
    for (int b = 0; b < num_batches; ++b) {
        const __half* Kb = hK.data() + static_cast<size_t>(b) * Kdim * seq_len;
        __half*      Tb  = hK_T.data() + static_cast<size_t>(b) * seq_len * Kdim;
        for (int k = 0; k < Kdim; ++k) {
            for (int j = 0; j < seq_len; ++j) {
                Tb[j*Kdim + k] = Kb[k*seq_len + j];
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like (v12: warp-full softmax, M=4, N_TILE=32)\n");

    constexpr int NUM_BATCH   = 1024;
    constexpr int NUM_TILES   = 4;               // seq_len = 32 * 4 = 128 (same as before)
    constexpr int SEQ_LEN     = N_TILE * NUM_TILES;

    const int num_batches = NUM_BATCH;
    const int seq_len     = SEQ_LEN;

    float scale = 1.0f / std::sqrt((float)Kdim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__half> hQ(num_batches * M * Kdim);
    std::vector<__half> hK(num_batches * Kdim * seq_len);      // original layout
    std::vector<__half> hK_T(num_batches * seq_len * Kdim);    // transposed
    std::vector<__half> hV(num_batches * seq_len * Dv);
    std::vector<float>  hO_ref(num_batches * M * Dv);
    std::vector<float>  hO(num_batches * M * Dv);

    for (int i = 0; i < (int)hQ.size(); ++i) hQ[i] = __float2half(dist(rng));
    for (int i = 0; i < (int)hK.size(); ++i) hK[i] = __float2half(dist(rng));
    for (int i = 0; i < (int)hV.size(); ++i) hV[i] = __float2half(dist(rng));

    transpose_K_host(hK, hK_T, num_batches, seq_len);
    flashattn_cpu_ref(hQ, hK, hV, hO_ref, num_batches, seq_len, scale);

    __half *dQ=nullptr, *dKT=nullptr, *dV=nullptr;
    float *dO=nullptr;

    CHECK_CUDA(cudaMalloc(&dQ,  hQ.size()   * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dKT, hK_T.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV,  hV.size()   * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO,  hO.size()   * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ,  hQ.data(),   hQ.size()   * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dKT, hK_T.data(), hK_T.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV,  hV.data(),   hV.size()   * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(5 * WARP_SIZE, 1, 1); // 5 warps = 160 threads
    dim3 grid(num_batches, 1, 1);

    // correctness
    flashattn_warp_full_softmax_v12<<<grid, block>>>(dQ, dKT, dV, dO, num_batches, seq_len, scale);
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
    for(int j=0;j<8;++j) std::printf("%f ", hO[b0*M*Dv + 0*Dv + j]);
    std::printf("\nBatch %d, O_ref[0, 0..7] (CPU): ", b0);
    for(int j=0;j<8;++j) std::printf("%f ", hO_ref[b0*M*Dv + 0*Dv + j]);
    std::printf("\nRelative L2 error over all batches: %.12e\n", rel_l2);
    std::printf("NUM_BATCH=%d, SEQ_LEN=%d, M=%d, Kdim=%d, Dv=%d, N_TILE=%d\n",
                num_batches, seq_len, M, Kdim, Dv, N_TILE);

    // perf (IMPORTANT: do NOT trust this under ncu full replay; run exe alone)
    const int NUM_ITERS = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(int it=0; it<NUM_ITERS; ++it){
        flashattn_warp_full_softmax_v12<<<grid, block>>>(dQ, dKT, dV, dO, num_batches, seq_len, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms=0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS;

    // FLOPs rough:
    // QK: per row i, per token j: Kdim mul-add => ~2*Kdim
    // so QK flops: 2*M*seq_len*Kdim
    // PV: for each row-token-d: mul-add => ~2
    // so PV flops: 2*M*seq_len*Dv
    double flops_per_batch = 2.0 * M * seq_len * Kdim + 2.0 * M * seq_len * Dv;
    double total_flops = flops_per_batch * num_batches;
    double sec = ms * 1e-3;
    double tflops = (total_flops / sec) / 1e12;

    std::printf("Avg kernel time: %f ms (per launch)\n", ms);
    std::printf("Approx TFLOPS  : %f\n", tflops);

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dKT));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
