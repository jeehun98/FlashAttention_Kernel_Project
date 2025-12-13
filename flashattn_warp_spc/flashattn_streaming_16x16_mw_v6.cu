// flashattn_streaming_16x16_mw_v6_1.cu
// v6.1: v6 정합성 fix (cp.async wait는 발사한 warp0가 해야 함)
// - block: 2 warps (64 threads)
// - warp0: loader (cp.async global->shared + wait_group0)
// - warp1: compute (WMMA QK^T + streaming softmax + PV)
// - NOTE: mbarrier 없이 안전하게 가려고 tile마다 __syncthreads() 유지
//
// Build:
// nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo \
//   -o flashattn_streaming_16x16_mw_v6_1.exe \
//   flashattn_streaming_16x16_mw_v6_1.cu
//
// Profile:
// ncu --set full --launch-skip 10 --launch-count 1 \
//   ./flashattn_streaming_16x16_mw_v6_1.exe

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int M        = 16;
constexpr int Kdim     = 16;
constexpr int Dv       = 16;
constexpr int N_TILE   = 16;

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS       = 1e-6f;

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
// NOTE: shared operand must be 32-bit shared address
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
// Q: [B, M, Kdim] half
// K: [B, Kdim, L] half  (row-major, ld = L)
// V: [B, L, Dv]   half
// O: [B, M, Dv]   float
__global__ void flashattn_streaming_16x16_kernel_mw_v6_1(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int num_batches,
    int seq_len,   // multiple of 16
    float scale
) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid >> 5; // 0,1

    const __half* Q_b = Q + static_cast<size_t>(batch_id) * M * Kdim;
    const __half* K_b = K + static_cast<size_t>(batch_id) * Kdim * seq_len;
    const __half* V_b = V + static_cast<size_t>(batch_id) * seq_len * Dv;
    float*       O_b  = O + static_cast<size_t>(batch_id) * M * Dv;

    // warp1: Q fragment
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    if (warp == 1) {
        wmma::load_matrix_sync(q_frag, Q_b, Kdim);
    }

    // running state
    __shared__ float sh_m_running[M];
    __shared__ float sh_l_running[M];
    __shared__ float sh_y_running[M][Dv];

    for (int i = tid; i < M; i += blockDim.x) {
        sh_m_running[i] = NEG_LARGE;
        sh_l_running[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < Dv; ++d) sh_y_running[i][d] = 0.0f;
    }
    __syncthreads();

    // ping-pong buffers
    __shared__ __half shK[2][Kdim][N_TILE]; // [Kdim][N_TILE]
    __shared__ __half shV[2][N_TILE][Dv];   // [N_TILE][Dv]
    __shared__ float  s_scores[M * N_TILE];

    int num_tiles = seq_len / N_TILE;

    // ---- helper lambda: warp0 loads one tile t into buf ----
    auto load_tile_cpasync = [&](int t, int buf) {
        int col_start = t * N_TILE;

        // lane 0..31 -> (k,rowseg) mapping
        // Each lane copies 16B from K row and 16B from V row.
        int idx   = lane;       // 0..31
        int k     = idx >> 1;   // 0..15
        int seg   = idx & 1;    // 0/1
        int off_h = seg * 8;    // 8 half = 16B

        // K: row k, columns col_start + off_h .. + off_h+7
        const __half* gK = K_b + k * seq_len + (col_start + off_h);
        __half* sK = &shK[buf][k][off_h];
        cp_async_cg_16B(sK, gK);

        // V: row n=k, columns off_h..off_h+7 in Dv
        int n = k;
        const __half* gV = V_b + (col_start + n) * Dv + off_h;
        __half* sV = &shV[buf][n][off_h];
        cp_async_cg_16B(sV, gV);

        cp_async_commit_group();
    };

    // ---- preload tile0 into buf0 ----
    if (warp == 0) {
        load_tile_cpasync(0, 0);
    }

    // IMPORTANT: wait must be executed by issuer warp0
    __syncthreads();
    if (warp == 0) cp_async_wait_group0();
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = cur ^ 1;

        // prefetch next tile into nxt
        if (warp == 0 && (t + 1) < num_tiles) {
            load_tile_cpasync(t + 1, nxt);
        }

        // compute cur
        if (warp == 1) {
            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::row_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            wmma::load_matrix_sync(k_frag, &shK[cur][0][0], N_TILE);
            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);

            // streaming softmax + PV
            for (int i = 0; i < M; ++i) {
                float x = (lane < N_TILE) ? (s_scores[i * N_TILE + lane] * scale) : NEG_LARGE;
                float maxv = warp_allreduce_max(x);

                float l_part = 0.0f;
                float u_part[Dv];
                #pragma unroll
                for (int d = 0; d < Dv; ++d) u_part[d] = 0.0f;

                if (lane < N_TILE) {
                    float s = s_scores[i * N_TILE + lane] * scale;
                    float e = __expf(s - maxv);
                    l_part = e;
                    #pragma unroll
                    for (int d = 0; d < Dv; ++d) {
                        u_part[d] = e * __half2float(shV[cur][lane][d]);
                    }
                }

                float l_t = warp_allreduce_sum(l_part);

                float u[Dv];
                #pragma unroll
                for (int d = 0; d < Dv; ++d) u[d] = warp_allreduce_sum(u_part[d]);

                if (lane == 0 && l_t > 0.0f) {
                    float m_old = sh_m_running[i];
                    float l_old = sh_l_running[i];
                    float m_t   = maxv;

                    if (m_old == NEG_LARGE) {
                        sh_m_running[i] = m_t;
                        sh_l_running[i] = l_t;
                        #pragma unroll
                        for (int d = 0; d < Dv; ++d) sh_y_running[i][d] = u[d];
                    } else {
                        float m_new = fmaxf(m_old, m_t);
                        float alpha = __expf(m_old - m_new);
                        float beta  = __expf(m_t   - m_new);
                        float l_new = l_old * alpha + l_t * beta;

                        #pragma unroll
                        for (int d = 0; d < Dv; ++d) {
                            float y_old = sh_y_running[i][d];
                            sh_y_running[i][d] = y_old * alpha + u[d] * beta;
                        }
                        sh_m_running[i] = m_new;
                        sh_l_running[i] = l_new;
                    }
                }
            }
        }

        // before next tile compute, ensure nxt is ready (issuer warp0 waits)
        if ((t + 1) < num_tiles) {
            __syncthreads();
            if (warp == 0) cp_async_wait_group0();
            __syncthreads();
        }
    }

    // write O
    if (warp == 1) {
        for (int i = 0; i < M; ++i) {
            float inv_l = 1.0f / (sh_l_running[i] + EPS);
            for (int d = lane; d < Dv; d += WARP_SIZE) {
                if (d < Dv) O_b[i * Dv + d] = sh_y_running[i][d] * inv_l;
            }
        }
    }
}

// ---------------- CPU reference ----------------
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
                    acc += h2f(Q_b[i * Kdim + k]) * h2f(K_b[k * seq_len + j]);
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
                    acc += P[i * seq_len + j] * h2f(V_b[j * Dv + d]);
                }
                O_b[i * Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like Multi-Warp (v6.1: cp.async wait fix)\n");

    constexpr int NUM_BATCH   = 1024;
    constexpr int NUM_K_TILES = 8;
    constexpr int SEQ_LEN     = N_TILE * NUM_K_TILES;

    const int num_batches = NUM_BATCH;
    const int seq_len     = SEQ_LEN;

    float scale = 1.0f / std::sqrt((float)Kdim);

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
    float *dO=nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(64,1,1);
    dim3 grid(num_batches,1,1);

    // correctness
    flashattn_streaming_16x16_kernel_mw_v6_1<<<grid, block>>>(dQ,dK,dV,dO,num_batches,seq_len,scale);
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
    std::printf("NUM_BATCH=%d, SEQ_LEN=%d, M=Kdim=Dv=16\n", num_batches, seq_len);

    // perf
    const int NUM_ITERS=50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(int it=0; it<NUM_ITERS; ++it){
        flashattn_streaming_16x16_kernel_mw_v6_1<<<grid, block>>>(dQ,dK,dV,dO,num_batches,seq_len,scale);
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
