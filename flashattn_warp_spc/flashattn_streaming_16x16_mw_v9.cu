// flashattn_streaming_16x16_mw_v9_1.cu
// v9.1: mbarrier try_wait + per-tile re-init (no parity, no per-tile __syncthreads)

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cstdint>

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
constexpr int M = 16;
constexpr int Kdim = 16;
constexpr int Dv = 16;
constexpr int N_TILE = 16;

// cp.async alignment requirements
constexpr int PAD_K = 8; // 8 half = 16B
constexpr int PAD_V = 8;

constexpr float NEG_LARGE = -1e30f;
constexpr float EPS = 1e-6f;

// ---------------- warp reductions ----------------
__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned m = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1)
        v = fmaxf(v, __shfl_xor_sync(m, v, d));
    return v;
}
__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned m = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_xor_sync(m, v, d);
    return v;
}

// ---------------- cp.async ----------------
__device__ __forceinline__ void cp_async_cg_16B(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
    unsigned smem_u32 = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_u32), "l"(gmem) : "memory");
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// ---------------- mbarrier ----------------
__device__ __forceinline__ void mbarrier_init_shared(void* bar, unsigned count) {
#if __CUDA_ARCH__ >= 800
    unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(bar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "l"(addr), "r"(count) : "memory");
#endif
}
__device__ __forceinline__ void mbarrier_arrive_shared(void* bar) {
#if __CUDA_ARCH__ >= 800
    unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(bar);
    unsigned long long state;
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
        : "=l"(state) : "l"(addr) : "memory");
#endif
}
__device__ __forceinline__ bool mbarrier_try_wait_shared(void* bar) {
#if __CUDA_ARCH__ >= 800
    unsigned long long addr = (unsigned long long)__cvta_generic_to_shared(bar);
    int out;
    asm volatile(
        "{ .reg .pred p; "
        "  mbarrier.try_wait.shared::cta.b64 p, [%1]; "
        "  selp.b32 %0, 1, 0, p; }"
        : "=r"(out) : "l"(addr) : "memory");
    return out != 0;
#else
    return true;
#endif
}
__device__ __forceinline__ void nanosleep_u32(unsigned t) {
#if __CUDA_ARCH__ >= 800
    asm volatile("nanosleep.u32 %0;\n" :: "r"(t));
#endif
}

// ---------------- kernel ----------------
__global__ void flashattn_streaming_16x16_kernel_mw_v9_1(
    const __half* Q,
    const __half* K_T,   // [B, L, Kdim]
    const __half* V,
    float* O,
    int num_batches,
    int seq_len,
    float scale
) {
#if __CUDA_ARCH__ < 800
    return;
#endif
    int batch = blockIdx.x;
    if (batch >= num_batches) return;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5; // 0,1

    const __half* Qb  = Q   + (size_t)batch * M * Kdim;
    const __half* KTb = K_T + (size_t)batch * seq_len * Kdim;
    const __half* Vb  = V   + (size_t)batch * seq_len * Dv;
    float*       Ob   = O   + (size_t)batch * M * Dv;

    __shared__ alignas(16) unsigned long long sh_bar;
    __shared__ float sh_m[M], sh_l[M], sh_y[M][Dv];
    __shared__ alignas(16) __half shK[2][N_TILE][Kdim + PAD_K];
    __shared__ alignas(16) __half shV[2][N_TILE][Dv   + PAD_V];
    __shared__ float s_scores[M * N_TILE];

    // init running state
    for (int i = tid; i < M; i += blockDim.x) {
        sh_m[i] = NEG_LARGE;
        sh_l[i] = 0.f;
        for (int d = 0; d < Dv; ++d) sh_y[i][d] = 0.f;
    }
    __syncthreads();

    // load Q
    wmma::fragment<wmma::matrix_a, M, N_TILE, Kdim, __half, wmma::row_major> q_frag;
    if (warp == 1)
        wmma::load_matrix_sync(q_frag, Qb, Kdim);

    int tiles = seq_len / N_TILE;

    auto load_tile = [&](int t, int buf) {
        int col = t * N_TILE;
        int idx = lane;
        int n = idx >> 1;
        int seg = idx & 1;
        int off = seg * 8;

        cp_async_cg_16B(&shK[buf][n][off], KTb + (col + n) * Kdim + off);
        cp_async_cg_16B(&shV[buf][n][off], Vb  + (col + n) * Dv   + off);
        cp_async_commit();
    };

    // preload tile 0
    if (warp == 0) {
        if (lane == 0) mbarrier_init_shared(&sh_bar, 1);
        load_tile(0, 0);
        cp_async_wait0();
        asm volatile("membar.cta;\n" ::: "memory");
        if (lane == 0) mbarrier_arrive_shared(&sh_bar);
    }

    for (int t = 0; t < tiles; ++t) {
        int cur = t & 1;
        int nxt = cur ^ 1;

        if (warp == 1) {
            while (!mbarrier_try_wait_shared(&sh_bar))
                nanosleep_u32(20);
            asm volatile("membar.cta;\n" ::: "memory");

            wmma::fragment<wmma::matrix_b, M, N_TILE, Kdim, __half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, M, N_TILE, Kdim, float> c_frag;
            wmma::fill_fragment(c_frag, 0.f);

            wmma::load_matrix_sync(k_frag, &shK[cur][0][0], Kdim + PAD_K);
            wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
            wmma::store_matrix_sync(s_scores, c_frag, N_TILE, wmma::mem_row_major);

            for (int i = 0; i < M; ++i) {
                float x = (lane < N_TILE) ? s_scores[i*N_TILE + lane] * scale : NEG_LARGE;
                float maxv = warp_allreduce_max(x);

                float lp = 0.f, up[Dv] = {};
                if (lane < N_TILE) {
                    float e = __expf(x - maxv);
                    lp = e;
                    for (int d = 0; d < Dv; ++d)
                        up[d] = e * __half2float(shV[cur][lane][d]);
                }
                float lt = warp_allreduce_sum(lp);
                float u[Dv];
                for (int d = 0; d < Dv; ++d)
                    u[d] = warp_allreduce_sum(up[d]);

                if (lane == 0) {
                    float mo = sh_m[i], lo = sh_l[i];
                    if (mo == NEG_LARGE) {
                        sh_m[i] = maxv;
                        sh_l[i] = lt;
                        for (int d = 0; d < Dv; ++d) sh_y[i][d] = u[d];
                    } else {
                        float mn = fmaxf(mo, maxv);
                        float a = __expf(mo - mn);
                        float b = __expf(maxv - mn);
                        sh_l[i] = lo * a + lt * b;
                        for (int d = 0; d < Dv; ++d)
                            sh_y[i][d] = sh_y[i][d] * a + u[d] * b;
                        sh_m[i] = mn;
                    }
                }
            }
        }

        if (warp == 0 && t + 1 < tiles) {
            if (lane == 0) mbarrier_init_shared(&sh_bar, 1);
            load_tile(t + 1, nxt);
            cp_async_wait0();
            asm volatile("membar.cta;\n" ::: "memory");
            if (lane == 0) mbarrier_arrive_shared(&sh_bar);
        }
    }

    if (warp == 1) {
        for (int i = 0; i < M; ++i) {
            float inv = 1.f / (sh_l[i] + EPS);
            for (int d = lane; d < Dv; d += 32)
                Ob[i*Dv + d] = sh_y[i][d] * inv;
        }
    }
}
