// flashattn_streaming_16x32_2warp_softmax_v12i.cu
// v12i (A안): smemW 제거, warp1이 score 재계산해서 weight 즉시 적용
// - warp0: online softmax stats (m,l)만 계산
// - warp1: score 재계산 -> exp(score - m_new) -> O_num 누적 -> 마지막 normalize + coalesced store
//
// Build:
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 flashattn_streaming_16x32_2warp_softmax_v12i.cu -o flashattn_streaming_16x32_2warp_softmax_v12i.exe
//
// Run:
//   .\flashattn_streaming_16x32_2warp_softmax_v12i.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(cmd)                                                      \
    do {                                                                     \
        cudaError_t e = (cmd);                                               \
        if (e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(e));                                  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

static inline float frand_uniform(std::mt19937 &gen) {
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(gen);
}

// ---------------- warp helpers ----------------
__device__ __forceinline__ float warp_allreduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// ---------------- kernel ----------------
// Layouts (all contiguous):
// Q: [B, M_TOTAL, KDIM] half
// K: [B, SEQ_LEN, KDIM] half
// V: [B, SEQ_LEN, DV]   half
// O: [B, M_TOTAL, DV]   float
//
// Each block: (b=blockIdx.x, row_block=blockIdx.y), ROWS_PER_BLOCK rows.
// blockDim.x=64 = 2 warps.
//
// Shared: smemK [N_TILE*KDIM] half, smemV [N_TILE*DV] half, stats [m,l,scale_old] float
template<int KDIM, int DV, int ROWS_PER_BLOCK, int N_TILE>
__global__ void flashattn_2warp_nosmemW_online_softmax(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int m_total,
    float scale
) {
    static_assert(N_TILE == 32, "This kernel assumes N_TILE == 32.");

    const int tid  = threadIdx.x;   // 0..63
    const int warp = tid >> 5;      // 0 or 1
    const int lane = tid & 31;      // 0..31

    const int b = (int)blockIdx.x;
    const int row_block = (int)blockIdx.y;
    const int row_base = row_block * ROWS_PER_BLOCK;

    // shared
    extern __shared__ unsigned char smem_raw[];
    half*  smemK = reinterpret_cast<half*>(smem_raw);                       // N_TILE*KDIM
    half*  smemV = smemK + (N_TILE * KDIM);                                 // N_TILE*DV
    float* smem_m = reinterpret_cast<float*>(smemV + (N_TILE * DV));        // ROWS
    float* smem_l = smem_m + ROWS_PER_BLOCK;                                 // ROWS
    float* smem_scale_old = smem_l + ROWS_PER_BLOCK;                        // ROWS

    // pointers
    const half* Qb = Q + (size_t)b * m_total * KDIM;
    const half* Kb = K + (size_t)b * seq_len * KDIM;
    const half* Vb = V + (size_t)b * seq_len * DV;
    float*      Ob = O + (size_t)b * m_total * DV;

    // warp0 stats
    float m_row[ROWS_PER_BLOCK];
    float l_row[ROWS_PER_BLOCK];
    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            m_row[r] = -INFINITY;
            l_row[r] = 0.0f;
        }
    }

    // warp1 accumulators (lane maps dv + rowpair)
    const int dv = lane % DV;        // 0..15
    const int rowpair = lane / DV;   // 0 or 1 (DV=16)

    float o_num_a = 0.0f; // row = row_base + rowpair
    float o_num_b = 0.0f; // row = row_base + rowpair + 2

    for (int t0 = 0; t0 < seq_len; t0 += N_TILE) {

        // global -> shared (K tile)
        for (int idx = tid; idx < N_TILE * KDIM; idx += blockDim.x) {
            int t = idx / KDIM;
            int d = idx - t * KDIM;
            int g_t = t0 + t;
            half val = __float2half(0.0f);
            if (g_t < seq_len) val = Kb[g_t * KDIM + d];
            smemK[t * KDIM + d] = val;
        }

        // global -> shared (V tile)
        for (int idx = tid; idx < N_TILE * DV; idx += blockDim.x) {
            int t = idx / DV;
            int d = idx - t * DV;
            int g_t = t0 + t;
            half val = __float2half(0.0f);
            if (g_t < seq_len) val = Vb[g_t * DV + d];
            smemV[t * DV + d] = val;
        }

        __syncthreads();

        // warp0: compute online (m,l) for each row
        if (warp == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
                const int row = row_base + r;

                float score = -INFINITY;
                if (row < m_total) {
                    const half* qptr = Qb + row * KDIM;
                    const half* kptr = smemK + lane * KDIM; // token-lane
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < KDIM; ++i) {
                        acc += __half2float(qptr[i]) * __half2float(kptr[i]);
                    }
                    score = acc * scale;
                }

                float tile_m = warp_allreduce_max(score);

                float e = 0.0f;
                const int g_t = t0 + lane;
                if (g_t < seq_len && row < m_total) e = __expf(score - tile_m);
                float tile_l = warp_allreduce_sum(e);

                float m_old = m_row[r];
                float l_old = l_row[r];
                float m_new = fmaxf(m_old, tile_m);

                float scale_old = (l_old == 0.0f) ? 0.0f : __expf(m_old - m_new);
                float scale_tile = (tile_l == 0.0f) ? 0.0f : __expf(tile_m - m_new);

                float l_new = l_old * scale_old + tile_l * scale_tile;

                if (lane == 0) {
                    smem_m[r] = m_new;
                    smem_l[r] = l_new;
                    smem_scale_old[r] = scale_old;
                }

                m_row[r] = m_new;
                l_row[r] = l_new;
            }
        }

        __syncthreads();

        // warp1: rescale old accumulators + recompute scores -> accumulate O_num
        if (warp == 1) {
            int ra = rowpair;      // 0 or 1
            int rb = rowpair + 2;  // 2 or 3
            int rowA = row_base + ra;
            int rowB = row_base + rb;

            float scaleA = (ra < ROWS_PER_BLOCK) ? smem_scale_old[ra] : 0.0f;
            float scaleB = (rb < ROWS_PER_BLOCK) ? smem_scale_old[rb] : 0.0f;

            o_num_a *= scaleA;
            o_num_b *= scaleB;

            #pragma unroll
            for (int t = 0; t < N_TILE; ++t) {
                int g_t = t0 + t;
                if (g_t >= seq_len) break;

                float v = __half2float(smemV[t * DV + dv]);

                if (rowA < m_total) {
                    const half* qptr = Qb + rowA * KDIM;
                    const half* kptr = smemK + t * KDIM;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < KDIM; ++i)
                        acc += __half2float(qptr[i]) * __half2float(kptr[i]);
                    float w = __expf(acc * scale - smem_m[ra]);
                    o_num_a += w * v;
                }

                if (rowB < m_total) {
                    const half* qptr = Qb + rowB * KDIM;
                    const half* kptr = smemK + t * KDIM;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < KDIM; ++i)
                        acc += __half2float(qptr[i]) * __half2float(kptr[i]);
                    float w = __expf(acc * scale - smem_m[rb]);
                    o_num_b += w * v;
                }
            }
        }

        __syncthreads();
    }

    // final normalize + store
    if (warp == 1) {
        int ra = rowpair;
        int rb = rowpair + 2;
        int rowA = row_base + ra;
        int rowB = row_base + rb;

        if (rowA < m_total) Ob[rowA * DV + dv] = o_num_a / smem_l[ra];
        if (rowB < m_total) Ob[rowB * DV + dv] = o_num_b / smem_l[rb];
    }
}

// ---------------- CPU reference ----------------
static void cpu_reference(
    const std::vector<half>& hQ,
    const std::vector<half>& hK,
    const std::vector<half>& hV,
    std::vector<float>& hOref,
    int B, int M_TOTAL, int SEQ_LEN, int KDIM, int DV,
    float scale
) {
    auto idxQ = [&](int b, int m, int k){ return ((size_t)b*M_TOTAL + m)*KDIM + k; };
    auto idxK = [&](int b, int t, int k){ return ((size_t)b*SEQ_LEN + t)*KDIM + k; };
    auto idxV = [&](int b, int t, int d){ return ((size_t)b*SEQ_LEN + t)*DV + d; };
    auto idxO = [&](int b, int m, int d){ return ((size_t)b*M_TOTAL + m)*DV + d; };

    std::vector<float> scores(SEQ_LEN);

    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M_TOTAL; ++m) {
            float mmax = -INFINITY;
            for (int t = 0; t < SEQ_LEN; ++t) {
                float acc = 0.0f;
                for (int k = 0; k < KDIM; ++k) {
                    float q = __half2float(hQ[idxQ(b,m,k)]);
                    float kk = __half2float(hK[idxK(b,t,k)]);
                    acc += q * kk;
                }
                float s = acc * scale;
                scores[t] = s;
                mmax = std::max(mmax, s);
            }
            float denom = 0.0f;
            for (int t = 0; t < SEQ_LEN; ++t) denom += std::exp(scores[t] - mmax);

            for (int d = 0; d < DV; ++d) {
                float out = 0.0f;
                for (int t = 0; t < SEQ_LEN; ++t) {
                    float w = std::exp(scores[t] - mmax) / denom;
                    float vv = __half2float(hV[idxV(b,t,d)]);
                    out += w * vv;
                }
                hOref[idxO(b,m,d)] = out;
            }
        }
    }
}

static double rel_l2_error(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = (double)a[i] - (double)b[i];
        num += diff * diff;
        den += (double)b[i] * (double)b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

int main() {
    const int NUM_BATCH = 1024;
    const int SEQ_LEN   = 128;
    const int M_TOTAL   = 16;
    const int KDIM      = 16;
    const int DV        = 16;
    const int ROWS_PER_BLOCK = 4;
    const int N_TILE    = 32;

    const float scale = 1.0f / std::sqrt((float)KDIM);

    printf("Streaming Softmax FlashAttention-like (v12i: A안, no smemW, recompute score)\n");

    std::vector<half>  hQ((size_t)NUM_BATCH * M_TOTAL * KDIM);
    std::vector<half>  hK((size_t)NUM_BATCH * SEQ_LEN * KDIM);
    std::vector<half>  hV((size_t)NUM_BATCH * SEQ_LEN * DV);
    std::vector<float> hO((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);
    std::vector<float> hOref((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);

    std::mt19937 gen(123);
    for (auto &x : hQ) x = __float2half(frand_uniform(gen));
    for (auto &x : hK) x = __float2half(frand_uniform(gen));
    for (auto &x : hV) x = __float2half(frand_uniform(gen));

    half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
    float *dO=nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dO, 0, hO.size()*sizeof(float)));

    dim3 grid(NUM_BATCH, (M_TOTAL + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    dim3 block(64, 1, 1);

    // shared bytes: K + V + (m,l,scale_old)
    size_t shmem_bytes =
        (size_t)N_TILE * KDIM * sizeof(half) +
        (size_t)N_TILE * DV   * sizeof(half) +
        (size_t)3 * ROWS_PER_BLOCK * sizeof(float);

    // warmup
    for (int i = 0; i < 10; ++i) {
        flashattn_2warp_nosmemW_online_softmax<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
            <<<grid, block, shmem_bytes>>>(dQ, dK, dV, dO, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    const int iters = 200;
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        flashattn_2warp_nosmemW_online_softmax<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
            <<<grid, block, shmem_bytes>>>(dQ, dK, dV, dO, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size()*sizeof(float), cudaMemcpyDeviceToHost));

    cpu_reference(hQ, hK, hV, hOref, NUM_BATCH, M_TOTAL, SEQ_LEN, KDIM, DV, scale);

    printf("Batch 0, O[0, 0..7] (GPU): ");
    for (int i = 0; i < 8; ++i) printf("% .6f ", hO[(size_t)0*M_TOTAL*DV + 0*DV + i]);
    printf("\n");

    printf("Batch 0, O_ref[0, 0..7] (CPU): ");
    for (int i = 0; i < 8; ++i) printf("% .6f ", hOref[(size_t)0*M_TOTAL*DV + 0*DV + i]);
    printf("\n");

    double err = rel_l2_error(hO, hOref);
    printf("Relative L2 error: %.12e\n", err);
    printf("NUM_BATCH=%d, SEQ_LEN=%d, M_TOTAL=%d, KDIM=%d, DV=%d, N_TILE=%d\n",
           NUM_BATCH, SEQ_LEN, M_TOTAL, KDIM, DV, N_TILE);
    printf("Avg kernel time: %.6f ms (per launch)\n", avg_ms);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    return 0;
}

/*
추천 ncu:
ncu --metrics "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active.pct" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12i.exe
*/
