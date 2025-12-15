// flashattn_streaming_16x32_2warp_softmax_v12iA.cu
// A안(v12iA): no smemW(float), no score recompute
// - Fix shared bank conflicts by K transpose+pad in smem (half2) : smemK2[(KDIM/2)][N_TILE+1]
// - Store exp(score - tile_m) as half in smemE (ROWS_PER_BLOCK x N_TILE)
// - warp0: compute score + online softmax stats (m,l) + store e_half + scale_old/scale_tile
// - warp1: rescale old O_num with scale_old + accumulate O_num with (e * scale_tile) * V + final normalize + coalesced store
//
// Build:
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 flashattn_streaming_16x32_2warp_softmax_v12iA.cu -o flashattn_streaming_16x32_2warp_softmax_v12iA.exe
//
// Run:
//   .\flashattn_streaming_16x32_2warp_softmax_v12iA.exe

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
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------- kernel ----------------
// Layouts (all contiguous):
// Q: [B, M_TOTAL, KDIM] half
// K: [B, SEQ_LEN, KDIM] half
// V: [B, SEQ_LEN, DV]   half
// O: [B, M_TOTAL, DV]   float

template<int KDIM, int DV, int ROWS_PER_BLOCK, int N_TILE>
__global__ void flashattn_2warp_iA_kpad_ehalf(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int m_total,
    float scale
) {
    static_assert(N_TILE == 32, "Assume N_TILE==32");
    static_assert(KDIM % 2 == 0, "Assume KDIM even for half2");

    const int tid  = threadIdx.x;     // 0..63
    const int warp = tid >> 5;        // 0 or 1
    const int lane = tid & 31;        // 0..31

    const int b = (int)blockIdx.x;
    const int row_block = (int)blockIdx.y;
    const int row_base = row_block * ROWS_PER_BLOCK;

    // ---- Shared memory layout ----
    // K2: [KDIM/2][N_TILE + 1] half2   (transpose + pad to break bank pattern)
    // V : [N_TILE][DV] half
    // E : [ROWS_PER_BLOCK][N_TILE] half   (e = exp(score - tile_m))
    // stats: m,l,scale_old,scale_tile : float[ROWS_PER_BLOCK] each

    extern __shared__ unsigned char smem_raw[];
    unsigned char* p = smem_raw;

    half2* smemK2 = reinterpret_cast<half2*>(p);
    p += (size_t)(KDIM/2) * (N_TILE + 1) * sizeof(half2);

    half* smemV = reinterpret_cast<half*>(p);
    p += (size_t)N_TILE * DV * sizeof(half);

    half* smemE = reinterpret_cast<half*>(p);
    p += (size_t)ROWS_PER_BLOCK * N_TILE * sizeof(half);

    float* smem_m = reinterpret_cast<float*>(p);
    p += (size_t)ROWS_PER_BLOCK * sizeof(float);

    float* smem_l = reinterpret_cast<float*>(p);
    p += (size_t)ROWS_PER_BLOCK * sizeof(float);

    float* smem_scale_old = reinterpret_cast<float*>(p);
    p += (size_t)ROWS_PER_BLOCK * sizeof(float);

    float* smem_scale_tile = reinterpret_cast<float*>(p);
    // p += ROWS_PER_BLOCK * sizeof(float);

    // Pointers
    const half* Qb = Q + (size_t)b * m_total * KDIM;
    const half* Kb = K + (size_t)b * seq_len * KDIM;
    const half* Vb = V + (size_t)b * seq_len * DV;
    float*      Ob = O + (size_t)b * m_total * DV;

    // warp0 online stats (owner)
    float m_row[ROWS_PER_BLOCK];
    float l_row[ROWS_PER_BLOCK];
    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            m_row[r] = -INFINITY;
            l_row[r] = 0.0f;
        }
    }

    // warp1 accumulators: lane maps dv + rowpair
    float o_num_a = 0.0f;
    float o_num_b = 0.0f;
    const int dv = lane % DV;        // 0..15
    const int rowpair = lane / DV;   // 0 or 1 (DV=16)
    // rowA = rowpair, rowB = rowpair+2 when ROWS_PER_BLOCK=4

    for (int t0 = 0; t0 < seq_len; t0 += N_TILE) {
        // ---- Load K tile into smemK2 (transpose+pad) ----
        // total elements: (KDIM/2) * N_TILE half2
        for (int idx = tid; idx < (KDIM/2) * N_TILE; idx += blockDim.x) {
            int i = idx / N_TILE;      // 0..(KDIM/2-1)
            int t = idx - i * N_TILE;  // 0..31
            int g_t = t0 + t;

            half2 val = __float2half2_rn(0.0f);
            if (g_t < seq_len) {
                const half2* k2 = reinterpret_cast<const half2*>(Kb + (size_t)g_t * KDIM);
                val = k2[i];
            }
            // store transpose: [i][t] with pad (N_TILE+1)
            smemK2[i * (N_TILE + 1) + t] = val;
        }

        // ---- Load V tile into shared ----
        for (int idx = tid; idx < N_TILE * DV; idx += blockDim.x) {
            int t = idx / DV;
            int d = idx - t * DV;
            int g_t = t0 + t;

            half val = __float2half(0.0f);
            if (g_t < seq_len) val = Vb[(size_t)g_t * DV + d];
            smemV[t * DV + d] = val;
        }

        __syncthreads();

        // ---- warp0: score + online softmax + store e_half + stats ----
        if (warp == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
                const int row = row_base + r;

                float score = -INFINITY;
                if (row < m_total) {
                    const half2* q2 = reinterpret_cast<const half2*>(Qb + (size_t)row * KDIM);

                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < KDIM/2; ++i) {
                        half2 k = smemK2[i * (N_TILE + 1) + lane]; // padded stride breaks bank pattern
                        float2 qf = __half22float2(q2[i]);
                        float2 kf = __half22float2(k);
                        acc += qf.x * kf.x + qf.y * kf.y;
                    }
                    score = acc * scale;
                }

                float tile_m = warp_allreduce_max(score);

                float e = 0.0f;
                const int g_t = t0 + lane;
                if (g_t < seq_len && row < m_total) {
                    e = __expf(score - tile_m);
                }
                float tile_l = warp_allreduce_sum(e);

                // online update
                float m_old = m_row[r];
                float l_old = l_row[r];
                float m_new = fmaxf(m_old, tile_m);

                float scale_old = (l_old == 0.0f) ? 0.0f : __expf(m_old - m_new);
                float scale_tile = (tile_l == 0.0f) ? 0.0f : __expf(tile_m - m_new);
                float l_new = l_old * scale_old + tile_l * scale_tile;

                // store e_half for this row/token (exp(score - tile_m))
                smemE[r * N_TILE + lane] = __float2half_rn(e);

                // publish stats once per row
                if (lane == 0) {
                    smem_m[r] = m_new;
                    smem_l[r] = l_new;
                    smem_scale_old[r] = scale_old;
                    smem_scale_tile[r] = scale_tile;
                }

                // update owner regs
                m_row[r] = m_new;
                l_row[r] = l_new;
            }
        }

        __syncthreads();

        // ---- warp1: rescale old accum + accumulate with (e * scale_tile) ----
        if (warp == 1) {
            int ra = rowpair;       // 0 or 1
            int rb = rowpair + 2;   // 2 or 3
            int rowA = row_base + ra;
            int rowB = row_base + rb;

            float sOldA = (ra < ROWS_PER_BLOCK) ? smem_scale_old[ra] : 0.0f;
            float sOldB = (rb < ROWS_PER_BLOCK) ? smem_scale_old[rb] : 0.0f;
            float sTileA = (ra < ROWS_PER_BLOCK) ? smem_scale_tile[ra] : 0.0f;
            float sTileB = (rb < ROWS_PER_BLOCK) ? smem_scale_tile[rb] : 0.0f;

            o_num_a *= sOldA;
            o_num_b *= sOldB;

            #pragma unroll
            for (int t = 0; t < N_TILE; ++t) {
                int g_t = t0 + t;
                if (g_t >= seq_len) break;

                float v = __half2float(smemV[t * DV + dv]);

                // w = exp(score - tile_m) * exp(tile_m - m_new) = e * scale_tile
                float wA = 0.0f;
                float wB = 0.0f;
                if (rowA < m_total) wA = __half2float(smemE[ra * N_TILE + t]) * sTileA;
                if (rowB < m_total) wB = __half2float(smemE[rb * N_TILE + t]) * sTileB;

                o_num_a += wA * v;
                o_num_b += wB * v;
            }
        }

        __syncthreads();
    }

    // ---- final normalize + store ----
    if (warp == 1) {
        int ra = rowpair;
        int rb = rowpair + 2;
        int rowA = row_base + ra;
        int rowB = row_base + rb;

        float lA = (rowA < m_total) ? smem_l[ra] : 1.0f;
        float lB = (rowB < m_total) ? smem_l[rb] : 1.0f;

        float outA = (rowA < m_total) ? (o_num_a / lA) : 0.0f;
        float outB = (rowB < m_total) ? (o_num_b / lB) : 0.0f;

        if (rowA < m_total) Ob[(size_t)rowA * DV + dv] = outA;
        if (rowB < m_total) Ob[(size_t)rowB * DV + dv] = outB;
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

    printf("Streaming Softmax FlashAttention-like (v12iA: Kpad+Ehalf, no smemW, no recompute)\n");

    // Host buffers
    std::vector<half>  hQ((size_t)NUM_BATCH * M_TOTAL * KDIM);
    std::vector<half>  hK((size_t)NUM_BATCH * SEQ_LEN * KDIM);
    std::vector<half>  hV((size_t)NUM_BATCH * SEQ_LEN * DV);
    std::vector<float> hO((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);
    std::vector<float> hOref((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);

    std::mt19937 gen(123);
    for (auto &x : hQ) x = __float2half(frand_uniform(gen));
    for (auto &x : hK) x = __float2half(frand_uniform(gen));
    for (auto &x : hV) x = __float2half(frand_uniform(gen));

    // Device buffers
    half* dQ = nullptr;
    half* dK = nullptr;
    half* dV = nullptr;
    float* dO = nullptr;

    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dO, 0, hO.size() * sizeof(float)));

    dim3 grid(NUM_BATCH, (M_TOTAL + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    dim3 block(64, 1, 1);

    // Shared memory bytes:
    // K2: (KDIM/2)*(N_TILE+1)*half2
    // V : N_TILE*DV*half
    // E : ROWS*N_TILE*half
    // stats: 4 arrays of ROWS floats
    size_t shmem_bytes =
        (size_t)(KDIM/2) * (N_TILE + 1) * sizeof(half2) +
        (size_t)N_TILE * DV * sizeof(half) +
        (size_t)ROWS_PER_BLOCK * N_TILE * sizeof(half) +
        (size_t)4 * ROWS_PER_BLOCK * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        flashattn_2warp_iA_kpad_ehalf<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
            <<<grid, block, shmem_bytes>>>(dQ, dK, dV, dO, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    const int iters = 200;
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        flashattn_2warp_iA_kpad_ehalf<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
            <<<grid, block, shmem_bytes>>>(dQ, dK, dV, dO, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

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
추천 NCU 체크 (A안 효과 확인용)

1) bank conflict가 smemK에서 내려갔는지:
ncu --metrics "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12iA.exe

2) long scoreboard(L1TEX) 비중:
ncu --metrics "smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12iA.exe

3) global load/store 요청 대비 섹터(코얼레싱 간접 확인):
ncu --metrics "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12iA.exe
*/
