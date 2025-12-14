// flashattn_streaming_16x32_2warp_softmax_v12h.cu
// 2-warp fixed-role FlashAttention-like streaming softmax
// - warp0: scores + online softmax stats (m,l) + weights (unnormalized exp)
// - warp1: O_num accumulation + final normalize + coalesced store
//
// Build (Windows, example):
//   nvcc -O3 -lineinfo -arch=sm_86 flashattn_streaming_16x32_2warp_softmax_v12h.cu -o flashattn_streaming_16x32_2warp_softmax_v12h.exe
//
// Run:
//   .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

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
    // full mask
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
//
// Each block handles:
//   b = blockIdx.x
//   row_block = blockIdx.y  (ROWS_PER_BLOCK rows)
//   rows = row_block*ROWS_PER_BLOCK + r (r=0..ROWS_PER_BLOCK-1)
// Each lane corresponds to a token in the tile (N_TILE=32).

template<int KDIM, int DV, int ROWS_PER_BLOCK, int N_TILE>
__global__ void flashattn_2warp_fixedrole_online_softmax(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int m_total,
    float scale  // usually 1/sqrt(KDIM)
) {
    static_assert(N_TILE == 32, "This kernel assumes N_TILE == 32 (one warp per tile tokens).");
    const int tid  = threadIdx.x;     // 0..63
    const int warp = tid >> 5;        // 0 or 1
    const int lane = tid & 31;        // 0..31

    const int b = (int)blockIdx.x;
    const int row_block = (int)blockIdx.y;
    const int row_base = row_block * ROWS_PER_BLOCK;

    // shared: Ktile [N_TILE][KDIM], Vtile [N_TILE][DV], weights [ROWS_PER_BLOCK][N_TILE], stats [ROWS_PER_BLOCK]
    extern __shared__ unsigned char smem_raw[];
    half*  smemK = reinterpret_cast<half*>(smem_raw);                                         // N_TILE*KDIM
    half*  smemV = smemK + (N_TILE * KDIM);                                                    // N_TILE*DV
    float* smemW = reinterpret_cast<float*>(smemV + (N_TILE * DV));                            // ROWS_PER_BLOCK*N_TILE
    float* smem_m = smemW + (ROWS_PER_BLOCK * N_TILE);                                         // ROWS_PER_BLOCK
    float* smem_l = smem_m + ROWS_PER_BLOCK;                                                   // ROWS_PER_BLOCK
    float* smem_scale_old = smem_l + ROWS_PER_BLOCK;                                           // ROWS_PER_BLOCK

    // Pointers
    const half* Qb = Q + (size_t)b * m_total * KDIM;
    const half* Kb = K + (size_t)b * seq_len * KDIM;
    const half* Vb = V + (size_t)b * seq_len * DV;
    float*      Ob = O + (size_t)b * m_total * DV;

    // warp0 keeps online stats in registers (only warp0 uses them as owner)
    float m_row[ROWS_PER_BLOCK];
    float l_row[ROWS_PER_BLOCK];

    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            m_row[r] = -INFINITY;
            l_row[r] = 0.0f;
        }
    }

    // warp1 keeps O_num accumulators for two rows per lane (no divergence)
    // lane maps dv = lane % DV (DV=16), rowpair = lane / DV (0 or 1)
    // This covers rows (rowpair) and (rowpair+2) for ROWS_PER_BLOCK=4.
    float o_num_a = 0.0f; // row = rowpair
    float o_num_b = 0.0f; // row = rowpair + 2

    const int dv = lane % DV;
    const int rowpair = lane / DV; // 0 or 1 when DV=16

    // Streaming over tokens in tiles of N_TILE
    for (int t0 = 0; t0 < seq_len; t0 += N_TILE) {
        // Load K/V tile into shared (all 64 threads cooperate, coalesced)
        // Each thread loads multiple elements in a strided way.
        // K tile: N_TILE*KDIM halfs
        for (int idx = tid; idx < N_TILE * KDIM; idx += blockDim.x) {
            int t = idx / KDIM;           // 0..31
            int d = idx - t * KDIM;       // 0..KDIM-1
            int g_t = t0 + t;
            half val = __float2half(0.0f);
            if (g_t < seq_len) val = Kb[g_t * KDIM + d];
            smemK[t * KDIM + d] = val;
        }
        // V tile: N_TILE*DV halfs
        for (int idx = tid; idx < N_TILE * DV; idx += blockDim.x) {
            int t = idx / DV;            // 0..31
            int d = idx - t * DV;        // 0..DV-1
            int g_t = t0 + t;
            half val = __float2half(0.0f);
            if (g_t < seq_len) val = Vb[g_t * DV + d];
            smemV[t * DV + d] = val;
        }

        __syncthreads();

        // ---------------- warp0: scores + online softmax + weights ----------------
        if (warp == 0) {
            // For each row r, each lane computes score for token (t0+lane)
            #pragma unroll
            for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
                const int row = row_base + r;
                float score = -INFINITY;

                if (row < m_total) {
                    // dot(Q[row], K[t]) in float
                    // KDIM is small (16), unroll, use half2 for speed
                    const half* qptr = Qb + row * KDIM;
                    const half* kptr = smemK + lane * KDIM; // token-local K in shared

                    float acc = 0.0f;
                    // KDIM=16 -> 8 half2
                    const half2* q2 = reinterpret_cast<const half2*>(qptr);
                    const half2* k2 = reinterpret_cast<const half2*>(kptr);
                    #pragma unroll
                    for (int i = 0; i < KDIM/2; ++i) {
                        half2 a = q2[i];
                        half2 b2 = k2[i];
                        float2 af = __half22float2(a);
                        float2 bf = __half22float2(b2);
                        acc += af.x * bf.x + af.y * bf.y;
                    }
                    score = acc * scale;
                }

                // tile max over 32 tokens
                float tile_m = warp_allreduce_max(score);

                // compute exp(score - tile_m) (0 for invalid tokens)
                float e = 0.0f;
                const int g_t = t0 + lane;
                if (g_t < seq_len && row < m_total) {
                    e = __expf(score - tile_m);
                }
                float tile_l = warp_allreduce_sum(e);

                // online update: m_new, l_new
                float m_old = m_row[r];
                float l_old = l_row[r];

                float m_new = fmaxf(m_old, tile_m);
                float scale_old = (l_old == 0.0f) ? 0.0f : __expf(m_old - m_new);
                float scale_tile = (tile_l == 0.0f) ? 0.0f : __expf(tile_m - m_new);
                float l_new = l_old * scale_old + tile_l * scale_tile;

                // store scale_old + m_new + l_new to shared so warp1 can rescale/normalize
                if (lane == 0) {
                    smem_m[r] = m_new;
                    smem_l[r] = l_new;
                    smem_scale_old[r] = scale_old;
                }

                // compute unnormalized weights exp(score - m_new) for this token and store
                float w = 0.0f;
                if (g_t < seq_len && row < m_total) {
                    // exp(score - m_new) = exp(score - tile_m) * exp(tile_m - m_new)
                    w = e * scale_tile;
                }
                smemW[r * N_TILE + lane] = w;

                // update owner registers
                m_row[r] = m_new;
                l_row[r] = l_new;
            }
        }

        __syncthreads();

        // ---------------- warp1: O_num update (no divergence) ----------------
        if (warp == 1) {
            // rows for this lane
            int ra = rowpair;       // 0 or 1
            int rb = rowpair + 2;   // 2 or 3
            int rowA = row_base + ra;
            int rowB = row_base + rb;

            // rescale old accumulators to new m (scale_old)
            float scaleA = (ra < ROWS_PER_BLOCK) ? smem_scale_old[ra] : 0.0f;
            float scaleB = (rb < ROWS_PER_BLOCK) ? smem_scale_old[rb] : 0.0f;

            // if l_old was zero, scale_old is 0 -> accumulator becomes 0 (correct)
            o_num_a *= scaleA;
            o_num_b *= scaleB;

            // accumulate over tokens in this tile
            #pragma unroll
            for (int t = 0; t < N_TILE; ++t) {
                int g_t = t0 + t;
                if (g_t >= seq_len) break;

                // load V[t][dv] from shared (coalesced across lanes: dv is contiguous in lane groups)
                float v = __half2float(smemV[t * DV + dv]);

                float wA = (rowA < m_total) ? smemW[ra * N_TILE + t] : 0.0f;
                float wB = (rowB < m_total) ? smemW[rb * N_TILE + t] : 0.0f;

                o_num_a += wA * v;
                o_num_b += wB * v;
            }
        }

        __syncthreads();
    }

    // ---------------- final normalize + store ----------------
    if (warp == 1) {
        int ra = rowpair;       // 0 or 1
        int rb = rowpair + 2;   // 2 or 3
        int rowA = row_base + ra;
        int rowB = row_base + rb;

        float lA = (rowA < m_total) ? smem_l[ra] : 1.0f;
        float lB = (rowB < m_total) ? smem_l[rb] : 1.0f;

        float outA = (rowA < m_total) ? (o_num_a / lA) : 0.0f;
        float outB = (rowB < m_total) ? (o_num_b / lB) : 0.0f;

        // Coalesced store over DV: for each row, lanes with same row write contiguous dv
        if (rowA < m_total) {
            Ob[rowA * DV + dv] = outA;
        }
        if (rowB < m_total) {
            Ob[rowB * DV + dv] = outB;
        }
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
            // scores
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
            // softmax denom
            float denom = 0.0f;
            for (int t = 0; t < SEQ_LEN; ++t) denom += std::exp(scores[t] - mmax);

            // output
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
    // Match your typical test config
    const int NUM_BATCH = 1024;
    const int SEQ_LEN   = 128;
    const int M_TOTAL   = 16;
    const int KDIM      = 16;
    const int DV        = 16;
    const int ROWS_PER_BLOCK = 4;
    const int N_TILE    = 32;

    const float scale = 1.0f / std::sqrt((float)KDIM);

    printf("Streaming Softmax FlashAttention-like (v12h: 2-warp fixed-role online softmax, N_TILE=32)\n");

    // Host buffers
    std::vector<half>  hQ((size_t)NUM_BATCH * M_TOTAL * KDIM);
    std::vector<half>  hK((size_t)NUM_BATCH * SEQ_LEN * KDIM);
    std::vector<half>  hV((size_t)NUM_BATCH * SEQ_LEN * DV);
    std::vector<float> hO((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);
    std::vector<float> hOref((size_t)NUM_BATCH * M_TOTAL * DV, 0.0f);

    // Init random
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

    // Launch config
    dim3 grid(NUM_BATCH, (M_TOTAL + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    dim3 block(64, 1, 1);

    // Shared memory bytes:
    // K: N_TILE*KDIM half
    // V: N_TILE*DV half
    // W: ROWS_PER_BLOCK*N_TILE float
    // m,l,scale_old: 3*ROWS_PER_BLOCK float
    size_t shmem_bytes =
        (size_t)N_TILE * KDIM * sizeof(half) +
        (size_t)N_TILE * DV   * sizeof(half) +
        (size_t)ROWS_PER_BLOCK * N_TILE * sizeof(float) +
        (size_t)3 * ROWS_PER_BLOCK * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        flashattn_2warp_fixedrole_online_softmax<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
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
        flashattn_2warp_fixedrole_online_softmax<KDIM, DV, ROWS_PER_BLOCK, N_TILE>
            <<<grid, block, shmem_bytes>>>(dQ, dK, dV, dO, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    float avg_ms = ms / iters;

    // Copy back
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (slow but exact)
    cpu_reference(hQ, hK, hV, hOref, NUM_BATCH, M_TOTAL, SEQ_LEN, KDIM, DV, scale);

    // Print sample
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

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));

    return 0;
}
/*

ncu --metrics "smsp__average_warp_latency_issue_stalled_short_scoreboard,smsp__average_warp_latency_issue_stalled_long_scoreboard,smsp__average_warp_latency_issue_stalled_long_scoreboard_pipe_l1tex"   --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active"   --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active"   --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active"   --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --set full --section SourceCounters --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --set full --section "L2 Theoretical Sectors Global Excessive" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --set full --section "Warp Stall Sampling (All Samples)" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe
ncu --set full --section "L2 Theoretical Sectors Global Excessive" --section "Warp Stall Sampling (All Samples)" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --list-sections | findstr /i "stall warp sampling scoreboard l2 sectors excessive theoretical"
ncu --set pmsampling --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --set pmsampling --section PmSampling_WarpStates --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --set full --section MemoryWorkloadAnalysis_Tables --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe


ncu --metrics "smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --metrics "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum" --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

ncu --set full --section SourceCounters --page details --print-details all     --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_2warp_softmax_v12h.exe

*/