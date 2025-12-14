// flashattn_streaming_16x32_warp_softmax_v12f.cu
// v12f: Correct online softmax (FlashAttention formula) + shared transpose (half2) + padding
// - One CTA handles: (batch b, row_block rb) where ROWS_PER_BLOCK=4, each warp handles one query row
// - Warp4 loads K/V tiles (N_TILE=32 tokens) into shared as half2 (token-major) with padding
// - Warps 0..3 do warp-full softmax over 32 tokens (one lane per token), and online-softmax across tiles

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(cmd)                                                       \
    do {                                                                      \
        cudaError_t e = (cmd);                                                \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(e));                                   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

static inline float frand_uniform(std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return __shfl_sync(mask, v, 0);
}

__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return __shfl_sync(mask, v, 0);
}

// reduce for float2 by reducing x and y separately
__inline__ __device__ float2 warp_allreduce_sum2(float2 v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(mask, v.x, offset);
        v.y += __shfl_down_sync(mask, v.y, offset);
    }
    v.x = __shfl_sync(mask, v.x, 0);
    v.y = __shfl_sync(mask, v.y, 0);
    return v;
}

struct alignas(16) ShmemPack {
    // half2 layout: [token][d2]
    // Add padding in d2 stride to reduce bank conflicts when many warps read same token index pattern.
    // Kdim=16 => d2=8. We pad to 10.
    half2 K2[32][10];
    half2 V2[32][10];
};

constexpr int KDIM = 16;
constexpr int DV   = 16;
constexpr int KDIM2 = KDIM / 2; // 8
constexpr int DV2   = DV / 2;   // 8

constexpr int N_TILE = 32;
constexpr int ROWS_PER_BLOCK = 4;

// Q: [B, M_TOTAL, KDIM] half
// K: [B, SEQ_LEN, KDIM] half
// V: [B, SEQ_LEN, DV]   half
// O: [B, M_TOTAL, DV]   float
__global__ void flashattn_warp_full_softmax_v12f(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int B, int SEQ_LEN, int M_TOTAL, float scale)
{
    // block config: 160 threads = 5 warps
    int tid = threadIdx.x;
    int warp_id = tid >> 5;   // 0..4
    int lane    = tid & 31;

    int b = (int)blockIdx.x;
    int rb = (int)blockIdx.y; // row-block id
    int row_base = rb * ROWS_PER_BLOCK;

    if (b >= B) return;
    if (row_base >= M_TOTAL) return;

    extern __shared__ unsigned char smem_raw[];
    ShmemPack* smem = reinterpret_cast<ShmemPack*>(smem_raw);

    // pointers
    const __half* Qb = Q + (size_t)b * M_TOTAL * KDIM;
    const __half* Kb = K + (size_t)b * SEQ_LEN * KDIM;
    const __half* Vb = V + (size_t)b * SEQ_LEN * DV;
    float* Ob = O + (size_t)b * M_TOTAL * DV;

    // each warp 0..3 handles one row
    int row = row_base + warp_id;
    bool active_row_warp = (warp_id < ROWS_PER_BLOCK) && (row < M_TOTAL);

    // Load query row into registers (broadcast within warp via shfl)
    // We'll load by lane 0..7 as half2, then broadcast each half2 to all lanes.
    half2 q2[KDIM2]; // 8 half2 = 16 half
    if (active_row_warp) {
        // only lanes < KDIM2 load, then broadcast
        half2 local = __float2half2_rn(0.f);
        if (lane < KDIM2) {
            const half2* qptr2 = reinterpret_cast<const half2*>(Qb + (size_t)row * KDIM);
            local = qptr2[lane];
        }
        // broadcast each q2[i] from lane i
        #pragma unroll
        for (int i = 0; i < KDIM2; i++) {
            unsigned mask = 0xffffffffu;
            // treat half2 as 32-bit
            int v = __shfl_sync(mask, (int)reinterpret_cast<int&>(local), i);
            q2[i] = reinterpret_cast<half2&>(v);
        }
    }

    // online softmax accumulators per row warp
    float m = -INFINITY;
    float l = 0.0f;
    float o[DV];
    #pragma unroll
    for (int d = 0; d < DV; d++) o[d] = 0.0f;

    // iterate tiles along sequence
    for (int t0 = 0; t0 < SEQ_LEN; t0 += N_TILE) {
        // loader warp loads K/V tile into shared (token-major half2)
        if (warp_id == ROWS_PER_BLOCK) { // warp 4
            int token = t0 + lane; // one token per lane
            if (token < SEQ_LEN) {
                const half2* kptr2 = reinterpret_cast<const half2*>(Kb + (size_t)token * KDIM);
                const half2* vptr2 = reinterpret_cast<const half2*>(Vb + (size_t)token * DV);

                #pragma unroll
                for (int i = 0; i < KDIM2; i++) {
                    smem->K2[lane][i] = kptr2[i];
                }
                #pragma unroll
                for (int i = KDIM2; i < 10; i++) smem->K2[lane][i] = __float2half2_rn(0.f);

                #pragma unroll
                for (int i = 0; i < DV2; i++) {
                    smem->V2[lane][i] = vptr2[i];
                }
                #pragma unroll
                for (int i = DV2; i < 10; i++) smem->V2[lane][i] = __float2half2_rn(0.f);
            } else {
                // out-of-range token: fill zeros
                #pragma unroll
                for (int i = 0; i < 10; i++) {
                    smem->K2[lane][i] = __float2half2_rn(0.f);
                    smem->V2[lane][i] = __float2half2_rn(0.f);
                }
            }
        }

        __syncthreads();

        if (active_row_warp) {
            // Each lane corresponds to one token inside tile: token = t0 + lane
            // If token >= SEQ_LEN, we mask score to -inf so it doesn't contribute.
            int token = t0 + lane;
            float score = -INFINITY;

            if (token < SEQ_LEN) {
                // dot(q, k_token)
                float acc = 0.f;
                #pragma unroll
                for (int i = 0; i < KDIM2; i++) {
                    half2 k2 = smem->K2[lane][i];
                    half2 qq = q2[i];
                    float2 kf = __half22float2(k2);
                    float2 qf = __half22float2(qq);
                    acc += qf.x * kf.x + qf.y * kf.y;
                }
                score = acc * scale;
            }

            // tile max
            float tile_max = warp_allreduce_max(score);
            float m_new = fmaxf(m, tile_max);
            float alpha = expf(m - m_new);         // rescale old
            float p = (score == -INFINITY) ? 0.f : expf(score - m_new); // new weights
            float tile_l = warp_allreduce_sum(p);
            float l_new = l * alpha + tile_l;

            // rescale old o by alpha, then add sum(p * v)
            #pragma unroll
            for (int d2 = 0; d2 < DV2; d2++) {
                half2 v2 = smem->V2[lane][d2];
                float2 vf = __half22float2(v2);
                float2 contrib;
                contrib.x = p * vf.x;
                contrib.y = p * vf.y;
                contrib = warp_allreduce_sum2(contrib);

                // lane0 holds reduced contrib; broadcast already in function
                float sumx = contrib.x;
                float sumy = contrib.y;

                int d = d2 * 2;
                o[d + 0] = o[d + 0] * alpha + sumx;
                o[d + 1] = o[d + 1] * alpha + sumy;
            }

            m = m_new;
            l = l_new;
        }

        __syncthreads();
    }

    // write output: o / l
    if (active_row_warp) {
        float inv_l = 1.0f / l;
        // only lane 0 writes (DV=16 floats)
        if (lane == 0) {
            float* out = Ob + (size_t)row * DV;
            #pragma unroll
            for (int d = 0; d < DV; d++) out[d] = o[d] * inv_l;
        }
    }
}

// CPU reference: full softmax over SEQ_LEN for each (b,row)
static void cpu_reference(
    const std::vector<__half>& hQ,
    const std::vector<__half>& hK,
    const std::vector<__half>& hV,
    std::vector<float>& hO,
    int B, int SEQ_LEN, int M_TOTAL, float scale)
{
    for (int b = 0; b < B; b++) {
        for (int r = 0; r < M_TOTAL; r++) {
            // scores
            std::vector<float> s(SEQ_LEN);
            float m = -1e30f;
            for (int t = 0; t < SEQ_LEN; t++) {
                float acc = 0.f;
                for (int d = 0; d < KDIM; d++) {
                    float q = __half2float(hQ[(size_t)b*M_TOTAL*KDIM + (size_t)r*KDIM + d]);
                    float k = __half2float(hK[(size_t)b*SEQ_LEN*KDIM + (size_t)t*KDIM + d]);
                    acc += q * k;
                }
                float sc = acc * scale;
                s[t] = sc;
                m = std::max(m, sc);
            }
            // softmax
            float l = 0.f;
            std::vector<float> p(SEQ_LEN);
            for (int t = 0; t < SEQ_LEN; t++) {
                float e = std::exp(s[t] - m);
                p[t] = e;
                l += e;
            }
            float inv_l = 1.f / l;
            // output
            for (int d = 0; d < DV; d++) {
                float out = 0.f;
                for (int t = 0; t < SEQ_LEN; t++) {
                    float v = __half2float(hV[(size_t)b*SEQ_LEN*DV + (size_t)t*DV + d]);
                    out += (p[t] * inv_l) * v;
                }
                hO[(size_t)b*M_TOTAL*DV + (size_t)r*DV + d] = out;
            }
        }
    }
}

static double rel_l2_error(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double da = (double)a[i];
        double db = (double)b[i];
        double diff = da - db;
        num += diff * diff;
        den += db * db;
    }
    return std::sqrt(num / (den + 1e-30));
}

int main() {
    // match your common setup
    const int B = 1024;
    const int SEQ_LEN = 128;
    const int M_TOTAL = 16;
    const float scale = 1.0f / std::sqrt((float)KDIM);

    printf("Streaming Softmax FlashAttention-like (v12f: correct online softmax + shared half2 transpose + padding)\n");

    std::mt19937 rng(123);

    std::vector<__half> hQ((size_t)B * M_TOTAL * KDIM);
    std::vector<__half> hK((size_t)B * SEQ_LEN * KDIM);
    std::vector<__half> hV((size_t)B * SEQ_LEN * DV);
    std::vector<float>  hO_ref((size_t)B * M_TOTAL * DV, 0.f);
    std::vector<float>  hO_gpu((size_t)B * M_TOTAL * DV, 0.f);

    for (auto &x : hQ) x = __float2half(frand_uniform(rng));
    for (auto &x : hK) x = __float2half(frand_uniform(rng));
    for (auto &x : hV) x = __float2half(frand_uniform(rng));

    cpu_reference(hQ, hK, hV, hO_ref, B, SEQ_LEN, M_TOTAL, scale);

    __half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
    float *dO=nullptr;
    CHECK_CUDA(cudaMalloc(&dQ, hQ.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, hK.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, hO_gpu.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dO, 0, hO_gpu.size()*sizeof(float)));

    dim3 block(160, 1, 1);
    dim3 grid(B, (M_TOTAL + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    size_t shmem = sizeof(ShmemPack);

    // warmup
    for (int i = 0; i < 20; i++) {
        flashattn_warp_full_softmax_v12f<<<grid, block, shmem>>>(dQ, dK, dV, dO, B, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));
    CHECK_CUDA(cudaEventRecord(st));

    const int iters = 200;
    for (int i = 0; i < iters; i++) {
        flashattn_warp_full_softmax_v12f<<<grid, block, shmem>>>(dQ, dK, dV, dO, B, SEQ_LEN, M_TOTAL, scale);
    }
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
    ms /= iters;

    CHECK_CUDA(cudaMemcpy(hO_gpu.data(), dO, hO_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost));

    // print sanity
    printf("Batch 0, O[0, 0..7] (GPU): ");
    for (int i = 0; i < 8; i++) printf("% .6f ", hO_gpu[(size_t)0*M_TOTAL*DV + 0*DV + i]);
    printf("\n");

    printf("Batch 0, O_ref[0, 0..7] (CPU): ");
    for (int i = 0; i < 8; i++) printf("% .6f ", hO_ref[(size_t)0*M_TOTAL*DV + 0*DV + i]);
    printf("\n");

    double err = rel_l2_error(hO_gpu, hO_ref);
    printf("Relative L2 error: %.12e\n", err);
    printf("NUM_BATCH=%d, SEQ_LEN=%d, M_TOTAL=%d, KDIM=%d, DV=%d, N_TILE=%d\n",
           B, SEQ_LEN, M_TOTAL, KDIM, DV, N_TILE);
    printf("Avg kernel time: %.6f ms (per launch)\n", ms);

    // cleanup
    CHECK_CUDA(cudaEventDestroy(st));
    CHECK_CUDA(cudaEventDestroy(ed));
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    return 0;
}
/*
nvcc -O3 -arch=sm_86 flashattn_streaming_16x32_warp_softmax_v12f.cu -o flashattn_streaming_16x32_warp_softmax_v12f.exe
.\flashattn_streaming_16x32_warp_softmax_v12f.exe
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_streaming_16x32_warp_softmax_v12f.exe

*/