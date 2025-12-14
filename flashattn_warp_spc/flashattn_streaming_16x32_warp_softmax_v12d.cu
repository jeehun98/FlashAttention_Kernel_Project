#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#define CHECK_CUDA(cmd) do {                          \
  cudaError_t e = (cmd);                              \
  if (e != cudaSuccess) {                             \
    printf("CUDA error %s:%d: %s\n",                  \
      __FILE__, __LINE__, cudaGetErrorString(e));     \
    std::exit(EXIT_FAILURE);                          \
  }                                                   \
} while(0)

constexpr int WARP = 32;
constexpr int Kdim = 16;
constexpr int Dv   = 16;
constexpr int N_TILE = 32;                 // warp-full tile
constexpr int ROWS_PER_BLOCK = 4;          // 4 compute warps
constexpr int WARPS_PER_BLOCK = 1 + ROWS_PER_BLOCK; // warp0 load + 4 compute
constexpr float NEG_LARGE = -1e30f;
constexpr float EPS = 1e-6f;

__device__ __forceinline__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  8));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  4));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  2));
    v = fmaxf(v, __shfl_xor_sync(mask, v,  1));
    return v;
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v,  8);
    v += __shfl_xor_sync(mask, v,  4);
    v += __shfl_xor_sync(mask, v,  2);
    v += __shfl_xor_sync(mask, v,  1);
    return v;
}

// Q:   [B, M, Kdim]
// K_T: [B, L, Kdim]   (token-major)
// V_T: [B, L, Dv]     (token-major)
// O:   [B, M, Dv] float
__global__ void flashattn_warp_full_softmax_v12d_fixed2(
    const half* __restrict__ Q,
    const half* __restrict__ K_T,
    const half* __restrict__ V_T,
    float* __restrict__ O,
    int M, int L, float scale
) {
    __shared__ half sK[N_TILE][Kdim];
    __shared__ half sV[N_TILE][Dv];

    int tid  = threadIdx.x;
    int warp = tid >> 5;     // 0..4
    int lane = tid & 31;     // 0..31
    unsigned mask = 0xffffffffu;

    int b = blockIdx.x;
    int row_base = blockIdx.y * ROWS_PER_BLOCK;

    // base pointers
    const half* Qb   = Q   + (size_t)b * M * Kdim;
    const half* KTb  = K_T + (size_t)b * L * Kdim;
    const half* VTb  = V_T + (size_t)b * L * Dv;
    float*      Ob   = O   + (size_t)b * M * Dv;

    // compute warp maps to one row
    int row = row_base + (warp - 1); // warp1->row_base+0 ... warp4->row_base+3
    bool is_compute = (warp >= 1) && (warp <= ROWS_PER_BLOCK) && (row < M);

    // Q row: lane0~15만 로드, 이후 shfl로 모든 lane이 q[k]를 읽게 함
    half q_lane = __float2half(0.0f);
    if (is_compute && lane < Kdim) {
        q_lane = Qb[row * Kdim + lane];
    }

    // running (online softmax state): lane0만 유지 (레지스터 폭발 방지 + 명확)
    float m_running = NEG_LARGE;
    float l_running = 0.0f;
    float y_running[Dv];
    if (is_compute && lane == 0) {
#pragma unroll
        for (int d = 0; d < Dv; ++d) y_running[d] = 0.0f;
    }

    // iterate tiles of 32 tokens
    for (int t0 = 0; t0 < L; t0 += N_TILE) {

        // warp0 loads tile (global->shared)
        if (warp == 0) {
            int token = t0 + lane;
            if (token < L) {
                const half* ksrc = KTb + (size_t)token * Kdim;
                const half* vsrc = VTb + (size_t)token * Dv;
#pragma unroll
                for (int k = 0; k < Kdim; ++k) sK[lane][k] = ksrc[k];
#pragma unroll
                for (int d = 0; d < Dv; ++d)   sV[lane][d] = vsrc[d];
            } else {
#pragma unroll
                for (int k = 0; k < Kdim; ++k) sK[lane][k] = __float2half(0.0f);
#pragma unroll
                for (int d = 0; d < Dv; ++d)   sV[lane][d] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if (is_compute) {
            int token = t0 + lane;
            bool valid = (token < L);

            // each lane computes one score for its token in tile
            float dot = 0.0f;
            if (valid) {
#pragma unroll
                for (int k = 0; k < Kdim; ++k) {
                    // q[k]는 lane k (0..15)에서 가져와서 warp 전체에 broadcast
                    half qk_h = __shfl_sync(mask, q_lane, k);
                    float qk  = __half2float(qk_h);
                    float kk  = __half2float(sK[lane][k]);
                    dot += qk * kk;
                }
                dot *= scale;
            } else {
                dot = NEG_LARGE;
            }

            float m_t = warp_allreduce_max(dot);

            float e = valid ? __expf(dot - m_t) : 0.0f;
            float l_t = warp_allreduce_sum(e);

            float u[Dv];
#pragma unroll
            for (int d = 0; d < Dv; ++d) {
                float v = valid ? __half2float(sV[lane][d]) : 0.0f;
                u[d] = warp_allreduce_sum(e * v);
            }

            // online update는 lane0만 수행
            if (lane == 0) {
                if (m_running == NEG_LARGE) {
                    m_running = m_t;
                    l_running = l_t;
#pragma unroll
                    for (int d = 0; d < Dv; ++d) y_running[d] = u[d];
                } else {
                    float m_new = fmaxf(m_running, m_t);
                    float alpha = __expf(m_running - m_new);
                    float beta  = __expf(m_t       - m_new);
                    float l_new = l_running * alpha + l_t * beta;

#pragma unroll
                    for (int d = 0; d < Dv; ++d) {
                        y_running[d] = y_running[d] * alpha + u[d] * beta;
                    }
                    m_running = m_new;
                    l_running = l_new;
                }
            }
        }

        __syncthreads();
    }

    // write output (lane0 only)
    if (is_compute && lane == 0) {
        float inv = 1.0f / (l_running + EPS);
#pragma unroll
        for (int d = 0; d < Dv; ++d) {
            Ob[row * Dv + d] = y_running[d] * inv;
        }
    }
}

// ---------------- CPU reference (full softmax) ----------------
static void cpu_ref(
    const std::vector<half>& Q,
    const std::vector<half>& K_T,
    const std::vector<half>& V_T,
    std::vector<float>& O,
    int B, int M, int L, float scale
) {
    auto h2f = [] (half x) { return __half2float(x); };

    for (int b = 0; b < B; ++b) {
        const half* Qb  = Q.data()   + (size_t)b * M * Kdim;
        const half* KTb = K_T.data() + (size_t)b * L * Kdim;
        const half* VTb = V_T.data() + (size_t)b * L * Dv;
        float* Ob       = O.data()   + (size_t)b * M * Dv;

        for (int i = 0; i < M; ++i) {
            float maxv = NEG_LARGE;
            std::vector<float> s(L);
            for (int t = 0; t < L; ++t) {
                float dot = 0.f;
                for (int k = 0; k < Kdim; ++k) {
                    dot += h2f(Qb[i*Kdim+k]) * h2f(KTb[t*Kdim+k]);
                }
                dot *= scale;
                s[t] = dot;
                maxv = fmaxf(maxv, dot);
            }
            float sum = 0.f;
            for (int t = 0; t < L; ++t) sum += std::exp(s[t] - maxv);
            float inv = 1.f / (sum + EPS);

            for (int d = 0; d < Dv; ++d) {
                float acc = 0.f;
                for (int t = 0; t < L; ++t) {
                    float p = std::exp(s[t] - maxv) * inv;
                    acc += p * h2f(VTb[t*Dv + d]);
                }
                Ob[i*Dv + d] = acc;
            }
        }
    }
}

int main() {
    std::printf("Streaming Softmax FlashAttention-like (v12d_fixed2: warp-full + online softmax, Q shfl broadcast)\n");

    constexpr int B = 1024;
    constexpr int M = 16;
    constexpr int L = 128;

    float scale = 1.0f / std::sqrt((float)Kdim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);

    std::vector<half>  hQ((size_t)B * M * Kdim);
    std::vector<half>  hK_T((size_t)B * L * Kdim);
    std::vector<half>  hV_T((size_t)B * L * Dv);
    std::vector<float> hO_ref((size_t)B * M * Dv);
    std::vector<float> hO((size_t)B * M * Dv);

    for (auto& x : hQ)   x = __float2half(dist(rng));
    for (auto& x : hK_T) x = __float2half(dist(rng));
    for (auto& x : hV_T) x = __float2half(dist(rng));

    cpu_ref(hQ, hK_T, hV_T, hO_ref, B, M, L, scale);

    half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
    float* dO=nullptr;
    CHECK_CUDA(cudaMalloc(&dQ, hQ.size()*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dK, hK_T.size()*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dV, hV_T.size()*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dO, hO.size()*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK_T.data(), hK_T.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV_T.data(), hV_T.size()*sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(WARPS_PER_BLOCK * WARP, 1, 1);  // 160 threads
    dim3 grid(B, (M + ROWS_PER_BLOCK - 1)/ROWS_PER_BLOCK, 1);

    flashattn_warp_full_softmax_v12d_fixed2<<<grid, block>>>(dQ, dK, dV, dO, M, L, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size()*sizeof(float), cudaMemcpyDeviceToHost));

    double num=0.0, den=0.0;
    for (size_t i=0;i<hO.size();++i){
        double diff = (double)hO[i] - (double)hO_ref[i];
        num += diff*diff;
        den += (double)hO_ref[i]*(double)hO_ref[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    std::printf("Batch 0, O[0, 0..7] (GPU): ");
    for (int j=0;j<8;++j) std::printf("%f ", hO[0*M*Dv + 0*Dv + j]);
    std::printf("\nBatch 0, O_ref[0, 0..7] (CPU): ");
    for (int j=0;j<8;++j) std::printf("%f ", hO_ref[0*M*Dv + 0*Dv + j]);
    std::printf("\nRelative L2 error: %.12e\n", rel_l2);

    // perf
    const int ITERS = 50;
    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));

    CHECK_CUDA(cudaEventRecord(st));
    for (int it=0; it<ITERS; ++it) {
        flashattn_warp_full_softmax_v12d_fixed2<<<grid, block>>>(dQ, dK, dV, dO, M, L, scale);
    }
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));

    float ms=0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
    ms /= ITERS;

    // FLOPs rough
    double flops_per_batch =
        2.0 * M * L * Kdim +   // QK
        6.0 * M * L * Dv;      // softmax+PV rough
    double total_flops = flops_per_batch * B;
    double tflops = (total_flops / (ms*1e-3)) / 1e12;

    std::printf("Avg kernel time: %f ms\n", ms);
    std::printf("Approx TFLOPS  : %f\n", tflops);

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    CHECK_CUDA(cudaEventDestroy(st));
    CHECK_CUDA(cudaEventDestroy(ed));
    return 0;
}
