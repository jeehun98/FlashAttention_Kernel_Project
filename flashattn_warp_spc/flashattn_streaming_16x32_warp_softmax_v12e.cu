// flashattn_streaming_16x32_warp_softmax_v12e.cu
// v12e: K_T token-major + shared transpose to remove bank conflict

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

constexpr int WARP_SIZE = 32;
constexpr int Kdim = 16;
constexpr int Dv   = 16;
constexpr int N_TILE = 32;
constexpr int ROWS_PER_BLOCK = 4;

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float warp_allreduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__global__ void flashattn_warp_full_softmax_v12e(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_T,
    const __half* __restrict__ V_T,
    float* __restrict__ O,
    int seq_len,
    int stride_q,
    float scale
) {
    __shared__ __half sK[Kdim][N_TILE];
    __shared__ __half sV[Dv][N_TILE];

    int batch = blockIdx.x;
    int row_block = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (warp >= ROWS_PER_BLOCK) return;

    int row = row_block * ROWS_PER_BLOCK + warp;
    if (row >= seq_len) return;

    // ---- load Q (lane0 only, then shfl)
    float q[Kdim];
    if (lane == 0) {
        const __half* q_ptr = Q + batch * stride_q + row * Kdim;
        #pragma unroll
        for (int k = 0; k < Kdim; ++k)
            q[k] = __half2float(q_ptr[k]);
    }
    #pragma unroll
    for (int k = 0; k < Kdim; ++k)
        q[k] = __shfl_sync(0xffffffff, q[k], 0);

    float m = -1e20f;
    float l = 0.f;
    float out[Dv] = {0.f};

    for (int col0 = 0; col0 < seq_len; col0 += N_TILE) {
        int col = col0 + lane;

        // ---- load K_T / V_T into shared (token-major -> shared transpose)
        if (lane < N_TILE && col < seq_len) {
            const __half* k_ptr = K_T + batch * seq_len * Kdim + col * Kdim;
            const __half* v_ptr = V_T + batch * seq_len * Dv   + col * Dv;

            #pragma unroll
            for (int k = 0; k < Kdim; ++k)
                sK[k][lane] = k_ptr[k];

            #pragma unroll
            for (int d = 0; d < Dv; ++d)
                sV[d][lane] = v_ptr[d];
        }
        __syncthreads();

        // ---- compute attention
        float score = -1e20f;
        if (lane < N_TILE && col < seq_len) {
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < Kdim; ++k)
                acc += q[k] * __half2float(sK[k][lane]);
            score = acc * scale;
        }

        float tile_max = warp_allreduce_max(score);
        float exp_score = (lane < N_TILE && col < seq_len)
                            ? expf(score - tile_max)
                            : 0.f;
        float tile_sum = warp_allreduce_sum(exp_score);

        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);
        float beta  = expf(tile_max - m_new);

        #pragma unroll
        for (int d = 0; d < Dv; ++d) {
            float v = (lane < N_TILE && col < seq_len)
                        ? __half2float(sV[d][lane])
                        : 0.f;
            float contrib = warp_allreduce_sum(exp_score * v);
            out[d] = out[d] * alpha + beta * contrib;
        }

        l = l * alpha + beta * tile_sum;
        m = m_new;

        __syncthreads();
    }

    // ---- write output (lane0)
    if (lane == 0) {
        float inv_l = 1.f / l;
        float* o_ptr = O + batch * seq_len * Dv + row * Dv;
        #pragma unroll
        for (int d = 0; d < Dv; ++d)
            o_ptr[d] = out[d] * inv_l;
    }
}

// ====================== host ======================

void cpu_reference(
    const std::vector<__half>& Q,
    const std::vector<__half>& K,
    const std::vector<__half>& V,
    std::vector<float>& O,
    int B, int L
) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < L; ++i) {
            float m = -1e20f;
            std::vector<float> scores(L);
            for (int j = 0; j < L; ++j) {
                float acc = 0.f;
                for (int k = 0; k < Kdim; ++k)
                    acc += __half2float(Q[b*L*Kdim + i*Kdim + k]) *
                           __half2float(K[b*L*Kdim + j*Kdim + k]);
                scores[j] = acc;
                m = fmaxf(m, acc);
            }
            float sum = 0.f;
            for (int j = 0; j < L; ++j) {
                scores[j] = expf(scores[j] - m);
                sum += scores[j];
            }
            for (int d = 0; d < Dv; ++d) {
                float acc = 0.f;
                for (int j = 0; j < L; ++j)
                    acc += scores[j] * __half2float(V[b*L*Dv + j*Dv + d]);
                O[b*L*Dv + i*Dv + d] = acc / sum;
            }
        }
    }
}

int main() {
    int B = 1024;
    int L = 128;
    float scale = 1.f / sqrtf((float)Kdim);

    std::vector<__half> Q(B*L*Kdim), K(B*L*Kdim), V(B*L*Dv);
    std::vector<float> O(B*L*Dv), Oref(B*L*Dv);

    for (auto& x : Q) x = __float2half(((rand()%100)/100.f - 0.5f));
    for (auto& x : K) x = __float2half(((rand()%100)/100.f - 0.5f));
    for (auto& x : V) x = __float2half(((rand()%100)/100.f - 0.5f));

    cpu_reference(Q, K, V, Oref, B, L);

    // build K_T, V_T
    std::vector<__half> K_T(B*L*Kdim), V_T(B*L*Dv);
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < Kdim; ++k)
                K_T[b*L*Kdim + j*Kdim + k] = K[b*L*Kdim + j*Kdim + k];
            for (int d = 0; d < Dv; ++d)
                V_T[b*L*Dv + j*Dv + d] = V[b*L*Dv + j*Dv + d];
        }

    __half *dQ, *dK, *dV;
    float* dO;
    CHECK_CUDA(cudaMalloc(&dQ, Q.size()*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, K_T.size()*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, V_T.size()*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, O.size()*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K_T.data(), K_T.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V_T.data(), V_T.size()*sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(ROWS_PER_BLOCK * WARP_SIZE);
    dim3 grid(B, (L + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    flashattn_warp_full_softmax_v12e<<<grid, block>>>(
        dQ, dK, dV, dO, L, L*Kdim, scale
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(O.data(), dO, O.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float err = 0.f, ref = 0.f;
    for (size_t i = 0; i < O.size(); ++i) {
        float diff = O[i] - Oref[i];
        err += diff * diff;
        ref += Oref[i] * Oref[i];
    }
    printf("Relative L2 error: %.3e\n", sqrt(err/ref));

    printf("v12e done\n");
    return 0;
}
