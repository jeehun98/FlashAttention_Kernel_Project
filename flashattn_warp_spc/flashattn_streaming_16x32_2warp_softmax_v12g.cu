// flashattn_streaming_16x32_2warp_softmax_v12g_fix.cu
// FIX: When l_w/acc_w are computed using m_tile, DO NOT rescale again by exp(m_w - m_t).
//      m_t = m_tile, l_t = l_w0 + l_w1, acc_t = acc0 + acc1.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(cmd) do {                                              \
  cudaError_t e = (cmd);                                                  \
  if (e != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
            cudaGetErrorString(e));                                       \
    std::exit(EXIT_FAILURE);                                              \
  }                                                                       \
} while (0)

static inline float frand(std::mt19937 &g) {
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(g);
}

__device__ __forceinline__ float warp_reduce_max_16(float v, unsigned mask16) {
  v = fmaxf(v, __shfl_down_sync(mask16, v, 8));
  v = fmaxf(v, __shfl_down_sync(mask16, v, 4));
  v = fmaxf(v, __shfl_down_sync(mask16, v, 2));
  v = fmaxf(v, __shfl_down_sync(mask16, v, 1));
  return v;
}

__device__ __forceinline__ float warp_reduce_sum_16(float v, unsigned mask16) {
  v += __shfl_down_sync(mask16, v, 8);
  v += __shfl_down_sync(mask16, v, 4);
  v += __shfl_down_sync(mask16, v, 2);
  v += __shfl_down_sync(mask16, v, 1);
  return v;
}

__device__ __forceinline__ float2 warp_reduce_sum2_16(float2 v, unsigned mask16) {
  float2 t;
  t.x = __shfl_down_sync(mask16, v.x, 8);
  t.y = __shfl_down_sync(mask16, v.y, 8);
  v.x += t.x; v.y += t.y;
  t.x = __shfl_down_sync(mask16, v.x, 4);
  t.y = __shfl_down_sync(mask16, v.y, 4);
  v.x += t.x; v.y += t.y;
  t.x = __shfl_down_sync(mask16, v.x, 2);
  t.y = __shfl_down_sync(mask16, v.y, 2);
  v.x += t.x; v.y += t.y;
  t.x = __shfl_down_sync(mask16, v.x, 1);
  t.y = __shfl_down_sync(mask16, v.y, 1);
  v.x += t.x; v.y += t.y;
  return v;
}

// Kernel config
constexpr int KDIM = 16;
constexpr int DV   = 16;
constexpr int N_TILE = 32;
constexpr int ROWS_PER_BLOCK = 4;

struct __align__(16) Shared {
  float q_sh[KDIM];
  float m_w[2];
  float l_w[2];
  float acc_w[2][DV];
  float m_tile;
  float pad[3];
};

__global__ void flashattn_2warp_split_softmax_v12g_fix(
    const __half* __restrict__ Q,     // [B, M, KDIM] row-major
    const __half* __restrict__ K_T,   // [B, KDIM, S] token-major
    const __half* __restrict__ V_T,   // [B, DV,   S] token-major
    float* __restrict__ O,            // [B, M, DV] float
    int S, int M_total,
    float sm_scale)
{
  __shared__ Shared sh;

  const int tid  = threadIdx.x;      // 0..63
  const int warp = tid >> 5;         // 0 or 1
  const int lane = tid & 31;         // 0..31
  const unsigned mask16 = 0x0000FFFFu;

  const int b = blockIdx.x;
  const int row_block = blockIdx.y;
  const int row0 = row_block * ROWS_PER_BLOCK;

  #pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int row = row0 + r;
    if (row >= M_total) break;

    if (tid < KDIM) {
      int q_idx = (b * M_total + row) * KDIM + tid;
      sh.q_sh[tid] = __half2float(Q[q_idx]);
    }
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.f;
    float out_acc[DV];
    if (warp == 0 && lane == 0) {
      #pragma unroll
      for (int d = 0; d < DV; ++d) out_acc[d] = 0.f;
    }

    for (int t0 = 0; t0 < S; t0 += N_TILE) {
      const int token = t0 + (warp * 16) + lane;

      float score = -INFINITY;
      bool active = (lane < 16) && (token < S);
      if (active) {
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < KDIM; ++k) {
          const int k_idx = (b * KDIM + k) * S + token;
          float kv = __half2float(K_T[k_idx]);
          acc = fmaf(sh.q_sh[k], kv, acc);
        }
        score = acc * sm_scale;
      }

      float m_w = warp_reduce_max_16(score, mask16);
      if (lane == 0) sh.m_w[warp] = m_w;
      __syncthreads();

      if (tid == 0) {
        sh.m_tile = fmaxf(sh.m_w[0], sh.m_w[1]);
      }
      __syncthreads();
      const float m_tile = sh.m_tile;

      float p = 0.f;
      if (active) p = __expf(score - m_tile);

      float l_w_sum = warp_reduce_sum_16(p, mask16);
      if (lane == 0) sh.l_w[warp] = l_w_sum;

      float2 acc2[DV/2];
      #pragma unroll
      for (int i = 0; i < DV/2; ++i) acc2[i] = make_float2(0.f, 0.f);

      if (active) {
        #pragma unroll
        for (int i = 0; i < DV/2; ++i) {
          const int d0 = 2*i + 0;
          const int d1 = 2*i + 1;
          float v0 = __half2float(V_T[(b * DV + d0) * S + token]);
          float v1 = __half2float(V_T[(b * DV + d1) * S + token]);
          acc2[i].x = fmaf(p, v0, acc2[i].x);
          acc2[i].y = fmaf(p, v1, acc2[i].y);
        }
      }

      #pragma unroll
      for (int i = 0; i < DV/2; ++i) {
        acc2[i] = warp_reduce_sum2_16(acc2[i], mask16);
      }

      if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < DV/2; ++i) {
          sh.acc_w[warp][2*i+0] = acc2[i].x;
          sh.acc_w[warp][2*i+1] = acc2[i].y;
        }
      }
      __syncthreads();

      // FIXED combine (no extra rescale)
      if (warp == 0 && lane == 0) {
        const float m_t = m_tile;
        const float l_t = sh.l_w[0] + sh.l_w[1];

        float acc_t[DV];
        #pragma unroll
        for (int d = 0; d < DV; ++d) {
          acc_t[d] = sh.acc_w[0][d] + sh.acc_w[1][d];
        }

        // Online softmax update
        float m_new = fmaxf(m_i, m_t);
        float alpha = (l_i == 0.f) ? 0.f : __expf(m_i - m_new);
        float beta  = __expf(m_t - m_new);
        float l_new = l_i * alpha + l_t * beta;

        #pragma unroll
        for (int d = 0; d < DV; ++d) {
          out_acc[d] = out_acc[d] * alpha + acc_t[d] * beta;
        }

        m_i = m_new;
        l_i = l_new;
      }
      __syncthreads();
    }

    if (warp == 0 && lane == 0) {
      float inv = 1.f / (l_i + 1e-20f);
      int out_base = (b * M_total + row) * DV;
      #pragma unroll
      for (int d = 0; d < DV; ++d) {
        O[out_base + d] = out_acc[d] * inv;
      }
    }
    __syncthreads();
  }
}

static void cpu_ref_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int B, int M, int S,
    float sm_scale)
{
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      float maxv = -INFINITY;
      std::vector<float> s(S);
      for (int t = 0; t < S; ++t) {
        float acc = 0.f;
        for (int k = 0; k < KDIM; ++k) {
          float qv = Q[(b*M + m)*KDIM + k];
          float kv = K[(b*S + t)*KDIM + k];
          acc += qv * kv;
        }
        float sc = acc * sm_scale;
        s[t] = sc;
        maxv = std::max(maxv, sc);
      }
      float denom = 0.f;
      std::vector<float> p(S);
      for (int t = 0; t < S; ++t) {
        float e = std::exp(s[t] - maxv);
        p[t] = e;
        denom += e;
      }
      float inv = 1.f / (denom + 1e-20f);
      for (int d = 0; d < DV; ++d) {
        float out = 0.f;
        for (int t = 0; t < S; ++t) {
          out += (p[t] * inv) * V[(b*S + t)*DV + d];
        }
        O[(b*M + m)*DV + d] = out;
      }
    }
  }
}

static void to_half(const std::vector<float>& in, std::vector<__half>& out) {
  out.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) out[i] = __float2half(in[i]);
}

static void build_transposed(
    const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& K_T, std::vector<float>& V_T,
    int B, int S)
{
  K_T.assign(B * KDIM * S, 0.f);
  V_T.assign(B * DV   * S, 0.f);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < S; ++t) {
      for (int k = 0; k < KDIM; ++k) {
        K_T[(b*KDIM + k)*S + t] = K[(b*S + t)*KDIM + k];
      }
      for (int d = 0; d < DV; ++d) {
        V_T[(b*DV + d)*S + t] = V[(b*S + t)*DV + d];
      }
    }
  }
}

static double rel_l2_error(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = double(a[i]) - double(b[i]);
    num += diff * diff;
    den += double(b[i]) * double(b[i]);
  }
  return std::sqrt(num / (den + 1e-30));
}

int main() {
  const int B = 1024;
  const int S = 128;
  const int M = 16;
  const float sm_scale = 1.0f / std::sqrt(float(KDIM));

  printf("Streaming Softmax FlashAttention-like (v12g_fix: 2-warp split, correct combine)\n");

  std::mt19937 rng(0);

  std::vector<float> hQ(B * M * KDIM);
  std::vector<float> hK(B * S * KDIM);
  std::vector<float> hV(B * S * DV);
  for (auto& x : hQ) x = frand(rng);
  for (auto& x : hK) x = frand(rng);
  for (auto& x : hV) x = frand(rng);

  std::vector<float> hK_T, hV_T;
  build_transposed(hK, hV, hK_T, hV_T, B, S);

  // 기존: hQh, hKTh, hVTh만 만들었음
std::vector<__half> hQh, hKTh, hVTh;
to_half(hQ,   hQh);
to_half(hK_T, hKTh);
to_half(hV_T, hVTh);

// ✅ 추가: 원본 K/V도 half로 내려서 CPU ref가 같은 입력을 보게 만들기
std::vector<__half> hKh, hVh;
to_half(hK, hKh);   // K: [B,S,KDIM] row-major
to_half(hV, hVh);   // V: [B,S,DV]   row-major

// ✅ 추가: half->float로 복원한 입력으로 CPU ref 수행
std::vector<float> hQ_ref(hQ.size()), hK_ref(hK.size()), hV_ref(hV.size());
for (size_t i = 0; i < hQ_ref.size(); ++i) hQ_ref[i] = __half2float(hQh[i]);
for (size_t i = 0; i < hK_ref.size(); ++i) hK_ref[i] = __half2float(hKh[i]);
for (size_t i = 0; i < hV_ref.size(); ++i) hV_ref[i] = __half2float(hVh[i]);



  __half *dQ = nullptr, *dK_T = nullptr, *dV_T = nullptr;
  float *dO = nullptr;
  CHECK_CUDA(cudaMalloc(&dQ,   hQh.size()  * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dK_T, hKTh.size() * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dV_T, hVTh.size() * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dO,   (size_t)B * M * DV * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dQ,   hQh.data(),  hQh.size()  * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK_T, hKTh.data(), hKTh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV_T, hVTh.data(), hVTh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dO, 0, (size_t)B * M * DV * sizeof(float)));

  dim3 block(64, 1, 1);
  dim3 grid(B, (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

  for (int i = 0; i < 20; ++i) {
    flashattn_2warp_split_softmax_v12g_fix<<<grid, block>>>(dQ, dK_T, dV_T, dO, S, M, sm_scale);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t st, ed;
  CHECK_CUDA(cudaEventCreate(&st));
  CHECK_CUDA(cudaEventCreate(&ed));
  CHECK_CUDA(cudaEventRecord(st));
  const int iters = 200;
  for (int i = 0; i < iters; ++i) {
    flashattn_2warp_split_softmax_v12g_fix<<<grid, block>>>(dQ, dK_T, dV_T, dO, S, M, sm_scale);
  }
  CHECK_CUDA(cudaEventRecord(ed));
  CHECK_CUDA(cudaEventSynchronize(ed));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
  ms /= iters;

  std::vector<float> hO(B * M * DV);
  CHECK_CUDA(cudaMemcpy(hO.data(), dO, (size_t)B * M * DV * sizeof(float), cudaMemcpyDeviceToHost));

  
  std::vector<float> hOref(B * M * DV);
  cpu_ref_attention(hQ_ref, hK_ref, hV_ref, hOref, B, M, S, sm_scale);

  printf("Batch 0, O[0, 0..7] (GPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hO[(0*M + 0)*DV + i]);
  printf("\n");
  printf("Batch 0, O_ref[0, 0..7] (CPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hOref[(0*M + 0)*DV + i]);
  printf("\n");

  double err = rel_l2_error(hO, hOref);
  printf("Relative L2 error: %.12e\n", err);
  printf("NUM_BATCH=%d, SEQ_LEN=%d, M_TOTAL=%d, KDIM=%d, DV=%d, N_TILE=%d\n", B, S, M, KDIM, DV, N_TILE);
  printf("Avg kernel time: %.6f ms (per launch)\n", ms);

  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK_T));
  CHECK_CUDA(cudaFree(dV_T));
  CHECK_CUDA(cudaFree(dO));
  CHECK_CUDA(cudaEventDestroy(st));
  CHECK_CUDA(cudaEventDestroy(ed));
  return 0;
}
