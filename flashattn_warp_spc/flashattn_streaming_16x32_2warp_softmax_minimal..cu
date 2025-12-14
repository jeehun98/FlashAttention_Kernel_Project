// flashattn_streaming_16x32_2warp_softmax_minimal.cu
// 2-warp fixed-role online softmax (warp0 load K/V tile, warp1 compute)
// Target-ish: M=16, KDIM=16, DV=16, N_TILE=32, SEQ_LEN=128
// Build (example):
//   nvcc -O3 -lineinfo -arch=sm_86 flashattn_streaming_16x32_2warp_softmax_minimal.cu -o v12h_min.exe

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(x) do {                                  \
  cudaError_t e = (x);                                      \
  if (e != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",               \
            __FILE__, __LINE__, cudaGetErrorString(e));     \
    std::exit(1);                                           \
  }                                                         \
} while(0)

static inline float frand(std::mt19937& g) {
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(g);
}

__device__ __forceinline__ float warp_allreduce_max(float v) {
  for (int d = 16; d > 0; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
  return v;
}
__device__ __forceinline__ float warp_allreduce_sum(float v) {
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  return v;
}

// Layout assumption (simple):
// Q: [B, M, KDIM] half
// K: [B, SEQ, KDIM] half
// V: [B, SEQ, DV]   half
// O: [B, M, DV]     float

template<int M, int KDIM, int DV, int N_TILE>
__global__ void flashattn_2warp_fixedrole_online_softmax(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    float scale)
{
  constexpr int WARPS_PER_BLOCK = 2;
  constexpr int WARP_SIZE = 32;
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  int block = blockIdx.x;       // block in [0, B*M)
  int b = block / M;
  int m = block - b * M;

  // Shared: K_tile [N_TILE, KDIM], V_tile [N_TILE, DV]
  extern __shared__ __half smem[];
  __half* sK = smem;                         // size N_TILE*KDIM
  __half* sV = sK + N_TILE * KDIM;           // size N_TILE*DV

  // Pointers
  const __half* Qbm = Q + ( (b * M + m) * KDIM );
  const __half* Kb  = K + ( b * seq_len * KDIM );
  const __half* Vb  = V + ( b * seq_len * DV );
  float* Obm        = O + ( (b * M + m) * DV );

  // Load Q into regs (warp1 only needs it, but cheap here)
  float qf[KDIM];
  if (warp_id == 1) {
    // lane 0..15 load two halves at a time (half2), then broadcast via shfl
    // Simpler: every lane loads all KDIM (still OK for KDIM=16)
    #pragma unroll
    for (int i = 0; i < KDIM; ++i) qf[i] = __half2float(Qbm[i]);
  }

  // Online softmax state (warp1)
  float m_prev = -INFINITY;
  float l_prev = 0.0f;
  float o_prev[DV];
  if (warp_id == 1 && lane == 0) {
    #pragma unroll
    for (int d = 0; d < DV; ++d) o_prev[d] = 0.0f;
  }

  // Iterate tiles
  for (int t0 = 0; t0 < seq_len; t0 += N_TILE) {
    // ---- warp0: load K/V tile into shared ----
    if (warp_id == 0) {
      int j = lane; // 0..31
      int gj = t0 + j;
      bool valid = (gj < seq_len);

      // Load K[gj, :]
      #pragma unroll
      for (int i = 0; i < KDIM; ++i) {
        __half hv = valid ? Kb[gj * KDIM + i] : __float2half(0.0f);
        sK[j * KDIM + i] = hv;
      }
      // Load V[gj, :]
      #pragma unroll
      for (int d = 0; d < DV; ++d) {
        __half hv = valid ? Vb[gj * DV + d] : __float2half(0.0f);
        sV[j * DV + d] = hv;
      }
    }

    __syncthreads();

    // ---- warp1: compute scores, online softmax update ----
    if (warp_id == 1) {
      int j = lane;
      int gj = t0 + j;
      bool valid = (gj < seq_len);

      // score = dot(Q, K[j]) * scale
      float s = -INFINITY;
      if (valid) {
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < KDIM; ++i) {
          acc += qf[i] * __half2float(sK[j * KDIM + i]);
        }
        s = acc * scale;
      }

      // tile max
      float tile_max = warp_allreduce_max(s);

      // exp(s - tile_max), tile sumexp
      float e = valid ? __expf(s - tile_max) : 0.0f;
      float tile_sum = warp_allreduce_sum(e);

      // rescale factors for online softmax
      float m_new = fmaxf(m_prev, tile_max);
      float a = __expf(m_prev - m_new);        // old scale
      float bscale = __expf(tile_max - m_new); // tile scale
      float l_new = l_prev * a + tile_sum * bscale;

      // tile_o = sum_j e_j * V_j  (reduce across warp)
      // We reduce each DV dimension; lane0 receives final and updates o_prev.
      float tile_o_d;

      if (lane == 0) {
        #pragma unroll
        for (int d = 0; d < DV; ++d) {
          // Start with lane0's contribution then add others via shfl
          float partial = e * __half2float(sV[j * DV + d]); // lane0's j=0
          // gather other lanes
          #pragma unroll
          for (int off = 1; off < 32; ++off) {
            float e_other = __shfl_sync(0xffffffff, e, off);
            float v_other = __half2float(sV[off * DV + d]);
            partial += e_other * v_other;
          }
          tile_o_d = partial;

          // online update: o_prev = o_prev*a + tile_o*bscale
          o_prev[d] = o_prev[d] * a + tile_o_d * bscale;
        }
        m_prev = m_new;
        l_prev = l_new;
      }

      __syncwarp();
    }

    __syncthreads();
  }

  // Write output: O = o_prev / l_prev
  if (warp_id == 1 && lane == 0) {
    float inv_l = 1.0f / (l_prev + 1e-20f);
    #pragma unroll
    for (int d = 0; d < DV; ++d) Obm[d] = o_prev[d] * inv_l;
  }
}

// CPU reference
static void ref_attention(
    const std::vector<__half>& Q,
    const std::vector<__half>& K,
    const std::vector<__half>& V,
    std::vector<float>& O,
    int B, int M, int S, int KDIM, int DV, float scale)
{
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      // scores
      std::vector<float> s(S);
      float mx = -1e30f;
      for (int j = 0; j < S; ++j) {
        float acc = 0.f;
        for (int i = 0; i < KDIM; ++i) {
          float q = __half2float(Q[(b*M + m)*KDIM + i]);
          float k = __half2float(K[(b*S + j)*KDIM + i]);
          acc += q*k;
        }
        float val = acc * scale;
        s[j] = val;
        mx = std::max(mx, val);
      }
      // softmax
      float den = 0.f;
      for (int j = 0; j < S; ++j) den += std::exp(s[j] - mx);
      // output
      for (int d = 0; d < DV; ++d) {
        float out = 0.f;
        for (int j = 0; j < S; ++j) {
          float p = std::exp(s[j] - mx) / den;
          float v = __half2float(V[(b*S + j)*DV + d]);
          out += p * v;
        }
        O[(b*M + m)*DV + d] = out;
      }
    }
  }
}

int main() {
  // Match your typical params
  constexpr int B = 1024;
  constexpr int M = 16;
  constexpr int S = 128;
  constexpr int KDIM = 16;
  constexpr int DV = 16;
  constexpr int N_TILE = 32;

  float scale = 1.0f / std::sqrt((float)KDIM);

  std::mt19937 rng(0);

  std::vector<__half> hQ(B*M*KDIM), hK(B*S*KDIM), hV(B*S*DV);
  for (auto& x : hQ) x = __float2half(frand(rng));
  for (auto& x : hK) x = __float2half(frand(rng));
  for (auto& x : hV) x = __float2half(frand(rng));

  std::vector<float> hO_ref(B*M*DV), hO(B*M*DV);

  // CPU ref
  ref_attention(hQ, hK, hV, hO_ref, B, M, S, KDIM, DV, scale);

  // Device
  __half *dQ, *dK, *dV;
  float *dO;
  CHECK_CUDA(cudaMalloc(&dQ, hQ.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dK, hK.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dV, hV.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dO, hO.size()*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size()*sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size()*sizeof(__half), cudaMemcpyHostToDevice));

  dim3 block(64);
  dim3 grid(B*M);
  size_t smem = (N_TILE*KDIM + N_TILE*DV) * sizeof(__half);

  // Warmup
  for (int i = 0; i < 5; ++i) {
    flashattn_2warp_fixedrole_online_softmax<M,KDIM,DV,N_TILE><<<grid, block, smem>>>(
      dQ, dK, dV, dO, S, scale);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t st, ed;
  CHECK_CUDA(cudaEventCreate(&st));
  CHECK_CUDA(cudaEventCreate(&ed));
  CHECK_CUDA(cudaEventRecord(st));
  int iters = 50;
  for (int i = 0; i < iters; ++i) {
    flashattn_2warp_fixedrole_online_softmax<M,KDIM,DV,N_TILE><<<grid, block, smem>>>(
      dQ, dK, dV, dO, S, scale);
  }
  CHECK_CUDA(cudaEventRecord(ed));
  CHECK_CUDA(cudaEventSynchronize(ed));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
  printf("Avg kernel time: %.6f ms (per launch)\n", ms / iters);

  CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // Error
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < hO.size(); ++i) {
    double a = (double)hO[i];
    double b = (double)hO_ref[i];
    double diff = a - b;
    num += diff*diff;
    den += b*b;
  }
  double rel = std::sqrt(num / (den + 1e-30));
  printf("Relative L2 error: %.12e\n", rel);

  printf("Batch 0, O[0, 0..7] (GPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hO[i]);
  printf("\nBatch 0, O_ref[0, 0..7] (CPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hO_ref[i]);
  printf("\n");

  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dO));
  return 0;
}
