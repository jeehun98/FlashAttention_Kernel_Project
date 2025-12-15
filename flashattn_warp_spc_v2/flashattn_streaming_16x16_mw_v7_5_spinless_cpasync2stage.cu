// flashattn_streaming_16x16_mw_v7_5_spinless_cpasync2stage.cu
//
// v7.5 (spinless): remove ready/consumed spin loops
// - Use cp.async double-buffer (2-stage) for K/V tiles
// - Use cp.async.commit_group + cp.async.wait_group 0 + __syncthreads() at tile boundaries
// - Compute: streaming softmax over SEQ_LEN keys for a single query (D=16), output D=16
//
// Build (Ampere sm_86):
//   nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 flashattn_streaming_16x16_mw_v7_5_spinless_cpasync2stage.cu -o flashattn_streaming_16x16_mw_v7_5.exe
//
// Run:
//   ./flashattn_streaming_16x16_mw_v7_5.exe
//
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>


#define CHECK_CUDA(cmd) do {                                      \
  cudaError_t e = (cmd);                                          \
  if (e != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(e));           \
    std::exit(EXIT_FAILURE);                                      \
  }                                                               \
} while(0)

static inline __host__ __device__ float f2(float x) { return x*x; }

// ------------------------- cp.async wrappers (sm80+) -------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
  // cp.async requires shared address space pointer (u32 or u64 depending); we pass u64 via "l"
  unsigned long long smem_addr = static_cast<unsigned long long>(__cvta_generic_to_shared(smem_dst));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
      :: "l"(smem_addr), "l"(gmem_src)
      : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n" :: : "memory");
}
#else
// Fallback (won't be used for sm_86 build)
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
  *reinterpret_cast<int4*>(smem_dst) = *reinterpret_cast<const int4*>(gmem_src);
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait0() {}
#endif

// ------------------------- warp reduce (mask) -------------------------
__device__ __forceinline__ float warp_reduce_max_mask(float v, unsigned mask) {
  for (int off = 16; off > 0; off >>= 1) {
    float o = __shfl_xor_sync(mask, v, off);
    v = fmaxf(v, o);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum_mask(float v, unsigned mask) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(mask, v, off);
  }
  return v;
}

// ------------------------- kernel -------------------------
constexpr int D = 16;
constexpr int N_TILE = 16; // tile along sequence dimension
constexpr int PIPE = 2;

__global__ void flashattn_streaming_16x16_kernel_mw_v7_5(
    const __half* __restrict__ Q,   // [B, D]
    const __half* __restrict__ K,   // [B, S, D]
    const __half* __restrict__ V,   // [B, S, D]
    float* __restrict__ O,          // [B, D]
    int S,
    float inv_sqrt_d
) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;         // 0..63
  const int lane = tid & 31;           // 0..31
  const int warp = tid >> 5;           // 0 or 1

  // Shared: double-buffered K and V
  __shared__ __align__(16) __half smemK[PIPE][N_TILE][D];
  __shared__ __align__(16) __half smemV[PIPE][N_TILE][D];

  // Load q into registers for compute lanes (warp1 lanes 0..15 do compute)
  // (We keep it simple; overhead is tiny at D=16.)
  float qf[D];
  if (warp == 1 && lane < D) {
    const __half* qptr = Q + b * D;
    #pragma unroll
    for (int i = 0; i < D; ++i) qf[i] = __half2float(qptr[i]);
  }

  auto issue_tile_prefetch = [&](int t, int buf) {
    // Each tile has K: 16*16 half = 512B, V: 512B, total 1024B.
    // We use 64 threads * 16B = 1024B, mapping:
    // - tid 0..31: K (512B) as 32 chunks of 16B
    // - tid 32..63: V (512B) as 32 chunks of 16B
    const int base = t * N_TILE;
    if (base >= S) return;

    if (tid < 32) {
      const int chunk = tid;                 // 0..31
      const int byte_off = chunk * 16;       // 16B each
      const char* g = reinterpret_cast<const char*>(K + (b * S + base) * D);
      char* s = reinterpret_cast<char*>(smemK[buf]);
      cp_async_16B(s + byte_off, g + byte_off);
    } else {
      const int chunk = tid - 32;            // 0..31
      const int byte_off = chunk * 16;
      const char* g = reinterpret_cast<const char*>(V + (b * S + base) * D);
      char* s = reinterpret_cast<char*>(smemV[buf]);
      cp_async_16B(s + byte_off, g + byte_off);
    }
  };

  const int T = (S + N_TILE - 1) / N_TILE;

  // Preload tile 0 -> buf0
  issue_tile_prefetch(0, 0);
  cp_async_commit();
  cp_async_wait0();
  __syncthreads();

  // Streaming softmax state (only warp1 lanes < 16 are active compute lanes)
  float m = -CUDART_INF_F;
  float l = 0.f;
  float o[D];
  if (warp == 1 && lane < D) {
    #pragma unroll
    for (int i = 0; i < D; ++i) o[i] = 0.f;
  }

  // We reduce only across lanes 0..15 of warp1
  const unsigned mask16 = 0x0000ffffu;

  int cur = 0;

  for (int t = 0; t < T; ++t) {
    // Issue prefetch of next tile early (2-stage pipeline)
    const int nxt = cur ^ 1;
    if (t + 1 < T) {
      issue_tile_prefetch(t + 1, nxt);
      cp_async_commit();
    }

    // Compute on current tile (warp1 lanes 0..15)
    if (warp == 1 && lane < D) {
      // Each lane computes one score for key j=lane (0..15)
      const int j = lane;
      const int seq = t * N_TILE + j;

      float score = -CUDART_INF_F;
      if (seq < S) {
        // dot(q, k_j)
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < D; ++i) {
          acc += qf[i] * __half2float(smemK[cur][j][i]);
        }
        score = acc * inv_sqrt_d;
      }

      // tile max over 16 lanes
      float tile_m = warp_reduce_max_mask(score, mask16);

      // tile sum exp(score - tile_m)
      float e = (seq < S) ? __expf(score - tile_m) : 0.f;
      float tile_l = warp_reduce_sum_mask(e, mask16);

      // online softmax update
      float m_new = fmaxf(m, tile_m);
      float scale_old = (m == -CUDART_INF_F) ? 0.f : __expf(m - m_new);
      float scale_tile = __expf(tile_m - m_new);
      float l_new = l * scale_old + tile_l * scale_tile;

      // Update output vector: o = o*scale_old + scale_tile * sum_j (e_j * v_j)
      // We do it in a lane-per-dimension manner:
      // lane d accumulates output component o[d], reading V[j][d] across j=0..15.
      // Here lane is both "j" and "d"; to avoid extra warp roles, we compute full o[] per lane<16.
      // Since D=16 is tiny, this is fine.
      // (Yes, this is redundant work; correctness first. If you want faster, we can split roles.)
      float new_o[D];
      #pragma unroll
      for (int di = 0; di < D; ++di) new_o[di] = o[di] * scale_old;

      // Accumulate tile contribution
      // NOTE: each lane recomputes e_j for all j (redundant). Still small at D=16.
      #pragma unroll
      for (int jj = 0; jj < N_TILE; ++jj) {
        const int sidx = t * N_TILE + jj;
        if (sidx < S) {
          // score_jj
          float acc2 = 0.f;
          #pragma unroll
          for (int i = 0; i < D; ++i) {
            acc2 += qf[i] * __half2float(smemK[cur][jj][i]);
          }
          float sc = acc2 * inv_sqrt_d;
          float ej = __expf(sc - tile_m);               // exp(score - tile_m)
          float w = ej * scale_tile;                    // weight in global-normalized space (up to division by l_new later)

          #pragma unroll
          for (int di = 0; di < D; ++di) {
            new_o[di] += w * __half2float(smemV[cur][jj][di]);
          }
        }
      }

      // commit state
      m = m_new;
      l = l_new;
      #pragma unroll
      for (int di = 0; di < D; ++di) o[di] = new_o[di];
    }

    // Ensure next tile data is ready before next iteration uses it
    if (t + 1 < T) {
      cp_async_wait0();
      __syncthreads();
      cur ^= 1;
    }
  }

  // Write output (normalize)
  if (warp == 1 && lane < D) {
    float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    float* out = O + b * D;
    #pragma unroll
    for (int di = 0; di < D; ++di) out[di] = o[di] * inv_l;
  }
}

// ------------------------- CPU reference -------------------------
static void cpu_ref_attention(
    const std::vector<__half>& Q,
    const std::vector<__half>& K,
    const std::vector<__half>& V,
    std::vector<float>& Oref,
    int B, int S
) {
  const float inv_sqrt_d = 1.0f / std::sqrt(float(D));
  for (int b = 0; b < B; ++b) {
    // scores
    std::vector<float> scores(S);
    for (int s = 0; s < S; ++s) {
      float acc = 0.f;
      for (int i = 0; i < D; ++i) {
        acc += __half2float(Q[b*D + i]) * __half2float(K[(b*S + s)*D + i]);
      }
      scores[s] = acc * inv_sqrt_d;
    }
    // softmax
    float m = -1e30f;
    for (int s = 0; s < S; ++s) m = std::max(m, scores[s]);
    float sum = 0.f;
    for (int s = 0; s < S; ++s) sum += std::exp(scores[s] - m);

    // output
    for (int i = 0; i < D; ++i) {
      float out = 0.f;
      for (int s = 0; s < S; ++s) {
        float w = std::exp(scores[s] - m) / sum;
        out += w * __half2float(V[(b*S + s)*D + i]);
      }
      Oref[b*D + i] = out;
    }
  }
}

// ------------------------- main -------------------------
int main() {
  const int B = 1024;
  const int S = 128;
  const float inv_sqrt_d = 1.0f / std::sqrt(float(D));

  printf("Streaming Softmax FlashAttention-like Multi-Warp (v7.5: spinless + cp.async 2-stage)\n");
  printf("NUM_BATCH=%d, SEQ_LEN=%d, D=%d\n", B, S, D);

  // Host init
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<__half> hQ(B*D), hK(B*S*D), hV(B*S*D);
  for (auto& x : hQ) x = __float2half(dist(rng));
  for (auto& x : hK) x = __float2half(dist(rng));
  for (auto& x : hV) x = __float2half(dist(rng));

  std::vector<float> hO(B*D, 0.f), hOref(B*D, 0.f);
  cpu_ref_attention(hQ, hK, hV, hOref, B, S);

  // Device alloc
  __half *dQ=nullptr, *dK=nullptr, *dV=nullptr;
  float *dO=nullptr;
  CHECK_CUDA(cudaMalloc(&dQ, hQ.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dK, hK.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dV, hV.size()*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dO, hO.size()*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, hK.data(), hK.size()*sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size()*sizeof(__half), cudaMemcpyHostToDevice));

  // Launch
  dim3 grid(B,1,1);
  dim3 block(64,1,1);
  flashattn_streaming_16x16_kernel_mw_v7_5<<<grid, block>>>(dQ, dK, dV, dO, S, inv_sqrt_d);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(hO.data(), dO, hO.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // Print small slice
  printf("Batch 0, O[0, 0..7] (GPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hO[i]);
  printf("\n");

  printf("Batch 0, O_ref[0, 0..7] (CPU): ");
  for (int i = 0; i < 8; ++i) printf("% .6f ", hOref[i]);
  printf("\n");

  // Relative L2 error
  double num=0.0, den=0.0;
  for (int i = 0; i < B*D; ++i) {
    double diff = double(hO[i]) - double(hOref[i]);
    num += diff*diff;
    den += double(hOref[i])*double(hOref[i]);
  }
  double rel = std::sqrt(num / (den + 1e-12));
  printf("Relative L2 error over all batches: %.12e\n", rel);

  // Cleanup
  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dO));
  return 0;
}
/*
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -o flashattn_streaming_16x16_mw_v7_5_spinless_cpasync2stage.exe flashattn_streaming_16x16_mw_v7_5_spinless_cpasync2stage.cu

ncu   --section SpeedOfLight   --section SchedulerStats  --section WarpStateStats   --page details   ./flashattn_streaming_16x16_mw_v7_5_spinless_cpasync2stage.exe

*/