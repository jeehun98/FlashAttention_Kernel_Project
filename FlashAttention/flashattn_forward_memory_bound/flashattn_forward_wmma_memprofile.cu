// flashattn_forward_wmma_memprofile.cu
//
// 5.6.1: Global Memory Throughput / Memory-Bound 분석용 FlashAttention Forward (Tensor Core + Warp-stripe)
//
// - 커널은 5.5.4 v4 구조 그대로 사용
// - BH, N, D를 조금 키워서 실제 메모리 트래픽을 만들어줌
// - cudaEvent 로 평균 실행 시간, 대략적인 TFLOPS / GB/s 계산
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_memprofile.cu -o flashattn_wmma_memprofile.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_wmma_memprofile.exe

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

using namespace nvcuda;

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

constexpr int WARP_SIZE      = 32;
constexpr int NUM_WARPS      = 4;
constexpr int BLOCK_THREADS  = WARP_SIZE * NUM_WARPS;

// 타일 사이즈 (32x32)
constexpr int BLOCK_M = 32;   // row tile (queries per block)
constexpr int TILE_N  = 32;   // col tile (keys/values per step)

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * FlashAttention Forward (Tensor Core / WMMA, 32x32 tile, 4 warps, warp-stripe shared access)
 *
 * Q, K, V: [BH, N, D]  (half)
 * O:       [BH, N, D]  (float)
 *
 * BH: batch_size * num_heads
 * N:  sequence length   (32 배수 가정)
 * D:  head dimension    (16 배수 가정, 32 배수가 깔끔)
 *
 * scale: 1/sqrt(D) 정도
 */
__global__ void flashattn_forward_wmma_v4_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int N,
    int D,
    float scale
) {
#if __CUDA_ARCH__ < 700
    return; // WMMA는 Volta 이상
#endif

    if (D % 16 != 0) return;  // WMMA k dimension 제약

    // --- leading dimension 설정 (WMMA 조건 맞추기: 16의 배수) ---
    const int Q_LD  = D;        // Q_smem: [BLOCK_M, Q_LD]
    const int K_LD  = TILE_N;   // K_smem_T: [D, K_LD]
    const int V_LD  = D;        // V_smem: [TILE_N, V_LD]
    const int S_LD  = TILE_N;   // S_smem: [BLOCK_M, S_LD]
    const int W_LD  = TILE_N;   // W_smem: [BLOCK_M, W_LD]
    const int O_LD  = D;        // O_smem: [BLOCK_M, O_LD]
    const int PV_LD = D;        // PV_smem: [BLOCK_M, PV_LD]

    // shared memory base를 16B 정렬
    extern __shared__ __align__(16) char smem_raw[];
    char* smem_ptr = smem_raw;

    // Q tile: [BLOCK_M, Q_LD] (row-major, half)
    half* Q_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += BLOCK_M * Q_LD * sizeof(half);

    // K^T tile: [D, K_LD] (row-major, half)
    half* K_smem_T = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += D * K_LD * sizeof(half);

    // V tile: [TILE_N, V_LD] (row-major, half)
    half* V_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += TILE_N * V_LD * sizeof(half);

    // scores: [BLOCK_M, S_LD] float
    float* S_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * S_LD * sizeof(float);

    // softmax 상태: m, l, alpha [BLOCK_M]
    float* m_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * sizeof(float);

    float* l_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * sizeof(float);

    float* alpha_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * sizeof(float);

    // softmax weights: [BLOCK_M, W_LD] half
    half* W_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += BLOCK_M * W_LD * sizeof(half);

    // output tile: [BLOCK_M, O_LD] float
    float* O_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * O_LD * sizeof(float);

    // PV 임시: [BLOCK_M, PV_LD] float
    float* PV_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * PV_LD * sizeof(float);

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;   // 0..3
    const int lane_id = tid % WARP_SIZE;   // 0..31
    const unsigned FULL_MASK = 0xffffffffu;

    const int bh        = blockIdx.y;              // [0, BH)
    const int row_block = blockIdx.x;              // row tile index
    const int row_start = row_block * BLOCK_M;     // global row 시작

    if (row_start >= N) return;

    // --- softmax 상태 초기화 (작은 배열이므로 그냥 tid 기반) ---
    for (int r = tid; r < BLOCK_M; r += blockDim.x) {
        m_smem[r]     = -CUDART_INF_F;
        l_smem[r]     = 0.0f;
        alpha_smem[r] = 0.0f;
    }

    // O_smem 초기화 (유효 영역: D까지만 실제 사용, warp-stripe)
    for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            O_smem[r * O_LD + d] = 0.0f;
        }
    }
    __syncthreads();

    // --- Q tile 로드 ---
    const int q_block_base = (bh * N + row_start) * D; // Q[bh, row_start, 0]

    // row striping by warp, col striping by lane
    for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
        int global_row = row_start + r;
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            half qv = __float2half(0.0f);
            if (global_row < N) {
                qv = Q[q_block_base + r * D + d];
            }
            Q_smem[r * Q_LD + d] = qv;
        }
    }
    __syncthreads();

    // --- K,V 타일 루프 (col_start 기준) ---
    for (int col_start = 0; col_start < N; col_start += TILE_N) {
        const int tile_cols = min(TILE_N, N - col_start);

        // (1) K tile 로드: K[bh, col_start:..., :] -> K_smem_T[d, j]
        const int k_block_base = (bh * N + col_start) * D;

        // d striping by warp, j striping by lane
        for (int d = warp_id; d < D; d += NUM_WARPS) {
            for (int j = lane_id; j < tile_cols; j += WARP_SIZE) {
                half kv = K[k_block_base + j * D + d];
                K_smem_T[d * K_LD + j] = kv;
            }
            // tail padding (j >= tile_cols)
            for (int j = lane_id + tile_cols; j < TILE_N; j += WARP_SIZE) {
                K_smem_T[d * K_LD + j] = __float2half(0.0f);
            }
        }

        // (2) V tile 로드: V[bh, col_start:..., :] -> V_smem[j, d]
        const int v_block_base = (bh * N + col_start) * D;

        for (int j = warp_id; j < TILE_N; j += NUM_WARPS) {
            int global_col = col_start + j;
            for (int d = lane_id; d < D; d += WARP_SIZE) {
                half vv = __float2half(0.0f);
                if (j < tile_cols) {
                    vv = V[v_block_base + j * D + d];
                }
                V_smem[j * V_LD + d] = vv;
            }
        }

        __syncthreads();

        // ------------------------------------------------------
        // 1) QK^T via WMMA (Tensor Core)
        // ------------------------------------------------------
        if (warp_id < NUM_WARPS) {
            int warp_m = warp_id / 2; // 0 or 1
            int warp_n = warp_id % 2; // 0 or 1

            int row0 = warp_m * 16;
            int col0 = warp_n * 16;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> A_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> B_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>              C_frag;

            wmma::fill_fragment(C_frag, 0.0f);

            for (int kk = 0; kk < D; kk += 16) {
                const half* a_ptr = &Q_smem[row0 * Q_LD + kk];
                const half* b_ptr = &K_smem_T[kk * K_LD + col0];

                wmma::load_matrix_sync(A_frag, a_ptr, Q_LD);
                wmma::load_matrix_sync(B_frag, b_ptr, K_LD);
                wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
            }

            float* S_tile = &S_smem[row0 * S_LD + col0];
            wmma::store_matrix_sync(S_tile, C_frag, S_LD, wmma::mem_row_major);
        }
        __syncthreads();

        // ------------------------------------------------------
        // 2) 온라인 softmax 업데이트 (warp-stripe, warp reduce 사용)
        // ------------------------------------------------------
        for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
            int global_row = row_start + r;
            if (global_row >= N) continue;

            // m_old, l_old broadcast
            float m_old = 0.0f;
            float l_old = 0.0f;
            if (lane_id == 0) {
                m_old = m_smem[r];
                l_old = l_smem[r];
            }
            m_old = __shfl_sync(FULL_MASK, m_old, 0);
            l_old = __shfl_sync(FULL_MASK, l_old, 0);

            // (a) scale 적용 + tile 내 max_t
            float local_max = -CUDART_INF_F;
            for (int j = lane_id; j < tile_cols; j += WARP_SIZE) {
                float s = S_smem[r * S_LD + j] * scale;
                S_smem[r * S_LD + j] = s;
                local_max = fmaxf(local_max, s);
            }

            // warp reduce max
            float max_tile = local_max;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(FULL_MASK, max_tile, offset);
                max_tile = fmaxf(max_tile, other);
            }
            max_tile = __shfl_sync(FULL_MASK, max_tile, 0);

            float m_new   = fmaxf(m_old, max_tile);
            float exp_old = (m_old == -CUDART_INF_F)
                            ? 0.0f
                            : expf(m_old - m_new) * l_old;

            // (b) sum_exp_tile
            float local_sum = 0.0f;
            for (int j = lane_id; j < tile_cols; j += WARP_SIZE) {
                float s = S_smem[r * S_LD + j];
                local_sum += expf(s - m_new);
            }
            float sum_exp_tile = local_sum;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(FULL_MASK, sum_exp_tile, offset);
                sum_exp_tile += other;
            }
            sum_exp_tile = __shfl_sync(FULL_MASK, sum_exp_tile, 0);

            float l_new = exp_old + sum_exp_tile;
            float alpha = 0.0f;
            if (l_new > 0.0f && l_old > 0.0f) {
                alpha = exp_old / l_new;
            }

            if (lane_id == 0) {
                m_smem[r]     = m_new;
                l_smem[r]     = l_new;
                alpha_smem[r] = alpha;
            }
        }
        __syncthreads();

        // 3) O_smem *= alpha(row) (warp-stripe, D 방향 lane-stripe)
        for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
            int global_row = row_start + r;
            if (global_row >= N) continue;
            float alpha = alpha_smem[r];
            for (int d = lane_id; d < D; d += WARP_SIZE) {
                O_smem[r * O_LD + d] *= alpha;
            }
        }
        __syncthreads();

        // 4) softmax weight W_smem 계산 (warp-stripe)
        for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
            int global_row = row_start + r;
            if (global_row >= N) {
                for (int j = lane_id; j < TILE_N; j += WARP_SIZE) {
                    W_smem[r * W_LD + j] = __float2half(0.0f);
                }
                continue;
            }

            float m_new = m_smem[r];
            float l_new = l_smem[r];
            float inv_l = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            for (int j = lane_id; j < tile_cols; j += WARP_SIZE) {
                float s = S_smem[r * S_LD + j];
                float w = expf(s - m_new) * inv_l;
                W_smem[r * W_LD + j] = __float2half(w);
            }
            for (int j = lane_id + tile_cols; j < TILE_N; j += WARP_SIZE) {
                W_smem[r * W_LD + j] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ------------------------------------------------------
        // 5) PV = W * V via WMMA (Tensor Core)
        // ------------------------------------------------------
        // PV_smem 초기화 (warp-stripe)
        for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
            for (int d = lane_id; d < D; d += WARP_SIZE) {
                PV_smem[r * PV_LD + d] = 0.0f;
            }
        }
        __syncthreads();

        const int OUT_TILE_D = 32;
        for (int d_block = 0; d_block < D; d_block += OUT_TILE_D) {
            int d_rem = D - d_block;
            int cur_d = (d_rem >= OUT_TILE_D) ? OUT_TILE_D : d_rem;
            if (cur_d < 16) break; // tail 단순화

            if (warp_id < NUM_WARPS) {
                int warp_m = warp_id / 2; // 0 or 1
                int warp_n = warp_id % 2; // 0 or 1

                int row0 = warp_m * 16;
                int col0 = d_block + warp_n * 16;

                if (col0 < D) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> W_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> V_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              O_frag;

                    wmma::fill_fragment(O_frag, 0.0f);

                    // K dimension(TILE_N=32)을 16씩 나눠서 accumulate
                    for (int kk = 0; kk < TILE_N; kk += 16) {
                        const half* w_ptr = &W_smem[row0 * W_LD + kk];
                        const half* v_ptr = &V_smem[kk * V_LD + col0];

                        wmma::load_matrix_sync(W_frag, w_ptr, W_LD);
                        wmma::load_matrix_sync(V_frag, v_ptr, V_LD);
                        wmma::mma_sync(O_frag, W_frag, V_frag, O_frag);
                    }

                    // PV_smem row-major 저장
                    float* pv_ptr = &PV_smem[row0 * PV_LD + col0];
                    wmma::store_matrix_sync(pv_ptr, O_frag, PV_LD, wmma::mem_row_major);
                }
            }
            __syncthreads();
        }

        // O_smem += PV_smem (warp-stripe)
        for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
            for (int d = lane_id; d < D; d += WARP_SIZE) {
                O_smem[r * O_LD + d] += PV_smem[r * PV_LD + d];
            }
        }
        __syncthreads();
    } // col_start loop

    // --- 최종 O_smem -> global O (warp-stripe) ---
    const int o_block_base = (bh * N + row_start) * D;
    for (int r = warp_id; r < BLOCK_M; r += NUM_WARPS) {
        int global_row = row_start + r;
        if (global_row >= N) continue;
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            O[o_block_base + r * D + d] = O_smem[r * O_LD + d];
        }
    }
}

// =========================
// 5.6.1용 메인: 타이밍 + 대략적인 TFLOPS, GB/s 계산
// =========================
int main() {
    // N, D를 약간 키워서 실제 메모리/연산량이 있는 상태로 프로파일
    int BH = 4;    // batch_size * num_heads
    int N  = 512;  // seq len (32 배수)
    int D  = 64;   // head dim (16/32 배수)

    float scale = 1.0f / std::sqrt((float)D);

    size_t num_elem = (size_t)BH * N * D;

    size_t bytes_half  = num_elem * sizeof(half);
    size_t bytes_float = num_elem * sizeof(float);

    std::vector<half>  hQ(num_elem), hK(num_elem), hV(num_elem);
    std::vector<float> hO(num_elem);

    // deterministic 초기화
    for (size_t i = 0; i < num_elem; ++i) {
        float v = (float)(i % 13) * 0.01f;
        hQ[i] = __float2half(v);
        hK[i] = __float2half(0.5f * v);
        hV[i] = __float2half(0.3f * v);
    }

    half  *dQ, *dK, *dV;
    float *dO;
    CHECK_CUDA(cudaMalloc(&dQ, bytes_half));
    CHECK_CUDA(cudaMalloc(&dK, bytes_half));
    CHECK_CUDA(cudaMalloc(&dV, bytes_half));
    CHECK_CUDA(cudaMalloc(&dO, bytes_float));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dO, 0, bytes_float));

    dim3 grid(ceil_div(N, BLOCK_M), BH, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    int Q_LD  = D;
    int K_LD  = TILE_N;
    int V_LD  = D;
    int S_LD  = TILE_N;
    int W_LD  = TILE_N;
    int O_LD  = D;
    int PV_LD = D;

    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * Q_LD  * sizeof(half);   // Q_smem
    smem_bytes += D        * K_LD  * sizeof(half);  // K_smem_T
    smem_bytes += TILE_N   * V_LD  * sizeof(half);  // V_smem
    smem_bytes += BLOCK_M  * S_LD  * sizeof(float); // S_smem
    smem_bytes += BLOCK_M  * sizeof(float);         // m_smem
    smem_bytes += BLOCK_M  * sizeof(float);         // l_smem
    smem_bytes += BLOCK_M  * sizeof(float);         // alpha_smem
    smem_bytes += BLOCK_M  * W_LD  * sizeof(half);  // W_smem
    smem_bytes += BLOCK_M  * O_LD  * sizeof(float); // O_smem
    smem_bytes += BLOCK_M  * PV_LD * sizeof(float); // PV_smem

    printf("Launching kernel (memprofile) with BH=%d, N=%d, D=%d\n", BH, N, D);
    printf("grid=(%d,%d), block=%d, smem=%zu bytes\n", grid.x, grid.y, block.x, smem_bytes);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        flashattn_forward_wmma_v4_kernel<<<grid, block, smem_bytes>>>(
            dQ, dK, dV, dO, BH, N, D, scale
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing
    int iters = 50;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        flashattn_forward_wmma_v4_kernel<<<grid, block, smem_bytes>>>(
            dQ, dK, dV, dO, BH, N, D, scale
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, bytes_float, cudaMemcpyDeviceToHost));

    printf("O_sample[0,0,0..7]: ");
    for (int j = 0; j < 8 && j < D; ++j) {
        printf("%f ", hO[j]);
    }
    printf("\n");

    // 대략적인 FLOPs / bytes 추정
    // Attention per head: QK^T + PV
    // FLOPs ≈ 2 * BH * N * N * D (QK^T + PV 둘 다 D차원 inner product)
    double flops = 2.0 * (double)BH * N * N * D * 2.0; // QK^T + PV
    double t_sec = avg_ms * 1e-3;
    double tflops = flops / t_sec / 1e12;

    // 매우 러프한 DRAM bytes 추정:
    //   Q: BH*N*D*2  (한 번)
    //   K: BH*N*D*2  (한 번)
    //   V: BH*N*D*2  (한 번)
    //   O: BH*N*D*4  (write)
    // 실제론 더 많지만 "최소" 수준 트래픽
    double bytes_dram_min =
        3.0 * (double)BH * N * D * 2.0 + // Q,K,V read
        1.0 * (double)BH * N * D * 4.0;  // O write
    double gbps = bytes_dram_min / t_sec / 1e9;

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx FLOPs: %.3e\n", flops);
    printf("Approx TFLOPS: %.3f\n", tflops);
    printf("Approx DRAM traffic (min): %.3f GB/s\n", gbps);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_memprofile.cu -o flashattn_wmma_memprofile.exe

# 실행
.\flashattn_wmma_memprofile.exe

# Nsight Compute (10번 워밍업 후 1번 런 프로파일)
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_wmma_memprofile.exe

# 5.6.1에서 볼 포인트
- GPU Speed Of Light Throughput:
    - DRAM Throughput (%)
    - Memory Throughput (%)
- Memory Workload Analysis:
    - Mem Busy, Max Bandwidth
    - L2 Hit Rate, L1/TEX Hit Rate
- Warp State / Scheduler:
    - No Eligible Warp %
    - SM Busy %
- 이 코드의 printf TFLOPS / GB/s와 Nsight의 DRAM Throughput을 함께 보면서
  "FlashAttention Forward는 Tensor Core가 아니라 softmax/shared memory 쪽이 병목" 이라는 결론을
  실제 수치로 확인할 수 있다.
*/
