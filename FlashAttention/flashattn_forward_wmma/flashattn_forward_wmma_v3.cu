// flashattn_forward_wmma_v3.cu
// 5.5.3: 멀티 워프 (4 warps) + 32x32 WMMA 타일
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_v3.cu -o flashattn_wmma_v3.exe

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
 * FlashAttention Forward (Tensor Core / WMMA, 32x32 tile, 4 warps)
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
__global__ void flashattn_forward_wmma_v3_kernel(
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
    (void)lane_id;

    const int bh        = blockIdx.y;              // [0, BH)
    const int row_block = blockIdx.x;              // row tile index
    const int row_start = row_block * BLOCK_M;     // global row 시작

    if (row_start >= N) return;

    // --- softmax 상태 초기화 ---
    for (int r = tid; r < BLOCK_M; r += blockDim.x) {
        m_smem[r]     = -CUDART_INF_F;
        l_smem[r]     = 0.0f;
        alpha_smem[r] = 0.0f;
    }

    // O_smem 초기화 (유효 영역: D까지만 실제 사용)
    for (int idx = tid; idx < BLOCK_M * O_LD; idx += blockDim.x) {
        O_smem[idx] = 0.0f;
    }
    __syncthreads();

    // --- Q tile 로드 ---
    const int q_block_base = (bh * N + row_start) * D; // Q[bh, row_start, 0]

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;     // 0..BLOCK_M-1
        int d = idx % D;     // 0..D-1
        int global_row = row_start + r;

        half qv = __float2half(0.0f);
        if (global_row < N) {
            qv = Q[q_block_base + r * D + d];
        }
        Q_smem[r * Q_LD + d] = qv;
    }
    __syncthreads();

    // --- K,V 타일 루프 (col_start 기준) ---
    for (int col_start = 0; col_start < N; col_start += TILE_N) {
        const int tile_cols = min(TILE_N, N - col_start);

        // (1) K tile 로드: K[bh, col_start:..., :] -> K_smem_T[d, j]
        const int k_block_base = (bh * N + col_start) * D;
        for (int idx = tid; idx < tile_cols * D; idx += blockDim.x) {
            int j = idx / D; // 0..tile_cols-1
            int d = idx % D;
            half kv = K[k_block_base + j * D + d];
            K_smem_T[d * K_LD + j] = kv;
        }
        // tail padding (j >= tile_cols)
        for (int idx = tid + tile_cols * D; idx < TILE_N * D; idx += blockDim.x) {
            int j = idx / D;
            int d = idx % D;
            K_smem_T[d * K_LD + j] = __float2half(0.0f);
        }

        // (2) V tile 로드: V[bh, col_start:..., :] -> V_smem[j, d]
        const int v_block_base = (bh * N + col_start) * D;
        for (int idx = tid; idx < tile_cols * D; idx += blockDim.x) {
            int j = idx / D;
            int d = idx % D;
            V_smem[j * V_LD + d] = V[v_block_base + j * D + d];
        }
        // tail padding for V
        for (int idx = tid + tile_cols * D; idx < TILE_N * D; idx += blockDim.x) {
            int j = idx / D;
            int d = idx % D;
            V_smem[j * V_LD + d] = __float2half(0.0f);
        }

        __syncthreads();

        // ------------------------------------------------------
        // 1) QK^T via WMMA (Tensor Core)
        //
        //   Q_smem:   [BLOCK_M, Q_LD]   (32 x Q_LD)
        //   K_smem_T: [D, K_LD]        (D x K_LD)
        //   -> S_smem: [BLOCK_M, S_LD] float (32 x S_LD)
        //
        //   warp 매핑 (2x2 fragment grid, 16x16):
        //     warp 0: rows 0..15,  cols 0..15
        //     warp 1: rows 16..31, cols 0..15
        //     warp 2: rows 0..15,  cols 16..31
        //     warp 3: rows 16..31, cols 16..31
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
        // 2) 온라인 softmax 업데이트 (row별)
        // ------------------------------------------------------
        for (int r = tid; r < BLOCK_M; r += blockDim.x) {
            int global_row = row_start + r;
            if (global_row >= N) continue;

            float m_old = m_smem[r];
            float l_old = l_smem[r];

            // (a) scale 적용 + tile 내 max_t
            float max_tile = -CUDART_INF_F;
            for (int j = 0; j < tile_cols; ++j) {
                float s = S_smem[r * S_LD + j] * scale;
                S_smem[r * S_LD + j] = s;
                max_tile = fmaxf(max_tile, s);
            }

            float m_new = fmaxf(m_old, max_tile);
            float exp_old = (m_old == -CUDART_INF_F)
                            ? 0.0f
                            : expf(m_old - m_new) * l_old;

            float sum_exp_tile = 0.0f;
            for (int j = 0; j < tile_cols; ++j) {
                float s = S_smem[r * S_LD + j];
                sum_exp_tile += expf(s - m_new);
            }

            float l_new = exp_old + sum_exp_tile;

            float alpha = 0.0f;
            if (l_new > 0.0f && l_old > 0.0f) {
                alpha = exp_old / l_new;
            }

            m_smem[r]     = m_new;
            l_smem[r]     = l_new;
            alpha_smem[r] = alpha;
        }
        __syncthreads();

        // 3) O_smem *= alpha(row) (유효 D 영역만)
        for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
            int r = idx / D;
            int d = idx % D;
            int global_row = row_start + r;
            if (global_row >= N) continue;
            float alpha = alpha_smem[r];
            O_smem[r * O_LD + d] *= alpha;
        }
        __syncthreads();

        // 4) softmax weight W_smem 계산
        for (int r = tid; r < BLOCK_M; r += blockDim.x) {
            int global_row = row_start + r;
            if (global_row >= N) {
                for (int j = 0; j < TILE_N; ++j) {
                    W_smem[r * W_LD + j] = __float2half(0.0f);
                }
                continue;
            }

            float m_new = m_smem[r];
            float l_new = l_smem[r];
            float inv_l = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            for (int j = 0; j < tile_cols; ++j) {
                float s = S_smem[r * S_LD + j];
                float w = expf(s - m_new) * inv_l;
                W_smem[r * W_LD + j] = __float2half(w);
            }
            for (int j = tile_cols; j < TILE_N; ++j) {
                W_smem[r * W_LD + j] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ------------------------------------------------------
        // 5) PV = W * V via WMMA (Tensor Core)
        //
        //   W_smem: [BLOCK_M, W_LD] (32 x W_LD)
        //   V_smem: [TILE_N, V_LD]  (32 x V_LD)
        //   -> PV_smem: [BLOCK_M, PV_LD]  (32 x PV_LD)
        //
        //   D 방향을 32씩 타일링 (OUT_TILE_D = 32) 하고,
        //   warp 4개가 2x2 fragment grid로 출력 조각 담당.
        // ------------------------------------------------------
        // PV_smem 초기화
        for (int idx = tid; idx < BLOCK_M * PV_LD; idx += blockDim.x) {
            PV_smem[idx] = 0.0f;
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

        // O_smem += PV_smem (유효 D 영역만)
        for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
            int r = idx / D;
            int d = idx % D;
            O_smem[r * O_LD + d] += PV_smem[r * PV_LD + d];
        }
        __syncthreads();
    } // col_start loop

    // --- 최종 O_smem -> global O ---
    const int o_block_base = (bh * N + row_start) * D;
    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        int global_row = row_start + r;
        if (global_row >= N) continue;
        O[o_block_base + r * D + d] = O_smem[r * O_LD + d];
    }
}

// =========================
// 테스트 메인 (5.5.3용)
// =========================
int main() {
    int BH = 1;
    int N  = 128; // 32 배수
    int D  = 64;  // 16/32 배수

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

    // host에서도 LD 기준으로 shared memory 크기 계산 (커널과 동일한 LD 사용)
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

    printf("Launching kernel v3 with grid=(%d,%d), block=%d, smem=%zu bytes\n",
           grid.x, grid.y, block.x, smem_bytes);

    int iters = 50;
    for (int i = 0; i < iters; ++i) {
        flashattn_forward_wmma_v3_kernel<<<grid, block, smem_bytes>>>(
            dQ, dK, dV, dO, BH, N, D, scale
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, bytes_float, cudaMemcpyDeviceToHost));

    printf("O_v3[0,0,0..7]: ");
    for (int j = 0; j < 8 && j < D; ++j) {
        printf("%f ", hO[j]);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_v3.cu -o flashattn_wmma_v3.exe

# 실행
.\flashattn_wmma_v3.exe

# Nsight Compute (31번째 런만 프로파일)
ncu --set full --launch-skip 30 --launch-count 1 .\flashattn_wmma_v3.exe
*/
