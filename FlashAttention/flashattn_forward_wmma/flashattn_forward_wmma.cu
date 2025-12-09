// flashattn_forward_wmma.cu
// nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma.cu -o flashattn_wmma

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

constexpr int WARP_SIZE = 32;

// 타일 사이즈 예시 (단일 warp, 단일 fragment)
constexpr int BLOCK_M = 16;   // row tile (queries per block)
constexpr int TILE_N  = 16;   // col tile (keys/values per step)

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * FlashAttention Forward (Tensor Core / WMMA)
 *
 * Q, K, V: [BH, N, D]  (half)
 * O:       [BH, N, D]  (float)
 *
 * BH: batch_size * num_heads
 * N:  sequence length
 * D:  head dimension (16의 배수)
 *
 * scale: 1/sqrt(D) 정도
 */
__global__ void flashattn_forward_wmma_kernel(
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

    if (D % 16 != 0) return; // WMMA k dimension 제약

    // --- shared memory layout ---
    extern __shared__ char smem_raw[];
    char* smem_ptr = smem_raw;

    // Q tile: [BLOCK_M, D] (row-major)
    half* Q_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += BLOCK_M * D * sizeof(half);

    // K tile (transposed): [D, TILE_N] row-major (K^T tile)
    half* K_smem_T = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += D * TILE_N * sizeof(half);

    // V tile: [TILE_N, D] row-major
    half* V_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += TILE_N * D * sizeof(half);

    // score tile: [BLOCK_M, TILE_N] float
    float* S_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * TILE_N * sizeof(float);

    // softmax 상태 (row별): m, l, alpha
    float* m_smem = reinterpret_cast<float*>(smem_ptr); // [BLOCK_M]
    smem_ptr += BLOCK_M * sizeof(float);

    float* l_smem = reinterpret_cast<float*>(smem_ptr); // [BLOCK_M]
    smem_ptr += BLOCK_M * sizeof(float);

    float* alpha_smem = reinterpret_cast<float*>(smem_ptr); // [BLOCK_M]
    smem_ptr += BLOCK_M * sizeof(float);

    // weight tile (softmax 결과): [BLOCK_M, TILE_N] half
    half* W_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += BLOCK_M * TILE_N * sizeof(half);

    // output tile: [BLOCK_M, D] float
    float* O_smem = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += BLOCK_M * D * sizeof(float);

    // PV 타일 임시 저장: [BLOCK_M, D] float
    float* PV_smem = reinterpret_cast<float*>(smem_ptr);
    // smem_ptr += BLOCK_M * D * sizeof(float); // 필요시

    const int lane = threadIdx.x;      // 0..31 (단일 warp 가정)

    const int bh        = blockIdx.y;              // [0, BH)
    const int row_block = blockIdx.x;              // row tile index
    const int row_start = row_block * BLOCK_M;     // global row 시작

    if (row_start >= N) return;

    // --- softmax 상태 초기화 ---
    if (lane < BLOCK_M) {
        m_smem[lane]     = -CUDART_INF_F;  // row별 max
        l_smem[lane]     = 0.0f;           // row별 sum(exp)
        alpha_smem[lane] = 0.0f;           // scaling factor
    }

    // O_smem 초기화
    for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
        O_smem[idx] = 0.0f;
    }
    __syncthreads();

    // --- Q tile 로드 (한 block이 담당하는 BLOCK_M rows) ---
    const int q_block_base = (bh * N + row_start) * D; // Q[bh, row_start, 0]

    for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
        int r = idx / D;     // 0..BLOCK_M-1
        int d = idx % D;     // 0..D-1
        int global_row = row_start + r;
        half qv = __float2half(0.0f);
        if (global_row < N) {
            qv = Q[q_block_base + r * D + d];
        }
        Q_smem[r * D + d] = qv;
    }
    __syncthreads();

    // --- K,V 타일 루프 (col_start 기준) ---
    for (int col_start = 0; col_start < N; col_start += TILE_N) {
        const int tile_cols = min(TILE_N, N - col_start);

        // (1) K tile 로드: global K[bh, col_start : col_start + tile_cols, :]
        //     -> shared K_smem_T[d, j] (j: tile 내 key index)
        const int k_block_base = (bh * N + col_start) * D;
        for (int idx = lane; idx < tile_cols * D; idx += WARP_SIZE) {
            int j = idx / D;  // 0..tile_cols-1
            int d = idx % D;
            half kv = K[k_block_base + j * D + d];
            // K^T 타일: [D, TILE_N] row-major
            K_smem_T[d * TILE_N + j] = kv;
        }
        // tail tile 패딩
        for (int idx = lane + tile_cols * D; idx < TILE_N * D; idx += WARP_SIZE) {
            K_smem_T[idx] = __float2half(0.0f);
        }

        // (2) V tile 로드: global V[bh, col_start : col_start + tile_cols, :]
        //     -> shared V_smem[j, d], [TILE_N, D] row-major
        const int v_block_base = (bh * N + col_start) * D;
        for (int idx = lane; idx < tile_cols * D; idx += WARP_SIZE) {
            int j = idx / D;
            int d = idx % D;
            V_smem[j * D + d] = V[v_block_base + j * D + d];
        }
        // 패딩
        for (int idx = lane + tile_cols * D; idx < TILE_N * D; idx += WARP_SIZE) {
            V_smem[idx] = __float2half(0.0f);
        }

        __syncthreads();

        // ------------------------------------------------------
        // 1) QK^T via WMMA (Tensor Core)
        //   Q_smem:     [BLOCK_M, D]   (M x K)
        //   K_smem_T:   [D, TILE_N]    (K x N)
        //   -> S_smem:  [BLOCK_M, TILE_N] float
        // ------------------------------------------------------
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

        wmma::fill_fragment(C_frag, 0.0f);

        // k dimension(=D)을 16씩 쪼개서 accumulate
        for (int kk = 0; kk < D; kk += 16) {
            const half* a_ptr = &Q_smem[0 * D + kk];          // [BLOCK_M x D]
            const half* b_ptr = &K_smem_T[kk * TILE_N + 0];   // [D x TILE_N]
            wmma::load_matrix_sync(A_frag, a_ptr, D);
            wmma::load_matrix_sync(B_frag, b_ptr, TILE_N);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }

        // C_frag -> S_smem (row-major float) + scale 적용
        wmma::store_matrix_sync(&S_smem[0], C_frag, TILE_N, wmma::mem_row_major);
        __syncthreads();

        // ------------------------------------------------------
        // 2) 온라인 softmax 업데이트 (row별)
        //    m_smem[row], l_smem[row] 유지
        // ------------------------------------------------------
        if (lane < BLOCK_M) {
            int r = lane;
            int global_row = row_start + r;
            if (global_row < N) {
                float m_old = m_smem[r];
                float l_old = l_smem[r];

                // (a) tile 내 score에 scale 적용 + max_t 찾기
                float max_tile = -CUDART_INF_F;
                for (int j = 0; j < tile_cols; ++j) {
                    float s = S_smem[r * TILE_N + j] * scale;
                    S_smem[r * TILE_N + j] = s; // scaled score 저장
                    max_tile = fmaxf(max_tile, s);
                }

                // (b) 새로운 max m_new, 이전 sum의 rescale
                float m_new = fmaxf(m_old, max_tile);
                float exp_old = (m_old == -CUDART_INF_F)
                                ? 0.0f
                                : expf(m_old - m_new) * l_old;

                // (c) 이번 tile의 exp sum
                float sum_exp_tile = 0.0f;
                for (int j = 0; j < tile_cols; ++j) {
                    float s = S_smem[r * TILE_N + j];
                    sum_exp_tile += expf(s - m_new);
                }

                float l_new = exp_old + sum_exp_tile;

                // O_old에 곱할 scaling factor alpha = exp_old / l_new
                float alpha = 0.0f;
                if (l_new > 0.0f && l_old > 0.0f) {
                    alpha = exp_old / l_new;
                }

                m_smem[r]     = m_new;
                l_smem[r]     = l_new;
                alpha_smem[r] = alpha;
            } else {
                alpha_smem[r] = 0.0f;
            }
        }
        __syncthreads();

        // ------------------------------------------------------
        // 3) 기존 O_smem에 alpha(row) 곱해줌 (행마다 다름)
        // ------------------------------------------------------
        for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
            int r = idx / D;
            int d = idx % D;
            int global_row = row_start + r;
            if (global_row >= N) continue;
            float alpha = alpha_smem[r];
            O_smem[r * D + d] *= alpha;
        }
        __syncthreads();

        // ------------------------------------------------------
        // 4) softmax weight W_smem 계산 (row별, tile 내 columns)
        //    w_ij = exp(s_ij - m_new) / l_new
        // ------------------------------------------------------
        if (lane < BLOCK_M) {
            int r = lane;
            int global_row = row_start + r;
            if (global_row < N) {
                float m_new = m_smem[r];
                float l_new = l_smem[r];
                float inv_l = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

                for (int j = 0; j < tile_cols; ++j) {
                    float s = S_smem[r * TILE_N + j];
                    float w = expf(s - m_new) * inv_l;
                    W_smem[r * TILE_N + j] = __float2half(w);
                }
                // tail padding
                for (int j = tile_cols; j < TILE_N; ++j) {
                    W_smem[r * TILE_N + j] = __float2half(0.0f);
                }
            } else {
                for (int j = 0; j < TILE_N; ++j) {
                    W_smem[r * TILE_N + j] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ------------------------------------------------------
        // 5) PV = W * V 타일을 WMMA로 계산, PV_smem에 저장 후 O_smem += PV
        //
        //    W_smem: [BLOCK_M, TILE_N] (M x K)
        //    V_smem: [TILE_N, D]       (K x N)
        //    -> PV_smem: [BLOCK_M, D]  float
        // ------------------------------------------------------
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> W_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> V_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_frag;

        // PV_smem 초기화
        for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
            PV_smem[idx] = 0.0f;
        }
        __syncthreads();

        for (int d0 = 0; d0 < D; d0 += 16) {
            const half* w_ptr = &W_smem[0 * TILE_N + 0]; // [BLOCK_M x TILE_N]
            const half* v_ptr = &V_smem[0 * D + d0];     // [TILE_N x D]
            float*      pv_ptr = &PV_smem[0 * D + d0];   // [BLOCK_M x D]

            // A, B load
            wmma::load_matrix_sync(W_frag, w_ptr, TILE_N);
            wmma::load_matrix_sync(V_frag, v_ptr, D);

            wmma::fill_fragment(O_frag, 0.0f);
            // O_frag = W_frag * V_frag
            wmma::mma_sync(O_frag, W_frag, V_frag, O_frag);

            // PV_smem tile에 저장 (row-major, leading dim = D)
            wmma::store_matrix_sync(pv_ptr, O_frag, D, wmma::mem_row_major);
        }
        __syncthreads();

        // O_smem += PV_smem
        for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
            O_smem[idx] += PV_smem[idx];
        }
        __syncthreads();
    } // col_start loop

    // ------------------------------------------------------
    // 6) O_smem -> global O 쓰기
    // ------------------------------------------------------
    const int o_block_base = (bh * N + row_start) * D;
    for (int idx = lane; idx < BLOCK_M * D; idx += WARP_SIZE) {
        int r = idx / D;
        int d = idx % D;
        int global_row = row_start + r;
        if (global_row >= N) continue;
        O[o_block_base + r * D + d] = O_smem[r * D + d];
    }
}

// =========================
// 테스트 메인
// =========================
int main() {
    // 간단한 테스트 파라미터
    int BH = 1;
    int N  = 128;
    int D  = 64;   // 16의 배수

    float scale = 1.0f / std::sqrt((float)D);

    size_t num_elem = (size_t)BH * N * D;

    size_t bytes_half  = num_elem * sizeof(half);
    size_t bytes_float = num_elem * sizeof(float);

    // Host 메모리
    std::vector<half>  hQ(num_elem), hK(num_elem), hV(num_elem);
    std::vector<float> hO(num_elem);

    // 간단 초기화
    for (size_t i = 0; i < num_elem; ++i) {
        float v = (float)(i % 13) * 0.01f;
        hQ[i] = __float2half(v);
        hK[i] = __float2half(0.5f * v);
        hV[i] = __float2half(0.3f * v);
    }

    // Device 메모리
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

    // grid/block 설정
    dim3 grid(ceil_div(N, BLOCK_M), BH, 1);
    dim3 block(WARP_SIZE, 1, 1);

    // shared memory 크기 계산
    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * D * sizeof(half);        // Q_smem
    smem_bytes += D * TILE_N * sizeof(half);         // K_smem_T
    smem_bytes += TILE_N * D * sizeof(half);         // V_smem
    smem_bytes += BLOCK_M * TILE_N * sizeof(float);  // S_smem
    smem_bytes += BLOCK_M * sizeof(float);           // m_smem
    smem_bytes += BLOCK_M * sizeof(float);           // l_smem
    smem_bytes += BLOCK_M * sizeof(float);           // alpha_smem
    smem_bytes += BLOCK_M * TILE_N * sizeof(half);   // W_smem
    smem_bytes += BLOCK_M * D * sizeof(float);       // O_smem
    smem_bytes += BLOCK_M * D * sizeof(float);       // PV_smem

    printf("Launching kernel with grid=(%d,%d), block=%d, smem=%zu bytes\n",
           grid.x, grid.y, block.x, smem_bytes);

    // 워밍업 + 반복 실행 (ncu로 볼 때도 유리)
    int iters = 50;
    for (int i = 0; i < iters; ++i) {
        flashattn_forward_wmma_kernel<<<grid, block, smem_bytes>>>(
            dQ, dK, dV, dO, BH, N, D, scale
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, bytes_float, cudaMemcpyDeviceToHost));

    // 결과 일부 출력
    printf("O[0,0,0..7]: ");
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
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17      flashattn_forward_wmma.cu -o flashattn_wmma.exe

ncu --set full --launch-skip 30 --launch-count 1 .\flashattn_wmma.exe

  .\flashattn_wmma.exe

*/