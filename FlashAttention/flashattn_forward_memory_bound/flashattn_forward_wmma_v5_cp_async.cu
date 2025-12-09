// flashattn_forward_wmma_v5_cp_async.cu
//
// 5.x: FlashAttention Forward + WMMA + cp.async 2-stage pipeline
//
// - v4에서 사용한 32x32 타일 구조/softmax/PV는 그대로 유지
// - 차이점: K,V 타일 로드를 cp.async 2-stage ping-pong으로 이식
//   * Q tile: 여전히 block 당 1회 global load
//   * K,V tile: cp.async.cg.shared.global 로 double-buffer prefetch
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_v5_cp_async.cu -o flashattn_wmma_v5_cp_async.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 30 --launch-count 1 .\flashattn_wmma_v5_cp_async.exe

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

// =====================
// cp.async helper 들
// =====================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// shared generic ptr -> shared 주소(u32)로 변환
__device__ __forceinline__ unsigned cvta_to_shared_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16B(void* smem_dst, const void* gmem_src) {
    unsigned smem_addr = cvta_to_shared_u32(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

#define CP_ASYNC_CG(dst, src)       cp_async_cg_16B((void*)(dst), (const void*)(src))
#define CP_ASYNC_COMMIT_GROUP()     cp_async_commit_group()
#define CP_ASYNC_WAIT_ALL()         cp_async_wait_all()

#else   // __CUDA_ARCH__ < 800 또는 host 컴파일 시

#define CP_ASYNC_CG(dst, src)       ((void)0)
#define CP_ASYNC_COMMIT_GROUP()     ((void)0)
#define CP_ASYNC_WAIT_ALL()         ((void)0)

#endif

/**
 * FlashAttention Forward (Tensor Core / WMMA, 32x32 tile, 4 warps, cp.async)
 *
 * Q, K, V: [BH, N, D]  (half)
 * O:       [BH, N, D]  (float)
 *
 * BH: batch_size * num_heads
 * N:  sequence length   (TILE_N=32 배수 가정)
 * D:  head dimension    (16 배수 가정, 32/64 추천)
 *
 * scale: 1/sqrt(D) 정도
 */
__global__ void flashattn_forward_wmma_v5_cp_async_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int N,
    int D,
    float scale
) {
#if __CUDA_ARCH__ < 800
    return; // cp.async + WMMA는 Ampere 이상
#endif

    if (D % 16 != 0) return;
    if (N % TILE_N != 0) return; // cp.async 단순화를 위해 TILE_N 배수 가정

    // leading dimension 설정
    const int Q_LD  = D;        // Q_smem: [BLOCK_M, Q_LD]
    const int V_LD  = D;        // V_smem: [TILE_N, V_LD]
    const int S_LD  = TILE_N;   // S_smem: [BLOCK_M, S_LD]
    const int W_LD  = TILE_N;   // W_smem: [BLOCK_M, W_LD]
    const int O_LD  = D;        // O_smem: [BLOCK_M, O_LD]
    const int PV_LD = D;        // PV_smem: [BLOCK_M, PV_LD]

    // shared memory base (16B aligned)
    extern __shared__ __align__(16) char smem_raw[];
    char* smem_ptr = smem_raw;

    // Q tile: [BLOCK_M, Q_LD] (row-major, half)
    half* Q_smem = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += BLOCK_M * Q_LD * sizeof(half);

    // K tile (double-buffer): [TILE_N, D] (row-major, half)
    //   -> WMMA matrix_b(col_major) 로는 [D, TILE_N] col-major 로 해석
    half* K_smem[2];
    size_t K_tile_elems = (size_t)TILE_N * D;
    size_t K_tile_bytes = K_tile_elems * sizeof(half);
    K_smem[0] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += K_tile_bytes;
    K_smem[1] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += K_tile_bytes;

    // V tile (double-buffer): [TILE_N, V_LD] (row-major, half)
    half* V_smem[2];
    size_t V_tile_elems = (size_t)TILE_N * V_LD;
    size_t V_tile_bytes = V_tile_elems * sizeof(half);
    V_smem[0] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += V_tile_bytes;
    V_smem[1] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += V_tile_bytes;

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

    if (bh >= BH || row_start >= N) return;

    // --- softmax 상태 초기화 ---
    for (int r = tid; r < BLOCK_M; r += blockDim.x) {
        m_smem[r]     = -CUDART_INF_F;
        l_smem[r]     = 0.0f;
        alpha_smem[r] = 0.0f;
    }

    // O_smem 초기화
    for (int idx = tid; idx < BLOCK_M * O_LD; idx += blockDim.x) {
        O_smem[idx] = 0.0f;
    }
    __syncthreads();

    // --- Q tile 로드 (global -> shared, 일반 load) ---
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

    // --- K,V 타일 cp.async 2-stage 파이프라인 ---
    int num_tiles = N / TILE_N;  // N은 TILE_N 배수 가정

    // tile loop: load(tile) + compute(tile-1) 형태
    for (int tile = 0; tile < num_tiles + 1; ++tile) {
        int load_tile    = tile;
        int compute_tile = tile - 1;

        int s_load = load_tile & 1;    // 0 or 1
        int s_comp = compute_tile & 1; // 0 or 1

        // --- load 단계: cp.async로 K/V tile을 shared로 prefetch ---
        if (load_tile < num_tiles) {
            int col_start = load_tile * TILE_N;

            const half* K_gptr = K + ((size_t)bh * N + col_start) * D;  // [TILE_N, D]
            const half* V_gptr = V + ((size_t)bh * N + col_start) * D;  // [TILE_N, D]

            char*       K_smem_bytes = reinterpret_cast<char*>(K_smem[s_load]);
            const char* K_glob_bytes = reinterpret_cast<const char*>(K_gptr);

            char*       V_smem_bytes = reinterpret_cast<char*>(V_smem[s_load]);
            const char* V_glob_bytes = reinterpret_cast<const char*>(V_gptr);

            int total_16B_chunks = (int)(K_tile_bytes / 16);  // == V_tile_bytes/16

            for (int idx = tid; idx < total_16B_chunks; idx += BLOCK_THREADS) {
                size_t byte_offset = (size_t)idx * 16;

                CP_ASYNC_CG(K_smem_bytes + byte_offset, K_glob_bytes + byte_offset);
                CP_ASYNC_CG(V_smem_bytes + byte_offset, V_glob_bytes + byte_offset);
            }

            CP_ASYNC_COMMIT_GROUP();
        }

        // --- compute 단계: 이전 타일에 대해 QKᵀ + softmax + PV 수행 ---
        if (compute_tile >= 0) {
            CP_ASYNC_WAIT_ALL();
            __syncthreads();

            int col_start = compute_tile * TILE_N;
            (void)col_start; // 필요하면 마스크/디버그용으로 사용

            half* K_tile = K_smem[s_comp];  // [TILE_N, D] row-major
            half* V_tile = V_smem[s_comp];  // [TILE_N, D] row-major

            // ------------------------------------------------------
            // 1) QKᵀ via WMMA (Tensor Core)
            //
            //   Q_smem: [BLOCK_M, Q_LD]   (32 x D)
            //   K_tile: [TILE_N, D] row-major
            //     -> WMMA matrix_b (col_major) 로는 [D, TILE_N] 로 해석
            //   -> S_smem: [BLOCK_M, S_LD] float (32 x 32)
            //
            // warp 매핑 (2x2 fragment grid, 16x16):
            //   warp 0: rows 0..15,  cols 0..15
            //   warp 1: rows 16..31, cols 0..15
            //   warp 2: rows 0..15,  cols 16..31
            //   warp 3: rows 16..31, cols 16..31
            // ------------------------------------------------------
            if (warp_id < NUM_WARPS) {
                int warp_m = warp_id / 2; // 0 or 1
                int warp_n = warp_id % 2; // 0 or 1

                int row0 = warp_m * 16;
                int col0 = warp_n * 16;   // 0 or 16 (TILE_N=32)

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> A_frag;
                // B: col_major, ld = D
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::col_major> B_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float>              C_frag;

                wmma::fill_fragment(C_frag, 0.0f);

                for (int kk = 0; kk < D; kk += 16) {
                    const half* a_ptr = &Q_smem[row0 * Q_LD + kk];
                    // K_tile: [TILE_N, D] row-major => [D, TILE_N] col-major with ld = D
                    const half* b_ptr = &K_tile[col0 * D + kk];

                    wmma::load_matrix_sync(A_frag, a_ptr, Q_LD);
                    wmma::load_matrix_sync(B_frag, b_ptr, D);      // ldB = D
                    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
                }

                float* S_tile = &S_smem[row0 * S_LD + col0];
                wmma::store_matrix_sync(S_tile, C_frag, S_LD, wmma::mem_row_major);
            }
            __syncthreads();

            // ------------------------------------------------------
            // 2) 온라인 softmax 업데이트 (row별)
            // ------------------------------------------------------
            const int tile_cols = TILE_N;

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
            }
            __syncthreads();

            // ------------------------------------------------------
            // 5) PV = W * V via WMMA (Tensor Core)
            //
            //   W_smem: [BLOCK_M, W_LD] (32 x 32)
            //   V_tile: [TILE_N, V_LD]  (32 x D)
            //   -> PV_smem: [BLOCK_M, PV_LD]  (32 x D)
            //
            //   OUT_TILE_D = 32 으로 나누고,
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
                            const half* v_ptr = &V_tile[kk * V_LD + col0];

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
        } // if (compute_tile >= 0)

        __syncthreads();
    } // tile loop

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
// 테스트 메인 (cp.async 이식 버전)
// =========================
int main() {
    int BH = 1;
    int N  = 128; // TILE_N=32 배수
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

    int Q_LD  = D;
    int V_LD  = D;
    int S_LD  = TILE_N;
    int W_LD  = TILE_N;
    int O_LD  = D;
    int PV_LD = D;

    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * Q_LD  * sizeof(half);   // Q_smem
    smem_bytes += 2 * (TILE_N * D * sizeof(half));  // K_smem[2]
    smem_bytes += 2 * (TILE_N * V_LD * sizeof(half));// V_smem[2]
    smem_bytes += BLOCK_M * S_LD  * sizeof(float);  // S_smem
    smem_bytes += BLOCK_M * sizeof(float);          // m_smem
    smem_bytes += BLOCK_M * sizeof(float);          // l_smem
    smem_bytes += BLOCK_M * sizeof(float);          // alpha_smem
    smem_bytes += BLOCK_M * W_LD  * sizeof(half);   // W_smem
    smem_bytes += BLOCK_M * O_LD  * sizeof(float);  // O_smem
    smem_bytes += BLOCK_M * PV_LD * sizeof(float);  // PV_smem

    printf("Launching kernel v5 (cp.async) with grid=(%d,%d), block=%d, smem=%zu bytes\n",
           grid.x, grid.y, block.x, smem_bytes);

    int iters = 50;
    for (int i = 0; i < iters; ++i) {
        flashattn_forward_wmma_v5_cp_async_kernel<<<grid, block, smem_bytes>>>(
            dQ, dK, dV, dO, BH, N, D, scale
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, bytes_float, cudaMemcpyDeviceToHost));

    printf("O_v5_cp_async[0,0,0..7]: ");
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
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_wmma_v5_cp_async.cu -o flashattn_wmma_v5_cp_async.exe

# 실행
.\flashattn_wmma_v5_cp_async.exe

# Nsight Compute (31번째 런만 프로파일)
ncu --set full --launch-skip 30 --launch-count 1 .\flashattn_wmma_v5_cp_async.exe
*/
