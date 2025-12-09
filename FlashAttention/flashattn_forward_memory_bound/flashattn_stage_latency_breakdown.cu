// flashattn_stage_latency_breakdown.cu
//
// 5.6.6: Stage-by-Stage Latency 분석용 마이크로 벤치마크
//
// - Stage A: QKᵀ Tensor Core GEMM (WMMA)
// - Stage B: Softmax (row-wise, numerically stable)
// - Stage C: PV Tensor Core GEMM (WMMA)
//
// 각 스테이지를 독립 커널로 분리해서
//   - 평균 kernel time
//   - FLOPs / TFLOPS
// 를 출력해서 “어디서 시간이 새는지”를 확인하는 용도.
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_stage_latency_breakdown.cu -o flashattn_stage_latency.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_stage_latency.exe
//

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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

constexpr int WARP_SIZE     = 32;
constexpr int BLOCK_THREADS = 128;

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// =============================
// 1. WMMA Tensor Core GEMM 커널
// =============================
//
// C[M,N] = A[M,K] * B[K,N]
// A,B: half (row-major)
// C  : float (row-major)
//
// M,N,K는 모두 16의 배수라고 가정.
//
__global__ void wmma_gemm_tensorcore_kernel(
    const half* __restrict__ A,  // [M,K], row-major
    const half* __restrict__ B,  // [K,N], row-major
    float* __restrict__ C,       // [M,N], row-major
    int M,
    int N,
    int K
) {
#if __CUDA_ARCH__ < 700
    return; // Tensor Core 미지원 아키텍처에서는 아무 것도 안 함
#endif

    // block당 4 warp, warp당 16x16 tile 하나 처리
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int warps_per_block = blockDim.x / WARP_SIZE;

    // grid.x: N 방향 타일, grid.y: M 방향 타일
    int tile_m = blockIdx.y * warps_per_block + warp_id; // tile row idx
    int tile_n = blockIdx.x;                             // tile col idx

    int row = tile_m * 16;
    int col = tile_n * 16;

    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // K 방향으로 16씩 진행
    for (int k = 0; k < K; k += 16) {
        const half* a_tile = A + row * K + k; // [row, k]
        const half* b_tile = B + k * N + col; // [k, col]

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + row * N + col;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

// ===================================
// 2. Softmax bottleneck (row-wise)
// ===================================
//
// Scores: [NUM_ROWS, K_tot]
// 각 row에 대해:
//   m = max_j scores
//   l = sum_j exp(scores - m)
//   scores = exp(scores - m) / l
//
__global__ void softmax_rowwise_kernel(
    float* __restrict__ Scores, // [num_rows, K_tot]
    float* __restrict__ m_out,  // [num_rows]
    float* __restrict__ l_out,  // [num_rows]
    int num_rows,
    int K_tot
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) return;

    float* row_ptr = Scores + (size_t)row * K_tot;

    __shared__ float sdata[BLOCK_THREADS];

    // 1) row-wise max
    float local_max = -1e30f;
    for (int j = tid; j < K_tot; j += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[j]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float m = sdata[0];
    if (tid == 0) m_out[row] = m;
    __syncthreads();

    // 2) exp + sum
    float local_sum = 0.0f;
    for (int j = tid; j < K_tot; j += blockDim.x) {
        float e = expf(row_ptr[j] - m);
        row_ptr[j] = e; // 나중에 normalize에 사용
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float l = sdata[0];
    if (tid == 0) l_out[row] = l;
    __syncthreads();

    // 3) normalize
    for (int j = tid; j < K_tot; j += blockDim.x) {
        row_ptr[j] = row_ptr[j] / l;
    }
}

// ========================
// 3. 타이밍 helper 함수
// ========================
template <typename Kernel, typename... Args>
float benchmark_kernel(int iters, Kernel kernel, dim3 grid, dim3 block, size_t smem, Args... args) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 5; ++i) {
        kernel<<<grid, block, smem>>>(args...);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block, smem>>>(args...);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iters;
}

// ==========
// 4. main()
// ==========
int main() {
    // FlashAttention-like 스케일 (하나의 예시)
    // Stage A: QKᵀ  ~ [N,D] x [D,N] = [N,N]
    // Stage B: Softmax on [NUM_ROWS, K_tot] ~ [BH*N, N]
    // Stage C: PV    ~ [N,N] x [N,D] = [N,D]

    const int N   = 512;
    const int D   = 64;
    const int BH  = 4;

    // Stage A (QKᵀ GEMM) => M = N, Ncol = N, Kdim = D
    const int M_qk = N;
    const int N_qk = N;
    const int K_qk = D;

    // Stage B (softmax) => NUM_ROWS = BH * N, K_tot = N
    const int NUM_ROWS = BH * N;  // 4 * 512 = 2048
    const int K_tot    = N;

    // Stage C (PV GEMM) => M = N, Ncol = D, Kdim = N
    const int M_pv = N;
    const int N_pv = D;
    const int K_pv = N;

    printf("Stage-by-Stage Latency Breakdown (N=%d, D=%d, BH=%d)\n", N, D, BH);
    printf("  Stage A: QK^T   -> GEMM [%d x %d] x [%d x %d]\n", M_qk, K_qk, K_qk, N_qk);
    printf("  Stage B: softmax -> matrix [%d x %d]\n", NUM_ROWS, K_tot);
    printf("  Stage C: PV     -> GEMM [%d x %d] x [%d x %d]\n\n", M_pv, K_pv, K_pv, N_pv);

    // ----------------------
    // Stage A: QKᵀ GEMM
    // ----------------------
    size_t bytes_A_qk = (size_t)M_qk * K_qk * sizeof(half);
    size_t bytes_B_qk = (size_t)K_qk * N_qk * sizeof(half);
    size_t bytes_C_qk = (size_t)M_qk * N_qk * sizeof(float);

    std::vector<half>  hA_qk(M_qk * K_qk);
    std::vector<half>  hB_qk(K_qk * N_qk);
    std::vector<float> hC_qk(M_qk * N_qk);

    // 간단한 deterministic init
    for (size_t i = 0; i < hA_qk.size(); ++i) {
        float v = (float)((i % 13) - 6) * 0.01f;
        hA_qk[i] = __float2half(v);
    }
    for (size_t i = 0; i < hB_qk.size(); ++i) {
        float v = (float)((i % 17) - 8) * 0.01f;
        hB_qk[i] = __float2half(v);
    }

    half  *dA_qk, *dB_qk;
    float *dC_qk;
    CHECK_CUDA(cudaMalloc(&dA_qk, bytes_A_qk));
    CHECK_CUDA(cudaMalloc(&dB_qk, bytes_B_qk));
    CHECK_CUDA(cudaMalloc(&dC_qk, bytes_C_qk));

    CHECK_CUDA(cudaMemcpy(dA_qk, hA_qk.data(), bytes_A_qk, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_qk, hB_qk.data(), bytes_B_qk, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_qk, 0, bytes_C_qk));

    dim3 block_tc(BLOCK_THREADS, 1, 1);
    int warps_per_block = BLOCK_THREADS / WARP_SIZE;
    dim3 grid_qk(ceil_div(N_qk, 16), ceil_div(M_qk, 16 * warps_per_block), 1);

    printf("Stage A (QK^T GEMM, Tensor Core):\n");
    printf("  grid=(%d,%d), block=%d\n", grid_qk.x, grid_qk.y, block_tc.x);

    float avg_ms_qk = benchmark_kernel(
        100,
        wmma_gemm_tensorcore_kernel,
        grid_qk,
        block_tc,
        0,
        dA_qk, dB_qk, dC_qk, M_qk, N_qk, K_qk
    );

    CHECK_CUDA(cudaMemcpy(hC_qk.data(), dC_qk, bytes_C_qk, cudaMemcpyDeviceToHost));

    printf("  C_qk[0,0..7]: ");
    for (int j = 0; j < 8; ++j) {
        printf("%f ", hC_qk[j]);
    }
    printf("\n");

    double flops_qk = 2.0 * (double)M_qk * N_qk * K_qk;
    double tflops_qk = flops_qk / (avg_ms_qk * 1e-3) / 1e12;

    printf("  Avg kernel time: %.3f ms\n", avg_ms_qk);
    printf("  Approx FLOPs   : %.3e\n", flops_qk);
    printf("  Approx TFLOPS  : %.3f\n\n", tflops_qk);

    // ----------------------
    // Stage B: Softmax
    // ----------------------
    size_t bytes_scores = (size_t)NUM_ROWS * K_tot * sizeof(float);
    size_t bytes_vec    = (size_t)NUM_ROWS * sizeof(float);

    std::vector<float> hScores(NUM_ROWS * K_tot);
    std::vector<float> hM(NUM_ROWS), hL(NUM_ROWS);

    for (size_t i = 0; i < hScores.size(); ++i) {
        float v = (float)((i % 23) - 11) * 0.01f;
        hScores[i] = v;
    }

    float *dScores, *dM, *dL;
    CHECK_CUDA(cudaMalloc(&dScores, bytes_scores));
    CHECK_CUDA(cudaMalloc(&dM, bytes_vec));
    CHECK_CUDA(cudaMalloc(&dL, bytes_vec));

    CHECK_CUDA(cudaMemcpy(dScores, hScores.data(), bytes_scores, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dM, 0, bytes_vec));
    CHECK_CUDA(cudaMemset(dL, 0, bytes_vec));

    dim3 grid_soft(NUM_ROWS, 1, 1);
    dim3 block_soft(BLOCK_THREADS, 1, 1);

    printf("Stage B (Softmax row-wise bottleneck):\n");
    printf("  grid=(%d,%d), block=%d\n", grid_soft.x, grid_soft.y, block_soft.x);

    float avg_ms_soft = benchmark_kernel(
        50,
        softmax_rowwise_kernel,
        grid_soft,
        block_soft,
        0,
        dScores, dM, dL, NUM_ROWS, K_tot
    );

    CHECK_CUDA(cudaMemcpy(hScores.data(), dScores, bytes_scores, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hM.data(),      dM,      bytes_vec,    cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hL.data(),      dL,      bytes_vec,    cudaMemcpyDeviceToHost));

    printf("  Row[0..3] m/l:\n");
    for (int r = 0; r < 4; ++r) {
        printf("    row %d: m=%f, l=%f\n", r, hM[r], hL[r]);
    }

    // softmax FLOPs 대략: (max + exp + sum + div) ~ 6 * NUM_ROWS * K_tot
    double flops_soft = 6.0 * (double)NUM_ROWS * K_tot;
    double tflops_soft = flops_soft / (avg_ms_soft * 1e-3) / 1e12;

    printf("  Avg kernel time: %.3f ms\n", avg_ms_soft);
    printf("  Approx FLOPs   : %.3e (rough)\n", flops_soft);
    printf("  Approx TFLOPS  : %.3f\n\n", tflops_soft);

    // ----------------------
    // Stage C: PV GEMM
    // ----------------------
    size_t bytes_A_pv = (size_t)M_pv * K_pv * sizeof(half);
    size_t bytes_B_pv = (size_t)K_pv * N_pv * sizeof(half);
    size_t bytes_C_pv = (size_t)M_pv * N_pv * sizeof(float);

    std::vector<half>  hA_pv(M_pv * K_pv);
    std::vector<half>  hB_pv(K_pv * N_pv);
    std::vector<float> hC_pv(M_pv * N_pv);

    for (size_t i = 0; i < hA_pv.size(); ++i) {
        float v = (float)((i % 19) - 9) * 0.01f;
        hA_pv[i] = __float2half(v);
    }
    for (size_t i = 0; i < hB_pv.size(); ++i) {
        float v = (float)((i % 29) - 14) * 0.01f;
        hB_pv[i] = __float2half(v);
    }

    half  *dA_pv, *dB_pv;
    float *dC_pv;
    CHECK_CUDA(cudaMalloc(&dA_pv, bytes_A_pv));
    CHECK_CUDA(cudaMalloc(&dB_pv, bytes_B_pv));
    CHECK_CUDA(cudaMalloc(&dC_pv, bytes_C_pv));

    CHECK_CUDA(cudaMemcpy(dA_pv, hA_pv.data(), bytes_A_pv, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_pv, hB_pv.data(), bytes_B_pv, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_pv, 0, bytes_C_pv));

    dim3 grid_pv(ceil_div(N_pv, 16), ceil_div(M_pv, 16 * warps_per_block), 1);

    printf("Stage C (PV GEMM, Tensor Core):\n");
    printf("  grid=(%d,%d), block=%d\n", grid_pv.x, grid_pv.y, block_tc.x);

    float avg_ms_pv = benchmark_kernel(
        100,
        wmma_gemm_tensorcore_kernel,
        grid_pv,
        block_tc,
        0,
        dA_pv, dB_pv, dC_pv, M_pv, N_pv, K_pv
    );

    CHECK_CUDA(cudaMemcpy(hC_pv.data(), dC_pv, bytes_C_pv, cudaMemcpyDeviceToHost));

    printf("  C_pv[0,0..7]: ");
    for (int j = 0; j < 8 && j < M_pv * N_pv; ++j) {
        printf("%f ", hC_pv[j]);
    }
    printf("\n");

    double flops_pv = 2.0 * (double)M_pv * N_pv * K_pv;
    double tflops_pv = flops_pv / (avg_ms_pv * 1e-3) / 1e12;

    printf("  Avg kernel time: %.3f ms\n", avg_ms_pv);
    printf("  Approx FLOPs   : %.3e\n", flops_pv);
    printf("  Approx TFLOPS  : %.3f\n\n", tflops_pv);

    // ----------------------
    // 전체 요약
    // ----------------------
    printf("=== Stage Latency Summary ===\n");
    printf("  Stage A (QK^T GEMM):   %.3f ms, %.3f TFLOPS\n", avg_ms_qk,  tflops_qk);
    printf("  Stage B (Softmax):     %.3f ms, %.3f TFLOPS (rough)\n", avg_ms_soft, tflops_soft);
    printf("  Stage C (PV GEMM):     %.3f ms, %.3f TFLOPS\n", avg_ms_pv,  tflops_pv);
    printf("=============================\n");

    CHECK_CUDA(cudaFree(dA_qk));
    CHECK_CUDA(cudaFree(dB_qk));
    CHECK_CUDA(cudaFree(dC_qk));
    CHECK_CUDA(cudaFree(dScores));
    CHECK_CUDA(cudaFree(dM));
    CHECK_CUDA(cudaFree(dL));
    CHECK_CUDA(cudaFree(dA_pv));
    CHECK_CUDA(cudaFree(dB_pv));
    CHECK_CUDA(cudaFree(dC_pv));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_stage_latency_breakdown.cu -o flashattn_stage_latency.exe

# 실행
.\flashattn_stage_latency.exe

# Nsight Compute 예시 (Stage별 stall / IPC / roofline 관찰)
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_stage_latency.exe
*/
