// flashattn_forward_fused_5_4_2.cu
// 5.4.2 - Full FlashAttention Forward (FMA + shared double-buffer, numerically correct version)
//
// Q, K, V: [BH, N, D] (float32, row-major)
// O      : [BH, N, D] (float32)
//
// Stage1: QK^T tile
// Stage2: Incremental softmax (row_max, row_sum)
// Stage3: softmax weights * V tile → O
//
// 5.4.1에서 누락됐던 부분:
//  - row_max / row_sum 갱신 시, 이전까지 누적된 O_local도 동일한 스케일 팩터로 재조정해야 한다.
//  - alpha = exp(m_old - new_m) * l_old / new_l
//  - O_new = alpha * O_old + (1/new_l) * Σ_j exp(s_j - new_m) V_j

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int BLOCK_M = 32;   // query rows per block
constexpr int TILE_N  = 32;   // key/value columns per tile
constexpr int MAX_D   = 64;   // head_dim upper bound for this sample kernel

// --- Full FlashAttention Forward fused kernel (FMA version, numerically correct) ---
// Q, K, V, O: [BH, N, D], row-major
__global__ void flashattn_forward_fused_kernel_5_4_2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH, int N, int D,
    float scale  // 1/sqrt(D)
)
{
    if (D > MAX_D) return;  // sample guard

    // block mapping
    //  blockIdx.y → bh (0..BH-1)
    //  blockIdx.x → q_block index in [0, ceil(N/BLOCK_M))
    int bh      = blockIdx.y;
    int q_block = blockIdx.x;
    int q_start = q_block * BLOCK_M;

    if (bh >= BH || q_start >= N) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;  // e.g., 128

    // Shared memory layout
    extern __shared__ float smem[];
    float* Q_smem = smem;                                     // [BLOCK_M, MAX_D]
    float* K_smem = Q_smem + BLOCK_M * MAX_D;                 // [2, TILE_N, MAX_D] double-buffer
    float* V_smem = K_smem + 2 * TILE_N * MAX_D;              // [2, TILE_N, MAX_D]

    __shared__ float row_max[BLOCK_M];
    __shared__ float row_sum[BLOCK_M];

    // 1. Q block load: [bh, q_start : q_start+BLOCK_M, :]
    const float* Q_base = Q + static_cast<size_t>(bh) * N * D;
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        int r = i / D;   // row in [0, BLOCK_M)
        int c = i % D;
        int q_idx = q_start + r;
        if (q_idx < N && c < D) {
            Q_smem[r * MAX_D + c] = Q_base[q_idx * D + c];
        } else {
            Q_smem[r * MAX_D + c] = 0.0f;
        }
    }

    // row_max / row_sum 초기화
    for (int r = tid; r < BLOCK_M; r += num_threads) {
        row_max[r] = -INFINITY;
        row_sum[r] = 0.0f;
    }
    __syncthreads();

    // 각 thread가 담당할 row 구간
    int rows_per_thread = (BLOCK_M + num_threads - 1) / num_threads;
    int row_begin = tid * rows_per_thread;
    int row_end   = min(BLOCK_M, row_begin + rows_per_thread);

    // per-thread local accumulator for O: [row, D]
    float O_local[BLOCK_M][MAX_D];
    for (int r = row_begin; r < row_end; ++r) {
        for (int c = 0; c < D; ++c) {
            O_local[r][c] = 0.0f;
        }
    }

    const float* K_base = K + static_cast<size_t>(bh) * N * D;
    const float* V_base = V + static_cast<size_t>(bh) * N * D;

    int stage = 0;

    // 2. Loop over key/value tiles along sequence dimension
    for (int k0 = 0; k0 < N; k0 += TILE_N) {
        int tile_cols = min(TILE_N, N - k0);
        int next_stage = stage ^ 1;

        // 2-1. Load K/V tile into shared memory (double buffer)
        for (int i = tid; i < tile_cols * D; i += num_threads) {
            int row = i / D;  // 0..tile_cols-1
            int col = i % D;

            int kv_idx = k0 + row;
            float kv = K_base[kv_idx * D + col];
            float vv = V_base[kv_idx * D + col];

            K_smem[stage * TILE_N * MAX_D + row * MAX_D + col] = kv;
            V_smem[stage * TILE_N * MAX_D + row * MAX_D + col] = vv;
        }
        __syncthreads();

        // 2-2. Stage1 + Stage2 + Stage3 for rows handled by this thread
        for (int r = row_begin; r < row_end; ++r) {
            int q_idx = q_start + r;
            if (q_idx >= N) continue;

            // Q row를 레지스터에
            float q_vec[MAX_D];
            for (int c = 0; c < D; ++c) {
                q_vec[c] = Q_smem[r * MAX_D + c];
            }

            // 2-2-1. scores for this row and current tile
            float scores[TILE_N];
            for (int j = 0; j < tile_cols; ++j) {
                float acc = 0.0f;
                float* K_row_ptr = &K_smem[stage * TILE_N * MAX_D + j * MAX_D];
                for (int c = 0; c < D; ++c) {
                    acc += q_vec[c] * K_row_ptr[c];
                }
                scores[j] = acc * scale;  // QK^T / sqrt(D)
            }

            // --- Incremental Softmax (수학적으로 정확한 버전) ---

            // 기존 상태
            float old_m = row_max[r];
            float old_l = row_sum[r];

            // 새로운 타일의 최대값
            float tile_m = -INFINITY;
            for (int j = 0; j < tile_cols; ++j) {
                tile_m = fmaxf(tile_m, scores[j]);
            }

            float new_m = fmaxf(old_m, tile_m);

            // 이전 기여분 재스케일을 위한 exp_scale = exp(old_m - new_m) * old_l
            float exp_scale;
            if (old_l == 0.0f && old_m == -INFINITY) {
                exp_scale = 0.0f;
            } else {
                exp_scale = expf(old_m - new_m) * old_l;
            }

            // 새 타일의 e^{s_j - new_m} 합
            float tile_exp_sum = 0.0f;
            float exp_scores[TILE_N];
            for (int j = 0; j < tile_cols; ++j) {
                float e = expf(scores[j] - new_m);
                exp_scores[j] = e;
                tile_exp_sum += e;
            }

            float new_l = exp_scale + tile_exp_sum;

            // --- 여기서 O_local도 함께 재스케일해야 함 ---
            if (new_l > 0.0f) {
                float alpha = (new_l > 0.0f) ? (exp_scale / new_l) : 0.0f;
                // 기존까지의 O_old는 softmax_old 기준
                // 새 기준(new_m, new_l)에서 old 기여분은 alpha * O_old 로 들어간다
                for (int c = 0; c < D; ++c) {
                    O_local[r][c] *= alpha;
                }

                // 새 타일 기여: (1/new_l) * Σ_j exp(scores_j - new_m) * V_j
                for (int j = 0; j < tile_cols; ++j) {
                    float w = exp_scores[j] / new_l;  // w_ij
                    float* V_row_ptr = &V_smem[stage * TILE_N * MAX_D + j * MAX_D];
                    for (int c = 0; c < D; ++c) {
                        O_local[r][c] += w * V_row_ptr[c];
                    }
                }
            }

            // row-wise softmax state 업데이트
            row_max[r] = new_m;
            row_sum[r] = new_l;
        }

        __syncthreads();
        stage = next_stage;
    }

    // 3. Store O_local to global O
    float* O_base = O + static_cast<size_t>(bh) * N * D;
    for (int r = row_begin; r < row_end; ++r) {
        int q_idx = q_start + r;
        if (q_idx >= N) continue;
        for (int c = 0; c < D; ++c) {
            O_base[q_idx * D + c] = O_local[r][c];
        }
    }
}

// ---- CPU reference attention: naive implementation ----
// Q, K, V: [BH, N, D], float
void flashattn_cpu_ref(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int BH, int N, int D
)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int bh = 0; bh < BH; ++bh) {
        const float* Qb = Q.data() + static_cast<size_t>(bh) * N * D;
        const float* Kb = K.data() + static_cast<size_t>(bh) * N * D;
        const float* Vb = V.data() + static_cast<size_t>(bh) * N * D;
        float*       Ob = O.data() + static_cast<size_t>(bh) * N * D;

        for (int i = 0; i < N; ++i) {
            // 1) scores
            std::vector<float> scores(N);
            float max_s = -INFINITY;
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < D; ++k) {
                    acc += static_cast<double>(Qb[i*D + k]) * Kb[j*D + k];
                }
                acc *= scale;
                scores[j] = static_cast<float>(acc);
                max_s = std::max(max_s, scores[j]);
            }

            // 2) softmax
            double sum_e = 0.0;
            for (int j = 0; j < N; ++j) {
                sum_e += std::exp(static_cast<double>(scores[j] - max_s));
            }

            // 3) output
            for (int k = 0; k < D; ++k) {
                double out = 0.0;
                for (int j = 0; j < N; ++j) {
                    double w = std::exp(static_cast<double>(scores[j] - max_s)) / sum_e;
                    out += w * Vb[j*D + k];
                }
                Ob[i*D + k] = static_cast<float>(out);
            }
        }
    }
}

int main() {
    // ---- 테스트 파라미터 ----
    int BH = 2;   // B * H
    int N  = 128; // sequence length
    int D  = 64;  // head dim (<= MAX_D)

    if (D > MAX_D) {
        printf("D must be <= %d for this sample. Got D=%d\n", MAX_D, D);
        return 0;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    printf("FlashAttention Forward Fused 5.4.2 (FMA + shared double-buffer, numerically correct)\n");
    printf("BH=%d, N=%d, D=%d\n", BH, N, D);
    printf("BLOCK_M=%d, TILE_N=%d\n", BLOCK_M, TILE_N);

    size_t size_Q = static_cast<size_t>(BH) * N * D;
    size_t size_K = static_cast<size_t>(BH) * N * D;
    size_t size_V = static_cast<size_t>(BH) * N * D;
    size_t size_O = static_cast<size_t>(BH) * N * D;

    std::vector<float> h_Q(size_Q);
    std::vector<float> h_K(size_K);
    std::vector<float> h_V(size_V);
    std::vector<float> h_O(size_O);
    std::vector<float> h_O_ref(size_O);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < size_Q; ++i) h_Q[i] = dist(rng);
    for (size_t i = 0; i < size_K; ++i) h_K[i] = dist(rng);
    for (size_t i = 0; i < size_V; ++i) h_V[i] = dist(rng);

    // Device 메모리
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, size_Q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, size_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, size_V * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, size_O * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), size_Q * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), size_K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), size_V * sizeof(float), cudaMemcpyHostToDevice));

    // 커널 설정
    dim3 block(128, 1, 1);
    dim3 grid(
        (N + BLOCK_M - 1) / BLOCK_M,
        BH,
        1
    );

    size_t smem_bytes =
        BLOCK_M * MAX_D * sizeof(float) +   // Q_smem
        2 * TILE_N * MAX_D * sizeof(float) +// K_smem
        2 * TILE_N * MAX_D * sizeof(float); // V_smem

    printf("Grid = (%d, %d, %d), Block = (%d, %d, %d)\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);
    printf("Shared mem = %zu bytes\n", smem_bytes);

    // Warm-up
    flashattn_forward_fused_kernel_5_4_2<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O,
        BH, N, D, scale
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // 시간 측정
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    flashattn_forward_fused_kernel_5_4_2<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O,
        BH, N, D, scale
    );
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, size_O * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    printf("Running CPU reference attention...\n");
    flashattn_cpu_ref(h_Q, h_K, h_V, h_O_ref, BH, N, D);

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < size_O; ++i) {
        double diff = std::fabs(static_cast<double>(h_O[i]) - h_O_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    // FLOP 계산: QK^T (2*N*N*D) + softmax + P×V(2*N*N*D) ≈ 4*N*N*D*BH
    double flop = 4.0 * static_cast<double>(BH) * N * N * D;
    double gflops = flop / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS (approx): %.3f\n", ms, gflops);
    printf("Max abs diff vs CPU: %.6e\n", max_abs_diff);

    // 정리
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 flashattn_forward_fused_5_4_2.cu -o flashattn_forward_fused_5_4_2.exe

ncu --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:flashattn_forward_fused_kernel_5_4_2     --section ComputeWorkloadAnalysis     --section MemoryWorkloadAnalysis     --section SpeedOfLight_RooflineChart     .\flashattn_forward_fused_5_4_2.exe

.\flashattn_forward_fused_5_4_2.exe

*/