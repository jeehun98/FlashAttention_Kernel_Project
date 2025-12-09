// flashattn_warp_stall_breakdown.cu
//
// 5.6.5: Warp Stall Breakdown 마이크로 커널
//
// - FlashAttention에서 softmax / tile pipeline 구현 시 생기는 warp stall을
//   단순화된 형태로 재현하는 예제.
// - 동일한 연산량을 가진 두 커널을 비교:
//   (1) block-wide __syncthreads()를 매 iteration마다 사용하는 버전
//       -> Warp Stall: Barrier 가 크게 잡힘
//   (2) warp-local only 버전 (각 warp 독립, __syncthreads() 미사용)
//       -> Warp Stall: Execution Dependency 중심
//
// Nsight Compute로 두 커널을 각각 프로파일링하여
// Warp State Statistics 의 stall breakdown을 비교하는 것이 목적.
//
// 빌드 예시 (Windows PowerShell, Ampere/그 이상/그 이하 모두 가능):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_warp_stall_breakdown.cu -o flashattn_warp_stall_breakdown.exe
//
// 실행:
//   .\flashattn_warp_stall_breakdown.exe
//
// Nsight Compute 예시:
//   # barrier-heavy 버전만 프로파일
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_warp_stall_breakdown.exe --kernel-type barrier
//
//   # warp-only 버전만 프로파일
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_warp_stall_breakdown.exe --kernel-type warp
//
// (커맨드라인 인자에 따라 하나만 여러 번 실행하도록 구현함)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

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
constexpr int BLOCK_THREADS = 128;   // 4 warps/block

// row 개수 (FlashAttention의 BH*N에 대응)
constexpr int NUM_ROWS = 4096;

// softmax-like 반복 횟수 (K_tot 역할, 연산 chain 길이)
constexpr int K_TOT = 512;

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// ===============================================
// 1) Barrier-heavy 버전
//    - block-wide __syncthreads() 를 매 iteration마다 사용
//    - warp마다 work 양이 약간 다르도록 해서 barrier stall 유도
// ===============================================
__global__ void softmax_barrier_kernel(
    float* __restrict__ Out_acc,
    float* __restrict__ Out_m,
    float* __restrict__ Out_l,
    int num_rows,
    int k_tot
) {
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;      // 0..3
    int lane_id = tid % WARP_SIZE;

    int row_per_block = blockDim.x;     // 128
    int row_start     = blockIdx.x * row_per_block;
    int row_id        = row_start + tid;

    if (row_id >= num_rows) {
        return;
    }

    // softmax 상태
    float m = -1e9f;
    float l = 0.0f;
    float acc = 0.0f;

    // 공유 메모리: barrier를 통해 warp 간 의존성 추가
    extern __shared__ float smem[];
    float* tmp_buf = smem; // 크기 >= BLOCK_THREADS

    for (int k = 0; k < k_tot; ++k) {
        // (a) score/value 생성 (global load 대신 deterministic 함수)
        //     row_id와 k에 따라 pseudo-random-like 값 생성
        float s = 0.01f * (float)((row_id * 131 + k * 17) % 113);
        float v = 0.02f * (float)((row_id * 17  + k * 31) % 97);

        // (b) warp별로 일부 "쓸데없는 work"를 추가해서
        //     barrier 직전 warp 간 도달 시간 차이를 유도
        //     (warp_id가 클수록 더 많은 dummy 연산 수행)
        int extra_iters = (warp_id + 1) * 8;  // 8, 16, 24, 32
        float dummy = s;
        for (int it = 0; it < extra_iters; ++it) {
            dummy = dummy * 1.0001f + 0.0001f;
        }
        // dummy를 softmax 계산에 섞어서 dead-code 제거 방지
        s += 0.001f * dummy;

        // (c) 한 번의 block-wide barrier
        //     -> 일부 warp는 빨리 끝나고 대기, 일부는 늦게 도착
        tmp_buf[tid] = s;
        __syncthreads();

        // (d) 온라인 softmax 업데이트 (row별)
        //     (FlashAttention에서 사용하는 형태와 동일한 구조)
        float m_old = m;
        float l_old = l;

        float s_scaled = s * 0.1f;  // scale ~ 1/sqrt(D) 흉내
        float m_new = fmaxf(m_old, s_scaled);

        float exp_old = (m_old <= -1e8f) ? 0.0f : expf(m_old - m_new) * l_old;
        float exp_new = expf(s_scaled - m_new);
        float l_new   = exp_old + exp_new;

        float alpha = (l_new > 0.0f && l_old > 0.0f) ? (exp_old / l_new) : 0.0f;
        float p_new = (l_new > 0.0f) ? (exp_new / l_new) : 0.0f;

        acc = alpha * acc + p_new * v;
        m   = m_new;
        l   = l_new;

        // (e) 또 barrier
        //     -> 한 iteration당 barrier 2개: warp stall barrier 유도
        __syncthreads();
    }

    // 결과 저장
    Out_acc[row_id] = acc;
    Out_m[row_id]   = m;
    Out_l[row_id]   = l;
}

// ===============================================
// 2) Warp-local only 버전
//    - block은 동일하게 128 threads이지만,
//      각 warp가 독립적으로 softmax를 수행 (warp-scope row)
//    - __syncthreads() 없이, warp마다 1개의 row를 처리
//    - barrier stall은 사라지고, Execution Dependency 위주로 측정
// ===============================================
__global__ void softmax_warp_local_kernel(
    float* __restrict__ Out_acc,
    float* __restrict__ Out_m,
    float* __restrict__ Out_l,
    int num_rows,
    int k_tot
) {
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;      // 0..3
    int lane_id = tid % WARP_SIZE;

    // block당 4 rows (warp당 한 row)
    int row_per_block = blockDim.x / WARP_SIZE; // 4
    int row_start     = blockIdx.x * row_per_block;

    int row_id = row_start + warp_id;
    if (row_id >= num_rows) return;

    // lane 0만 실제 softmax 상태를 업데이트,
    // 나머지 lanes는 동일한 상태를 share한다고 가정 (병목만 보기 위함)
    float m   = -1e9f;
    float l   = 0.0f;
    float acc = 0.0f;

    for (int k = 0; k < k_tot; ++k) {
        float s = 0.01f * (float)((row_id * 131 + k * 17) % 113);
        float v = 0.02f * (float)((row_id * 17  + k * 31) % 97);

        // warp 별로 조금 다른 dummy work (barrier 없음)
        int extra_iters = (warp_id + 1) * 8;
        float dummy = s;
        for (int it = 0; it < extra_iters; ++it) {
            dummy = dummy * 1.0001f + 0.0001f;
        }
        s += 0.001f * dummy;

        // lane 0만 실제 softmax 상태를 업데이트
        if (lane_id == 0) {
            float m_old = m;
            float l_old = l;

            float s_scaled = s * 0.1f;
            float m_new = fmaxf(m_old, s_scaled);

            float exp_old = (m_old <= -1e8f) ? 0.0f : expf(m_old - m_new) * l_old;
            float exp_new = expf(s_scaled - m_new);
            float l_new   = exp_old + exp_new;

            float alpha = (l_new > 0.0f && l_old > 0.0f) ? (exp_old / l_new) : 0.0f;
            float p_new = (l_new > 0.0f) ? (exp_new / l_new) : 0.0f;

            acc = alpha * acc + p_new * v;
            m   = m_new;
            l   = l_new;
        }

        // warp 내 broadcast (굳이 필요하진 않지만 형태적으로 맞춰줌)
        unsigned mask = 0xffffffffu;
        m   = __shfl_sync(mask, m,   0);
        l   = __shfl_sync(mask, l,   0);
        acc = __shfl_sync(mask, acc, 0);
    }

    if (lane_id == 0) {
        Out_acc[row_id] = acc;
        Out_m[row_id]   = m;
        Out_l[row_id]   = l;
    }
}

// ===============================================
// 메인: 두 커널을 각각 여러 번 실행, 시간 비교
// ===============================================
int main(int argc, char** argv) {
    bool run_barrier = true;
    bool run_warp    = true;

    // 간단한 커맨드라인 옵션:
    //   --kernel-type barrier
    //   --kernel-type warp
    if (argc >= 3 && std::strcmp(argv[1], "--kernel-type") == 0) {
        if (std::strcmp(argv[2], "barrier") == 0) {
            run_barrier = true;
            run_warp    = false;
        } else if (std::strcmp(argv[2], "warp") == 0) {
            run_barrier = false;
            run_warp    = true;
        }
    }

    int num_rows = NUM_ROWS;
    int k_tot    = K_TOT;

    size_t rows_bytes = (size_t)num_rows * sizeof(float);

    float *dAcc_barrier, *dM_barrier, *dL_barrier;
    float *dAcc_warp,    *dM_warp,    *dL_warp;

    CHECK_CUDA(cudaMalloc(&dAcc_barrier, rows_bytes));
    CHECK_CUDA(cudaMalloc(&dM_barrier,   rows_bytes));
    CHECK_CUDA(cudaMalloc(&dL_barrier,   rows_bytes));

    CHECK_CUDA(cudaMalloc(&dAcc_warp, rows_bytes));
    CHECK_CUDA(cudaMalloc(&dM_warp,   rows_bytes));
    CHECK_CUDA(cudaMalloc(&dL_warp,   rows_bytes));

    CHECK_CUDA(cudaMemset(dAcc_barrier, 0, rows_bytes));
    CHECK_CUDA(cudaMemset(dM_barrier,   0, rows_bytes));
    CHECK_CUDA(cudaMemset(dL_barrier,   0, rows_bytes));

    CHECK_CUDA(cudaMemset(dAcc_warp, 0, rows_bytes));
    CHECK_CUDA(cudaMemset(dM_warp,   0, rows_bytes));
    CHECK_CUDA(cudaMemset(dL_warp,   0, rows_bytes));

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid_barrier(ceil_div(num_rows, BLOCK_THREADS), 1, 1);
    // warp 버전은 block당 4 rows (warp 4개)
    dim3 grid_warp(ceil_div(num_rows, BLOCK_THREADS / WARP_SIZE), 1, 1);

    int iters = 50;

    printf("Softmax warp stall breakdown micro-kernel\n");
    printf("NUM_ROWS=%d, K_TOT=%d\n", num_rows, k_tot);
    printf("Barrier kernel: grid=(%d,1), block=%d\n", grid_barrier.x, block.x);
    printf("Warp-local kernel: grid=(%d,1), block=%d\n", grid_warp.x, block.x);

    // --- barrier 버전 ---
    if (run_barrier) {
        size_t smem_bytes = BLOCK_THREADS * sizeof(float); // tmp_buf 용

        printf("\n[Barrier kernel] Running %d iterations...\n", iters);

        // warmup
        for (int i = 0; i < 10; ++i) {
            softmax_barrier_kernel<<<grid_barrier, block, smem_bytes>>>(
                dAcc_barrier, dM_barrier, dL_barrier, num_rows, k_tot
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            softmax_barrier_kernel<<<grid_barrier, block, smem_bytes>>>(
                dAcc_barrier, dM_barrier, dL_barrier, num_rows, k_tot
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;
        printf("[Barrier] Avg kernel time: %.3f ms\n", avg_ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        // 일부 row 출력
        std::vector<float> hAcc(num_rows), hM(num_rows), hL(num_rows);
        CHECK_CUDA(cudaMemcpy(hAcc.data(), dAcc_barrier, rows_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hM.data(),   dM_barrier,   rows_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hL.data(),   dL_barrier,   rows_bytes, cudaMemcpyDeviceToHost));

        printf("[Barrier] Row[0..3] acc/m/l:\n");
        for (int r = 0; r < 4 && r < num_rows; ++r) {
            printf("  row %d: acc=%f, m=%f, l=%f\n", r, hAcc[r], hM[r], hL[r]);
        }
    }

    // --- warp-local 버전 ---
    if (run_warp) {
        printf("\n[Warp-local kernel] Running %d iterations...\n", iters);

        // warmup
        for (int i = 0; i < 10; ++i) {
            softmax_warp_local_kernel<<<grid_warp, block>>>(
                dAcc_warp, dM_warp, dL_warp, num_rows, k_tot
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            softmax_warp_local_kernel<<<grid_warp, block>>>(
                dAcc_warp, dM_warp, dL_warp, num_rows, k_tot
            );
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;
        printf("[Warp-local] Avg kernel time: %.3f ms\n", avg_ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        std::vector<float> hAcc(num_rows), hM(num_rows), hL(num_rows);
        CHECK_CUDA(cudaMemcpy(hAcc.data(), dAcc_warp, rows_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hM.data(),   dM_warp,   rows_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hL.data(),   dL_warp,   rows_bytes, cudaMemcpyDeviceToHost));

        printf("[Warp-local] Row[0..3] acc/m/l:\n");
        for (int r = 0; r < 4 && r < num_rows; ++r) {
            printf("  row %d: acc=%f, m=%f, l=%f\n", r, hAcc[r], hM[r], hL[r]);
        }
    }

    CHECK_CUDA(cudaFree(dAcc_barrier));
    CHECK_CUDA(cudaFree(dM_barrier));
    CHECK_CUDA(cudaFree(dL_barrier));
    CHECK_CUDA(cudaFree(dAcc_warp));
    CHECK_CUDA(cudaFree(dM_warp));
    CHECK_CUDA(cudaFree(dL_warp));

    return 0;
}
