// flashattn_forward_softmax_bottleneck.cu
//
// 5.6.4: Softmax reduction 병목 관찰용 마이크로 커널
//
// - FlashAttention 온라인 softmax 구조만 따로 떼어낸 버전
// - K/V, WMMA, cp.async 전혀 없음 → 오로지 expf / fmaxf / FMA 체인만 존재
// - 각 thread가 하나의 (bh, row)을 담당하고, K_tot 개의 key에 대해
//   온라인 softmax + PV-like accumulate를 수행
//   (score / value 는 전부 on-the-fly로 생성, global load 거의 없음)
//
// 목표:
//   - Nsight Compute에서 "Execution Dependency" / "Math Pipe Latency" stall 이
//     어떻게 softmax 구간을 지배하는지 확인하는 용도
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_softmax_bottleneck.cu -o flashattn_softmax_bottleneck.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_softmax_bottleneck.exe

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include <math_constants.h>


#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

constexpr int BLOCK_THREADS = 128;

// 간단한 ceil_div
__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * 온라인 softmax + PV-like accumulate 전용 커널
 *
 * 각 thread:
 *   - 하나의 (bh, row)을 담당
 *   - k = 0..K_tot-1 에 대해:
 *       - score s_k, value v_k 를 on-the-fly로 생성
 *       - "온라인 softmax" 알고리즘으로 m, l 업데이트
 *       - p_k = exp(s_k - m_new) / l_new
 *       - acc = acc * alpha + p_k * v_k
 *
 * I/O:
 *   Out_acc: [BH * N]  (최종 acc 값)
 *   Out_m:   [BH * N]  (최종 softmax max)
 *   Out_l:   [BH * N]  (최종 softmax 평균 분모)
 *
 * BH: batch_size * num_heads
 * N : query length (row 개수)
 * K_tot: key 개수 (전체 seq length)
 */
__global__ void flashattn_softmax_bottleneck_kernel(
    float* __restrict__ Out_acc,
    float* __restrict__ Out_m,
    float* __restrict__ Out_l,
    int BH,
    int N,
    int K_tot
) {
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = BH * N;
    if (tid_global >= total_rows) return;

    int bh  = tid_global / N;
    int row = tid_global % N;

    // 온라인 softmax 상태
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc = 0.0f;

    // FlashAttention의 "score tile" 대신 단순 for k = 0..K_tot-1
    // 여기서는 DRAM load 대신 on-the-fly pseudo score / value 생성
    for (int k = 0; k < K_tot; ++k) {
        // pseudo score / value (deterministic, global memory 없이 생성)
        // score는 약 [-0.2, 0.2] 근방 값이 나오도록 조정
        int score_seed = (bh * 131 + row * 17 + k * 7) & 0x7fffffff;
        float s = ((float)(score_seed % 41) - 20.0f) * 0.01f;

        int value_seed = (bh * 19 + row * 11 + k * 5) & 0x7fffffff;
        float v = ((float)(value_seed % 31) - 15.0f) * 0.01f;

        // --- 온라인 softmax 업데이트 ---
        float m_old = m;
        float l_old = l;

        // 새로운 score와 기존 max 를 비교해서 m_new 결정
        float m_new = fmaxf(m_old, s);

        // 이전까지의 분모(l_old)를 새로운 기준 m_new 로 가져옴
        float exp_old = 0.0f;
        if (m_old != -CUDART_INF_F && l_old > 0.0f) {
            exp_old = expf(m_old - m_new) * l_old;
        }

        // 이번 score 의 contribution
        float exp_new = expf(s - m_new);

        // 새로운 분모
        float l_new = exp_old + exp_new;

        // 이전 누적 확률 질량이 차지하는 비율 (O 업데이트 스케일)
        float alpha = 0.0f;
        if (l_new > 0.0f && l_old > 0.0f) {
            alpha = exp_old / l_new;
        }

        // 이번 score 의 정규화된 softmax weight
        float p_new = 0.0f;
        if (l_new > 0.0f) {
            p_new = exp_new / l_new;
        }

        // FlashAttention 의 O = alpha * O + sum_j w_j * V_j 에 해당하는 부분을
        // 1D scalar 로 축소한 형태: acc = alpha * acc + p_new * v
        acc = alpha * acc + p_new * v;

        m = m_new;
        l = l_new;
    }

    Out_acc[tid_global] = acc;
    Out_m[tid_global]   = m;
    Out_l[tid_global]   = l;
}

// =========================
// 5.6.4용 메인: softmax 병목 관찰용
// =========================
int main() {
    // FlashAttention 과 비슷한 스케일로 설정
    int BH    = 4;    // batch * heads
    int N     = 512;  // query length
    int K_tot = 512;  // key length (전체 softmax 길이)

    int total_rows = BH * N;

    size_t bytes = (size_t)total_rows * sizeof(float);

    std::vector<float> hAcc(total_rows);
    std::vector<float> hM(total_rows);
    std::vector<float> hL(total_rows);

    float *dAcc, *dM, *dL;
    CHECK_CUDA(cudaMalloc(&dAcc, bytes));
    CHECK_CUDA(cudaMalloc(&dM,   bytes));
    CHECK_CUDA(cudaMalloc(&dL,   bytes));

    CHECK_CUDA(cudaMemset(dAcc, 0, bytes));
    CHECK_CUDA(cudaMemset(dM,   0, bytes));
    CHECK_CUDA(cudaMemset(dL,   0, bytes));

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(ceil_div(total_rows, BLOCK_THREADS), 1, 1);

    printf("Launching Softmax bottleneck kernel\n");
    printf("BH=%d, N=%d, K_tot=%d\n", BH, N, K_tot);
    printf("grid=(%d,%d), block=%d\n", grid.x, grid.y, block.x);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        flashattn_softmax_bottleneck_kernel<<<grid, block>>>(
            dAcc, dM, dL, BH, N, K_tot
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
        flashattn_softmax_bottleneck_kernel<<<grid, block>>>(
            dAcc, dM, dL, BH, N, K_tot
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hAcc.data(), dAcc, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hM.data(),   dM,   bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hL.data(),   dL,   bytes, cudaMemcpyDeviceToHost));

    // 샘플 몇 개 출력
    printf("Row[0..3] acc/m/l:\n");
    for (int r = 0; r < 4; ++r) {
        printf("  row %d: acc=%f, m=%f, l=%f\n",
               r, hAcc[r], hM[r], hL[r]);
    }

    // softmax 연산량 대략 계산
    // K_tot 개에 대해:
    //  - expf 2회 정도 (exp_old, exp_new) ≈ 2 FLOPs * K_tot
    //  - fmaxf, FMA, add 등 포함해서 대략 16 FLOPs/element 라고 가정
    double flops_per_elem = 16.0;
    double total_flops = (double)total_rows * (double)K_tot * flops_per_elem;
    double tflops = (total_flops * 1e-12) / (avg_ms * 1e-3);

    printf("Avg kernel time: %.3f ms\n", avg_ms);
    printf("Approx FLOPs: %.3e\n", total_flops);
    printf("Approx TFLOPS: %.3f\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dAcc));
    CHECK_CUDA(cudaFree(dM));
    CHECK_CUDA(cudaFree(dL));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_softmax_bottleneck.cu -o flashattn_softmax_bottleneck.exe

# 실행
.\flashattn_softmax_bottleneck.exe

# Nsight Compute (softmax execution dependency stall 분석)
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_softmax_bottleneck.exe

분석 포인트 (Warp State / Scheduler Statistics):
- Warp Cycles Per Issued Instruction
- Stall Reason: "Execution Dependency" / "Math Pipe" 비중
- Memory stall 은 거의 없고, expf 체인에 의한 latency 가 얼마나 지배적인지 확인
*/