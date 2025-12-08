// naive_gemm.cu
//  - Shared memory / tiling 전혀 없음
//  - 1 thread = C(row, col) 한 원소
//  - global memory에서 A, B를 그대로 매번 읽는 가장 단순한 형태
//  - NCU baseline용

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#define CHECK_CUDA(cmd)                                                         \
    do {                                                                        \
        cudaError_t e = (cmd);                                                  \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e));                                     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// === 설정 ===
// 검증 모드 (CPU GEMM으로 결과 확인, 크기 작게 추천)
constexpr int DO_VERIFY = 0;

// 문제 크기 (row-major, A: MxK, B: KxN, C: MxN)
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// 블록 사이즈
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// === naive GEMM kernel ===
// C[M, N] = A[M, K] * B[K, N]
__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, N)

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    // 완전 naive: 매번 global에서 A[row, k], B[k, col] 로드
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }

    C[row * N + col] = acc;
}

// CPU reference GEMM (검증용, 크기 작게 써야 함)
void cpu_gemm_ref(const std::vector<float>& A,
                  const std::vector<float>& B,
                  std::vector<float>& C,
                  int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

int main()
{
    printf("Naive GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    // Host 메모리
    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<float> hC(sizeC, 0.0f);

    // 난수 초기화
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

#if DO_VERIFY
    // 검증용 reference
    std::vector<float> hC_ref(sizeC, 0.0f);
    printf("Running CPU reference GEMM...\n");
    cpu_gemm_ref(hA, hB, hC_ref, M, N, K);
#endif

    // Device 메모리
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC * sizeof(float)));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BLOCK_X - 1) / BLOCK_X,
              (M + BLOCK_Y - 1) / BLOCK_Y);

    printf("Grid = (%d, %d), Block = (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warm-up (ncu에서는 launch-skip 로 스킵 가능하지만, 그냥 한 번 실행)
    naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 타이밍 측정
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // GFLOPS 계산: 2 * M * N * K / time
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms * 1.0e6);

    printf("Kernel time: %.3f ms, GFLOPS: %.3f\n", ms, gflops);

#if DO_VERIFY
    // 최대 절대 오차 체크
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double diff = std::fabs(static_cast<double>(hC[i]) -
                                static_cast<double>(hC_ref[i]));
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    printf("Max abs diff vs CPU ref: %.6e\n", max_abs_diff);
#endif

    // 정리
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
/*
nvcc -O3 -std=c++17 -lineinfo -o naive_gemm naive_gemm.cu

.\naive_gemm.exe

ncu --set full --launch-skip 1 --launch-count 1     --kernel-name-base demangled     --kernel-name regex:naive_gemm_kernel     .\naive_gemm.exe


*/