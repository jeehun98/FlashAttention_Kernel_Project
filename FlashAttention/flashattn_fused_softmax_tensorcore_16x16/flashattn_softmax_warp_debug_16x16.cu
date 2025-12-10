// 6.13 Warp-parallel Softmax Microkernel Debug (M=N=16)
//   - 입력: scores[16,16] (row-major, float)
//   - 연산: row-wise softmax (scale 적용) 를 warp-parallel 로 수행
//   - 출력: probs[16,16] (row-major, float)
//   - CPU ref 와 상대 L2 error 비교 + kernel 시간 측정

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd)                                                          \
    do {                                                                         \
        cudaError_t e = (cmd);                                                   \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(e));                                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int M = 16;   // rows
constexpr int N = 16;   // cols
constexpr int TILE_SIZE = M * N;

// === Warp-level reduction helpers ===
__inline__ __device__ float warp_allreduce_max(float v) {
    unsigned mask = 0xffffffffu;
    // N=16만 유효하지만, 32-lane 전체에 대해 XOR reduction 사용
    // lane>=16 은 -INF 로 초기화해서 결과에 영향 X
    v = fmaxf(v, __shfl_xor_sync(mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

__inline__ __device__ float warp_allreduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

// === Softmax warp microkernel for a single 16x16 tile in shared memory ===
// s_scores: [M,N] row-major
// s_probs : [M,N] row-major (output)
// scale   : scaling factor applied before softmax (e.g. 1/sqrt(K))
__device__ void softmax_warp_16x16(float* s_scores, float* s_probs, float scale) {
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // 한 warp 가 전체 16 rows를 순회하면서 처리
    for (int row = 0; row < M; ++row) {
        float x = -INFINITY;

        if (lane < N) {
            float v = s_scores[row * N + lane];
            x = v * scale;
        }

        // 1) row-wise max
        float maxv = warp_allreduce_max(x);

        // 2) exp(x - max) 와 sum
        float ex = 0.0f;
        if (lane < N) {
            ex = __expf(x - maxv);
        }
        float sumv = warp_allreduce_sum(ex);

        // 3) normalize
        if (lane < N) {
            float p = ex / (sumv + 1e-6f);
            s_probs[row * N + lane] = p;
        }
        // lane>=16 은 dummy, 아무 것도 저장 안 함
    }
}

// === Kernel: global scores -> shared -> warp softmax -> global probs ===
__global__ void softmax_warp_16x16_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    float scale
) {
    // 이 디버그 버전은 1 block, 1 warp 만 가정
    __shared__ float s_scores[TILE_SIZE];
    __shared__ float s_probs[TILE_SIZE];

    int lane = threadIdx.x & (WARP_SIZE - 1);

    // 1) global -> shared load
    for (int idx = lane; idx < TILE_SIZE; idx += WARP_SIZE) {
        s_scores[idx] = scores[idx];
    }
    __syncthreads();

    // 2) warp-parallel softmax
    softmax_warp_16x16(s_scores, s_probs, scale);
    __syncthreads();

    // 3) shared -> global store
    for (int idx = lane; idx < TILE_SIZE; idx += WARP_SIZE) {
        probs[idx] = s_probs[idx];
    }
}

// === CPU reference: row-wise softmax on [M,N] ===
void softmax_cpu_ref(const std::vector<float>& scores,
                     std::vector<float>& probs,
                     float scale)
{
    for (int row = 0; row < M; ++row) {
        // 1) max
        float maxv = -INFINITY;
        for (int col = 0; col < N; ++col) {
            float v = scores[row * N + col] * scale;
            if (v > maxv) maxv = v;
        }

        // 2) exp & sum
        float sumv = 0.0f;
        for (int col = 0; col < N; ++col) {
            float v = scores[row * N + col] * scale;
            float e = std::exp(v - maxv);
            probs[row * N + col] = e;
            sumv += e;
        }

        // 3) normalize
        float inv_sum = 1.0f / (sumv + 1e-6f);
        for (int col = 0; col < N; ++col) {
            probs[row * N + col] *= inv_sum;
        }
    }
}

int main() {
    std::printf("6.13 Warp-parallel Softmax Microkernel Debug (M=N=16)\n");

    // 1) 입력 scores 생성 (랜덤)
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_scores(TILE_SIZE);
    for (int i = 0; i < TILE_SIZE; ++i) {
        h_scores[i] = dist(rng);
    }

    // scaling factor (FlashAttention 에서의 1/sqrt(dk) 와 유사하게)
    float scale = 1.0f / std::sqrt(static_cast<float>(N));

    // 2) CPU ref softmax
    std::vector<float> h_probs_ref(TILE_SIZE);
    softmax_cpu_ref(h_scores, h_probs_ref, scale);

    // 3) GPU 메모리 할당 / 복사
    float *d_scores = nullptr, *d_probs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scores, TILE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_probs,  TILE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scores, h_scores.data(),
                          TILE_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 4) 커널 실행 (정확도 체크용 1회)
    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(1, 1, 1);

    softmax_warp_16x16_kernel<<<grid, block>>>(d_scores, d_probs, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_probs(TILE_SIZE);
    CHECK_CUDA(cudaMemcpy(h_probs.data(), d_probs,
                          TILE_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 5) Relative L2 error 계산
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < TILE_SIZE; ++i) {
        double diff = static_cast<double>(h_probs[i]) - static_cast<double>(h_probs_ref[i]);
        num += diff * diff;
        den += static_cast<double>(h_probs_ref[i]) * static_cast<double>(h_probs_ref[i]);
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));

    // 6) 앞부분 출력
    std::printf("P[0, 0..7] (GPU): ");
    for (int j = 0; j < 8; ++j) {
        std::printf("%f ", h_probs[0 * N + j]);
    }
    std::printf("\n");

    std::printf("P_ref[0, 0..7] (CPU): ");
    for (int j = 0; j < 8; ++j) {
        std::printf("%f ", h_probs_ref[0 * N + j]);
    }
    std::printf("\n");

    std::printf("Relative L2 error: %.9e\n", rel_l2);

    // 7) 성능 측정 (여러 번 반복)
    const int NUM_ITERS = 10000;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < NUM_ITERS; ++it) {
        softmax_warp_16x16_kernel<<<grid, block>>>(d_scores, d_probs, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_ITERS; // per-iteration

    // softmax flops 대략 계산:
    // row당: scale(1) + max(15) + exp(16) + sum(15) + div(16) ≈ 63 FLOPs
    // tile 전체: 63 * 16 ≈ 1008 FLOPs (rough)
    double flops = 1008.0;
    double t_sec = ms * 1e-3;
    double gflops = (flops / t_sec) / 1e9;

    std::printf("Avg kernel time: %f ms (over %d iters)\n", ms, NUM_ITERS);
    std::printf("Approx GFLOPS  : %.6f\n", gflops);

    // 8) 정리
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
/*
nvcc -O3 -std=c++17   -arch=sm_86   -o flashattn_softmax_warp_debug_16x16.exe   flashattn_softmax_warp_debug_16x16.cu

ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_softmax_warp_debug_16x16.exe

*/