// flashattn_forward_cp_async_stall.cu
//
// 5.6.2: cp.async stall 분석용 마이크로 커널
//
// - FlashAttention에서 K/V tile을 streaming 하는 패턴만 떼어낸 형태
// - K, V: [BH, N, D] (half)
// - 각 block: 하나의 head(bh)를 담당, N을 TILE_N 단위로 순차 접근
// - cp.async 2-stage ping-pong으로 global -> shared prefetch
// - shared tile에 올라온 K, V에 대해 간단한 FMA accumulate 수행
//   (실제 Attention은 아니고, cp.async pipeline 관찰용)
//
// 빌드 예시 (Ampere, Windows PowerShell):
//   nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_cp_async_stall.cu -o flashattn_cp_async_stall.exe
//
// Nsight Compute 예시:
//   ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_cp_async_stall.exe

#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// 타일 사이즈 (seq 방향)
constexpr int TILE_N = 64;  // N은 64 배수 가정

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
 * cp.async 2-stage ping-pong pipeline kernel
 *
 * K, V: [BH, N, D] (half)
 * Out:  [BH] (float)  -- 각 head별 전체 FMA 결과 하나만 저장
 *
 * BH: batch_size * num_heads
 * N : sequence length (TILE_N 배수)
 * D : head dimension  (16/32 배수 가정, 여기선 64 사용)
 */
__global__ void flashattn_cp_async_2stage_kernel(
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ Out,
    int BH,
    int N,
    int D
) {
#if __CUDA_ARCH__ < 800
    return;  // cp.async는 Ampere 이상
#endif

    extern __shared__ __align__(16) char smem_raw[];
    char* smem_ptr = smem_raw;

    // double-buffered shared memory:
    // stage 0,1 각각 K_tile, V_tile 보관
    size_t tile_elems = (size_t)TILE_N * D;      // half elements per K or V tile
    size_t tile_bytes = tile_elems * sizeof(half);

    half* K_smem[2];
    half* V_smem[2];

    K_smem[0] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += tile_bytes;
    K_smem[1] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += tile_bytes;

    V_smem[0] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += tile_bytes;
    V_smem[1] = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += tile_bytes;

    // block 당 하나의 head(bh) 처리
    int bh  = blockIdx.x;
    int tid = threadIdx.x;

    if (bh >= BH) return;

    int num_tiles = N / TILE_N;  // N은 TILE_N 배수 가정

    // 각 thread의 누적 FMA 결과
    float acc = 0.0f;

    // cp.async 2-stage pipeline:
    //   tile = 0..num_tiles-1 에 대해
    //   load( tile ) + compute( tile-1 ) 형태로 파이프
    for (int tile = 0; tile < num_tiles + 1; ++tile) {
        int load_tile    = tile;
        int compute_tile = tile - 1;

        int s_load = load_tile & 1;    // 0 or 1
        int s_comp = compute_tile & 1; // 0 or 1

        // --- load 단계: cp.async로 K/V를 shared로 prefetch ---
        if (load_tile < num_tiles) {
            int col_start = load_tile * TILE_N;

            const half* K_gptr = K + ((size_t)bh * N + col_start) * D;
            const half* V_gptr = V + ((size_t)bh * N + col_start) * D;

            // 16B(=8 half) 단위로 cp.async 발행
            char*       K_smem_bytes = reinterpret_cast<char*>(K_smem[s_load]);
            const char* K_glob_bytes = reinterpret_cast<const char*>(K_gptr);

            char*       V_smem_bytes = reinterpret_cast<char*>(V_smem[s_load]);
            const char* V_glob_bytes = reinterpret_cast<const char*>(V_gptr);

            int total_16B_chunks = (int)(tile_bytes / 16);  // tile_bytes는 16B 배수

            for (int idx = tid; idx < total_16B_chunks; idx += BLOCK_THREADS) {
                size_t byte_offset = (size_t)idx * 16;

                CP_ASYNC_CG(K_smem_bytes + byte_offset, K_glob_bytes + byte_offset);
                CP_ASYNC_CG(V_smem_bytes + byte_offset, V_glob_bytes + byte_offset);
            }

            CP_ASYNC_COMMIT_GROUP();
        }

        // --- compute 단계: 이전 타일에 대해 K_smem, V_smem 사용 ---
        if (compute_tile >= 0) {
            CP_ASYNC_WAIT_ALL();
            __syncthreads();

            half* K_tile = K_smem[s_comp];
            half* V_tile = V_smem[s_comp];

            // 간단한 per-element FMA:
            //   acc += float(K_tile[e]) * float(V_tile[e])
            for (size_t e = tid; e < tile_elems; e += BLOCK_THREADS) {
                float kf = __half2float(K_tile[e]);
                float vf = __half2float(V_tile[e]);
                acc += kf * vf;
            }
        }

        __syncthreads();
    }

    // block 전체에서 acc를 reduce 해서 Out[bh]에 저장
    __shared__ float acc_smem[BLOCK_THREADS];
    acc_smem[tid] = acc;
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            acc_smem[tid] += acc_smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        Out[bh] = acc_smem[0];
    }
}

// =========================
// 5.6.2용 메인: cp.async stall 관찰용
// =========================
int main() {
    // cp.async 파이프라인이 충분히 길도록 N을 크게 설정
    int BH = 4;     // batch*heads
    int N  = 1024;  // seq len (TILE_N=64의 배수)
    int D  = 64;    // head dim

    size_t num_elem = (size_t)BH * N * D;

    size_t bytes_half  = num_elem * sizeof(half);
    size_t bytes_out   = BH * sizeof(float);

    std::vector<half>  hK(num_elem), hV(num_elem);
    std::vector<float> hOut(BH);

    // deterministic 초기화
    for (size_t i = 0; i < num_elem; ++i) {
        float v = (float)(i % 17) * 0.01f;
        hK[i] = __float2half(v);
        hV[i] = __float2half(0.5f * v);
    }

    half  *dK, *dV;
    float *dOut;
    CHECK_CUDA(cudaMalloc(&dK, bytes_half));
    CHECK_CUDA(cudaMalloc(&dV, bytes_half));
    CHECK_CUDA(cudaMalloc(&dOut, bytes_out));

    CHECK_CUDA(cudaMemcpy(dK, hK.data(), bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOut, 0, bytes_out));

    dim3 grid(BH, 1, 1);           // 각 block = 하나의 head
    dim3 block(BLOCK_THREADS, 1, 1);

    // shared memory: 2(stage) * (K_tile + V_tile)
    size_t tile_elems = (size_t)TILE_N * D;
    size_t tile_bytes = tile_elems * sizeof(half);
    size_t smem_bytes = 2 * (tile_bytes + tile_bytes);  // K0,K1,V0,V1

    printf("Launching cp.async kernel with BH=%d, N=%d, D=%d\n", BH, N, D);
    printf("grid=(%d,%d), block=%d, smem=%zu bytes\n",
           grid.x, grid.y, block.x, smem_bytes);

    // warmup
    int warmup = 10;
    for (int i = 0; i < warmup; ++i) {
        flashattn_cp_async_2stage_kernel<<<grid, block, smem_bytes>>>(
            dK, dV, dOut, BH, N, D
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
        flashattn_cp_async_2stage_kernel<<<grid, block, smem_bytes>>>(
            dK, dV, dOut, BH, N, D
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, bytes_out, cudaMemcpyDeviceToHost));

    printf("Out[bh=0..3]: ");
    for (int b = 0; b < BH; ++b) {
        printf("%f ", hOut[b]);
    }
    printf("\n");

    printf("Avg kernel time: %.3f ms\n", avg_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dOut));

    return 0;
}

/*
# 빌드
nvcc -arch=sm_86 -O3 -lineinfo -std=c++17 flashattn_forward_cp_async_stall.cu -o flashattn_cp_async_stall.exe

# 실행
.\flashattn_cp_async_stall.exe

# Nsight Compute (cp.async stall 분석)
ncu --set full --launch-skip 10 --launch-count 1 .\flashattn_cp_async_stall.exe
*/
