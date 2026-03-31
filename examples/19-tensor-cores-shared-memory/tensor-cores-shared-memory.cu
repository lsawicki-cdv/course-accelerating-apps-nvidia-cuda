#include <math.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define M 1024
#define N_DIM 1024
#define K 1024
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_PER_BLOCK_M 2
#define WARPS_PER_BLOCK_N 2
#define BLOCK_M (WARPS_PER_BLOCK_M * WMMA_M)
#define BLOCK_N (WARPS_PER_BLOCK_N * WMMA_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N * 32)

// Naive WMMA kernel — each warp loads its fragments directly from global
// memory. Reproduced from example 18 as the baseline for comparison.
__global__ void matMulWmmaNaive(half *A, half *B, float *C, int m, int n, int k) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int ki = 0; ki < k; ki += WMMA_K) {
        int aRow = warpM * WMMA_M, aCol = ki;
        int bRow = ki,             bCol = warpN * WMMA_N;

        if (aRow < m && aCol < k && bRow < k && bCol < n) {
            wmma::load_matrix_sync(a_frag, A + aRow * k + aCol, k);
            wmma::load_matrix_sync(b_frag, B + bRow * n + bCol, n);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M, cCol = warpN * WMMA_N;
    if (cRow < m && cCol < n)
        wmma::store_matrix_sync(C + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
}

// Shared-memory WMMA kernel. Each block of 4 warps cooperatively loads a
// BLOCK_M×WMMA_K strip of A and WMMA_K×BLOCK_N strip of B into shared memory
// for each K-phase. Warps then load their 16x16 fragments from shared memory,
// replacing repeated global-memory reads with a single cooperative load.
__global__ void matMulWmmaShared(half *A, half *B, float *C, int m, int n, int k) {
    // Shared buffers for one K-phase. All warps in the block fill these together
    // before any WMMA operation proceeds, ensuring data is ready for all warps.
    __shared__ half sh_A[BLOCK_M][WMMA_K];
    __shared__ half sh_B[WMMA_K][BLOCK_N];

    int blockMStart = blockIdx.y * BLOCK_M;
    int blockNStart = blockIdx.x * BLOCK_N;

    // Each warp within the block handles one 16×16 sub-tile of the 32×32 block tile.
    int warpId      = threadIdx.x / 32;
    int warpM_local = warpId / WARPS_PER_BLOCK_N;
    int warpN_local = warpId % WARPS_PER_BLOCK_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int ki = 0; ki < k; ki += WMMA_K) {
        // All THREADS_PER_BLOCK threads collaborate to fill sh_A and sh_B.
        // This amortizes the global-memory bandwidth cost across all 4 warps.

        // Load sh_A: BLOCK_M × WMMA_K = 32×16 = 512 elements, 128 threads → 4 each.
        for (int idx = threadIdx.x; idx < BLOCK_M * WMMA_K; idx += THREADS_PER_BLOCK) {
            int r = idx / WMMA_K, c = idx % WMMA_K;
            int globalRow = blockMStart + r, globalCol = ki + c;
            sh_A[r][c] = (globalRow < m && globalCol < k)
                         ? A[globalRow * k + globalCol]
                         : __float2half(0.0f);
        }

        // Load sh_B: WMMA_K × BLOCK_N = 16×32 = 512 elements, 128 threads → 4 each.
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int r = idx / BLOCK_N, c = idx % BLOCK_N;
            int globalRow = ki + r, globalCol = blockNStart + c;
            sh_B[r][c] = (globalRow < k && globalCol < n)
                         ? B[globalRow * n + globalCol]
                         : __float2half(0.0f);
        }

        // All threads must finish writing shared memory before any warp reads it.
        __syncthreads();

        // Each warp loads its 16×16 sub-tile from the shared buffer.
        // ld for sh_A = WMMA_K (columns per row); ld for sh_B = BLOCK_N.
        wmma::load_matrix_sync(a_frag, &sh_A[warpM_local * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &sh_B[0][warpN_local * WMMA_N], BLOCK_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Barrier before the next phase overwrites shared memory.
        __syncthreads();
    }

    int cRow = blockMStart + warpM_local * WMMA_M;
    int cCol = blockNStart + warpN_local * WMMA_N;
    if (cRow < m && cCol < n)
        wmma::store_matrix_sync(C + cRow * n + cCol, c_frag, n, wmma::mem_row_major);
}

int main() {
    int m = M, n = N_DIM, k = K;
    size_t sizeFloat = (size_t)m * n * sizeof(float);
    size_t sizeHalf  = (size_t)m * n * sizeof(half);

    float *C_naive, *C_shared;
    half  *A_fp16, *B_fp16;

    cudaMallocManaged(&A_fp16,   sizeHalf);
    cudaMallocManaged(&B_fp16,   sizeHalf);
    cudaMallocManaged(&C_naive,  sizeFloat);
    cudaMallocManaged(&C_shared, sizeFloat);

    for (int i = 0; i < m * k; i++)
        A_fp16[i] = __float2half((float)(i % 100) / 100.0f);
    for (int i = 0; i < k * n; i++)
        B_fp16[i] = __float2half((float)((i + 1) % 100) / 100.0f);
    for (int i = 0; i < m * n; i++) {
        C_naive[i] = 0.0f;  C_shared[i] = 0.0f;
    }

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaMemPrefetchAsync(A_fp16,   sizeHalf,  deviceId);
    cudaMemPrefetchAsync(B_fp16,   sizeHalf,  deviceId);
    cudaMemPrefetchAsync(C_naive,  sizeFloat, deviceId);
    cudaMemPrefetchAsync(C_shared, sizeFloat, deviceId);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Naive WMMA ---
    dim3 naiveThreads(32);
    dim3 naiveBlocks(n / WMMA_N, m / WMMA_M);

    cudaEventRecord(start);
    matMulWmmaNaive<<<naiveBlocks, naiveThreads>>>(A_fp16, B_fp16, C_naive, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeNaive;
    cudaEventElapsedTime(&timeNaive, start, stop);

    // --- Shared-memory WMMA ---
    dim3 sharedThreads(THREADS_PER_BLOCK);
    dim3 sharedBlocks(n / BLOCK_N, m / BLOCK_M);

    cudaEventRecord(start);
    matMulWmmaShared<<<sharedBlocks, sharedThreads>>>(A_fp16, B_fp16, C_shared, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeShared;
    cudaEventElapsedTime(&timeShared, start, stop);

    // Both kernels use identical FP16 inputs so results should match closely.
    bool error = false;
    for (int i = 0; i < m * n && !error; i++) {
        if (fabsf(C_naive[i] - C_shared[i]) > 1e-2f) {
            printf("MISMATCH at index %d: naive=%.6f, shared=%.6f\n", i, C_naive[i], C_shared[i]);
            error = true;
        }
    }
    if (!error)
        printf("Results match!\n");

    double flops = 2.0 * m * n * k;
    double gflopsNaive  = flops / timeNaive  / 1e6;
    double gflopsShared = flops / timeShared / 1e6;

    printf("Matrix size:                %d x %d x %d\n", m, n, k);
    printf("Block tile size:            %d x %d  (%d warps)\n",
           BLOCK_M, BLOCK_N, WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N);
    printf("Naive WMMA time:            %.3f ms  (%.1f GFLOPS)\n", timeNaive,  gflopsNaive);
    printf("Shared-memory WMMA time:    %.3f ms  (%.1f GFLOPS)\n", timeShared, gflopsShared);
    printf("Speedup (shared/naive):     %.2fx\n", timeNaive / timeShared);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_fp16);  cudaFree(B_fp16);
    cudaFree(C_naive); cudaFree(C_shared);

    return 0;
}
