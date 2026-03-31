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

// Standard CUDA-core FP32 matrix multiply — identical pattern to example 17's
// matMulBasic, kept here for a direct apples-to-apples timing comparison.
__global__ void matMulCuda(float *A, float *B, float *C, int m, int n, int k) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < m && col < n) {
    float value = 0.0f;
    for (int i = 0; i < k; i++)
      value += A[row * k + i] * B[i * n + col];
    C[row * n + col] = value;
  }
}

// Tensor core kernel using the WMMA API. One warp per block computes one 16x16
// output tile. The grid maps blocks 1-to-1 onto output tiles so every warp
// owns exactly the tile identified by its block index.
__global__ void matMulWmma(half *A, half *B, float *C, int m, int n, int k) {
  int warpM = blockIdx.y;
  int warpN = blockIdx.x;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  // Iterate over the K dimension in WMMA_K-wide strips. Each iteration loads
  // one 16x16 strip of A and one 16x16 strip of B, then accumulates into
  // c_frag.
  for (int ki = 0; ki < k; ki += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = ki;
    int bRow = ki;
    int bCol = warpN * WMMA_N;

    if (aRow < m && aCol < k && bRow < k && bCol < n) {
      wmma::load_matrix_sync(a_frag, A + aRow * k + aCol, k);
      wmma::load_matrix_sync(b_frag, B + bRow * n + bCol, n);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  if (cRow < m && cCol < n)
    wmma::store_matrix_sync(C + cRow * n + cCol, c_frag, n,
                            wmma::mem_row_major);
}

int main() {
  int m = M, n = N_DIM, k = K;
  size_t sizeFloat = (size_t)m * n * sizeof(float);
  size_t sizeHalf = (size_t)m * n * sizeof(half);

  float *A, *B, *C_cuda, *C_wmma;
  half *A_fp16, *B_fp16;

  cudaMallocManaged(&A, sizeFloat);
  cudaMallocManaged(&B, sizeFloat);
  cudaMallocManaged(&C_cuda, sizeFloat);
  cudaMallocManaged(&C_wmma, sizeFloat);
  cudaMallocManaged(&A_fp16, sizeHalf);
  cudaMallocManaged(&B_fp16, sizeHalf);

  // Initialize with small values so FP16 rounding error stays bounded.
  for (int i = 0; i < m * k; i++) {
    A[i] = (float)(i % 100) / 100.0f;
    A_fp16[i] = __float2half(A[i]);
  }
  for (int i = 0; i < k * n; i++) {
    B[i] = (float)((i + 1) % 100) / 100.0f;
    B_fp16[i] = __float2half(B[i]);
  }
  for (int i = 0; i < m * n; i++) {
    C_cuda[i] = 0.0f;
    C_wmma[i] = 0.0f;
  }

  int deviceId;
  cudaGetDevice(&deviceId);

  // Prefetch all arrays to device so page migrations don't inflate timing.
  cudaMemPrefetchAsync(A, sizeFloat, deviceId);
  cudaMemPrefetchAsync(B, sizeFloat, deviceId);
  cudaMemPrefetchAsync(C_cuda, sizeFloat, deviceId);
  cudaMemPrefetchAsync(A_fp16, sizeHalf, deviceId);
  cudaMemPrefetchAsync(B_fp16, sizeHalf, deviceId);
  cudaMemPrefetchAsync(C_wmma, sizeFloat, deviceId);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- CUDA core kernel ---
  dim3 cudaThreads(16, 16);
  dim3 cudaBlocks((n + cudaThreads.x - 1) / cudaThreads.x,
                  (m + cudaThreads.y - 1) / cudaThreads.y);

  cudaEventRecord(start);
  matMulCuda<<<cudaBlocks, cudaThreads>>>(A, B, C_cuda, m, n, k);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeCuda;
  cudaEventElapsedTime(&timeCuda, start, stop);

  // --- WMMA tensor core kernel ---
  // One warp (32 threads) per block; grid tiles cover the M×N output matrix.
  dim3 wmmaThreads(32);
  dim3 wmmaBlocks(n / WMMA_N, m / WMMA_M);

  cudaEventRecord(start);
  matMulWmma<<<wmmaBlocks, wmmaThreads>>>(A_fp16, B_fp16, C_wmma, m, n, k);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeWmma;
  cudaEventElapsedTime(&timeWmma, start, stop);

  // FP16 arithmetic accumulates rounding error proportional to K.
  // With K=1024 elements and FP16 machine epsilon ~1e-3, deviations up to 0.1
  // are expected for values in [0,1] — this tolerance detects real bugs.
  bool error = false;
  for (int i = 0; i < m * n && !error; i++) {
    if (fabsf(C_cuda[i] - C_wmma[i]) > 1e-1f) {
      printf("MISMATCH at index %d: cuda=%.6f, wmma=%.6f\n", i, C_cuda[i],
             C_wmma[i]);
      error = true;
    }
  }
  if (!error)
    printf("Results match!\n");

  double flops = 2.0 * m * n * k;
  double gflopsCuda = flops / timeCuda / 1e6;
  double gflopsWmma = flops / timeWmma / 1e6;

  printf("Matrix size:            %d x %d x %d\n", m, n, k);
  printf("CUDA core kernel time:  %.3f ms  (%.1f GFLOPS)\n", timeCuda,
         gflopsCuda);
  printf("WMMA tensor core time:  %.3f ms  (%.1f GFLOPS)\n", timeWmma,
         gflopsWmma);
  printf("Speedup (WMMA/CUDA):    %.2fx\n", timeCuda / timeWmma);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_cuda);
  cudaFree(C_wmma);
  cudaFree(A_fp16);
  cudaFree(B_fp16);

  return 0;
}
