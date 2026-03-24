#include <math.h>
#include <stdio.h>

#define N 1024

/*
 * Coalesced: threadIdx.x maps to column (consecutive threads access
 * consecutive memory locations in B).
 */
__global__ void matMulCoalesced(float *A, float *B, float *C, int n) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++)
      value += A[row * n + k] * B[k * n + col];
    C[row * n + col] = value;
  }
}

/*
 * Uncoalesced: threadIdx.x maps to row (consecutive threads access
 * memory locations N elements apart in A — strided access).
 */
__global__ void matMulUncoalesced(float *A, float *B, float *C, int n) {
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++)
      value += A[row * n + k] * B[k * n + col];
    C[row * n + col] = value;
  }
}

int main() {
  int n = N;
  size_t size = (size_t)n * n * sizeof(float);

  float *A, *B, *C_coal, *C_uncoal;
  cudaMallocManaged(&A, size);
  cudaMallocManaged(&B, size);
  cudaMallocManaged(&C_coal, size);
  cudaMallocManaged(&C_uncoal, size);

  for (int i = 0; i < n * n; i++) {
    A[i] = (float)(i % 100) / 100.0f;
    B[i] = (float)((i + 1) % 100) / 100.0f;
    C_coal[i] = 0.0f;
    C_uncoal[i] = 0.0f;
  }

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Prefetch all arrays to the GPU so page migrations don't skew timing
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaMemPrefetchAsync(A, size, deviceId);
  cudaMemPrefetchAsync(B, size, deviceId);
  cudaMemPrefetchAsync(C_coal, size, deviceId);
  cudaMemPrefetchAsync(C_uncoal, size, deviceId);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matMulCoalesced<<<numBlocks, threadsPerBlock>>>(A, B, C_coal, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeCoalesced;
  cudaEventElapsedTime(&timeCoalesced, start, stop);

  cudaEventRecord(start);
  matMulUncoalesced<<<numBlocks, threadsPerBlock>>>(A, B, C_uncoal, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeUncoalesced;
  cudaEventElapsedTime(&timeUncoalesced, start, stop);

  bool error = false;
  for (int i = 0; i < n * n && !error; i++) {
    if (fabsf(C_coal[i] - C_uncoal[i]) > 1e-3f) {
      printf("MISMATCH at index %d: coalesced=%.6f, uncoalesced=%.6f\n", i,
             C_coal[i], C_uncoal[i]);
      error = true;
    }
  }
  if (!error)
    printf("Results match!\n");

  printf("Matrix size: %d x %d\n", n, n);
  printf("Block size: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
  printf("Coalesced kernel time:           %.3f ms\n", timeCoalesced);
  printf("Uncoalesced kernel time:         %.3f ms\n", timeUncoalesced);
  printf("Speedup (uncoalesced/coalesced): %.2fx\n",
         timeUncoalesced / timeCoalesced);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_coal);
  cudaFree(C_uncoal);

  return 0;
}
