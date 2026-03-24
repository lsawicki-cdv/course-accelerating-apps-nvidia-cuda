#include <math.h>
#include <stdio.h>

#define N 4096
#define TILE_WIDTH 32

__global__ void matMulBasic(float *A, float *B, float *C, int n) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++)
      value += A[row * n + k] * B[k * n + col];
    C[row * n + col] = value;
  }
}

__global__ void matMulTiled(float *A, float *B, float *C, int n) {
  int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
  int col = TILE_WIDTH * blockIdx.x + threadIdx.x;

  __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

  float value = 0.0f;
  int numPhases = (n + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int phase = 0; phase < numPhases; phase++) {
    // Load tiles into shared memory with boundary checks
    if (row < n && (phase * TILE_WIDTH + threadIdx.x) < n)
      sh_A[threadIdx.y][threadIdx.x] =
          A[row * n + phase * TILE_WIDTH + threadIdx.x];
    else
      sh_A[threadIdx.y][threadIdx.x] = 0.0f;

    if ((phase * TILE_WIDTH + threadIdx.y) < n && col < n)
      sh_B[threadIdx.y][threadIdx.x] =
          B[(phase * TILE_WIDTH + threadIdx.y) * n + col];
    else
      sh_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads(); // Wait for all threads to finish loading

    for (int k = 0; k < TILE_WIDTH; k++)
      value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];

    __syncthreads(); // Wait before loading next tile
  }

  if (row < n && col < n)
    C[row * n + col] = value;
}

int main() {
  int n = N;
  size_t size = (size_t)n * n * sizeof(float);

  float *A, *B, *C_basic, *C_tiled;
  cudaMallocManaged(&A, size);
  cudaMallocManaged(&B, size);
  cudaMallocManaged(&C_basic, size);
  cudaMallocManaged(&C_tiled, size);

  for (int i = 0; i < n * n; i++) {
    A[i] = (float)(i % 100) / 100.0f;
    B[i] = (float)((i + 1) % 100) / 100.0f;
    C_basic[i] = 0.0f;
    C_tiled[i] = 0.0f;
  }

  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks((n + TILE_WIDTH - 1) / TILE_WIDTH,
                 (n + TILE_WIDTH - 1) / TILE_WIDTH);

  // Prefetch all arrays to the GPU so page migrations don't skew timing
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaMemPrefetchAsync(A, size, deviceId);
  cudaMemPrefetchAsync(B, size, deviceId);
  cudaMemPrefetchAsync(C_basic, size, deviceId);
  cudaMemPrefetchAsync(C_tiled, size, deviceId);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matMulBasic<<<numBlocks, threadsPerBlock>>>(A, B, C_basic, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeBasic;
  cudaEventElapsedTime(&timeBasic, start, stop);

  cudaEventRecord(start);
  matMulTiled<<<numBlocks, threadsPerBlock>>>(A, B, C_tiled, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float timeTiled;
  cudaEventElapsedTime(&timeTiled, start, stop);

  bool error = false;
  for (int i = 0; i < n * n && !error; i++) {
    if (fabsf(C_basic[i] - C_tiled[i]) > 1e-3f) {
      printf("MISMATCH at index %d: basic=%.6f, tiled=%.6f\n", i, C_basic[i],
             C_tiled[i]);
      error = true;
    }
  }
  if (!error)
    printf("Results match!\n");

  // Computational intensity analysis:
  //   Basic:  2N FLOPs per output, reads 2N floats from global memory
  //           => 2N / (2N * 4 bytes) = 0.25 FLOP/B
  //   Tiled:  same FLOPs, but each float is reused TILE_WIDTH times
  //           => (2 * TILE_WIDTH) / 8 FLOP/B
  float basicIntensity = 0.25f;
  float tiledIntensity = (2.0f * TILE_WIDTH) / 8.0f;

  printf("Matrix size:       %d x %d\n", n, n);
  printf("Tile width:        %d\n", TILE_WIDTH);
  printf("Basic kernel time: %.3f ms\n", timeBasic);
  printf("Tiled kernel time: %.3f ms\n", timeTiled);
  printf("Speedup (basic/tiled): %.2fx\n", timeBasic / timeTiled);
  printf("Basic computational intensity: %.4f FLOP/B\n", basicIntensity);
  printf("Tiled computational intensity: %.4f FLOP/B\n", tiledIntensity);
  printf("Shared memory per block: %lu bytes\n",
         2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_basic);
  cudaFree(C_tiled);

  return 0;
}
