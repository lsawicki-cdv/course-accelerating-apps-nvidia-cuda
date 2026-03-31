#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

#define N 4096       // matrix dimension (must be divisible by BLOCK_S and BM/BN/BK)
#define BLOCK_S 32   // tile size for naive / coalesced / tiled kernels

// Coarsening kernel tile parameters
#define BM 64   // block tile rows
#define BN 64   // block tile columns
#define BK 8    // block tile K-depth
#define TM 8    // output rows per thread (1D + 2D coarsening)
#define TN 8    // output cols per thread (2D coarsening only)

// ============================================================
// Kernel 1: Naive (uncoalesced)
//
// threadIdx.x → row: consecutive threads in a warp stride across
// rows, so A accesses are stride-N (one cache line per thread) and
// C writes are also stride-N.  This is the worst-case baseline.
//
// Arithmetic intensity: 0.25 FLOP/byte
// ============================================================
__global__ void matMulNaive(float *A, float *B, float *C, int n) {
  int row = blockIdx.x * BLOCK_S + threadIdx.x;
  int col = blockIdx.y * BLOCK_S + threadIdx.y;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++)
      value += A[row * n + k] * B[k * n + col];
    C[row * n + col] = value;
  }
}

// ============================================================
// Kernel 2: Coalesced
//
// threadIdx.x → col: consecutive threads access consecutive
// elements of B (stride-1) and write C stride-1 — both coalesced.
// A is broadcast (all threads share the same row) — L1 friendly.
//
// Arithmetic intensity: 0.25 FLOP/byte (same as naive, but far
// fewer DRAM transactions due to coalesced access pattern)
// ============================================================
__global__ void matMulCoalesced(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * BLOCK_S + threadIdx.y;
  int col = blockIdx.x * BLOCK_S + threadIdx.x;

  if (row < n && col < n) {
    float value = 0.0f;
    for (int k = 0; k < n; k++)
      value += A[row * n + k] * B[k * n + col];
    C[row * n + col] = value;
  }
}

// ============================================================
// Kernel 3: Tiled shared memory
//
// Loads BLOCK_S×BLOCK_S tiles of A and B into shared memory so
// each float is reused BLOCK_S times instead of being re-fetched
// from global memory on every K step.
//
// Arithmetic intensity: (2 * BLOCK_S) / 8 = 8.0 FLOP/byte
// ============================================================
__global__ void matMulTiled(float *A, float *B, float *C, int n) {
  int row = BLOCK_S * blockIdx.y + threadIdx.y;
  int col = BLOCK_S * blockIdx.x + threadIdx.x;

  __shared__ float sh_A[BLOCK_S][BLOCK_S];
  __shared__ float sh_B[BLOCK_S][BLOCK_S];

  float value = 0.0f;
  int numPhases = (n + BLOCK_S - 1) / BLOCK_S;

  for (int phase = 0; phase < numPhases; phase++) {
    if (row < n && (phase * BLOCK_S + threadIdx.x) < n)
      sh_A[threadIdx.y][threadIdx.x] = A[row * n + phase * BLOCK_S + threadIdx.x];
    else
      sh_A[threadIdx.y][threadIdx.x] = 0.0f;

    if ((phase * BLOCK_S + threadIdx.y) < n && col < n)
      sh_B[threadIdx.y][threadIdx.x] = B[(phase * BLOCK_S + threadIdx.y) * n + col];
    else
      sh_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
    for (int k = 0; k < BLOCK_S; k++)
      value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];
    __syncthreads();
  }

  if (row < n && col < n)
    C[row * n + col] = value;
}

// ============================================================
// Kernel 4: 1D thread coarsening
//
// Each thread computes TM=8 consecutive output elements along the
// row dimension for one fixed column.  The B column value is
// cached in a register (regB) before the TM inner MACs, reducing
// shared-memory reads by TM×.
//
// Block: dim3(BN=64, BM/TM=8) = 512 threads
// Grid:  dim3(N/BN=64, N/BM=64)
// Shared: sh_A[BM][BK] + sh_B[BK][BN] = 4096 bytes
//
// Each of the 512 threads loads exactly 1 element of sh_A and
// 1 element of sh_B → no stride loop required.
// ============================================================
__global__ void matMulCoarse1D(float *A, float *B, float *C, int n) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int tx  = threadIdx.x;          // 0..BN-1, output column within tile
  int ty  = threadIdx.y;          // 0..BM/TM-1, row-group within tile
  int tid = ty * BN + tx;         // linear thread index 0..511

  __shared__ float sh_A[BM][BK]; // A tile:   BM rows × BK cols
  __shared__ float sh_B[BK][BN]; // B tile:   BK rows × BN cols

  float regC[TM];
  for (int tm = 0; tm < TM; tm++)
    regC[tm] = 0.0f;

  for (int phase = 0; phase < n / BK; phase++) {
    // Load tiles: 512 threads, 512 elements each → 1 load per thread
    sh_A[tid / BK][tid % BK] =
        A[(blockRow * BM + tid / BK) * n + phase * BK + tid % BK];
    sh_B[tid / BN][tid % BN] =
        B[(phase * BK + tid / BN) * n + blockCol * BN + tid % BN];

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      float regB = sh_B[k][tx]; // cache B element once for TM MACs
      for (int tm = 0; tm < TM; tm++)
        regC[tm] += sh_A[ty * TM + tm][k] * regB;
    }

    __syncthreads();
  }

  for (int tm = 0; tm < TM; tm++)
    C[(blockRow * BM + ty * TM + tm) * n + blockCol * BN + tx] = regC[tm];
}

// ============================================================
// Kernel 5: 2D thread coarsening (register tiling / outer product)
//
// Each thread computes a TM×TN = 8×8 output sub-tile.  The inner
// k-loop loads a column of the A tile (regA[TM]) and a row of the
// B tile (regB[TN]) into registers, then accumulates the outer
// product into regC[TM][TN].  This gives 2*TM*TN MACs per pair
// of shared-memory reads — maximum register reuse.
//
// Block: dim3(BN/TN=8, BM/TM=8) = 64 threads
// Grid:  dim3(N/BN=64, N/BM=64)
// Shared: sh_A[BM][BK] + sh_B[BK][BN] = 4096 bytes
//
// 64 threads fill 512-element tiles → 8 global loads per thread
// via a stride loop.
// ============================================================
__global__ void matMulCoarse2D(float *A, float *B, float *C, int n) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int tx  = threadIdx.x;              // 0..BN/TN-1 = 0..7
  int ty  = threadIdx.y;              // 0..BM/TM-1 = 0..7
  int tid = ty * (BN / TN) + tx;     // linear thread index 0..63

  __shared__ float sh_A[BM][BK];
  __shared__ float sh_B[BK][BN];

  float regA[TM], regB[TN];
  float regC[TM][TN];
  for (int tm = 0; tm < TM; tm++)
    for (int tn = 0; tn < TN; tn++)
      regC[tm][tn] = 0.0f;

  int nThreads = (BN / TN) * (BM / TM); // 64

  for (int phase = 0; phase < n / BK; phase++) {
    // Fill sh_A: BM*BK = 512 elements, 64 threads → 8 loads each
    for (int idx = tid; idx < BM * BK; idx += nThreads)
      sh_A[idx / BK][idx % BK] =
          A[(blockRow * BM + idx / BK) * n + phase * BK + idx % BK];

    // Fill sh_B: BK*BN = 512 elements, 64 threads → 8 loads each
    for (int idx = tid; idx < BK * BN; idx += nThreads)
      sh_B[idx / BN][idx % BN] =
          B[(phase * BK + idx / BN) * n + blockCol * BN + idx % BN];

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      // Cache A column and B row into registers
      for (int tm = 0; tm < TM; tm++) regA[tm] = sh_A[ty * TM + tm][k];
      for (int tn = 0; tn < TN; tn++) regB[tn] = sh_B[k][tx * TN + tn];
      // Outer product: TM*TN MACs per k step
      for (int tm = 0; tm < TM; tm++)
        for (int tn = 0; tn < TN; tn++)
          regC[tm][tn] += regA[tm] * regB[tn];
    }

    __syncthreads();
  }

  for (int tm = 0; tm < TM; tm++)
    for (int tn = 0; tn < TN; tn++)
      C[(blockRow * BM + ty * TM + tm) * n + blockCol * BN + tx * TN + tn] =
          regC[tm][tn];
}

// ============================================================
// Kernel 6: Vectorized (float4 loads and stores)
//
// Same outer structure as kernel 5 but uses 128-bit float4
// transactions to load 4 floats per instruction when filling
// shared memory and writing the output tile.  This doubles the
// effective memory bandwidth by halving the number of load/store
// instructions.
//
// Alignment guarantees:
//   - cudaMallocManaged aligns to 256 bytes
//   - each row of A/B is N*4 = 16384 bytes (multiple of 16)
//   - phase*BK*4 = phase*32 bytes (multiple of 16) → always aligned
//   - tx*TN*4 = tx*32 bytes (multiple of 16) → stores always aligned
//
// Block: dim3(BN/TN=8, BM/TM=8) = 64 threads
// Grid:  dim3(N/BN=64, N/BM=64)
// ============================================================
__global__ void matMulVectorized(float *A, float *B, float *C, int n) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int tx  = threadIdx.x;
  int ty  = threadIdx.y;
  int tid = ty * (BN / TN) + tx; // 0..63

  // Flat shared memory with 16-byte alignment for float4 reinterpretation
  __shared__ __align__(16) float sh_A[BM * BK]; // 512 floats = 128 float4s
  __shared__ __align__(16) float sh_B[BK * BN]; // 512 floats = 128 float4s

  float regA[TM], regB[TN];
  float regC[TM][TN];
  for (int tm = 0; tm < TM; tm++)
    for (int tn = 0; tn < TN; tn++)
      regC[tm][tn] = 0.0f;

  int nThreads = (BN / TN) * (BM / TM); // 64

  for (int phase = 0; phase < n / BK; phase++) {
    // float4 load sh_A: 128 float4 ops, 64 threads → 2 loads per thread
    // Each float4 covers 4 consecutive cols within one row of the BK strip
    for (int i = tid; i < (BM * BK) / 4; i += nThreads) {
      int elem = i * 4;
      int row  = elem / BK; // row in [0, BM)
      int col  = elem % BK; // col offset: always 0 or 4 since BK=8
      reinterpret_cast<float4 *>(sh_A)[i] =
          *reinterpret_cast<const float4 *>(
              &A[(blockRow * BM + row) * n + phase * BK + col]);
    }
    // float4 load sh_B: 128 float4 ops, 64 threads → 2 loads per thread
    for (int i = tid; i < (BK * BN) / 4; i += nThreads) {
      int elem = i * 4;
      int row  = elem / BN; // row in [0, BK)
      int col  = elem % BN; // col offset: 0, 4, 8, ..., 60
      reinterpret_cast<float4 *>(sh_B)[i] =
          *reinterpret_cast<const float4 *>(
              &B[(phase * BK + row) * n + blockCol * BN + col]);
    }

    __syncthreads();

    // Compute: access flat shared memory via index arithmetic
    for (int k = 0; k < BK; k++) {
      for (int tm = 0; tm < TM; tm++)
        regA[tm] = sh_A[(ty * TM + tm) * BK + k];
      for (int tn = 0; tn < TN; tn++)
        regB[tn] = sh_B[k * BN + tx * TN + tn];
      for (int tm = 0; tm < TM; tm++)
        for (int tn = 0; tn < TN; tn++)
          regC[tm][tn] += regA[tm] * regB[tn];
    }

    __syncthreads();
  }

  // float4 store: 2 float4 writes per output row (TN=8 cols = 2 × float4)
  for (int tm = 0; tm < TM; tm++) {
    float *c_ptr =
        &C[(blockRow * BM + ty * TM + tm) * n + blockCol * BN + tx * TN];
    reinterpret_cast<float4 *>(c_ptr)[0] =
        make_float4(regC[tm][0], regC[tm][1], regC[tm][2], regC[tm][3]);
    reinterpret_cast<float4 *>(c_ptr)[1] =
        make_float4(regC[tm][4], regC[tm][5], regC[tm][6], regC[tm][7]);
  }
}

// ============================================================
// Helpers
// ============================================================
static double gflops(double flops, float ms) {
  return flops / ms / 1e6;
}

static void check(const char *label, float *ref, float *got, int nn,
                  float tol) {
  for (int i = 0; i < nn; i++) {
    if (fabsf(ref[i] - got[i]) > tol) {
      printf("MISMATCH (%s) at %d: ref=%.6f got=%.6f\n", label, i, ref[i],
             got[i]);
      return;
    }
  }
  printf("  %-20s OK\n", label);
}

// ============================================================
// Main
// ============================================================
int main() {
  int n = N;
  size_t sz = (size_t)n * n * sizeof(float);
  double flops = 2.0 * n * n * n; // 2*N^3 FLOPs for C = A*B

  float *A, *B;
  float *C_naive, *C_coal, *C_tiled, *C_1d, *C_2d, *C_vec, *C_cublas;

  cudaMallocManaged(&A,        sz);
  cudaMallocManaged(&B,        sz);
  cudaMallocManaged(&C_naive,  sz);
  cudaMallocManaged(&C_coal,   sz);
  cudaMallocManaged(&C_tiled,  sz);
  cudaMallocManaged(&C_1d,     sz);
  cudaMallocManaged(&C_2d,     sz);
  cudaMallocManaged(&C_vec,    sz);
  cudaMallocManaged(&C_cublas, sz);

  for (int i = 0; i < n * n; i++) {
    A[i] = (float)(i % 100) / 100.0f;
    B[i] = (float)((i + 1) % 100) / 100.0f;
  }

  // Prefetch all arrays to GPU so page migrations don't skew timing
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaMemPrefetchAsync(A,        sz, deviceId);
  cudaMemPrefetchAsync(B,        sz, deviceId);
  cudaMemPrefetchAsync(C_naive,  sz, deviceId);
  cudaMemPrefetchAsync(C_coal,   sz, deviceId);
  cudaMemPrefetchAsync(C_tiled,  sz, deviceId);
  cudaMemPrefetchAsync(C_1d,     sz, deviceId);
  cudaMemPrefetchAsync(C_2d,     sz, deviceId);
  cudaMemPrefetchAsync(C_vec,    sz, deviceId);
  cudaMemPrefetchAsync(C_cublas, sz, deviceId);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- Kernel 1: Naive ---
  dim3 blockBasic(BLOCK_S, BLOCK_S);
  dim3 gridBasic(n / BLOCK_S, n / BLOCK_S);

  cudaEventRecord(start);
  matMulNaive<<<gridBasic, blockBasic>>>(A, B, C_naive, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tNaive;
  cudaEventElapsedTime(&tNaive, start, stop);

  // --- Kernel 2: Coalesced ---
  cudaEventRecord(start);
  matMulCoalesced<<<gridBasic, blockBasic>>>(A, B, C_coal, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tCoal;
  cudaEventElapsedTime(&tCoal, start, stop);

  // --- Kernel 3: Tiled ---
  cudaEventRecord(start);
  matMulTiled<<<gridBasic, blockBasic>>>(A, B, C_tiled, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tTiled;
  cudaEventElapsedTime(&tTiled, start, stop);

  // --- Kernel 4: 1D coarsening ---
  // Block covers BM rows × BN cols; each thread handles TM rows × 1 col
  dim3 block1D(BN, BM / TM);   // (64, 8) = 512 threads
  dim3 grid1D(n / BN, n / BM); // (64, 64)

  cudaEventRecord(start);
  matMulCoarse1D<<<grid1D, block1D>>>(A, B, C_1d, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t1D;
  cudaEventElapsedTime(&t1D, start, stop);

  // --- Kernel 5: 2D coarsening ---
  // Block covers BM rows × BN cols; each thread handles TM×TN outputs
  dim3 block2D(BN / TN, BM / TM); // (8, 8) = 64 threads
  dim3 grid2D(n / BN, n / BM);    // (64, 64)

  cudaEventRecord(start);
  matMulCoarse2D<<<grid2D, block2D>>>(A, B, C_2d, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t2D;
  cudaEventElapsedTime(&t2D, start, stop);

  // --- Kernel 6: Vectorized ---
  cudaEventRecord(start);
  matMulVectorized<<<grid2D, block2D>>>(A, B, C_vec, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tVec;
  cudaEventElapsedTime(&tVec, start, stop);

  // --- cuBLAS baseline ---
  // cublasSgemm operates in column-major order.
  // Row-major C = A*B  ≡  column-major C^T = B^T * A^T.
  // Passing row-major B as cuBLAS matrix-A and row-major A as matrix-B
  // with CUBLAS_OP_N (no transpose) gives the correct row-major result.
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n, n, n, &alpha, B, n, A, n, &beta, C_cublas, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tCublas;
  cudaEventElapsedTime(&tCublas, start, stop);

  // --- Correctness (all kernels vs naive) ---
  printf("Correctness checks (vs naive, tol 1e-3; cuBLAS tol 2e-2):\n");
  check("Coalesced",     C_naive, C_coal,   n * n, 1e-3f);
  check("Tiled",         C_naive, C_tiled,  n * n, 1e-3f);
  check("1D Coarsening", C_naive, C_1d,     n * n, 1e-3f);
  check("2D Coarsening", C_naive, C_2d,     n * n, 1e-3f);
  check("Vectorized",    C_naive, C_vec,    n * n, 1e-3f);
  check("cuBLAS",        C_naive, C_cublas, n * n, 2e-2f);
  printf("\n");

  // --- Performance table ---
  double gCublas = gflops(flops, tCublas);

  printf("Matrix size: %d x %d   (%.0f GFLOP)\n\n", n, n, flops / 1e9);
  printf("%-22s  %10s  %10s  %10s\n",
         "Kernel", "Time (ms)", "GFLOPS", "% cuBLAS");
  printf("%-22s  %10s  %10s  %10s\n",
         "------", "---------", "------", "--------");

  #define ROW(label, t) \
    printf("%-22s  %10.1f  %10.1f  %9.1f%%\n", \
           label, (double)(t), gflops(flops, t), \
           100.0 * gflops(flops, t) / gCublas)

  ROW("Naive",          tNaive);
  ROW("Coalesced",      tCoal);
  ROW("Tiled",          tTiled);
  ROW("1D Coarsening",  t1D);
  ROW("2D Coarsening",  t2D);
  ROW("Vectorized",     tVec);
  ROW("cuBLAS",         tCublas);

  #undef ROW

  // --- Cleanup ---
  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_naive);
  cudaFree(C_coal);
  cudaFree(C_tiled);
  cudaFree(C_1d);
  cudaFree(C_2d);
  cudaFree(C_vec);
  cudaFree(C_cublas);

  return 0;
}
