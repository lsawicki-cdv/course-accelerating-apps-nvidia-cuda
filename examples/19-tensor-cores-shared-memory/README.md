# Example 19: Tensor Cores with Shared Memory

Extends example 18 by staging input tiles through shared memory before loading WMMA fragments. Mirrors the progression from example 16→17 (global memory → shared memory tiling), but applied to tensor cores.

## Key concepts

- **Naive WMMA**: each warp loads its 16×16 fragments independently from global memory
- **Shared-memory WMMA**: all 4 warps in a block cooperate to load a 32×32 tile of A and B into `__shared__` memory first, then each warp loads its fragment from shared memory — replacing 4 independent global-memory reads with 1 cooperative load

## Block topology

```
128 threads per block = 4 warps (2×2 arrangement)
Each block computes a 32×32 output tile (4 × 16×16 WMMA tiles)

  warp(0,0) → C[0:16, 0:16]
  warp(0,1) → C[0:16, 16:32]
  warp(1,0) → C[16:32, 0:16]
  warp(1,1) → C[16:32, 16:32]
```

## Shared memory layout

```
sh_A[BLOCK_M][WMMA_K]  →  32×16 half elements  (512 B)
sh_B[WMMA_K][BLOCK_N]  →  16×32 half elements  (512 B)
```

Two `__syncthreads()` per K-phase: one after filling shared memory, one before overwriting it in the next phase.

## Build & run

```bash
nvcc -arch=sm_89 -o output.bin tensor-cores-shared-memory.cu
./output.bin
```

## Expected output

```
Results match!
Matrix size:                1024 x 1024 x 1024
Block tile size:            32 x 32  (4 warps)
Naive WMMA time:            X.XXX ms  (XXXX.X GFLOPS)
Shared-memory WMMA time:    X.XXX ms  (XXXX.X GFLOPS)
Speedup (shared/naive):     X.XXx
```

The speedup at 1024×1024 is modest (~1.2x) because matrices fit in the RTX 4060's 24 MB L2 cache. The benefit of shared memory staging grows with larger matrices where L2 pressure increases.
