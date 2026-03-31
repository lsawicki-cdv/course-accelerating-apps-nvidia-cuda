# Example 18: Tensor Cores Intro (WMMA API)

Introduces NVIDIA tensor cores using the WMMA (Warp Matrix Multiply-Accumulate) API. A standard CUDA-core FP32 matrix multiply is placed side-by-side with a tensor core kernel to demonstrate the performance difference.

## Key concepts

- Tensor cores operate at the **warp level** — all 32 threads in a warp cooperate to compute one 16×16 matrix tile
- Inputs are **half precision (FP16)**, output accumulates in **FP32**
- `nvcuda::wmma` fragments abstract the register-level data movement
- Grid is sized so each block (1 warp) owns exactly one 16×16 output tile

## WMMA operations

```
fill_fragment   — initialize accumulator to zero
load_matrix_sync  — load a 16×16 tile from memory into a fragment
mma_sync          — D = A·B + C (one tensor core operation)
store_matrix_sync — write fragment result back to memory
```

## Build & run

```bash
nvcc -arch=sm_89 -o output.bin tensor-cores-intro.cu
./output.bin
```

## Expected output

```
Results match!
Matrix size:            1024 x 1024 x 1024
CUDA core kernel time:  X.XXX ms  (XXX.X GFLOPS)
WMMA tensor core time:  X.XXX ms  (XXXX.X GFLOPS)
Speedup (WMMA/CUDA):    X.XXx
```

Tensor cores typically show **5–10x** higher GFLOPS than CUDA cores for this matrix size on Ada Lovelace GPUs.
