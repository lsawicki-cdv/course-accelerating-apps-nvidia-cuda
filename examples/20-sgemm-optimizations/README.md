# Example 20: SGEMM Step-by-Step Optimizations

Six progressive kernels for single-precision matrix multiplication (SGEMM), each adding one GPU optimization technique to the previous.  All kernels run on the same 4096×4096 problem so you can see the cumulative effect.  Based on https://0mean1sigma.com/xgemm/.

## Optimization ladder

| # | Kernel | Key technique | ~% cuBLAS |
|---|--------|---------------|-----------|
| 1 | Naive | One thread per element, threadIdx.x → row (uncoalesced) | ~2% |
| 2 | Coalesced | Swap mapping: threadIdx.x → col (stride-1 B reads) | ~13% |
| 3 | Tiled | 32×32 shared-memory tiles — BLOCK_S reuses per byte | ~18% |
| 4 | 1D Coarsening | Each thread computes TM=8 rows; B cached in register | ~55% |
| 5 | 2D Coarsening | Each thread computes TM×TN=8×8 via outer product | ~64% |
| 6 | Vectorized | float4 loads/stores — 4× wider memory transactions | ~63% |
| — | cuBLAS | Vendor library (reference ceiling) | 100% |

## Key concepts

**Memory coalescing** (kernels 1 → 2): consecutive warp threads must touch consecutive DRAM addresses or the hardware issues one cache-line fetch per thread instead of one per warp.

**Shared memory tiling** (kernel 3): load a tile once into fast on-chip SRAM, then reuse it `TILE_SIZE` times before fetching the next tile.  Reduces global traffic by 32×.

**1D thread coarsening** (kernel 4): assigning each thread multiple output elements lets it cache a value from B in a register and amortise the shared-memory load across TM MACs.

**2D thread coarsening / register tiling** (kernel 5): extend coarsening to both dimensions.  The inner k-step loads `regA[TM]` and `regB[TN]` once and performs TM×TN multiply-accumulates — an *outer product* with TM×TN arithmetic operations per pair of shared-memory reads.

**Vectorized loads** (kernel 6): `float4` instructions issue a single 128-bit load/store instead of four separate 32-bit ones, increasing instruction throughput and hiding latency.

## Block topologies

### Kernels 1–3 (32×32, 1024 threads)

```
Each block computes a 32×32 tile of C.
Every thread computes exactly one output element.

  thread(0,0)  thread(1,0)  ...  thread(31,0)   → C row blockIdx.y*32
  thread(0,1)  thread(1,1)  ...  thread(31,1)   → C row blockIdx.y*32 + 1
  ...
  thread(0,31) thread(1,31) ...  thread(31,31)  → C row blockIdx.y*32 + 31
```

### Kernel 4 — 1D coarsening (64×8, 512 threads)

```
Each block covers a 64×64 output tile.
threadIdx.x (0..63) selects the output column.
threadIdx.y (0..7)  selects the row-group; each thread owns TM=8 rows.

  ty=0, tx=0..63  →  C rows  0.. 7, cols  0..63
  ty=1, tx=0..63  →  C rows  8..15, cols  0..63
  ...
  ty=7, tx=0..63  →  C rows 56..63, cols  0..63
```

### Kernels 5–6 — 2D coarsening (8×8, 64 threads)

```
Each block covers a 64×64 output tile.
Each thread owns a TM×TN = 8×8 sub-tile.

  ty=0,tx=0 → C[ 0.. 7][ 0.. 7]    ty=0,tx=7 → C[ 0.. 7][56..63]
  ty=1,tx=0 → C[ 8..15][ 0.. 7]    ...
  ...
  ty=7,tx=7 → C[56..63][56..63]
```

## Arithmetic intensity

| Kernel | Global loads per output element | FLOP/byte |
|--------|---------------------------------|-----------|
| Naive / Coalesced | 2N floats (A row + B col) | 0.25 |
| Tiled (tile=32) | 2N/BLOCK_S = 2N/32 floats | 8.0 |
| 1D Coarsening | same tiles, TM MACs per B load | 32+ |
| 2D Coarsening | outer product, TM×TN per AB pair | 128+ |

## Build & run

```bash
nvcc -arch=sm_89 -lcublas -o output.bin sgemm-optimizations.cu
./output.bin
```

## Expected output

```
Correctness checks (vs naive, tol 1e-3; cuBLAS tol 2e-2):
  Coalesced            OK
  Tiled                OK
  1D Coarsening        OK
  2D Coarsening        OK
  Vectorized           OK
  cuBLAS               OK

Matrix size: 4096 x 4096   (137 GFLOP)

Kernel                   Time (ms)      GFLOPS    % cuBLAS
------                   ---------      ------    --------
Naive                       XXXX.X        XXX.X        X.X%
Coalesced                    XXX.X        XXX.X       XX.X%
Tiled                        XXX.X       XXXX.X       XX.X%
1D Coarsening                 XX.X       XXXX.X       XX.X%
2D Coarsening                 XX.X       XXXX.X       XX.X%
Vectorized                    XX.X       XXXX.X       XX.X%
cuBLAS                        XX.X       XXXX.X      100.0%
```

> cuBLAS uses FMA and order-of-operations that differ from the scalar kernels; a tolerance of 2e-2 accommodates accumulated rounding across N=4096 additions.
