# Example 23: Image Convolution — Shared Memory Halo Pattern

Teaches the **halo (apron) tiling** technique for 2D stencil computations. Three kernel variants are benchmarked with both a 3×3 and a 15×15 filter to show that the benefit of shared memory scales with filter size — and can actually hurt performance for small filters.

## The problem: redundant global memory reads

In a naive convolution kernel, each thread fetches its own neighborhood from global memory. Threads in neighboring positions fetch **overlapping regions**, so the same input pixels are loaded multiple times:

- For a **3×3 filter**: each pixel is read up to 9 times across overlapping windows
- For a **15×15 filter**: each pixel is read up to 225 times

Shared memory lets a block load its required pixels **once cooperatively**, then all threads read from fast on-chip SRAM.

## The halo (apron) pattern

A 16×16 block of output pixels needs a slightly larger region of input pixels — the extra border is called the **halo** or **apron**:

```
┌─────────────────────────┐
│   halo  (top apron)     │  ← radius rows above block
├──────┬──────────┬────────┤
│ halo │  BLOCK   │ halo  │  ← radius cols on each side
│(left)│ 16 × 16  │(right)│
├──────┴──────────┴────────┤
│   halo  (bottom apron)  │  ← radius rows below block
└─────────────────────────┘

For radius=1 (3×3):  tile = 18×18 = 324 bytes per block
For radius=7 (15×15): tile = 30×30 = 900 bytes per block
```

The tile is declared in shared memory:
```cuda
__shared__ unsigned char tile[BLOCK_H + 2 * MAX_RADIUS][BLOCK_W + 2 * MAX_RADIUS];
```

## Cooperative tile loading with a stride loop

The tile has more pixels than the thread block, so each thread may need to load more than one pixel. This uses the same **stride loop** pattern from Example 5 (grid-stride):

```cuda
int tid = threadIdx.y * BLOCK_W + threadIdx.x;
for (int i = tid; i < tileW * tileH; i += BLOCK_W * BLOCK_H) {
    int ty = i / tileW,  tx = i % tileW;
    int gx = clamp(originX + tx, 0, W-1);
    int gy = clamp(originY + ty, 0, H-1);
    tile[ty][tx] = d_in[gy * W + gx];   // load from global
}
__syncthreads();  // barrier: wait until ALL threads finish loading
```

The `__syncthreads()` serves the same role as in Example 17 (tiled matrix multiply): no thread may read the tile until every thread has finished writing it.

After the barrier, each thread reads from `tile[threadIdx.y + ky][threadIdx.x + kx]` — no bounds checking needed since the halo was already padded with clamped values.

## Three kernel variants

| Variant | Memory for filter | Memory for pixels | Notes |
|---------|-----------------|-------------------|-------|
| V1 Naive global | global pointer | global memory | baseline from Ex. 21 |
| V2 Constant | `__constant__` cache | global memory | broadcast optimization from Ex. 22 |
| V3 Shared tile | `__constant__` cache | shared memory | halo pattern |

## Build & run

```bash
nvcc -o output.bin image-convolution-shared-memory.cu
./output.bin
```

## Expected output

```
Image: 2048 x 2048 grayscale

--- Filter: blur 3x3  (radius=1, 9 weights) ---
  Reuse ratio (redundant global reads without shared mem): 9x
  naive global                0.18   1.00x
  constant mem                0.14   1.21x
  Correctness vs naive: max diff = 0.0 (PASS)
  shared tile                 0.19   0.91x
  Correctness vs naive: max diff = 0.0 (PASS)
  Shared memory per block: 324 bytes  (tile: 18x18)

--- Filter: blur 15x15  (radius=7, 225 weights) ---
  Reuse ratio (redundant global reads without shared mem): 225x
  naive global                2.32   1.00x
  constant mem                2.24   1.04x
  Correctness vs naive: max diff = 0.0 (PASS)
  shared tile                 1.17   1.99x
  Correctness vs naive: max diff = 0.0 (PASS)
  Shared memory per block: 900 bytes  (tile: 30x30)
```

## Key takeaway: shared memory is not always faster

| Filter | Shared tile speedup | Why |
|--------|-------------------|-----|
| 3×3 (radius=1) | ~0.9× (slightly **slower**) | Halo-load overhead exceeds savings for only 9 weights |
| 15×15 (radius=7) | ~2× faster | 225 redundant global reads replaced by 1 shared read |

**Rule of thumb:** use shared memory tiling for convolution when `radius ≥ 4`. For small filters, the constant cache (`__constant__` memory, Ex. 22) is a better choice — zero cooperative-load overhead.

This is a general principle: the benefit of shared memory scales with the **ratio of redundant global reads to useful computation**. For small stencils that ratio is low; for large stencils it is high.

> The webcam filter (`templates/cuda-webcam-filter`) uses the naive global kernel. For `--kernel-size 15`, replacing it with `convSharedTile` would give ~2× kernel speedup.

## See the kernel size effect live — webcam filter template

The webcam filter's GPU FPS drops visibly as the kernel grows. Running both sizes back-to-back makes the cost of large convolutions concrete — and motivates the shared memory optimization.

### Build (once, from repo root)

```bash
cd templates/cuda-webcam-filter/
mkdir -p build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Compare 3×3 vs 15×15 kernel on a synthetic image

Watch how GPU FPS changes between runs as the kernel size grows from 9 to 225 weights:

```bash
# Small kernel — fast compute, GPU overhead dominated by transfers
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 3  --preview

# Large kernel — compute starts to dominate, GPU FPS drops noticeably
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 15 --preview
```

The gradient pattern makes blur and sharpen effects easy to spot: smooth transitions flatten further with blur and sharpen into ringing artefacts with sharpen.

### All filters with large kernel

```bash
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur    --kernel-size 15 --preview
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter sharpen --kernel-size 5  --preview
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter edge    --kernel-size 3  --preview
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter emboss  --kernel-size 3  --preview
```

### Run on your own image

```bash
./bin/cuda-webcam-filter --input image --path /path/to/photo.jpg --filter blur --kernel-size 15 --preview
```

A photograph shows the blur most clearly — fine detail (hair, grass, text) disappears with `--kernel-size 15` in a way that a synthetic pattern cannot demonstrate.
