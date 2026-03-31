# Example 22: Image Convolution — Memory Hierarchy Optimizations

Reveals that **data transfer, not kernel computation, is the bottleneck** for GPU image processing. Three progressively optimized versions are timed with per-phase breakdowns (H2D / kernel / D2H) to make the bottleneck explicit, then two targeted fixes are applied.

## The bottleneck insight

For a 2048×2048 grayscale image (~4 MB):

| Phase | Time | Share of total |
|-------|------|----------------|
| H2D transfer (pageable) | ~1–2 ms | ~40% |
| Kernel execution | ~0.2 ms | ~5% |
| D2H transfer (pageable) | ~2–3 ms | ~55% |

The kernel finishes in a fraction of a millisecond. Optimizing it further would barely matter — the bus transfers take 10–20× longer. This is a universal GPU pattern: **profile first, then optimize the bottleneck**.

## Optimization 1: `__constant__` memory for the filter

The filter kernel is small (9–225 floats), read-only, and — crucially — every thread in a warp reads the **same filter index at the same time** during the inner convolution loop. This is called a *broadcast* access pattern.

The 64 KB constant cache is optimized for exactly this: one memory transaction serves all 32 threads in a warp simultaneously, instead of 32 separate global reads.

```cuda
__constant__ float d_filter_const[225];   // declared at file scope
cudaMemcpyToSymbol(d_filter_const, filter, size);  // upload once
// kernel reads d_filter_const[...] directly — no pointer argument needed
```

## Optimization 2: Pinned (page-locked) host memory

Normal `malloc()` returns *pageable* memory — the OS can swap it to disk. When `cudaMemcpy` is called on pageable memory, CUDA must first copy data into an internal locked staging buffer before DMA can start. That doubles the transfer:

```
Pageable path:  CPU RAM ──stage──► locked buffer ──DMA──► GPU RAM
Pinned path:    CPU RAM ──────────────────DMA────────────► GPU RAM
```

`cudaHostAlloc()` pins the pages in physical RAM, skipping the staging step and roughly halving transfer time.

```cuda
cudaHostAlloc(&h_in,  imageBytes, cudaHostAllocDefault);
cudaHostAlloc(&h_out, imageBytes, cudaHostAllocDefault);
// use h_in / h_out exactly like normal host pointers
cudaFreeHost(h_in);
```

> **Caution:** Pinned pages cannot be swapped by the OS. Over-allocating pinned memory reduces the memory available to the rest of the system.

## Three versions compared

| Version | H2D memory | Filter memory | Fixes applied |
|---------|-----------|---------------|---------------|
| 1. Pageable + global | `malloc` + `cudaMemcpy` | global | baseline |
| 2. Pageable + constant | `malloc` + `cudaMemcpy` | `__constant__` | filter broadcast |
| 3. Pinned + constant | `cudaHostAlloc` | `__constant__` | both |

## Build & run

```bash
nvcc -o output.bin image-convolution-memory.cu
./output.bin
```

## Expected output

```
Image: 2048 x 2048 grayscale   Filter: blur 3x3

Version                  H2D(ms)  Kernel(ms)  D2H(ms)  Total(ms)  FPS eq.
──────────────────────  ────────  ──────────  ────────  ─────────  ───────
1. Pageable+global           1.3        0.24      2.4        3.9    259.6
2. Pageable+const            0.7        0.18      0.8        1.7    580.2
3. Pinned+const              0.5        0.14      0.5        1.1    927.0

KEY INSIGHT:
  The kernel runs in < 1 ms, but each memcpy takes ~10 ms.
  Pinned memory enables direct DMA: no staging copy → ~2x faster transfers.
  __constant__ cache: filter broadcast → one transaction per warp.
```

Actual numbers vary by GPU and system RAM speed. The key ratios to look for:
- **Kernel time is ≪ transfer time** regardless of version
- **Pinned memory halves H2D and D2H** compared to pageable
- **Total FPS roughly doubles** from Version 1 to Version 3

> The webcam filter (`templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu`) uses Version 1 (pageable `cudaMemcpy`). Adding `cudaHostAlloc` to `applyFilterGPU()` would push it from ~105 FPS toward ~200+ FPS.

## See the bottleneck live — webcam filter template

The webcam filter's FPS counter makes the memory bottleneck visible: even with a trivially fast kernel, FPS is capped by PCIe transfers. Run it on a synthetic image to observe this without a webcam.

### Build (once, from repo root)

```bash
cd templates/cuda-webcam-filter/
mkdir -p build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Observe the transfer bottleneck

Run blur with a small kernel (fast compute) and watch the GPU FPS in the overlay.
The GPU time shown includes H2D + kernel + D2H, not just the kernel:

```bash
# 3x3 blur — kernel is ~0.2 ms, but total frame time is much higher
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 3 --preview
```

### Compare small vs large kernel on the same image

A 15×15 kernel does 25× more work per pixel — note how GPU FPS drops relative to 3×3:

```bash
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 3  --preview
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 15 --preview
```

### Run on your own image

```bash
./bin/cuda-webcam-filter --input image --path /path/to/photo.jpg --filter blur --kernel-size 5 --preview
```

The FPS difference between CPU and GPU panels shows the compute speedup. The absolute GPU FPS ceiling is set by the PCIe transfer speed — the memory bottleneck this example teaches.
