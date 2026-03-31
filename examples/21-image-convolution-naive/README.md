# Example 21: Image Convolution — Naive GPU

Introduces 2D image convolution as a CUDA workload. A CPU reference implementation is compared against a naive GPU kernel that maps one thread to each output pixel. Five classic filters are demonstrated on a 2048×2048 synthetic grayscale image.

## What is convolution?

Each output pixel is a weighted sum of its neighbors in the input image. The weights come from a small "filter kernel" matrix:

```
output[y][x] = SUM over (ky, kx) of:
                 filter[ky + radius][kx + radius]
                 * input[clamp(y + ky, 0, H-1)][clamp(x + kx, 0, W-1)]
```

Pixels outside the image boundary are handled with **clamp-to-edge**: the nearest border pixel is reused. This is the same boundary strategy used in `templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu`.

## Why GPU acceleration fits perfectly

Every output pixel is computed **independently** — there are no data dependencies between outputs. That means all 4 million pixels of a 2048×2048 image can be computed simultaneously, one GPU thread per pixel.

## Thread-to-pixel mapping

```
threadIdx.x + blockIdx.x * blockDim.x  →  pixel column (x)
threadIdx.y + blockIdx.y * blockDim.y  →  pixel row    (y)

Block size:  16 × 16  (256 threads per block)
Grid size:   ceil(2048/16) × ceil(2048/16)  =  128 × 128 blocks
```

This is exactly the grid/block layout used in the webcam filter template.

## Filters demonstrated

| Name | Kernel (3×3) | Effect |
|------|-------------|--------|
| Identity | center=1, rest=0 | Pass-through, no change |
| Blur | all 1/9 | Average of neighborhood — smooths noise |
| Sharpen | `[0,-1,0,-1,5,-1,0,-1,0]` | Amplifies local contrast |
| Edge detection | `[-1,-1,-1,-1,8,-1,-1,-1,-1]` | Highlights boundaries |
| Emboss | diagonal gradient | Pseudo-3D relief effect |

## Key concept: only kernel time is measured here

Example 21 times only the GPU kernel execution. Example 22 reveals the full picture by also timing the memory transfers — which turn out to be the dominant cost.

## Build & run

```bash
nvcc -o output.bin image-convolution-naive.cu
./output.bin
```

## Expected output

```
Image: 2048 x 2048 grayscale (4194304 pixels)

Filter: identity          CPU:  160.0 ms   GPU:   0.2 ms   Speedup: 1000x   MaxDiff: 0.0 (PASS)
Filter: blur 3x3          CPU:  150.0 ms   GPU:   0.2 ms   Speedup:  950x   MaxDiff: 1.0 (PASS)
Filter: sharpen           CPU:  150.0 ms   GPU:   0.2 ms   Speedup:  980x   MaxDiff: 0.0 (PASS)
Filter: edge det.         CPU:  145.0 ms   GPU:   0.2 ms   Speedup:  950x   MaxDiff: 0.0 (PASS)
Filter: emboss            CPU:  150.0 ms   GPU:   0.2 ms   Speedup:  970x   MaxDiff: 0.0 (PASS)

Block size: 16x16  Grid: 128x128
```

**GPU speedup is ~1000x** for the kernel alone. The blur filter may show `MaxDiff: 1.0` — that is expected: `1/9` is not exactly representable in float, causing a rounding difference of at most 1 in the 0–255 byte range.

> The 1000x figure measures kernel execution only. Example 22 shows why the real-world FPS is much lower once memory transfers are counted.

## See the filters live — webcam filter template

The production app applies the same kernels to a real-time video stream and displays CPU vs GPU side-by-side with an FPS overlay.

### Build (once, from repo root)

```bash
cd templates/cuda-webcam-filter/
mkdir -p build && cd build
cmake ..
cmake --build . -j $(nproc)
```

### Run each filter on a synthetic image (no webcam required)

All commands below use `--input synthetic` so no camera or image file is needed.
Add `--preview` to show the original image alongside the filtered result.

```bash
# Blur — softens edges by averaging the neighborhood
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 3 --preview

# Sharpen — amplifies local contrast
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter sharpen --kernel-size 3 --preview

# Edge detection — shows only boundaries, black interior
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter edge --kernel-size 3 --preview

# Emboss — pseudo-3D relief along diagonal edges
./bin/cuda-webcam-filter --input synthetic --synthetic gradient --filter emboss --kernel-size 3 --preview
```

Other synthetic patterns (`gradient`, `noise`) can be substituted for `checkerboard`.

### Run on your own image

```bash
./bin/cuda-webcam-filter --input image --path /path/to/photo.jpg --filter edge --kernel-size 3 --preview
```

The window title shows per-frame timings and FPS for both CPU and GPU paths.
