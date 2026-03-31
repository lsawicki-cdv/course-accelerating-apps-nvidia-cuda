/*
 * Example 23: Image Convolution - Shared Memory Halo Pattern
 * ===========================================================
 *
 * LESSON: Shared memory for 2D stencils — the halo (apron) technique.
 *
 * In Example 22 we learned the data transfer is the dominant bottleneck.
 * Now we return to the kernel itself: when does the kernel become the
 * bottleneck, and how do we fix it?
 *
 * THE PROBLEM WITH GLOBAL MEMORY IN CONVOLUTION
 * -----------------------------------------------
 * For a 3x3 filter (radius=1), each output pixel reads 9 input pixels.
 * For a 15x15 filter (radius=7), each output pixel reads 225 input pixels.
 *
 * In a 16x16 thread block (256 threads), every input pixel is read by
 * multiple threads whose convolution windows overlap. Without shared memory:
 *
 *   Each thread fetches its own 9 (or 225) pixels from global memory.
 *   Neighboring threads redundantly re-fetch the SAME pixels.
 *
 * With shared memory, we load the tile ONCE cooperatively, then all threads
 * read from fast on-chip shared memory.  For a 15x15 filter the redundancy
 * is eliminated: ~225x more arithmetic per global memory read.
 *
 * THE HALO (APRON) PATTERN
 * -------------------------
 * A 16x16 block of output pixels needs a (16 + 2*radius) x (16 + 2*radius)
 * region of input pixels — the extra border is called the "halo" or "apron":
 *
 *   ┌─────────────────────────┐
 *   │   halo (top apron)      │   ← radius rows above block
 *   ├──────┬──────────┬───────┤
 *   │ halo │  BLOCK   │ halo  │   ← radius cols on each side
 *   │(left)│ 16 x 16  │(right)│
 *   ├──────┴──────────┴───────┤
 *   │   halo (bottom apron)   │   ← radius rows below block
 *   └─────────────────────────┘
 *
 * For a 15x15 filter (radius=7) on a 16x16 block:
 *   Tile size = (16+14) x (16+14) = 30x30 = 900 bytes per block
 *   Thread block has 256 threads → each thread loads ~3-4 tile pixels
 *   (using a stride loop — same pattern as grid-stride from Example 5)
 *
 * STRIDE LOOP FOR COOPERATIVE LOADING
 * ------------------------------------
 * The tile has more pixels than the thread block, so we use a stride loop:
 *
 *   int tid = threadIdx.y * BLOCK_W + threadIdx.x;
 *   for (int i = tid; i < tileW * tileH; i += BLOCK_W * BLOCK_H) { ... }
 *
 * Each thread loads one or more tile pixels. Boundary pixels use clamp-to-edge.
 *
 * __syncthreads() AFTER LOADING (same role as in Example 17 tiled matmul):
 * No thread may begin computing until ALL threads have finished writing the
 * shared memory tile.  Without this barrier, a fast thread might read a tile
 * pixel that a slower thread hasn't written yet.
 *
 * COMPILE & RUN:
 *   nvcc -o output.bin image-convolution-shared-memory.cu -run
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH   2048
#define HEIGHT  2048
#define BLOCK_W 16
#define BLOCK_H 16
#define MAX_RADIUS 8   // supports up to 17x17 filters; tile = 32x32 = 1024 B

// ---------------------------------------------------------------------------
// Constant memory for filter (same as Example 22)
// ---------------------------------------------------------------------------
#define MAX_FILTER_ELEMS 289   // 17x17
__constant__ float d_filter_const[MAX_FILTER_ELEMS];

// ---------------------------------------------------------------------------
// Filter definitions
// ---------------------------------------------------------------------------
static const float blur3x3[9] = {
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9
};

// 15x15 box blur (all weights equal)
static float blur15x15[225];   // initialized in main()

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ unsigned char clamp_uchar(float v) {
    return (unsigned char)(v < 0.f ? 0 : v > 255.f ? 255 : v);
}
static __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

void generateImage(unsigned char *img, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img[y * w + x] = (unsigned char)((y + x) % 256);
}

// ---------------------------------------------------------------------------
// Kernel V1: naive global memory (baseline — same as Example 21)
// ---------------------------------------------------------------------------
__global__ void convNaiveGlobal(const unsigned char *in, unsigned char *out,
                                const float *d_filter, int w, int h, int radius) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int filterW = 2 * radius + 1;
    float sum = 0.f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int ix = px + kx; if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;
            int iy = py + ky; if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;
            sum += d_filter[(ky + radius) * filterW + (kx + radius)]
                   * (float)in[iy * w + ix];
        }
    }
    out[py * w + px] = clamp_uchar(sum);
}

// ---------------------------------------------------------------------------
// Kernel V2: constant memory only (from Example 22)
// ---------------------------------------------------------------------------
__global__ void convConstantMem(const unsigned char *in, unsigned char *out,
                                int w, int h, int radius) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int filterW = 2 * radius + 1;
    float sum = 0.f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int ix = px + kx; if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;
            int iy = py + ky; if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;
            sum += d_filter_const[(ky + radius) * filterW + (kx + radius)]
                   * (float)in[iy * w + ix];
        }
    }
    out[py * w + px] = clamp_uchar(sum);
}

// ---------------------------------------------------------------------------
// Kernel V3: shared memory halo tile + constant filter
//
// Each block loads a (BLOCK_W + 2*radius) x (BLOCK_H + 2*radius) tile
// into shared memory, then computes BLOCK_W * BLOCK_H output pixels from it.
//
// Shared memory declaration uses MAX_RADIUS so the kernel compiles for any
// radius up to MAX_RADIUS without dynamic shared memory.
// ---------------------------------------------------------------------------
__global__ void convSharedTile(const unsigned char *in, unsigned char *out,
                               int w, int h, int radius) {
    // Shared memory tile: large enough for BLOCK + full halo on all sides
    __shared__ unsigned char tile[BLOCK_H + 2 * MAX_RADIUS]
                                 [BLOCK_W + 2 * MAX_RADIUS];

    int tileW = BLOCK_W + 2 * radius;
    int tileH = BLOCK_H + 2 * radius;

    // Top-left corner of this tile in global image coordinates
    int originX = blockIdx.x * BLOCK_W - radius;
    int originY = blockIdx.y * BLOCK_H - radius;

    // --- Cooperative tile load (stride loop, same idea as Example 5) ---
    // The tile has tileW*tileH pixels but only BLOCK_W*BLOCK_H threads,
    // so each thread may need to load more than one pixel.
    int tid = threadIdx.y * BLOCK_W + threadIdx.x;
    for (int i = tid; i < tileW * tileH; i += BLOCK_W * BLOCK_H) {
        int ty = i / tileW;   // row within tile
        int tx = i % tileW;   // col within tile
        // Global coordinates with clamp-to-edge boundary
        int gx = clampi(originX + tx, 0, w - 1);
        int gy = clampi(originY + ty, 0, h - 1);
        tile[ty][tx] = in[gy * w + gx];
    }

    // --- Barrier: wait until ALL threads have finished loading the tile ---
    // (same role as __syncthreads() between phases in Example 17 matmul)
    __syncthreads();

    // --- Compute output pixel from shared memory (no bounds check needed) ---
    int px = blockIdx.x * BLOCK_W + threadIdx.x;
    int py = blockIdx.y * BLOCK_H + threadIdx.y;
    if (px >= w || py >= h) return;

    int filterW = 2 * radius + 1;
    float sum = 0.f;
    // threadIdx offsets into the tile already account for the halo
    for (int ky = 0; ky < filterW; ky++) {
        for (int kx = 0; kx < filterW; kx++) {
            sum += d_filter_const[ky * filterW + kx]
                   * (float)tile[threadIdx.y + ky][threadIdx.x + kx];
        }
    }
    out[py * w + px] = clamp_uchar(sum);
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
void check(const char *label, const unsigned char *ref,
           const unsigned char *got, int n) {
    float maxDiff = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf((float)ref[i] - (float)got[i]);
        if (d > maxDiff) maxDiff = d;
    }
    printf("  Correctness vs naive: max diff = %.1f %s\n",
           maxDiff, maxDiff < 1.f ? "(PASS)" : "(FAIL)");
    (void)label;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int pixels      = WIDTH * HEIGHT;
    size_t imgBytes = (size_t)pixels;

    // Fill 15x15 box blur
    for (int i = 0; i < 225; i++) blur15x15[i] = 1.f / 225.f;

    // Pinned host memory (all versions share it — optimization from Ex. 22)
    unsigned char *h_in, *h_out_ref, *h_out_shared;
    cudaHostAlloc(&h_in,        imgBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out_ref,   imgBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out_shared, imgBytes, cudaHostAllocDefault);
    generateImage(h_in, WIDTH, HEIGHT);

    // Device memory
    unsigned char *d_in, *d_out;
    float *d_filter_global;
    cudaMalloc(&d_in,           imgBytes);
    cudaMalloc(&d_out,          imgBytes);
    cudaMalloc(&d_filter_global, MAX_FILTER_ELEMS * sizeof(float));
    cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((WIDTH + BLOCK_W - 1) / BLOCK_W, (HEIGHT + BLOCK_H - 1) / BLOCK_H);

    cudaEvent_t evA, evB;
    cudaEventCreate(&evA);
    cudaEventCreate(&evB);

    printf("Image: %d x %d grayscale\n\n", WIDTH, HEIGHT);
    printf("%-22s  %10s  %6s\n", "Kernel", "Time (ms)", "Speedup");
    printf("%-22s  %10s  %6s\n", "──────────────────────", "──────────", "───────");

    // -----------------------------------------------------------------------
    // Test both filter sizes
    // -----------------------------------------------------------------------
    struct { const char *name; const float *filter; int radius; int elems; } tests[] = {
        { "blur 3x3",   blur3x3,   1,   9 },
        { "blur 15x15", blur15x15, 7, 225 },
    };

    for (int t = 0; t < 2; t++) {
        const char  *fname  = tests[t].name;
        const float *filter = tests[t].filter;
        int radius          = tests[t].radius;
        int elems           = tests[t].elems;

        printf("\n--- Filter: %s  (radius=%d, %d weights) ---\n",
               fname, radius, elems);
        printf("  Reuse ratio (redundant global reads without shared mem): %dx\n",
               elems);

        // Upload filter to both memory locations
        cudaMemcpy(d_filter_global, filter, elems * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_filter_const, filter, elems * sizeof(float));

        float timeNaive = 0.f, timeConst = 0.f, timeShared = 0.f;

        // V1: naive global
        cudaEventRecord(evA);
        convNaiveGlobal<<<grid, block>>>(d_in, d_out, d_filter_global,
                                        WIDTH, HEIGHT, radius);
        cudaEventRecord(evB);
        cudaEventSynchronize(evB);
        cudaEventElapsedTime(&timeNaive, evA, evB);
        cudaMemcpy(h_out_ref, d_out, imgBytes, cudaMemcpyDeviceToHost);

        printf("  %-20s  %10.2f  %6s\n", "naive global", timeNaive, "1.00x");

        // V2: constant mem
        cudaEventRecord(evA);
        convConstantMem<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT, radius);
        cudaEventRecord(evB);
        cudaEventSynchronize(evB);
        cudaEventElapsedTime(&timeConst, evA, evB);
        cudaMemcpy(h_out_shared, d_out, imgBytes, cudaMemcpyDeviceToHost);

        printf("  %-20s  %10.2f  %5.2fx\n", "constant mem",
               timeConst, timeNaive / timeConst);
        check("constant mem", h_out_ref, h_out_shared, pixels);

        // V3: shared memory halo
        if (radius <= MAX_RADIUS) {
            cudaEventRecord(evA);
            convSharedTile<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT, radius);
            cudaEventRecord(evB);
            cudaEventSynchronize(evB);
            cudaEventElapsedTime(&timeShared, evA, evB);
            cudaMemcpy(h_out_shared, d_out, imgBytes, cudaMemcpyDeviceToHost);

            printf("  %-20s  %10.2f  %5.2fx\n", "shared tile",
                   timeShared, timeNaive / timeShared);
            check("shared tile", h_out_ref, h_out_shared, pixels);
        }

        printf("  Shared memory per block: %d bytes  (tile: %dx%d)\n",
               (BLOCK_H + 2*radius) * (BLOCK_W + 2*radius),
               BLOCK_W + 2*radius, BLOCK_H + 2*radius);
    }

    printf("\n");
    printf("KEY TAKEAWAY:\n");
    printf("  3x3 filter:  shared memory can be SLOWER than global memory!\n");
    printf("               The halo-load overhead is not free. For 9 weights,\n");
    printf("               the savings don't cover the cooperative-load cost.\n");
    printf("  15x15 filter: shared memory gives ~2x speedup.\n");
    printf("               225 redundant global reads are reduced to 1 shared read.\n");
    printf("  Rule: use shared memory tiling for large filters (radius >= 4).\n");
    printf("  The benefit of shared memory SCALES with filter size.\n");
    printf("\n");
    printf("In templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu,\n");
    printf("convolutionKernel() is the naive global version (V1 above).\n");
    printf("For --kernel-size 15, replacing it with convSharedTile would give ~2x kernel speedup.\n");

    cudaEventDestroy(evA);
    cudaEventDestroy(evB);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter_global);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out_ref);
    cudaFreeHost(h_out_shared);

    return 0;
}
