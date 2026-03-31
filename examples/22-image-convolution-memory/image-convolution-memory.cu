/*
 * Example 22: Image Convolution - Memory Hierarchy Optimizations
 * ==============================================================
 *
 * LESSON: Profile first, then optimize the bottleneck.
 *
 * In Example 21 the GPU kernel was already ~6x faster than CPU.
 * But where exactly does GPU time go?  We decompose it:
 *
 *   Total GPU time = H2D transfer + kernel execution + D2H transfer
 *
 * For a 2048x2048 image (~4 MB):
 *   H2D transfer:   ~10 ms   (over PCIe bus)
 *   Kernel:          ~0.8 ms  (actual compute!)
 *   D2H transfer:   ~10 ms   (over PCIe bus)
 *   ─────────────────────────
 *   Total:          ~21 ms   ← the kernel is only 4% of this!
 *
 * Optimizing the kernel further would barely move the needle.
 * We need to optimize the transfers.
 *
 * OPTIMIZATION 1: __constant__ memory for the filter
 * ---------------------------------------------------
 * WHY CONSTANT MEMORY?
 * The filter kernel is:
 *   - Small: 9-225 floats = 36-900 bytes (fits in 64 KB constant cache)
 *   - Read-only: never written during kernel execution
 *   - Broadcast: every thread in a warp reads the SAME filter index
 *     at the SAME time (all 32 threads compute the same (ky,kx) step)
 *
 * The constant cache is optimized for exactly this broadcast pattern:
 * ONE memory transaction serves all 32 threads simultaneously.
 * Global memory would require 32 separate reads (or rely on L1 luck).
 *
 * OPTIMIZATION 2: Pinned (page-locked) host memory
 * -------------------------------------------------
 * WHY PINNED MEMORY?
 * Normal malloc() returns pageable memory: the OS can swap pages to disk.
 * cudaMemcpy on pageable memory MUST first stage data through an internal
 * locked buffer before DMA can proceed. That staging step doubles the copy:
 *
 *   Pageable path:  CPU RAM → staging buffer → GPU  (2 copies)
 *   Pinned path:    CPU RAM ───────────────── → GPU  (1 DMA, no staging)
 *
 * cudaHostAlloc() pins pages in physical RAM so DMA can transfer directly,
 * bypassing the staging copy and roughly halving transfer time.
 *
 * Cost: pinned pages cannot be swapped by the OS — don't over-allocate.
 *
 * COMPILE & RUN:
 *   nvcc -o output.bin image-convolution-memory.cu -run
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH  2048
#define HEIGHT 2048
#define BLOCK_W 16
#define BLOCK_H 16

// ---------------------------------------------------------------------------
// Constant memory for the filter (up to 15x15 = 225 floats, 900 bytes)
// Fits entirely in the 64 KB constant cache.
// ---------------------------------------------------------------------------
#define MAX_FILTER_ELEMS 225
__constant__ float d_filter_const[MAX_FILTER_ELEMS];

// ---------------------------------------------------------------------------
// Filters (same as Example 21)
// ---------------------------------------------------------------------------
static const float blur3x3[9] = {
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ unsigned char clamp_uchar(float v) {
    return (unsigned char)(v < 0.f ? 0 : v > 255.f ? 255 : v);
}

void generateImage(unsigned char *img, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img[y * w + x] = (unsigned char)((y + x) % 256);
}

// ---------------------------------------------------------------------------
// Kernel V1: reads filter from global memory (baseline from Example 21)
// ---------------------------------------------------------------------------
__global__ void convGlobal(const unsigned char *in, unsigned char *out,
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
// Kernel V2/V3: reads filter from __constant__ cache (broadcast-optimized)
// No filter pointer argument — reads d_filter_const directly.
// ---------------------------------------------------------------------------
__global__ void convConstant(const unsigned char *in, unsigned char *out,
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
            // Read from constant cache — one transaction serves entire warp
            sum += d_filter_const[(ky + radius) * filterW + (kx + radius)]
                   * (float)in[iy * w + ix];
        }
    }
    out[py * w + px] = clamp_uchar(sum);
}

// ---------------------------------------------------------------------------
// Timing structure
// ---------------------------------------------------------------------------
typedef struct { float h2d, kernel, d2h, total; } Timing;

// ---------------------------------------------------------------------------
// Run version and measure each phase separately
// ---------------------------------------------------------------------------
Timing runVersion(const char *label,
                  unsigned char *h_in,   // host input  (pageable or pinned)
                  unsigned char *h_out,  // host output (pageable or pinned)
                  unsigned char *d_in, unsigned char *d_out,
                  float *d_filter_global,
                  bool useConstantMem,
                  int w, int h, int radius) {
    size_t imgBytes = (size_t)w * h;
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((w + BLOCK_W - 1) / BLOCK_W, (h + BLOCK_H - 1) / BLOCK_H);

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    // H2D
    cudaEventRecord(e0);
    cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    // Kernel
    cudaEventRecord(e1);
    if (useConstantMem)
        convConstant<<<grid, block>>>(d_in, d_out, w, h, radius);
    else
        convGlobal<<<grid, block>>>(d_in, d_out, d_filter_global, w, h, radius);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);

    // D2H
    cudaEventRecord(e2);
    cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    Timing t;
    cudaEventElapsedTime(&t.h2d,    e0, e1);
    cudaEventElapsedTime(&t.kernel, e1, e2);
    cudaEventElapsedTime(&t.d2h,    e2, e3);
    t.total = t.h2d + t.kernel + t.d2h;

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);

    (void)label;  // printed by caller
    return t;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int pixels      = WIDTH * HEIGHT;
    size_t imgBytes = (size_t)pixels;

    // Pageable host memory
    unsigned char *h_in_page  = (unsigned char*)malloc(imgBytes);
    unsigned char *h_out_page = (unsigned char*)malloc(imgBytes);
    generateImage(h_in_page, WIDTH, HEIGHT);

    // Pinned (page-locked) host memory — enables direct DMA, no staging copy
    unsigned char *h_in_pin  = NULL;
    unsigned char *h_out_pin = NULL;
    cudaHostAlloc(&h_in_pin,  imgBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out_pin, imgBytes, cudaHostAllocDefault);
    generateImage(h_in_pin, WIDTH, HEIGHT);

    // Device memory
    unsigned char *d_in, *d_out;
    float *d_filter_global;
    cudaMalloc(&d_in,           imgBytes);
    cudaMalloc(&d_out,          imgBytes);
    cudaMalloc(&d_filter_global, MAX_FILTER_ELEMS * sizeof(float));

    // Upload filter to both locations
    int radius = 1;           // 3x3 filter
    cudaMemcpy(d_filter_global, blur3x3, 9 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter_const, blur3x3, 9 * sizeof(float));

    printf("Image: %d x %d grayscale   Filter: blur 3x3\n\n", WIDTH, HEIGHT);
    printf("%-22s  %8s  %10s  %8s  %9s  %7s\n",
           "Version", "H2D(ms)", "Kernel(ms)", "D2H(ms)", "Total(ms)", "FPS eq.");
    printf("%-22s  %8s  %10s  %8s  %9s  %7s\n",
           "──────────────────────", "────────", "──────────", "────────",
           "─────────", "───────");

    Timing t;
    #define PRINT_ROW(label, t) \
        printf("%-22s  %8.1f  %10.2f  %8.1f  %9.1f  %7.1f\n", \
               label, t.h2d, t.kernel, t.d2h, t.total, 1000.f / t.total)

    t = runVersion("1. Pageable + global",
                   h_in_page, h_out_page, d_in, d_out, d_filter_global,
                   false, WIDTH, HEIGHT, radius);
    PRINT_ROW("1. Pageable+global", t);

    t = runVersion("2. Pageable + const",
                   h_in_page, h_out_page, d_in, d_out, d_filter_global,
                   true, WIDTH, HEIGHT, radius);
    PRINT_ROW("2. Pageable+const", t);

    t = runVersion("3. Pinned + const",
                   h_in_pin, h_out_pin, d_in, d_out, d_filter_global,
                   true, WIDTH, HEIGHT, radius);
    PRINT_ROW("3. Pinned+const", t);

    printf("\n");
    printf("KEY INSIGHT:\n");
    printf("  The kernel runs in < 1 ms, but each memcpy takes ~10 ms.\n");
    printf("  Pinned memory enables direct DMA: no staging copy → ~2x faster transfers.\n");
    printf("  __constant__ cache: filter broadcast → one transaction per warp.\n");
    printf("\n");
    printf("In templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu,\n");
    printf("applyFilterGPU() uses pageable cudaMemcpy (Version 1 above).\n");
    printf("Adding cudaHostAlloc would push it from ~105 FPS toward ~200+ FPS.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter_global);
    cudaFreeHost(h_in_pin);
    cudaFreeHost(h_out_pin);
    free(h_in_page);
    free(h_out_page);

    return 0;
}
