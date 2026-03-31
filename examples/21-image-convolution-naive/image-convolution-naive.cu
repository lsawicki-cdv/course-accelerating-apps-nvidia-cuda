/*
 * Example 21: Image Convolution - Naive GPU Implementation
 * =========================================================
 *
 * WHAT IS CONVOLUTION?
 * --------------------
 * Convolution applies a small "filter kernel" to every pixel of an image.
 * Each output pixel is a weighted sum of the corresponding input pixel and
 * its neighbors, where the weights come from the filter matrix:
 *
 *   output[y][x] = SUM over (ky, kx) of:
 *                    filter[ky + radius][kx + radius]
 *                    * input[clamp(y + ky, 0, H-1)][clamp(x + kx, 0, W-1)]
 *
 * The "clamp" handles image borders: pixels outside the image boundary
 * are replaced by the nearest edge pixel (clamp-to-edge).
 *
 * WHY IS GPU IDEAL FOR CONVOLUTION?
 * ----------------------------------
 * Each output pixel is computed INDEPENDENTLY — there are no dependencies
 * between output[y1][x1] and output[y2][x2]. This means we can compute
 * every output pixel simultaneously, one GPU thread per pixel.
 *
 * For a 2048x2048 image that's 4 million independent tasks — a perfect
 * fit for thousands of GPU threads running in parallel.
 *
 * THREAD-TO-PIXEL MAPPING:
 *   threadIdx.x + blockIdx.x * blockDim.x  →  pixel column (x)
 *   threadIdx.y + blockIdx.y * blockDim.y  →  pixel row    (y)
 *
 * Block size 16x16 = 256 threads per block.
 * Grid covers the full image: ceil(W/16) x ceil(H/16) blocks.
 *
 * This is exactly the same structure used in the production webcam filter:
 *   templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu
 *   → convolutionKernel()
 *
 * COMPILE & RUN:
 *   nvcc -o output.bin image-convolution-naive.cu -run
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Image dimensions
// ---------------------------------------------------------------------------
#define WIDTH  2048
#define HEIGHT 2048
#define BLOCK_W 16
#define BLOCK_H 16

// ---------------------------------------------------------------------------
// Filter definitions  (row-major, left-to-right, top-to-bottom)
// radius = (size - 1) / 2,  so 3x3 → radius = 1
// ---------------------------------------------------------------------------

// Identity: passes image through unchanged
static const float identity3x3[9] = {
    0.f, 0.f, 0.f,
    0.f, 1.f, 0.f,
    0.f, 0.f, 0.f
};

// Blur (box filter): average of 3x3 neighborhood
static const float blur3x3[9] = {
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9
};

// Sharpen: amplifies local contrast
// Center = 1 + 4*intensity,  cardinal neighbors = -intensity  (intensity=1.0)
static const float sharpen3x3[9] = {
     0.f, -1.f,  0.f,
    -1.f,  5.f, -1.f,
     0.f, -1.f,  0.f
};

// Edge detection (Laplacian): highlights rapid intensity changes
// Output is bright where edges exist, dark elsewhere
static const float edge3x3[9] = {
    -1.f, -1.f, -1.f,
    -1.f,  8.f, -1.f,
    -1.f, -1.f, -1.f
};

// Emboss: creates a pseudo-3D directional-gradient effect
static const float emboss3x3[9] = {
    -2.f, -1.f,  0.f,
    -1.f,  1.f,  1.f,
     0.f,  1.f,  2.f
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static __host__ __device__ __forceinline__ unsigned char clamp_uchar(float v) {
    return (unsigned char)(v < 0.f ? 0 : v > 255.f ? 255 : v);
}

// Diagonal gradient: pixel value = (row + col) % 256
// Makes edge detection produce visible output even without a real image.
void generateImage(unsigned char *img, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img[y * w + x] = (unsigned char)((y + x) % 256);
}

double wallTimeMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ---------------------------------------------------------------------------
// CPU reference implementation (single-threaded nested loops)
// ---------------------------------------------------------------------------
void convCPU(const unsigned char *in, unsigned char *out,
             const float *filter, int w, int h, int radius) {
    int filterW = 2 * radius + 1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.f;
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    // Clamp-to-edge boundary
                    if (ix < 0) ix = 0;
                    if (ix >= w) ix = w - 1;
                    if (iy < 0) iy = 0;
                    if (iy >= h) iy = h - 1;
                    sum += filter[(ky + radius) * filterW + (kx + radius)]
                           * (float)in[iy * w + ix];
                }
            }
            out[y * w + x] = clamp_uchar(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU naive kernel: one thread per output pixel
// ---------------------------------------------------------------------------
__global__ void convNaive(const unsigned char *in, unsigned char *out,
                          const float *d_filter, int w, int h, int radius) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int py = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (px >= w || py >= h) return;

    int filterW = 2 * radius + 1;
    float sum = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int ix = px + kx;
            int iy = py + ky;
            // Clamp-to-edge (same as the webcam filter kernel)
            if (ix < 0) ix = 0;
            if (ix >= w) ix = w - 1;
            if (iy < 0) iy = 0;
            if (iy >= h) iy = h - 1;

            sum += d_filter[(ky + radius) * filterW + (kx + radius)]
                   * (float)in[iy * w + ix];
        }
    }

    out[py * w + px] = clamp_uchar(sum);
}

// ---------------------------------------------------------------------------
// Run one filter: time CPU and GPU, print results, check correctness
// ---------------------------------------------------------------------------
void runFilter(const char *name,
               const unsigned char *h_in,
               const float *filter, int radius,
               unsigned char *d_in, unsigned char *d_filter_dev, unsigned char *d_out,
               unsigned char *h_out_cpu, unsigned char *h_out_gpu,
               int w, int h) {
    int pixels      = w * h;
    size_t imgBytes = (size_t)pixels * sizeof(unsigned char);
    int filterElems = (2*radius+1) * (2*radius+1);

    // Upload filter
    cudaMemcpy(d_filter_dev, filter, filterElems * sizeof(float), cudaMemcpyHostToDevice);

    // --- CPU ---
    double cpuStart = wallTimeMs();
    convCPU(h_in, h_out_cpu, filter, w, h, radius);
    double cpuMs = wallTimeMs() - cpuStart;

    // --- GPU ---
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((w + BLOCK_W - 1) / BLOCK_W, (h + BLOCK_H - 1) / BLOCK_H);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    cudaEventRecord(evStart);
    convNaive<<<grid, block>>>(d_in, d_out, (float*)d_filter_dev, w, h, radius);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);

    float gpuMs = 0.f;
    cudaEventElapsedTime(&gpuMs, evStart, evStop);

    cudaMemcpy(h_out_gpu, d_out, imgBytes, cudaMemcpyDeviceToHost);

    // Correctness check
    float maxDiff = 0.f;
    for (int i = 0; i < pixels; i++) {
        float d = fabsf((float)h_out_cpu[i] - (float)h_out_gpu[i]);
        if (d > maxDiff) maxDiff = d;
    }

    float speedup = (float)cpuMs / gpuMs;
    printf("Filter: %-16s  CPU: %6.1f ms   GPU: %5.1f ms   Speedup: %4.1fx   "
           "MaxDiff: %.1f %s\n",
           name, cpuMs, gpuMs, speedup, maxDiff,
           maxDiff <= 1.f ? "(PASS)" : "(FAIL)");

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int pixels      = WIDTH * HEIGHT;
    size_t imgBytes = (size_t)pixels * sizeof(unsigned char);

    // Host memory
    unsigned char *h_in      = (unsigned char*)malloc(imgBytes);
    unsigned char *h_out_cpu = (unsigned char*)malloc(imgBytes);
    unsigned char *h_out_gpu = (unsigned char*)malloc(imgBytes);

    generateImage(h_in, WIDTH, HEIGHT);

    // Device memory
    unsigned char *d_in, *d_out;
    unsigned char *d_filter_dev;     // reused for each filter (max 15x15 = 225 floats)
    cudaMalloc(&d_in,         imgBytes);
    cudaMalloc(&d_out,        imgBytes);
    cudaMalloc(&d_filter_dev, 225 * sizeof(float));

    cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice);

    // Warmup launch to avoid measuring GPU initialization in the first timing
    convNaive<<<dim3((WIDTH+BLOCK_W-1)/BLOCK_W,(HEIGHT+BLOCK_H-1)/BLOCK_H),
                dim3(BLOCK_W,BLOCK_H)>>>(d_in, d_out, (float*)d_filter_dev,
                                         WIDTH, HEIGHT, 1);
    cudaDeviceSynchronize();

    printf("Image: %d x %d grayscale (%d pixels)\n\n", WIDTH, HEIGHT, pixels);

    runFilter("identity",  h_in, identity3x3, 1,
              d_in, d_filter_dev, d_out, h_out_cpu, h_out_gpu, WIDTH, HEIGHT);
    runFilter("blur 3x3",  h_in, blur3x3, 1,
              d_in, d_filter_dev, d_out, h_out_cpu, h_out_gpu, WIDTH, HEIGHT);
    runFilter("sharpen",   h_in, sharpen3x3, 1,
              d_in, d_filter_dev, d_out, h_out_cpu, h_out_gpu, WIDTH, HEIGHT);
    runFilter("edge det.", h_in, edge3x3, 1,
              d_in, d_filter_dev, d_out, h_out_cpu, h_out_gpu, WIDTH, HEIGHT);
    runFilter("emboss",    h_in, emboss3x3, 1,
              d_in, d_filter_dev, d_out, h_out_cpu, h_out_gpu, WIDTH, HEIGHT);

    printf("\n");
    printf("Block size: %dx%d  Grid: %dx%d\n", BLOCK_W, BLOCK_H,
           (WIDTH + BLOCK_W - 1) / BLOCK_W, (HEIGHT + BLOCK_H - 1) / BLOCK_H);
    printf("\nSee templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu\n");
    printf("-> convolutionKernel() applies this exact structure to live webcam frames.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter_dev);
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);

    return 0;
}
