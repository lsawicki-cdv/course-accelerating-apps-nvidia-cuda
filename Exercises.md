# CUDA C++ Programming: Hands-on Lab Guide

This lab guide will walk you through a series of exercises designed to help you understand the fundamentals of CUDA C++ programming. By modifying and experimenting with the provided code examples, you'll gain hands-on experience with GPU programming concepts.

- [CUDA C++ Programming: Hands-on Lab Guide](#cuda-c-programming-hands-on-lab-guide)
  - [Exercise 1: Hello World from the GPU](#exercise-1-hello-world-from-the-gpu)
    - [Tasks:](#tasks)
  - [Exercise 2: Understanding Thread and Block Indices](#exercise-2-understanding-thread-and-block-indices)
    - [Tasks:](#tasks-1)
  - [Exercise 3: Single Block Loop Parallelization](#exercise-3-single-block-loop-parallelization)
    - [Tasks:](#tasks-2)
  - [Exercise 4: Multi-Block Loop Parallelization](#exercise-4-multi-block-loop-parallelization)
    - [Tasks:](#tasks-3)
  - [Exercise 5: Error Handling in CUDA](#exercise-5-error-handling-in-cuda)
    - [Tasks:](#tasks-4)
  - [Exercise 6: Optimizing with Grid-Stride Loops](#exercise-6-optimizing-with-grid-stride-loops)
    - [Tasks:](#tasks-5)
    - [Questions:](#questions)
  - [Exercise 7: Memory Prefetching and Initialization](#exercise-7-memory-prefetching-and-initialization)
    - [Tasks:](#tasks-6)
    - [Questions:](#questions-1)
  - [Exercise 8: Thread Configuration and Performance Analysis](#exercise-8-thread-configuration-and-performance-analysis)
    - [Tasks:](#tasks-7)
    - [Questions:](#questions-2)
  - [Exercise 9: CPU vs GPU Performance Comparison](#exercise-9-cpu-vs-gpu-performance-comparison)
    - [Tasks:](#tasks-8)
    - [Questions:](#questions-3)
  - [Exercise 10: Memory Optimization with Prefetching](#exercise-10-memory-optimization-with-prefetching)
    - [Tasks:](#tasks-9)
    - [Questions:](#questions-4)
  - [Exercise 11: GPU Initialization and Error Handling](#exercise-11-gpu-initialization-and-error-handling)
    - [Tasks:](#tasks-10)
    - [Questions:](#questions-5)
  - [Exercise 12: Matrix Multiplication Optimization](#exercise-12-matrix-multiplication-optimization)
    - [Tasks:](#tasks-11)
    - [Questions:](#questions-6)
  - [Exercise 13: CPU vs GPU Initialization](#exercise-13-cpu-vs-gpu-initialization)
    - [Tasks:](#tasks-12)
    - [Questions:](#questions-7)
  - [Exercise 14: Understanding CUDA Streams Basics](#exercise-14-understanding-cuda-streams-basics)
    - [Tasks:](#tasks-13)
    - [Questions:](#questions-8)
  - [Exercise 15: Stream-Based Initialization](#exercise-15-stream-based-initialization)
    - [Tasks:](#tasks-14)
    - [Questions:](#questions-9)
  - [Exercise 16: Stream-Sliced Vector Addition](#exercise-16-stream-sliced-vector-addition)
    - [Tasks:](#tasks-15)
    - [Questions:](#questions-10)
  - [Exercise 17: Advanced Stream Synchronization](#exercise-17-advanced-stream-synchronization)
    - [Tasks:](#tasks-16)
    - [Questions:](#questions-11)
  - [Exercise 18: Multi-Stage Pipeline with Streams](#exercise-18-multi-stage-pipeline-with-streams)
    - [Tasks:](#tasks-17)
    - [Questions:](#questions-12)
  - [Exercise 19: Optimizing Block Count Based on SM Count](#exercise-19-optimizing-block-count-based-on-sm-count)
    - [Tasks:](#tasks-18)
    - [Questions:](#questions-13)
  - [Exercise 20: Understanding Unified Memory Behavior](#exercise-20-understanding-unified-memory-behavior)
    - [Tasks:](#tasks-19)
    - [Questions:](#questions-14)
  - [Exercise 21: Memory Prefetching Optimization](#exercise-21-memory-prefetching-optimization)
    - [Tasks:](#tasks-20)
    - [Questions:](#questions-15)
  - [Exercise 22: Memory Coalescing](#exercise-22-memory-coalescing)
    - [Tasks:](#tasks-21)
    - [Questions:](#questions-16)
  - [Exercise 23: Tiled Matrix Multiplication with Shared Memory](#exercise-23-tiled-matrix-multiplication-with-shared-memory)
    - [Tasks:](#tasks-22)
    - [Questions:](#questions-17)
  - [Exercise 24: Tensor Cores Intro (WMMA API)](#exercise-24-tensor-cores-intro-wmma-api)
    - [Tasks:](#tasks-23)
    - [Questions:](#questions-18)
  - [Exercise 25: Tensor Cores with Shared Memory Staging](#exercise-25-tensor-cores-with-shared-memory-staging)
    - [Tasks:](#tasks-24)
    - [Questions:](#questions-19)
  - [Exercise 26: SGEMM Step-by-Step Optimization Ladder](#exercise-26-sgemm-step-by-step-optimization-ladder)
    - [Tasks:](#tasks-25)
    - [Questions:](#questions-20)
  - [Exercise 27: Image Convolution — Naive GPU Kernel](#exercise-27-image-convolution--naive-gpu-kernel)
    - [Tasks:](#tasks-26)
    - [Questions:](#questions-21)
  - [Exercise 28: Image Convolution — Constant & Pinned Memory](#exercise-28-image-convolution--constant--pinned-memory)
    - [Tasks:](#tasks-27)
    - [Questions:](#questions-22)
  - [Exercise 29: Image Convolution — Shared Memory Halo Pattern](#exercise-29-image-convolution--shared-memory-halo-pattern)
    - [Tasks:](#tasks-28)
    - [Questions:](#questions-23)

## Exercise 1: Hello World from the GPU

**File: [hello-world-gpu.cu](examples/1-gpu-hello-world/hello-world-gpu.cu)**

This first example demonstrates the basic structure of a CUDA program with both CPU and GPU functions.

```c
#include <stdio.h>

void helloCPU()
{
    printf("Hello from the CPU.\n");
}

/*
 * The addition of `__global__` signifies that this function
 * should be launced on the GPU.
 */

__global__ void helloGPU()
{
    printf("Hello from the GPU.\n");
}

int main()
{
    helloCPU();

    /*
     * Add an execution configuration with the <<<...>>> syntax
     * will launch this function as a kernel on the GPU.
     */

    helloGPU<<<1, 1>>>();

    /*
     * `cudaDeviceSynchronize` will block the CPU stream until
     * all GPU kernels have completed.
     */

    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program as is. Observe the output.
   ```
   nvcc hello-world-gpu.cu -o hello-gpu
   ./hello-gpu
   ```

2. Modify the program to print "Hello from the GPU" 5 times by changing the execution configuration to `<<<1, 5>>>`. 
   - What happens and why?

3. Change the execution configuration to `<<<5, 1>>>`.
   - How does the output differ from the previous configuration?
   - What does this tell you about the relationship between blocks and threads?

4. Experiment with removing `cudaDeviceSynchronize()`.
   - What happens when you run the program without it?
   - Why do you think this happens?

5. Create a new GPU function `helloGPU2` that prints "Hello again from the GPU". Call it after `helloGPU` with different execution configurations.
   - Try to make `helloGPU2` print before `helloGPU`

## Exercise 2: Understanding Thread and Block Indices

**File: [thread-and-block-idx.cu](examples/2-cuda-kernel-idx/thread-and-block-idx.cu)**

This example shows how to access thread and block indices within a kernel.

```c
#include <stdio.h>

__global__ void printHelloForCorrectExecutionConfiguration()
{
    if (threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Hello from GPU thread!\n");
    }
}

int main()
{
    // TODO: Change kernel parameters to print Hello message
    printHelloForCorrectExecutionConfiguration<<<2, 4>>>();

    cudaDeviceSynchronize();
}
```

### Tasks:

1. The current code doesn't print anything. Examine the if-condition in the kernel. What execution configuration would make this kernel print "Hello from GPU thread!"?

2. Modify the execution configuration to make the kernel print the message.

3. Change the kernel to print the thread ID and block ID for all threads:
   ```c
   printf("Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
   ```
   - Run with `<<<2, 4>>>` and observe the output
   - What do you notice about the order of execution?

4. Modify the kernel to only print when `threadIdx.x + blockIdx.x` equals 3.
   - Run with `<<<4, 4>>>` and count how many times the message prints
   - Explain why you get that number of messages

5. Add conditions to make only the first thread in each block print a message.
   - How would you identify the first thread in a block?
   - How would you identify the last thread in a block if the block size is variable?

## Exercise 3: Single Block Loop Parallelization

**File: [single-block-loop-gpu.cu](examples/3-loops/single-block-loop-gpu.cu)**

This example demonstrates how to parallelize a loop using a single block of threads.

```c
#include <stdio.h>

__global__ void loop()
{
    /*
     * This kernel does the work of only 1 iteration
     * of the original for loop. Indication of which
     * "iteration" is being executed by this kernel is
     * still available via `threadIdx.x`.
     */

    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
    loop<<<1, 10>>>();
    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program. Notice how each thread executes what would be a single iteration of a loop.

2. Modify the execution configuration to use 20 threads instead of 10.
   - What is the maximum number of threads you can have in a single block on your GPU?
   - Find this limit using `cudaGetDeviceProperties`

3. Add code to make the kernel perform some computation:
   ```c
   int result = threadIdx.x * threadIdx.x;
   printf("Thread %d computed: %d\n", threadIdx.x, result);
   ```

4. Create a new version that processes an array:
   - Define a global array `int results[1000]`
   - In the kernel, have each thread write its square to its position in the array
   - Add code to print the array after kernel execution
   - What's the problem with this approach? (Hint: think about scope and memory)

5. Fix the previous task by using proper CUDA memory management:
   ```c
   int *results;
   cudaMallocManaged(&results, 1000 * sizeof(int));
   // Initialize array values to 0
   // Launch kernel
   // Print values
   // Free memory
   ```

## Exercise 4: Multi-Block Loop Parallelization

**File: [multiple-block-loop-gpu.cu](examples/3-loops/multiple-block-loop-gpu.cu)**

This example shows how to use multiple blocks of threads to parallelize a loop.

```c
#include <stdio.h>

__global__ void loop()
{
    /*
     * This idiomatic expression gives each thread
     * a unique index within the entire grid.
     */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", i);
}

int main()
{
    loop<<<5, 5>>>();
    cudaDeviceSynchronize();
}
```

### Tasks:

1. Compile and run the program. How many numbers are printed? Verify that they go from 0 to 24.

2. Change the execution configuration to `<<<10, 10>>>`. 
   - How many numbers will this print?
   - What's the pattern of values?

3. Modify the kernel to print only even numbers:
   ```c
   if (i % 2 == 0) {
       printf("%d\n", i);
   }
   ```

4. Implement a grid-stride loop that allows a fixed number of threads to process a larger array:
   ```c
   __global__ void gridStrideLoop(int *data, int n)
   {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int i = idx; i < n; i += stride) {
           data[i] = i * i; // Square each element
       }
   }
   ```
   - Launch this with a small grid (e.g., `<<<2, 8>>>`) to process a large array (e.g., 100 elements)
   - Print the results to verify correctness

5. Compare the performance of a single large grid versus a grid-stride approach:
   - For an array of 1,000,000 elements, try:
     - Single launch with enough threads to cover all elements
     - Grid-stride with a smaller number of threads (e.g., 256 threads total)
   - Use `cudaEvent` timing to measure kernel execution time
   - Which is faster and why?


## Exercise 5: Error Handling in CUDA

**File:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu)

CUDA applications need proper error handling to ensure robustness. In this exercise, you'll modify a simple CUDA program to include comprehensive error checking.

### Tasks:

1. **Create an error checking helper function:**
   - Add a helper function that checks CUDA error codes and prints meaningful messages:
   ```c
   inline cudaError_t checkCuda(cudaError_t result) {
       if (result != cudaSuccess) {
           fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
           // Optionally add: assert(result == cudaSuccess);
       }
       return result;
   }
   ```

2. **Add error checking for memory allocation:**
   - Modify the `cudaMallocManaged` call to use your helper function:
   ```c
   checkCuda(cudaMallocManaged(&a, size));
   ```

3. **Add error checking for kernel launch:**
   - Add code after the kernel launch to check for synchronous errors:
   ```c
   cudaError_t syncErr = cudaGetLastError();
   if (syncErr != cudaSuccess) {
       printf("Kernel launch error: %s\n", cudaGetErrorString(syncErr));
   }
   ```

4. **Add error checking for kernel execution:**
   - Add code to capture asynchronous errors during kernel execution:
   ```c
   cudaError_t asyncErr = cudaDeviceSynchronize();
   if (asyncErr != cudaSuccess) {
       printf("Kernel execution error: %s\n", cudaGetErrorString(asyncErr));
   }
   ```

5. **Test with deliberate errors:**
   - Try launching the kernel with an invalid execution configuration (e.g., too many threads per block) and observe the error messages.
   - Study [error-handling.cu](examples/6-errors/error-handling.cu) to see how errors are caught and reported.

## Exercise 6: Optimizing with Grid-Stride Loops

**File:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu)

Grid-stride loops allow CUDA threads to process multiple elements of an array, enabling efficient handling of large datasets with a limited number of threads.

### Tasks:

1. **Increase the array size:**
   - Modify the program to handle a much larger array (e.g., N = 10,000,000).
   - Run the program and observe if it still works correctly.

2. **Add timing code:**
   - Add CUDA event timing to measure kernel execution time:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // Kernel launch here
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel execution time: %f ms\n", milliseconds);
   ```

3. **Implement a grid-stride loop:**
   - Study the implementation in [grid-stride-double.cu](examples/5-grid-stride/grid-stride-double.cu).
   - Modify the `doubleElements` kernel in your file to use a grid-stride loop:
   ```c
   __global__ void doubleElements(int *a, int N) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int i = idx; i < N; i += stride) {
           a[i] *= 2;
       }
   }
   ```

4. **Experiment with different grid sizes:**
   - Test the performance with various block counts (e.g., 32, 64, 128, 256).
   - Keep the number of threads per block constant (e.g., 256).
   - Record the execution time for each configuration.

5. **Optimize the number of blocks:**
   - Add code to query the device for its number of streaming multiprocessors (SMs):
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   int numberOfSMs;
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
   ```
   - Set the number of blocks to a multiple of the SM count (e.g., 2× or 4× the SM count).
   - Compare performance with your previous configurations.

### Questions:
- How does the execution time change with different numbers of blocks?
- Why is it beneficial to use a number of blocks that is a multiple of the SM count?
- What happens to performance if you use too few or too many blocks?

## Exercise 7: Memory Prefetching and Initialization

**File:** [block-config.cu](examples/4-allocation/block-config.cu)

CUDA Unified Memory can benefit significantly from prefetching data to the appropriate device before it's needed.

### Tasks:

1. **Add GPU initialization:**
   - Create a new kernel to initialize the array on the GPU:
   ```c
   __global__ void initializeElementsToGPU(int initialValue, int *a, int N) {
       int i = threadIdx.x + blockIdx.x * blockDim.x;
       int stride = blockDim.x * gridDim.x;
       
       for (int j = i; j < N; j += stride) {
           a[j] = initialValue;
       }
   }
   ```

2. **Add timing for both CPU and GPU initialization:**
   - Implement timing for both CPU and GPU initialization methods.
   - For CPU timing, use standard C `clock()`:
   ```c
   clock_t start = clock();
   // CPU initialization here
   clock_t end = clock();
   double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // in ms
   ```
   - For GPU timing, use CUDA events as shown in Exercise 2.

3. **Add memory prefetching:**
   - Add code to prefetch the array to the GPU before initialization:
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   cudaMemPrefetchAsync(a, size, deviceId);
   ```
   - Add timing to measure the impact of prefetching.

4. **Prefetch back to CPU for verification:**
   - Add code to prefetch the array back to the CPU before verification:
   ```c
   cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
   ```
   - Measure and compare verification time with and without this prefetch.

5. **Experiment with different array sizes:**
   - Test with various array sizes (e.g., 10K, 100K, 1M, 10M elements).
   - Record initialization and verification times for each size.
   - Determine at what array size GPU initialization becomes more efficient than CPU initialization.

### Questions:
- How does memory prefetching affect the performance of your program?
- At what array size does GPU initialization become faster than CPU initialization?
- What happens if you don't prefetch the data back to the CPU before verification?

## Exercise 8: Thread Configuration and Performance Analysis

**Files:** [double-elements-gpu.cu](examples/4-allocation/double-elements-gpu.cu) and [block-config.cu](examples/4-allocation/block-config.cu)

Understanding how thread configurations affect performance is crucial for optimizing CUDA applications.

### Tasks:

1. **Modify the programs to accept command-line arguments:**
   - Allow the user to specify array size, number of blocks, and threads per block:
   ```c
   int N = (argc > 1) ? atoi(argv[1]) : 1000;
   size_t threads_per_block = (argc > 2) ? atoi(argv[2]) : 256;
   size_t number_of_blocks = (argc > 3) ? atoi(argv[3]) : ((N + threads_per_block - 1) / threads_per_block);
   ```

2. **Create a performance testing loop:**
   - Write a script or modify your program to automatically test different configurations:
     - Various array sizes (e.g., 10K, 100K, 1M, 10M)
     - Different thread counts per block (e.g., 32, 64, 128, 256, 512, 1024)
     - Different block counts or calculation methods

3. **Add detailed timing measurements:**
   - Measure and report:
     - Memory allocation time
     - Data initialization time
     - Kernel execution time
     - Memory freeing time
     - Total application time

4. **Analyze occupancy:**
   - Add code to calculate theoretical occupancy based on your thread configuration:
   ```c
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, deviceId);
   int maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
   int maxBlocksPerMultiProcessor = props.maxBlocksPerMultiProcessor;
   int threadsPerSM = threads_per_block * number_of_blocks / props.multiProcessorCount;
   float occupancyPercentage = (float)threadsPerSM / maxThreadsPerMultiProcessor * 100.0f;
   printf("Theoretical occupancy: %.2f%%\n", occupancyPercentage);
   ```

### Questions:
- What is the optimal thread configuration for different array sizes?
- How does occupancy correlate with performance?
- What are the trade-offs between using many blocks with few threads vs. few blocks with many threads?

## Exercise 9: CPU vs GPU Performance Comparison

**Files:** [vector-add-cpu.cu](examples/7-vector/vector-add-cpu.cu) and [vector-add-gpu.cu](examples/7-vector/vector-add-gpu.cu)

These files implement vector addition on CPU and GPU respectively. This exercise will help you understand the performance differences between CPU and GPU implementations.

### Tasks:

1. **Run both implementations and record the execution times:**
   - Compile and run both programs:
   ```bash
   nvcc -o vector-add-cpu vector-add-cpu.cu
   nvcc -o vector-add-gpu vector-add-gpu.cu
   ./vector-add-cpu
   ./vector-add-gpu
   ```
   - Record the execution times for both implementations

2. **Modify both programs to try different array sizes:**
   - Change the `N` value in both programs to:
     - 2^20 (small)
     - 2^24 (medium)
     - 2^28 (large - already set)
   - For each size, record the execution times for both CPU and GPU implementations
   - Create a table or graph showing the relationship between array size and speedup (CPU time / GPU time)

3. **Analyze the performance crossover point:**
   - Find the approximate array size where GPU begins to outperform CPU
   - In your own words, explain why small arrays might not benefit from GPU acceleration

4. **Modify the GPU kernel to experiment with different block sizes:**
   - Try thread counts per block of: 32, 64, 128, 256, 512, 1024
   - Keep the total number of blocks formula the same: `(N + threadsPerBlock - 1) / threadsPerBlock`
   - Record the execution time for each configuration
   - Which block size gives the best performance? Why do you think that is?

### Questions:
- What factors contribute to the GPU outperforming the CPU for large arrays?
- What are the overheads associated with GPU computing that might make it slower for small arrays?
- How does the choice of block size affect performance?

## Exercise 10: Memory Optimization with Prefetching

**Files:** [vector-add-no-prefetch.cu](examples/09-vector-add-prefetch/vector-add-no-prefetch.cu) and [vector-add-prefetch.cu](examples/09-vector-add-prefetch/vector-add-prefetch.cu)

These files demonstrate the impact of memory prefetching on performance. This exercise will help you understand how memory movement affects GPU program performance.

### Tasks:

1. **Compare the execution time with and without prefetching:**
   - Run both implementations and record the execution times
   - Use the NVIDIA profiler to get more insight into memory operations:
   ```bash
   nsys profile --stats=true ./vector-add-no-prefetch
   nsys profile --stats=true ./vector-add-prefetch
   ```

2. **Modify the prefetching implementation:**
   - Add prefetching of result array back to the CPU before verification:
   ```c
   cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
   ```
   - Add this code before the `checkElementsAre` call
   - Measure execution time and compare with the previous version

3. **Create a hybrid prefetching strategy:**
   - Create a new version of the code that only prefetches part of each array
   - For example, prefetch the first half of each array to the GPU
   - Measure the impact on performance

4. **Add timing for memory operations:**
   - Add CUDA event timing to measure:
     - Initialization time
     - Prefetching time
     - Kernel execution time
     - Verification time
   - Use the following pattern:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // Operation to time
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Operation time: %f ms\n", milliseconds);
   ```

### Questions:
- How does prefetching affect the overall execution time?
- Which operation benefits most from prefetching?
- In what scenarios might prefetching be less beneficial?
- How does the memory transfer time compare to computation time?

## Exercise 11: GPU Initialization and Error Handling

**Files:** [vector-add-no-prefetch.cu](examples/09-vector-add-prefetch/vector-add-no-prefetch.cu) and [vector-add-prefetch.cu](examples/09-vector-add-prefetch/vector-add-prefetch.cu)

In this exercise, you'll improve the vector addition programs by adding GPU-based initialization and robust error handling.

### Tasks:

1. **Create a GPU initialization kernel:**
   - Add a new kernel to initialize arrays on the GPU:
   ```c
   __global__ void initWithGPU(float num, float *a, int N)
   {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int stride = blockDim.x * gridDim.x;
     
     for (int i = index; i < N; i += stride)
     {
       a[i] = num;
     }
   }
   ```
   - Replace CPU initialization with GPU initialization
   - Add timing to compare CPU vs GPU initialization speed

2. **Implement comprehensive error handling:**
   - Study the `checkCuda` function in `vector-add-gpu.cu`
   - Add error checking for all CUDA operations including:
     - Memory allocation
     - Memory prefetching (if used)
     - Kernel launches
     - Device synchronization
     - Memory freeing

3. **Test error detection:**
   - Intentionally create an error condition (e.g., specify too many threads per block)
   - Verify that your error handling catches and reports the error
   - Restore the correct configuration after testing

4. **Add device property queries:**
   - Query and print additional device properties:
   ```c
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, deviceId);
   printf("Device: %s\n", props.name);
   printf("Compute capability: %d.%d\n", props.major, props.minor);
   printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
   printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
   ```
   - Use these properties to optimize your block size

### Questions:
- How much faster is GPU initialization compared to CPU initialization?
- What are the most important device properties to consider when optimizing kernel launch parameters?
- What types of errors can occur in CUDA programs, and how can they be handled effectively?

## Exercise 12: Matrix Multiplication Optimization

**Files:** [matrix-multiply-2d-cpu.cu](examples/8-matrix-multiply/matrix-multiply-2d-cpu.cu) and [matrix-multiply-2d-gpu.cu](examples/8-matrix-multiply/matrix-multiply-2d-gpu.cu)

These files implement matrix multiplication on CPU and GPU. This exercise will have you optimize the GPU implementation for better performance.

### Tasks:

1. **Compare CPU and GPU performance:**
   - Run both implementations and record execution times
   - Try different matrix sizes (adjust the N definition):
     - N = 512
     - N = 1024 (default)
     - N = 2048 (be cautious with larger sizes)

2. **Add memory prefetching:**
   - Add code to prefetch matrices to the GPU before computation:
   ```c
   int deviceId;
   cudaGetDevice(&deviceId);
   cudaMemPrefetchAsync(a, size, deviceId);
   cudaMemPrefetchAsync(b, size, deviceId);
   cudaMemPrefetchAsync(c_gpu, size, deviceId);
   ```
   - Measure the impact on performance

3. **Implement a tiled matrix multiplication:**
   - Create a new kernel that uses shared memory for tiling:
   ```c
   __global__ void matrixMulTiled(int *a, int *b, int *c, int N)
   {
       __shared__ int aTile[TILE_SIZE][TILE_SIZE];
       __shared__ int bTile[TILE_SIZE][TILE_SIZE];

       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int col = blockIdx.x * blockDim.x + threadIdx.x;

       int sum = 0;

       // Loop over tiles
       for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
           // Load tiles into shared memory
           if (row < N && t * TILE_SIZE + threadIdx.x < N)
               aTile[threadIdx.y][threadIdx.x] = a[row * N + t * TILE_SIZE + threadIdx.x];
           else
               aTile[threadIdx.y][threadIdx.x] = 0;

           if (t * TILE_SIZE + threadIdx.y < N && col < N)
               bTile[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];
           else
               bTile[threadIdx.y][threadIdx.x] = 0;

           __syncthreads();

           // Compute partial sum for this tile
           for (int i = 0; i < TILE_SIZE; i++)
               sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];

           __syncthreads();
       }

       if (row < N && col < N)
           c[row * N + col] = sum;
   }
   ```
   - Define `TILE_SIZE` as 16 or 32
   - Replace the original kernel with this tiled version
   - Compare performance with the original implementation

4. **Experiment with different tile sizes:**
   - Try tile sizes of 8, 16, and 32
   - Record performance for each size
   - Find the optimal tile size for your GPU

### Questions:
- How does tiling affect the performance of matrix multiplication?
- Why does shared memory help with matrix multiplication performance?
- What factors determine the optimal tile size?
- How does the GPU speedup for matrix multiplication compare to the speedup for vector addition?

## Exercise 13: CPU vs GPU Initialization

**Files to use:** [init-kernel-cpu.cu](examples/10-init-kernel/init-kernel-cpu.cu) and [init-kernel-gpu.cu](examples/10-init-kernel/init-kernel-gpu.cu)

In this exercise, you'll compare initializing data on the CPU versus the GPU and understand the performance implications.

### Tasks:

1. **Compile and run both versions:**
   ```bash
   nvcc -o init-cpu init-kernel-cpu.cu
   nvcc -o init-gpu init-kernel-gpu.cu
   ./init-cpu
   ./init-gpu
   ```

2. **Add timing code to measure initialization performance:**
   - For both files, add CUDA event timing around the initialization sections:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   // CPU or GPU initialization here
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Initialization time: %f ms\n", milliseconds);
   ```

3. **Experiment with different array sizes:**
   - Modify both programs to test with these array sizes:
     - N = 2<<20 (small)
     - N = 2<<24 (medium - current setting)
     - N = 2<<28 (large)
   - Record initialization times for each size

4. **Analyze prefetching impact:**
   - In `init-kernel-cpu.cu`, move the prefetching code before the CPU initialization
   - Measure the performance difference and explain why it changed

### Questions:
- At what array size does GPU initialization become more efficient than CPU initialization?
- How does prefetching affect the performance of CPU initialization? Why?
- What happens if you remove the prefetching entirely? Test and explain.

## Exercise 14: Understanding CUDA Streams Basics

**Files to use:** [print-numbers-sync.cu](examples/11-stream-intro/print-numbers-sync.cu) and [print-numbers-async.cu](examples/11-stream-intro/print-numbers-async.cu)

These files demonstrate basic stream operations in CUDA. You'll modify them to understand how streams affect execution order.

### Tasks:

1. **Run both versions and observe the output ordering:**
   ```bash
   nvcc -o print-sync print-numbers-sync.cu
   nvcc -o print-async print-numbers-async.cu
   ./print-sync
   ./print-async
   ```
   - Note any differences in the output order

2. **Modify the async version:**
   - Instead of creating and destroying a stream for each number, create a fixed array of 3 streams:
   ```c
   cudaStream_t streams[3];
   for (int i = 0; i < 3; ++i) {
       cudaStreamCreate(&streams[i]);
   }
   
   for (int i = 0; i < 5; ++i) {
       printNumber<<<1, 1, 0, streams[i % 3]>>>(i);
   }
   
   cudaDeviceSynchronize();
   
   for (int i = 0; i < 3; ++i) {
       cudaStreamDestroy(streams[i]);
   }
   ```

3. **Add artificial delay to the kernel:**
   - Modify the `printNumber` kernel to include a delay:
   ```c
   __global__ void printNumber(int number) {
       // Add a small artificial delay
       int wait = 1000000;
       while(wait--) { }
       printf("%d\n", number);
   }
   ```
   - Run again and observe any changes in output order

4. **Create a priority stream version:**
   - Create streams with different priorities:
   ```c
   int priority_high, priority_low;
   cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
   
   cudaStream_t streamHigh, streamLow;
   cudaStreamCreateWithPriority(&streamHigh, cudaStreamNonBlocking, priority_high);
   cudaStreamCreateWithPriority(&streamLow, cudaStreamNonBlocking, priority_low);
   
   // Launch odd numbers on high priority
   // Launch even numbers on low priority
   for (int i = 0; i < 10; ++i) {
       if (i % 2 == 0)
           printNumber<<<1, 1, 0, streamLow>>>(i);
       else
           printNumber<<<1, 1, 0, streamHigh>>>(i);
   }
   ```

### Questions:
- How does the execution order differ between synchronous and asynchronous versions?
- Why might the output order be unpredictable even with separate streams?
- What impact does stream priority have? Is it guaranteed to execute in priority order?

## Exercise 15: Stream-Based Initialization

**Files to use:** [stream-init.cu](examples/12-stream-init/stream-init.cu) and [no-stream-init.cu](examples/12-stream-init/no-stream-init.cu)

This exercise focuses on the performance benefits of using streams for parallel initialization.

### Tasks:

1. **Add comprehensive timing code to both versions:**
   - Measure and record times for:
     - Memory allocation
     - Initialization of all three arrays
     - Prefetching
     - Vector addition
     - Verification
     - Overall execution

2. **Modify `no-stream-init.cu` to use streams:**
   - Replace the three sequential initialization calls with three streams like in `stream-init.cu`
   - Make sure to create/destroy streams properly

3. **Run nvprof to capture timeline information:**
   ```bash
   nsys profile --stats=true ./no-stream-init
   nsys profile --stats=true ./stream-init
   ```
   - Compare the timelines to see the difference in kernel execution patterns

4. **Experiment with stream count:**
   - Create a new version that uses 4 streams instead of 3
   - Use the 4th stream for the addition operation
   - Compare performance with the 3-stream version

### Questions:
- What performance improvement do you observe when using streams for initialization?
- Does using a separate stream for addition provide any benefit? Why or why not?
- What happens if you run the initialization kernels in the default stream but the addition kernel in a non-default stream?

## Exercise 16: Stream-Sliced Vector Addition

**File to use:** [stream-sliced.cu](examples/12-stream-init/stream-sliced.cu)

This exercise introduces a different approach to streaming: dividing the data into chunks and processing each chunk in its own stream.

### Tasks:

1. **Understand the existing code:**
   - Trace through the code and identify:
     - How the data is divided
     - How streams are assigned
     - The data processing flow

2. **Modify the stream count:**
   - Change the number of streams from 8 to:
     - 2 streams
     - 4 streams
     - 16 streams
   - Measure performance for each configuration

3. **Add error checking:**
   - Add proper error checking after each CUDA operation
   - Create a helper function for error checking:
   ```c
   inline void checkCuda(cudaError_t result) {
       if (result != cudaSuccess) {
           fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
           exit(EXIT_FAILURE);
       }
   }
   ```

4. **Implement overlapping computation and memory transfers:**
   - Modify the code to start copying results back to the host as soon as each chunk is processed
   - Use a loop structure like:
   ```c
   for (int i = 0; i < numberOfStreams; ++i) {
       // Initialize data
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(3, a[i], streamN);
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(4, b[i], streamN);
       initWith<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(0, c[i], streamN);
       
       // Process data
       addVectorsInto<<<numberOfBlocks, threadsPerBlock, 0, streams[i]>>>(c[i], a[i], b[i], streamN);
       
       // Start copying back to host (will be overlapped with next iteration's processing)
       cudaMemcpyAsync(h_c[i], c[i], streamSize, cudaMemcpyDeviceToHost, streams[i]);
   }
   ```

5. **Profile with different data sizes:**
   - Test with N = 2<<22, 2<<24, and 2<<26
   - Record performance metrics for each size

### Questions:
- What is the optimal number of streams for your GPU? Why?
- How does the performance of stream-sliced processing compare to processing the entire array at once?
- What factors limit the scalability of the stream-sliced approach?

## Exercise 17: Advanced Stream Synchronization 

For this exercise, create a new file named `stream-sync.cu` that demonstrates stream dependencies.

### Tasks:

1. **Create a program with the following structure:**
   - Allocate two input arrays and one output array
   - Create three streams: `streamA`, `streamB`, and `streamC`
   - Initialize array A in streamA and array B in streamB
   - Create CUDA events to mark completion of each initialization
   - Make streamC wait for both events before executing the vector addition kernel
   
   ```c
   cudaStream_t streamA, streamB, streamC;
   cudaStreamCreate(&streamA);
   cudaStreamCreate(&streamB);
   cudaStreamCreate(&streamC);
   
   cudaEvent_t eventA, eventB;
   cudaEventCreate(&eventA);
   cudaEventCreate(&eventB);
   
   // Initialize arrays in separate streams
   initWith<<<blocks, threads, 0, streamA>>>(3, a, N);
   initWith<<<blocks, threads, 0, streamB>>>(4, b, N);
   
   // Record events when initialization is complete
   cudaEventRecord(eventA, streamA);
   cudaEventRecord(eventB, streamB);
   
   // Make streamC wait for both initializations to complete
   cudaStreamWaitEvent(streamC, eventA, 0);
   cudaStreamWaitEvent(streamC, eventB, 0);
   
   // Launch addition in streamC
   addVectorsInto<<<blocks, threads, 0, streamC>>>(c, a, b, N);
   ```

2. **Add timing to measure:**
   - Initialization time for each array
   - Time between launching the init kernels and the addition kernel
   - Total execution time

3. **Create a comparison version without dependencies:**
   - Remove the event waiting
   - Launch the addition kernel in the default stream
   - Compare performance and correctness

4. **Implement priority-based scheduling:**
   - Modify the stream creation to use priorities
   - Assign higher priority to the addition kernel's stream
   - Measure the impact on performance

### Questions:
- How does the performance of the event-synchronized version compare to running kernels sequentially?
- What are the potential risks of not synchronizing streams when there are data dependencies?
- How much overhead do CUDA events add to the execution time?

## Exercise 18: Multi-Stage Pipeline with Streams

Create a new file `pipeline.cu` that implements a multi-stage data processing pipeline using CUDA streams.

### Tasks:

1. **Implement a three-stage pipeline:**
   - Stage 1: Initialize data (random numbers)
   - Stage 2: Process data (e.g., square each element)
   - Stage 3: Reduce data (e.g., sum all elements)

2. **Use stream synchronization to ensure correct execution order:**
   - Each stage must wait for the previous stage to complete
   - Different batches of data can be processed concurrently

3. **Process multiple batches of data:**
   - Divide a large array into smaller batches
   - Process each batch through the pipeline
   - Use multiple streams to overlap execution of different batches

4. **Compare with a sequential implementation:**
   - Implement the same functionality without streams
   - Measure and compare the performance

5. **Optimize for your GPU:**
   - Determine the optimal batch size and number of streams
   - Experiment with grid and block sizes
   - Profile and identify bottlenecks

### Questions:
- How much performance improvement does the pipelined approach provide?
- What is the optimal number of batches for your GPU?
- What factors limit the performance of the pipelined approach?

## Exercise 19: Optimizing Block Count Based on SM Count

**File to use:** [vector-add-SM-blocks.cu](examples/13-vector-add-sm-blocks/vector-add-SM-blocks.cu)

This exercise will help you understand how to optimize CUDA execution configuration based on the GPU's streaming multiprocessor (SM) count.

### Tasks:

1. **Fix the bug in the block count calculation:**
   - The current code has a bug in the `numberOfBlocks` calculation. It's trying to use `numberOfBlocks` to calculate itself!
   - Add code to query the number of SMs:
   ```c
   cudaGetDevice(&deviceId);
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
   printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);
   ```
   - Fix the calculation to make `numberOfBlocks` a multiple of the SM count:
   ```c
   numberOfBlocks = 32 * numberOfSMs; // Use 32 blocks per SM
   ```

2. **Add performance timing:**
   - Add CUDA event timing to measure kernel execution time:
   ```c
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel execution time: %f ms\n", milliseconds);
   ```

3. **Experiment with different multipliers:**
   - Try different multipliers for the SM count: 1, 2, 4, 8, 16, 32, 64
   - Record the execution time for each configuration
   - Create a table or graph of your results

4. **Experiment with different thread counts:**
   - Try different thread counts per block: 128, 256, 512, 1024
   - For each thread count, use your best SM multiplier from the previous step
   - Record execution times and compare

### Questions:
- Why is it beneficial to use a number of blocks that's a multiple of the SM count?
- What happens to performance if you use too few blocks? Too many?
- Is there an optimal threads-per-block size for your GPU? Why?

## Exercise 20: Understanding Unified Memory Behavior

**File to use:** [page-faults.cu](examples/14-unified-memory-page-faults/page-faults.cu)

This exercise will help you understand how Unified Memory behaves under different access patterns and how page faulting affects performance.

### Tasks:

1. **Experiment with GPU-only access:**
   - Complete the code to run only the GPU kernel:
   ```c
   // Experiment 1: GPU-only access
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("GPU-only access time: %f ms\n", milliseconds);
   ```
   - Profile this with `nsys profile --stats=true ./my_program`
   - Analyze the memory operations in the output

2. **Experiment with CPU-only access:**
   - Reset the code and run only the CPU function:
   ```c
   // Experiment 2: CPU-only access
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   cudaEventRecord(start);
   hostFunction(a, N);
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("CPU-only access time: %f ms\n", milliseconds);
   ```
   - Profile this and analyze the memory operations

3. **Experiment with GPU-then-CPU access:**
   - Run the GPU kernel followed by the CPU function:
   ```c
   // Experiment 3: GPU then CPU access
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   hostFunction(a, N);
   ```
   - Profile and analyze memory transfers

4. **Experiment with CPU-then-GPU access:**
   - Run the CPU function followed by the GPU kernel:
   ```c
   // Experiment 4: CPU then GPU access
   hostFunction(a, N);
   deviceKernel<<<256, 256>>>(a, N);
   cudaDeviceSynchronize();
   ```
   - Profile and analyze memory transfers

5. **Analyze and document your findings:**
   - For each experiment, record:
     - Number of page faults
     - Direction of memory transfers (HtoD or DtoH)
     - Total memory transferred
     - Execution time

### Questions:
- When do page faults occur in each scenario?
- How does the initial accessor of memory affect subsequent memory transfers?
- What pattern of memory access would be most efficient for a real application?
- How does the size of the data affect the page fault behavior?

## Exercise 21: Memory Prefetching Optimization

**File to use:** [vector-add-prefetch.cu](examples/15-unified-memory-prefetch/vector-add-prefetch.cu)

This exercise will show you how to use asynchronous memory prefetching to optimize performance by reducing page faults.

### Tasks:

1. **Add memory prefetching to GPU:**
   - Add code to prefetch arrays to the GPU before kernel execution:
   ```c
   // Prefetch arrays to the GPU
   cudaMemPrefetchAsync(a, size, deviceId);
   cudaMemPrefetchAsync(b, size, deviceId);
   cudaMemPrefetchAsync(c, size, deviceId);
   ```
   - Add this code after initialization but before kernel launch
   - Profile and measure the performance impact

2. **Add memory prefetching back to CPU:**
   - Add code to prefetch the result array back to the CPU before verification:
   ```c
   // Prefetch result back to CPU for verification
   cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
   ```
   - Add this after kernel execution but before verification
   - Profile and measure the impact

3. **Add comprehensive timing:**
   - Add timing for each phase of execution:
     - Memory allocation
     - Initialization
     - Prefetching to GPU
     - Kernel execution
     - Prefetching to CPU
     - Verification
   - Record and compare times for each phase

4. **Experiment with partial prefetching:**
   - Create a version that only prefetches part of each array
   - For example, prefetch the first half of each array:
   ```c
   cudaMemPrefetchAsync(a, size/2, deviceId);
   ```
   - Compare performance with full prefetching

5. **Try different array sizes:**
   - Modify the code to test with different array sizes:
     - N = 2<<20 (small)
     - N = 2<<24 (medium - current)
     - N = 2<<28 (large)
   - For each size, compare performance with and without prefetching

### Questions:
- At what data size does prefetching provide the most significant benefit?
- Which operation benefits most from prefetching?
- How does partial prefetching affect performance compared to full prefetching?
- How would you decide whether to use prefetching in a real application?

## Exercise 22: Memory Coalescing

**File to use:** [memory-coalescing.cu](examples/16-memory-coalescing/memory-coalescing.cu)

Memory coalescing is a technique that maximizes global memory bandwidth utilization. When all threads in a warp execute a load instruction, the hardware detects whether the memory accesses are consecutive. If they are, data is transferred in parallel via DRAM bursts, dramatically improving throughput.

This example compares two matrix multiplication kernels that are algorithmically identical but differ in how thread indices map to matrix rows and columns — one produces coalesced memory accesses, the other does not.

```c
/*
 * Coalesced: threadIdx.x maps to column (consecutive threads access
 * consecutive memory locations in B).
 */
__global__ void matMulCoalesced(float *A, float *B, float *C, int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n)
    {
        float value = 0.0f;
        for (int k = 0; k < n; k++)
            value += A[row * n + k] * B[k * n + col];
        C[row * n + col] = value;
    }
}

/*
 * Uncoalesced: threadIdx.x maps to row (consecutive threads access
 * memory locations N elements apart in A — strided access).
 */
__global__ void matMulUncoalesced(float *A, float *B, float *C, int n)
{
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n)
    {
        float value = 0.0f;
        for (int k = 0; k < n; k++)
            value += A[row * n + k] * B[k * n + col];
        C[row * n + col] = value;
    }
}
```

### Tasks:

1. **Compile and run the program, observe the speedup:**
   ```bash
   nvcc -o memory-coalescing memory-coalescing.cu
   ./memory-coalescing
   ```
   - Note the kernel times and the speedup factor

2. **Profile both kernels with Nsight Systems:**
   ```bash
   nsys profile --stats=true -o coalescing-report ./memory-coalescing
   ```
   - Compare global memory load/store throughput for each kernel
   - Look at the memory operations section to see the difference in bandwidth utilization

3. **Experiment with different matrix sizes:**
   - Change `N` to 512, 2048, and 4096
   - Record the speedup factor for each size
   - Does the coalescing advantage grow or shrink with larger matrices?

4. **Try different block sizes:**
   - Change `threadsPerBlock` from `(16, 16)` to `(32, 32)`, `(8, 8)`, and `(32, 8)`
   - How does the block shape affect coalescing performance?
   - Why does a `(32, 1)` block behave differently from a `(1, 32)` block?

5. **Analyze the memory access pattern on paper:**
   - For a warp of 32 threads with `threadIdx.x` = 0..31, trace which memory addresses are accessed in the inner loop iteration `k = 0`
   - For the coalesced kernel: what are the addresses for `B[k*N + col]`?
   - For the uncoalesced kernel: what are the addresses for `A[row*N + k]`?
   - Verify that the first pattern is consecutive and the second is strided by N

### Questions:
- Why does swapping row/col assignment between `threadIdx.x` and `threadIdx.y` cause such a large performance difference?
- What is the computational intensity (FLOP/Byte) of this kernel? How does it compare to the GPU's compute-to-bandwidth ratio?
- DRAM operates using bursts of consecutive locations. How does coalesced access exploit this hardware behavior?
- In what other types of kernels (beyond matrix multiplication) should you pay attention to coalescing?

## Exercise 23: Tiled Matrix Multiplication with Shared Memory

**File to use:** [tiled-matrix-multiply.cu](examples/17-tiled-matrix-multiply/tiled-matrix-multiply.cu)

Global memory is large but slow; shared memory is fast but small. Tiled matrix multiplication partitions matrices into tiles that fit into shared memory, reducing global memory traffic by a factor equal to the tile width.

This example compares a basic matrix multiplication kernel against a tiled version that uses shared memory. The tiled kernel handles non-divisible matrix sizes via boundary checks.

```c
__global__ void matMulTiled(float *A, float *B, float *C, int n)
{
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;
    int numPhases = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < numPhases; phase++)
    {
        // Load tiles into shared memory with boundary checks
        if (row < n && (phase * TILE_WIDTH + threadIdx.x) < n)
            sh_A[threadIdx.y][threadIdx.x] = A[row * n + phase * TILE_WIDTH + threadIdx.x];
        else
            sh_A[threadIdx.y][threadIdx.x] = 0.0f;

        if ((phase * TILE_WIDTH + threadIdx.y) < n && col < n)
            sh_B[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * n + col];
        else
            sh_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Wait for all threads to finish loading

        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];

        __syncthreads(); // Wait before loading next tile
    }

    if (row < n && col < n)
        C[row * n + col] = value;
}
```

### Tasks:

1. **Compile and run the program, observe the speedup:**
   ```bash
   nvcc -o tiled-matmul tiled-matrix-multiply.cu
   ./tiled-matmul
   ```
   - Note the kernel times for basic vs tiled and the reported computational intensity

2. **Profile both kernels with Nsight Systems:**
   ```bash
   nsys profile --stats=true -o tiled-report ./tiled-matmul
   ```
   - Compare global memory load throughput between the two kernels
   - How much did shared memory reduce global memory traffic?

3. **Experiment with different tile sizes:**
   - Change `TILE_WIDTH` to 8, 16, and 32 (remember: `TILE_WIDTH * TILE_WIDTH` must be <= 1024 max threads per block)
   - Record the execution time and shared memory usage for each
   - Verify the reported computational intensity matches: `(2 * TILE_WIDTH) / 8` FLOP/B
   - Which tile size gives the best performance on your GPU?

4. **Test with non-divisible matrix sizes:**
   - Change `N` to 1000, 1023, or 1025 (not divisible by 16 or 32)
   - Verify that results still match the basic kernel
   - Why are the boundary checks (`if row < n && ...`) and the `else 0.0f` assignments critical for correctness?

5. **Understand the two `__syncthreads()` calls:**
   - Try commenting out the first `__syncthreads()` (after loading). What happens? Why?
   - Try commenting out the second `__syncthreads()` (after computation). What happens? Why?
   - Explain what race condition each barrier prevents

6. **Calculate shared memory limits:**
   - Query your GPU's shared memory per block:
     ```c
     cudaDeviceProp props;
     cudaGetDeviceProperties(&props, deviceId);
     printf("Shared memory per block: %lu bytes\n", props.sharedMemPerBlock);
     ```
   - Calculate the maximum tile size that fits: `2 * TILE_WIDTH^2 * sizeof(float) <= sharedMemPerBlock`
   - What is the maximum `TILE_WIDTH` for your GPU? (also limited by 1024 threads per block)

7. **Implement a rectangular matrix version:**
   - Modify the tiled kernel to multiply A (N1 x N2) by B (N2 x N3) producing C (N1 x N3):
     - Replace the single `n` parameter with `n1`, `n2`, `n3`
     - Adjust the phase loop bound to `ceil(n2 / TILE_WIDTH)`
     - Update boundary checks to use the correct dimension for each matrix
   - Test with non-square matrices, e.g. A = 512x1024, B = 1024x768

### Questions:
- Without tiling, the computational intensity is 0.25 FLOP/B. With TILE_WIDTH=32 it becomes 8 FLOP/B. Show why, step by step.
- What limits the maximum tile size? (Consider both shared memory capacity and max threads per block.)
- Why must out-of-bounds shared memory elements be set to 0.0f instead of just skipping the load?
- The technique of breaking one long loop into multiple phases over smaller data chunks is called "strip mining." Why does this run faster even though the total work is the same?
- How does the tiled kernel's speedup compare to the coalescing speedup from Exercise 22? Which optimization has a bigger impact?

## Exercise 24: Tensor Cores Intro (WMMA API)

**File to use:** [tensor-cores-intro.cu](examples/18-tensor-cores-intro/tensor-cores-intro.cu)

Tensor cores are dedicated matrix-multiply-accumulate units present on Volta and newer NVIDIA GPUs. A single tensor-core operation computes `D = A * B + C` on 16×16 half-precision (FP16) tiles in one instruction, cooperatively executed by all 32 threads in a warp. This example pairs a standard CUDA-core FP32 matmul with a WMMA kernel so the speedup is directly observable.

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmmaKernel(const half *A, const half *B, float *C,
                           int M, int N, int K)
{
    int warpRow = blockIdx.y;                       // one warp per 16x16 tile
    int warpCol = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpRow * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpCol * 16,  N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + warpRow * 16 * N + warpCol * 16,
                            acc_frag, N, wmma::mem_row_major);
}
```

### Tasks

1. **Compile and run the baseline comparison:**
   ```bash
   nvcc -arch=sm_89 -o tensor-intro tensor-cores-intro.cu
   ./tensor-intro
   ```
   - Record GFLOPS for the CUDA-core and tensor-core kernels and the ratio.

2. **Profile both kernels:**
   ```bash
   nsys profile --stats=true ./tensor-intro
   ```
   - Confirm the tensor-core kernel finishes in a fraction of the CUDA-core time.

3. **Explore the FP16 / FP32 mix:**
   - `A` and `B` are `half`, the accumulator is `float`. Convert `A`/`B` to `float` and try to call `wmma::load_matrix_sync` — the compile error is part of the lesson: tensor cores require FP16 inputs in this API.

4. **Scale the problem:**
   - Try `M = N = K = 512`, `1024`, `2048`, `4096`.
   - Plot GFLOPS vs. matrix size for both kernels. Where does the tensor-core advantage peak?

5. **Verify accuracy:**
   - Compute a reference result on the CPU (or with the CUDA-core kernel) and measure the max absolute difference.
   - FP16 inputs cause small rounding error — what tolerance is acceptable? Why does the error grow with `K`?

### Questions

- Why must every member of a warp call `load_matrix_sync` / `mma_sync` / `store_matrix_sync`? What happens if only some threads enter the branch?
- The tensor core issues one 16×16×16 MMA per cycle per warp. Derive the theoretical FLOPS: 16×16×16×2 (MAC = 2 FLOPs) × clock rate × #TCs per SM × #SMs.
- Why does the speedup over CUDA cores shrink for very small problem sizes?

## Exercise 25: Tensor Cores with Shared Memory Staging

**File to use:** [tensor-cores-shared-memory.cu](examples/19-tensor-cores-shared-memory/tensor-cores-shared-memory.cu)

Mirrors the progression from Exercise 22 → 23, but for tensor cores: four warps in a block cooperatively stage a 32×32 tile of `A` and `B` through shared memory, replacing four independent global-memory loads per K-phase with one cooperative load.

### Tasks

1. **Compile and run, record both GFLOPS values:**
   ```bash
   nvcc -arch=sm_89 -o tc-shared tensor-cores-shared-memory.cu
   ./tc-shared
   ```
   - Compare naive-WMMA vs. shared-WMMA and compute the speedup.

2. **Understand the block topology:**
   - 128 threads per block = 4 warps arranged 2×2. Each warp owns one 16×16 output tile; the block covers a 32×32 region.
   - On paper, list which warp (identified by `threadIdx.y / 16`, `threadIdx.x / 16` or similar) writes which sub-tile of C.

3. **Inspect the two `__syncthreads()` calls:**
   - One after filling shared memory, one before overwriting it in the next K-phase. Comment out each in turn and describe the resulting race condition.

4. **Scale the problem size:**
   - Run with `M = N = K = 1024`, `2048`, `4096`, `8192`.
   - The shared-memory advantage is small at 1024 (matrices fit in L2) but grows with size. Plot the speedup vs. size.

5. **Compare with Exercise 26 (SGEMM):**
   - Kernel 3 of the SGEMM ladder in example 20 applies the same shared-memory tiling to FP32. How do the relative speedups differ? Why might tensor cores benefit less from tiling?

### Questions

- The tile is 32×32 half elements = 2048 bytes for A plus 2048 for B = 4 KB per block. How many blocks can run concurrently on one SM given its shared memory budget?
- Why is `half` (2 bytes) beneficial twice: once for tensor-core arithmetic, and again for shared-memory capacity?
- In what scenario would the naive WMMA version *beat* the shared-memory version?

## Exercise 26: SGEMM Step-by-Step Optimization Ladder

**File to use:** [sgemm-optimizations.cu](examples/20-sgemm-optimizations/sgemm-optimizations.cu)

Six progressively optimized SGEMM kernels running on the same 4096×4096 problem. Each kernel adds exactly one technique on top of the previous one, letting you isolate the contribution of each optimization and compare all six against cuBLAS.

| # | Kernel | Key technique |
|---|--------|---------------|
| 1 | Naive | one thread per output, uncoalesced |
| 2 | Coalesced | swap index mapping |
| 3 | Tiled | 32×32 shared-memory tile |
| 4 | 1D Coarsening | each thread computes TM=8 rows |
| 5 | 2D Coarsening | each thread computes TM×TN=8×8 (outer product) |
| 6 | Vectorized | `float4` loads/stores |

### Tasks

1. **Build with cuBLAS and run the ladder:**
   ```bash
   nvcc -arch=sm_89 -lcublas -o sgemm sgemm-optimizations.cu
   ./sgemm
   ```
   - Record GFLOPS and % of cuBLAS for each kernel. Build a bar chart.

2. **Identify the biggest jumps:**
   - Which single step gives the largest speedup? Explain why in terms of what it removes as the bottleneck (bandwidth vs. arithmetic intensity vs. instruction throughput).

3. **Profile kernel 4 (1D coarsening):**
   ```bash
   nsys profile --stats=true ./sgemm
   ```
   - How does the number of launched threads change between kernel 3 and kernel 4? How does occupancy change?

4. **Shrink the problem size:**
   - Re-run at 1024×1024 and 512×512. Do the later optimizations (5, 6) still help? Why might vectorization give a negative return on a small matrix?

5. **Ablation — disable vectorization:**
   - Replace the `float4` loads in kernel 6 with four scalar loads. Measure the difference. Is vectorization mainly saving instructions, memory transactions, or both?

6. **Compare with cuBLAS:**
   - cuBLAS reaches 100% by definition. Which of your kernels gets closest? What techniques are cuBLAS likely using that the 6-kernel ladder does not (hint: tensor cores, auto-tuning, warp-specialized producer/consumer pipelines)?

### Questions

- The arithmetic intensity climbs from 0.25 FLOP/byte (naive) to 128+ FLOP/byte (2D coarsening). Derive the exact number for TM=TN=8, tile=32.
- Why does 1D coarsening let you cache `B[...]` in a register and amortize the shared-memory load across TM MACs?
- At what point does the bottleneck shift from memory bandwidth to instruction throughput?
- Vectorized loads raise effective bandwidth but do not change arithmetic intensity. Why do they still help?

## Exercise 27: Image Convolution — Naive GPU Kernel

**File to use:** [image-convolution-naive.cu](examples/21-image-convolution-naive/image-convolution-naive.cu)

First hands-on 2D stencil kernel. One thread per output pixel reads a small filter-sized neighborhood from the input image and writes one pixel of the output. Clamp-to-edge boundary handling mirrors the production `templates/cuda-webcam-filter` implementation.

```cuda
__global__ void convKernel(const unsigned char *in, unsigned char *out,
                           const float *filter, int W, int H, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            sum += filter[(ky + radius) * (2*radius+1) + (kx + radius)]
                   * in[sy * W + sx];
        }
    }
    out[y * W + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
}
```

### Tasks

1. **Run the example and observe the five filters:**
   ```bash
   nvcc -o conv-naive image-convolution-naive.cu
   ./conv-naive
   ```
   - Record CPU vs. GPU times and the kernel-only speedup.

2. **Relate to the production app:**
   - Look at `templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu`. Identify the matching grid/block layout and clamp logic.
   - Run the webcam filter on a synthetic image:
     ```bash
     cd templates/cuda-webcam-filter/build
     ./cuda-webcam-filter --input synthetic --synthetic gradient --filter edge --kernel-size 3 --preview
     ```

3. **Change the block shape:**
   - Try `blockDim` of `(8,8)`, `(16,16)`, `(32,16)`, `(32,32)`. Record kernel time for each. Which shape best matches warp-coalescing of the input rows?

4. **Change the filter size:**
   - Modify the program to use a 5×5 and 7×7 blur. Cost scales as `(2r+1)^2` per output pixel — verify your measurements match that.

5. **Replace clamp-to-edge with zero padding:**
   - Change the boundary handling to return 0 for out-of-image samples. Run visually — what does the output look like near edges? Discuss trade-offs.

### Questions

- Why is 2D convolution a near-ideal GPU workload (no dependencies between output pixels)?
- The naive kernel reports ~1000× speedup versus CPU — but that is kernel time only. What is missing from that measurement? (See Exercise 28.)
- Why does a `(16, 16)` block shape usually outperform `(1, 256)` for this kernel, even though both have 256 threads?

## Exercise 28: Image Convolution — Constant & Pinned Memory

**File to use:** [image-convolution-memory.cu](examples/22-image-convolution-memory/image-convolution-memory.cu)

Exposes the real bottleneck in GPU image processing: PCIe transfer, not the kernel. Three progressively optimized versions break the per-phase cost (H2D / kernel / D2H), and two targeted fixes are applied — `__constant__` memory for the filter and pinned host memory for the image.

### Tasks

1. **Run the three versions and read the per-phase table:**
   ```bash
   nvcc -o conv-memory image-convolution-memory.cu
   ./conv-memory
   ```
   - Confirm: kernel time ≪ transfer time; pinned memory halves H2D and D2H.

2. **Declare `__constant__` memory yourself:**
   - In a copy of Exercise 27's kernel, add
     ```cuda
     __constant__ float d_filter_const[225];
     ```
     at file scope. Upload with `cudaMemcpyToSymbol(d_filter_const, filter, size);` and read `d_filter_const[...]` directly inside the kernel (no pointer parameter).
   - Measure the speedup for a 3×3 and a 15×15 filter.

3. **Switch from pageable to pinned:**
   - Replace `malloc` / `free` for the host image with `cudaHostAlloc` / `cudaFreeHost`.
   - Measure H2D and D2H time with each allocation strategy.

4. **Try the webcam filter at different kernel sizes:**
   ```bash
   ./cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 3  --preview
   ./cuda-webcam-filter --input synthetic --synthetic gradient --filter blur --kernel-size 15 --preview
   ```
   - Observe how GPU FPS drops at `--kernel-size 15`. Sketch which part of the pipeline (H2D / kernel / D2H) dominates at each size.

5. **Over-pinning warning:**
   - Allocate a very large pinned buffer (e.g., 2 GB) and observe system responsiveness. Why does over-pinning affect the whole OS?

### Questions

- Why is `__constant__` memory a perfect fit for filter weights specifically, but a bad fit for the input image?
- What is the broadcast pattern, and how does the constant cache exploit it?
- Pinned memory doubles DMA throughput in this example. Why is it not always the default?
- For a streaming webcam app at 60 FPS, what is the maximum image size that PCIe 4.0 (~32 GB/s one-way) can feed without becoming the bottleneck?

## Exercise 29: Image Convolution — Shared Memory Halo Pattern

**File to use:** [image-convolution-shared-memory.cu](examples/23-image-convolution-shared-memory/image-convolution-shared-memory.cu)

Teaches the **halo (apron) tiling** pattern for 2D stencils: each block cooperatively loads its output tile plus a border of width `radius` into shared memory, then all threads in the block read from fast on-chip SRAM. The benefit scales with filter size — for small filters the cooperative-load overhead can actually outweigh the savings.

```cuda
__shared__ unsigned char tile[BLOCK_H + 2*MAX_RADIUS][BLOCK_W + 2*MAX_RADIUS];

int tileW = BLOCK_W + 2 * radius;
int tileH = BLOCK_H + 2 * radius;
int tid   = threadIdx.y * BLOCK_W + threadIdx.x;

for (int i = tid; i < tileW * tileH; i += BLOCK_W * BLOCK_H) {
    int ty = i / tileW, tx = i % tileW;
    int gx = min(max(blockOriginX + tx - radius, 0), W - 1);
    int gy = min(max(blockOriginY + ty - radius, 0), H - 1);
    tile[ty][tx] = d_in[gy * W + gx];
}
__syncthreads();  // every thread must finish loading before anyone reads
```

### Tasks

1. **Compile and run for 3×3 and 15×15 filters:**
   ```bash
   nvcc -o conv-shared image-convolution-shared-memory.cu
   ./conv-shared
   ```
   - Record kernel times for naive global / constant memory / shared tile at each radius.

2. **Confirm the counter-intuitive result:**
   - Shared tile is ~0.9× (slower!) at 3×3 but ~2× faster at 15×15. Explain using the redundant-read ratio (`9×` vs `225×`).

3. **Vary the block size:**
   - Try `BLOCK_W × BLOCK_H` of `(16, 16)`, `(32, 8)`, `(8, 32)`. The tile size grows as `(B + 2r)²`. At what point do you hit the shared-memory-per-block limit?

4. **Query shared memory limits:**
   ```cuda
   cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
   printf("shmem/block: %lu B  shmem/SM: %lu B\n",
          props.sharedMemPerBlock, props.sharedMemPerMultiprocessor);
   ```
   - Compute the maximum `radius` you can support for a 16×16 block.

5. **Patch the webcam filter (optional):**
   - In `templates/cuda-webcam-filter/src/kernels/convolution_kernels.cu`, swap the naive kernel for a shared-tile version. Build and run with `--kernel-size 15`. Measure FPS change.

6. **Comment out the `__syncthreads()`:**
   - Remove the barrier after the cooperative load. Some threads will begin reading `tile[...]` before others finish writing. Describe the visible corruption and why it happens.

### Questions

- What are the two `__syncthreads()` calls in a tiled stencil kernel guarding against? (One after load, one before the next phase overwrites.)
- Why does shared-memory tiling hurt for `radius = 1` but help for `radius = 7`? Express the break-even in terms of `(2r+1)² / overhead_per_load`.
- How does the halo size affect the ratio of "useful" shared-memory elements (read by this block's outputs) to "halo" elements (only supporting neighbors)?
- For an anisotropic filter (wide in x, narrow in y), would a rectangular block shape give better shared-memory efficiency? Why?
