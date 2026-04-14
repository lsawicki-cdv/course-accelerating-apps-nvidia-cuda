# Accelerating Application with NVIDIA CUDA C++

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)

## Supporting Materials

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html)
- [CUDA Streams Best Practices](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
- [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [Unified Memory in CUDA](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)

## Laboratory exercises - [Exercises.md](Exercises.md)


### Tips for CUDA Optimization

Throughout these exercises, keep these optimization principles in mind:

1. **Memory optimization:**
   - Minimize host-device transfers
   - Use prefetching with Unified Memory
   - Ensure coalesced memory access
   - Use shared memory for frequently accessed data

2. **Execution configuration:**
   - Choose thread block size as a multiple of 32 (warp size)
   - Launch enough blocks to keep all SMs busy
   - Avoid thread divergence within warps

3. **Workload distribution:**
   - Use grid-stride loops for large datasets
   - Balance work evenly across threads
   - Avoid serial sections in parallel code

4. **Profiling-driven optimization:**
   - Use Nsight Systems or other profiling tools
   - Identify bottlenecks before optimizing
   - Make one change at a time and measure impact
   - Document all performance changes

### Examples

#### The examples were tested on the following environment
- Ubuntu 24.04.2 LTS
- `nvidia-smi` output
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   53C    P5             10W /   80W |     112MiB /   8188MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
- `nvcc --version` output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```
- `nsys -v` output
```
NVIDIA Nsight Systems version 2025.1.1.131-251135540420v0
```

#### Running examples

- open terminal (Bash/x64 Native Tools Command Prompt for VS 2022)
- go to the proper directory, e.g. `cd examples/1-gpu-hello-world`
- compile and run code with nvcc, e.g. `nvcc -o hello.bin hello-world-gpu.cu -run`
- profile application if needed, e.g. `nsys profile --stats=true -o hello-report ./hello.bin` (in case of issues on Windows please set the path to nsys in the command line)

#### Windows 11 + VS Code integrated terminal

The repo ships a `.vscode/settings.json` that registers **x64 Native Tools Command Prompt for VS 2022** as the default integrated terminal profile. When you open a new terminal in VS Code (`` Ctrl+` ``), `vcvars64.bat` runs automatically so `cl.exe`, `nvcc`, and `nsys` are on the `PATH` — no manual environment setup required.

If your Visual Studio 2022 is not the Community edition or lives at a non-default path, edit the `path`/`args` in `.vscode/settings.json` to point at your own `vcvars64.bat` (Professional / Enterprise / BuildTools variants).
