# CUDA 3×3 Image Convolution

A 3×3 image convolution implemented three ways — a CPU reference, a naive CUDA
kernel, and a shared-memory tiled CUDA kernel — with correctness checks and timing
for each. The same operation is also implemented as a custom FPGA accelerator,
which makes this a direct GPU-vs-custom-hardware comparison of one kernel.

## Contents

| File | Description |
| --- | --- |
| `01_vector_add.cu` | Minimal CUDA program (`C = A + B`): device allocation, host↔device copies, kernel launch, verification. |
| `02_convolution.cu` | 3×3 convolution on a 1024×1024 image: CPU reference, naive GPU kernel, tiled shared-memory GPU kernel. |

## Requirements

- An NVIDIA GPU with a recent driver
- CUDA Toolkit (provides `nvcc`)
- On Windows, the MSVC C++ build tools (Visual Studio "Desktop development with C++"), which `nvcc` uses as its host compiler

A [Google Colab](https://colab.research.google.com/) GPU runtime
already includes both the GPU and `nvcc` - can use if there is no access to an NVIDIA GPU. 

## Build and run

Linux / macOS shell:

```bash
nvcc -O3 01_vector_add.cu -o vadd && ./vadd
nvcc -O3 02_convolution.cu -o conv && ./conv
```

Windows (PowerShell, with CUDA and MSVC on `PATH`):

```powershell
nvcc -O3 01_vector_add.cu -o vadd.exe; .\vadd.exe
nvcc -O3 02_convolution.cu -o conv.exe; .\conv.exe
```

Google Colab: set **Runtime → Change runtime type → GPU**, then in cells:

```
%%writefile 02_convolution.cu
<paste file contents>
```

```
!nvcc -O3 02_convolution.cu -o conv && ./conv
```

`!nvidia-smi` shows which GPU the runtime was assigned.

## Implementation

**CPU reference (`convCPU`).** Straightforward nested loops. Pixels outside the
image are handled by clamping to the nearest edge pixel.

**Naive GPU (`convNaive`).** One thread per output pixel, 16×16 thread blocks. The
nine filter weights live in `__constant__` memory, which is cached and broadcast
when all threads in a warp read the same weight. Every thread reads its 3×3
neighborhood straight from global memory, so each input pixel is fetched up to
nine times by neighboring threads.

**Tiled GPU (`convTiled`).** Each 16×16 block cooperatively loads an 18×18 tile
(its pixels plus a 1-pixel halo) into `__shared__` memory. Since 324 values are
loaded by 256 threads, each thread loads a strided subset. After
`__syncthreads()`, every thread computes its output entirely from shared memory,
so each input pixel is read from global memory about once per block instead of up
to nine times.

**Timing.** The CPU path is timed with `std::chrono`. GPU kernels are timed with
`cudaEvent` and cover only kernel execution, not host↔device transfers. Both
kernels are launched once as a warm-up before timing, so one-time startup costs
are excluded.

## Changing the filter

The filter is `h_kernel` in `02_convolution.cu` (box blur by default). Other
common 3×3 filters:

| Filter | Weights |
| --- | --- |
| Gaussian blur | `{1,2,1, 2,4,2, 1,2,1} / 16` |
| Sobel (horizontal gradient) | `{-1,0,1, -2,0,2, -1,0,1}` |
| Sharpen | `{0,-1,0, -1,5,-1, 0,-1,0}` |
