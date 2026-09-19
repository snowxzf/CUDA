// Step 1 — "Hello GPU": C = A + B, one CUDA thread per element.
// This is the smallest complete CUDA program: read it, run it, understand every line.
//
// Compile & run:
//   nvcc 01_vector_add.cu -o vadd && ./vadd
// (In Google Colab: set Runtime -> GPU, then in a cell: !nvcc 01_vector_add.cu -o vadd && ./vadd)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// A "kernel" = a function that runs on the GPU, once per thread.
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    // Each thread computes its global index from its block + thread position.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // guard: last block may overshoot n
}

int main() {
    const int n = 1 << 20;               // ~1,000,000 elements
    size_t bytes = n * sizeof(float);

    // 1) allocate + fill host (CPU) arrays
    float *ha = (float*)malloc(bytes);
    float *hb = (float*)malloc(bytes);
    float *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { ha[i] = (float)i; hb[i] = 2.0f * i; }

    // 2) allocate device (GPU) arrays
    float *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    // 3) copy inputs host -> device
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

    // 4) launch: enough blocks of 256 threads to cover all n elements
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(da, db, dc, n);
    cudaDeviceSynchronize();             // wait for the GPU to finish

    // 5) copy result device -> host
    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

    // 6) verify
    bool ok = true;
    for (int i = 0; i < n; i++)
        if (hc[i] != ha[i] + hb[i]) { ok = false; break; }
    printf("vector add %s   (c[42] = %.1f, expected %.1f)\n",
           ok ? "PASSED" : "FAILED", hc[42], ha[42] + hb[42]);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
