// Step 2-4 — 3x3 image convolution: CPU baseline + naive GPU + (YOUR TODO) tiled GPU.
// This is the SAME operation as the FPGA accelerator, so it lets you compare
// GPU vs. custom-hardware acceleration of one kernel.
//
// Compile & run:
//   nvcc 02_convolution.cu -o conv && ./conv
//
// Out of the box: CPU baseline runs, naive GPU runs + verifies + prints a speedup.
// Your job: implement convTiled() (shared-memory tiling) -- that's the real CUDA lesson.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define W 1024
#define H 1024
#define TILE 16                       // each block is TILE x TILE threads

#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

// 3x3 kernel. Box blur by default; try Gaussian {1,2,1, 2,4,2, 1,2,1}/16, or a Sobel edge kernel.
const float h_kernel[9] = {1/9.f, 1/9.f, 1/9.f,
                           1/9.f, 1/9.f, 1/9.f,
                           1/9.f, 1/9.f, 1/9.f};
__constant__ float d_kernel[9];       // fast read-only GPU memory for the weights

// ---------- CPU reference (also the correctness baseline) ----------
void convCPU(const float* in, float* out) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float s = 0.f;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int yy = CLAMP(y + dy, 0, H - 1);
                    int xx = CLAMP(x + dx, 0, W - 1);
                    s += in[yy * W + xx] * h_kernel[(dy + 1) * 3 + (dx + 1)];
                }
            out[y * W + x] = s;
        }
}

// ---------- naive GPU: one thread per output pixel, reads global memory ----------
__global__ void convNaive(const float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float s = 0.f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int yy = CLAMP(y + dy, 0, H - 1);
            int xx = CLAMP(x + dx, 0, W - 1);
            s += in[yy * W + xx] * d_kernel[(dy + 1) * 3 + (dx + 1)];
        }
    out[y * W + x] = s;
}

// ---------- YOUR EXERCISE: tiled GPU using shared memory ----------
// Naive re-reads each pixel from slow global memory up to 9 times. Tiling loads a
// block's pixels (plus a 1-pixel halo border) into fast shared memory ONCE, then
// every thread reads its neighborhood from there. This is the key GPU optimization.
__global__ void convTiled(const float* in, float* out) {
    // TODO:
    //  1) __shared__ float tile[TILE + 2][TILE + 2];         // +2 = halo ring
    //  2) Load this block's pixel into tile[ty + 1][tx + 1].
    //  3) Threads on the block edge also load the halo (the surrounding ring),
    //     clamping at the image borders with CLAMP.
    //  4) __syncthreads();                                    // wait for the tile to fill
    //  5) Compute the 3x3 weighted sum reading ONLY from `tile`, write out[y*W + x].
    //  Hint: shared index = threadIdx + 1 (leaves room for the halo).

    // placeholder so the file compiles; delete once implemented (it will fail verification):
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) out[y * W + x] = 0.f;
}

float maxErr(const float* a, const float* b) {
    float m = 0.f;
    for (int i = 0; i < W * H; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    size_t n = (size_t)W * H, bytes = n * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_cpu = (float*)malloc(bytes);
    float *h_gpu = (float*)malloc(bytes);
    for (size_t i = 0; i < n; i++) h_in[i] = (float)(rand() % 256);   // synthetic 8-bit image

    // CPU baseline (timed)
    auto t0 = std::chrono::high_resolution_clock::now();
    convCPU(h_in, h_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU:       %.2f ms\n", cpu_ms);

    // device setup
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ms;

    // naive GPU
    cudaEventRecord(a); convNaive<<<grid, block>>>(d_in, d_out); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("GPU naive: %.3f ms  (%.1fx vs CPU)  max error %.5f\n", ms, cpu_ms / ms, maxErr(h_cpu, h_gpu));

    // tiled GPU (your exercise)
    cudaEventRecord(a); convTiled<<<grid, block>>>(d_in, d_out); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    float err = maxErr(h_cpu, h_gpu);
    printf("GPU tiled: %.3f ms  (%.1fx vs CPU)  max error %.5f  %s\n",
           ms, cpu_ms / ms, err, err < 1e-3f ? "(correct!)" : "(TODO: implement convTiled)");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_cpu); free(h_gpu);
    return 0;
}
