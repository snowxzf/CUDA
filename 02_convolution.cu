// Step 2-4 — 3x3 image convolution: CPU baseline + naive GPU + tiled (shared-memory) GPU.
// This is the SAME operation as the FPGA accelerator, so it lets you compare
// GPU vs. custom-hardware acceleration of one kernel.
//
// Compile & run:
//   nvcc 02_convolution.cu -o conv && ./conv
//
// Runs the CPU baseline, then the naive and tiled GPU kernels; each GPU result is
// verified against the CPU and printed with its speedup.

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

// ---------- tiled GPU using shared memory ----------
// Naive re-reads each pixel from slow global memory up to 9 times. Tiling loads a
// block's pixels (plus a 1-pixel halo border) into fast shared memory ONCE, then
// every thread reads its neighborhood from there. This is the key GPU optimization.
__global__ void convTiled(const float* in, float* out) {
    __shared__ float tile[TILE + 2][TILE + 2];   // +2 = 1-pixel halo ring on every side

    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = blockIdx.x * TILE - 1;              // global coords of tile[0][0] (top-left of halo)
    int y0 = blockIdx.y * TILE - 1;

    // (TILE+2)^2 = 324 values but only TILE*TILE = 256 threads, so each thread
    // loads a strided subset. Clamping here reproduces the CPU's edge handling.
    for (int i = ty * TILE + tx; i < (TILE + 2) * (TILE + 2); i += TILE * TILE) {
        int sy = i / (TILE + 2), sx = i % (TILE + 2);
        int gy = CLAMP(y0 + sy, 0, H - 1);
        int gx = CLAMP(x0 + sx, 0, W - 1);
        tile[sy][sx] = in[gy * W + gx];
    }
    __syncthreads();   // every load must land before any thread reads its neighbors

    int x = x0 + 1 + tx, y = y0 + 1 + ty;
    if (x >= W || y >= H) return;   // after the barrier, so all threads reach __syncthreads
    float s = 0.f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            s += tile[ty + 1 + dy][tx + 1 + dx] * d_kernel[(dy + 1) * 3 + (dx + 1)];
    out[y * W + x] = s;
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
    const int ITERS = 100;   // average each kernel over many launches for a stable number

    // warm-up: the first launch pays one-time startup costs that would skew the timings
    convNaive<<<grid, block>>>(d_in, d_out);
    convTiled<<<grid, block>>>(d_in, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    // host<->device transfer time: excluded from the kernel timings below, but it is
    // part of real end-to-end cost, so measure it explicitly instead of hiding it.
    cudaEventRecord(a); cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);  float h2d_ms = ms;
    cudaEventRecord(a); cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost); cudaEventRecord(b);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);  float d2h_ms = ms;
    printf("Transfers: H2D %.3f ms + D2H %.3f ms = %.3f ms  (not counted in kernel times)\n",
           h2d_ms, d2h_ms, h2d_ms + d2h_ms);

    // naive GPU: average over ITERS launches
    cudaEventRecord(a);
    for (int it = 0; it < ITERS; it++) convNaive<<<grid, block>>>(d_in, d_out);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    float naive_ms = ms / ITERS;
    cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    double naive_gpix = (double)n / (naive_ms * 1e-3) / 1e9;   // billions of pixels / second
    printf("GPU naive: %.4f ms/run  %.2f Gpix/s  (%.1fx vs CPU)  max error %.5f\n",
           naive_ms, naive_gpix, cpu_ms / naive_ms, maxErr(h_cpu, h_gpu));

    // tiled GPU: clear d_out first (so leftover naive output can't mask a tiled bug),
    // then average over ITERS launches
    cudaMemset(d_out, 0, bytes);
    cudaEventRecord(a);
    for (int it = 0; it < ITERS; it++) convTiled<<<grid, block>>>(d_in, d_out);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    float tiled_ms = ms / ITERS;
    cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    double tiled_gpix = (double)n / (tiled_ms * 1e-3) / 1e9;
    float err = maxErr(h_cpu, h_gpu);
    printf("GPU tiled: %.4f ms/run  %.2f Gpix/s  (%.1fx vs CPU, %.2fx vs naive)  max error %.5f  %s\n",
           tiled_ms, tiled_gpix, cpu_ms / tiled_ms, naive_ms / tiled_ms, err,
           err < 1e-3f ? "(correct!)" : "(MISMATCH)");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_cpu); free(h_gpu);
    return 0;
}
