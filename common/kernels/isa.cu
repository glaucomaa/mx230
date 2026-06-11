// Instruction-throughput probes for the ISA audit (stage-3 prep): each
// thread runs CHAINS independent dependency chains of one instruction so
// latency is hidden by ILP + warps, and the loop body is nothing but the
// instruction under test. Accumulators are summed into out[] so nothing is
// dead-code eliminated.
#include <cuda_fp16.h>

#define ITERS 4096
#define CHAINS 8

extern "C" __global__ void fma_f32(float *out, float a, float b) {
    float acc[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) acc[j] = (float)j;
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < CHAINS; ++j) acc[j] = fmaf(a, acc[j], b);
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) s += acc[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" __global__ void dp4a_s32(int *out, int a, int b) {
    int acc[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) acc[j] = j;
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < CHAINS; ++j) acc[j] = __dp4a(a, b, acc[j]);
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) s += acc[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" __global__ void fma_h2(float *out, float af, float bf) {
    __half2 a = __float2half2_rn(af), b = __float2half2_rn(bf);
    __half2 acc[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) acc[j] = __float2half2_rn((float)j);
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < CHAINS; ++j) acc[j] = __hfma2(a, acc[j], b);
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) s += __low2float(acc[j]) + __high2float(acc[j]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Streaming read bandwidth via float4 loads, grid-stride.
extern "C" __global__ void stream_f4(float *out, const float4 *x, int n4) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
        float4 v = x[i];
        s += v.x + v.y + v.z + v.w;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
