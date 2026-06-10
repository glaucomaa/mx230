// Naive attention baseline: O = softmax(Q K^T / sqrt(d)) V with the full
// N x N score matrix S materialized in global memory. Three kernels:
// scores -> row softmax (in-place) -> S*V. Head dim is fixed at D.
#include <math_constants.h>

#define D 64

extern "C" __global__ void attn_scores(const float *Q, const float *K, float *S,
                                       int N, int causal) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float s = 0.0f;
#pragma unroll
        for (int x = 0; x < D; ++x) {
            s += Q[row * D + x] * K[col * D + x];
        }
        S[(size_t)row * N + col] =
            (causal && col > row) ? -CUDART_INF_F : s * rsqrtf((float)D);
    }
}

// One block of 256 threads per row: max-reduce, exp+sum-reduce, normalize.
extern "C" __global__ void attn_softmax(float *S, int N) {
    __shared__ float red[256];
    float *row = S + (size_t)blockIdx.x * N;
    int tid = threadIdx.x;

    float m = -CUDART_INF_F;
    for (int j = tid; j < N; j += blockDim.x) {
        m = fmaxf(m, row[j]);
    }
    red[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float l = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float p = __expf(row[j] - m);
        row[j] = p;
        l += p;
    }
    red[tid] = l;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int j = tid; j < N; j += blockDim.x) {
        row[j] *= inv;
    }
}

extern "C" __global__ void attn_av(const float *P, const float *V, float *O, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 0..D
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && x < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            acc += P[(size_t)row * N + j] * V[j * D + x];
        }
        O[row * D + x] = acc;
    }
}
