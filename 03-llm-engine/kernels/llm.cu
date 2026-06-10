// GPT-2 decode-path kernels. The engine processes one token at a time
// (prompt prefill included), so every matmul is a GEMV: memory-bound by
// definition, which is exactly the regime a 40 GB/s bus punishes — and the
// reason int8 weights nearly quadruple tokens/sec.
//
// Weight matrices are [n_in, n_out] row-major (y = x @ W + b): consecutive
// threads read consecutive outputs of the same input row — fully coalesced.
#include <math_constants.h>

#define LN_EPS 1e-5f

// out = wte_t[:, tok] (+ scale if int8 path uses it) + wpe[pos]
// wte_t is the token embedding stored transposed: [n_embd, n_vocab].
extern "C" __global__ void embed(float *out, const float *wte_t, const float *wpe,
                                 int tok, int pos, int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = wte_t[(size_t)i * n_vocab + tok] + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int8(float *out, const signed char *wte_t,
                                      const float *scales, const float *wpe,
                                      int tok, int pos, int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = (float)wte_t[(size_t)i * n_vocab + tok] * scales[tok] + wpe[pos * n_embd + i];
    }
}

// One block; mean/var over n via shared-memory reduction.
extern "C" __global__ void layernorm(float *out, const float *x, const float *g,
                                     const float *b, int n) {
    __shared__ float red[256];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i];
    red[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    float mean = red[0] / n;
    __syncthreads();

    s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = x[i] - mean;
        s += d * d;
    }
    red[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / n + LN_EPS);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

// y[o] = sum_i x[i] * w[i*n_out+o] + b[o]; x staged through shared memory.
extern "C" __global__ void gemv(float *y, const float *x, const float *w,
                                const float *b, int n_in, int n_out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * w[(size_t)i * n_out + o];
        }
        y[o] = acc + (b ? b[o] : 0.0f);
    }
}

// int8 weights with one fp32 scale per output column (absmax quantization).
extern "C" __global__ void gemv_int8(float *y, const float *x, const signed char *w,
                                     const float *scales, const float *b,
                                     int n_in, int n_out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * (float)w[(size_t)i * n_out + o];
        }
        y[o] = acc * scales[o] + (b ? b[o] : 0.0f);
    }
}

// Causal attention for one new token over the KV cache (one block per head).
// Cache layout per layer: [t][n_embd] where each row is heads*head_dim.
// Scores for up to n_ctx cached positions live in shared memory.
extern "C" __global__ void attn_decode(float *out, const float *qkv,
                                       const float *kcache, const float *vcache,
                                       int t_cur, int n_head, int head_dim) {
    __shared__ float s[1024]; // n_ctx max
    __shared__ float red[128];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int e = n_head * head_dim;
    const float *q = qkv + h * head_dim;
    float scale = rsqrtf((float)head_dim);

    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        const float *k = kcache + (size_t)t * e + h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
        s[t] = dot * scale;
        m = fmaxf(m, s[t]);
    }
    red[tid] = m;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] = fmaxf(red[tid], red[tid + k]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float l = 0.0f;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        s[t] = __expf(s[t] - m);
        l += s[t];
    }
    red[tid] = l;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= t_cur; ++t) {
            acc += s[t] * vcache[(size_t)t * e + h * head_dim + d];
        }
        out[h * head_dim + d] = acc * inv;
    }
}

extern "C" __global__ void add_inplace(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" __global__ void gelu_inplace(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}
