// LLM engine kernels. Decode processes one token at a time, so every matmul
// is a GEMV: memory-bound by definition, which is exactly the regime a
// 40 GB/s bus punishes — and the reason int8 weights nearly quadruple
// tokens/sec. Prompt prefill and speculative verification instead batch T
// tokens through GEMM + flash-style attention (second half of this file).
//
// Weight matrices are [n_in, n_out] row-major (y = x @ W + b): consecutive
// threads read consecutive outputs of the same input row — fully coalesced.
#include <math_constants.h>
#include <cuda_fp16.h>

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

extern "C" __global__ void embed_half(float *out, const __half *wte_t, const float *wpe,
                                      int tok, int pos, int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = __half2float(wte_t[(size_t)i * n_vocab + tok]) + wpe[pos * n_embd + i];
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

extern "C" __global__ void embed_dyn(float *out, const float *wte_t, const float *wpe,
                                     const int *tok_ptr, const int *pos_ptr,
                                     int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = wte_t[(size_t)i * n_vocab + tok] + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_half_dyn(float *out, const __half *wte_t, const float *wpe,
                                          const int *tok_ptr, const int *pos_ptr,
                                          int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = __half2float(wte_t[(size_t)i * n_vocab + tok]) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int8_dyn(float *out, const signed char *wte_t,
                                          const float *scales, const float *wpe,
                                          const int *tok_ptr, const int *pos_ptr,
                                          int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
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

// fp16 storage, fp32 math: weights are loaded as half and immediately widened.
extern "C" __global__ void gemv_half(float *y, const float *x, const __half *w,
                                     const float *b, int n_in, int n_out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * __half2float(w[(size_t)i * n_out + o]);
        }
        y[o] = acc + (b ? b[o] : 0.0f);
    }
}

// int8 weights with one fp32 scale per output column (absmax quantization).
// For wide outputs (a multiple of 4) each thread produces 4 outputs from
// char4 loads — without this the kernel is instruction-bound on byte loads
// and loses to fp16 despite moving half the bytes. Narrow outputs keep one
// output per thread: dropping the thread count 4x there starves the SMs of
// latency-hiding warps and costs more than the wider loads gain.
extern "C" __global__ void gemv_int8(float *y, const float *x, const signed char *w,
                                     const float *scales, const float *b,
                                     int n_in, int n_out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    if (n_out % 4 == 0 && n_out >= 4096) {
        const char4 *w4 = (const char4 *)w;
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int i = 0; i < n_in; ++i) {
                char4 c = w4[(size_t)i * n4 + o4];
                float xi = xs[i];
                a0 += xi * (float)c.x;
                a1 += xi * (float)c.y;
                a2 += xi * (float)c.z;
                a3 += xi * (float)c.w;
            }
            int o = 4 * o4;
            y[o + 0] = a0 * scales[o + 0] + (b ? b[o + 0] : 0.0f);
            y[o + 1] = a1 * scales[o + 1] + (b ? b[o + 1] : 0.0f);
            y[o + 2] = a2 * scales[o + 2] + (b ? b[o + 2] : 0.0f);
            y[o + 3] = a3 * scales[o + 3] + (b ? b[o + 3] : 0.0f);
        }
        return;
    }
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * (float)w[(size_t)i * n_out + o];
        }
        y[o] = acc * scales[o] + (b ? b[o] : 0.0f);
    }
}

// ---- int4 weights (Q4_0-style) ---------------------------------------------
// Two weights per byte packed along n_in: byte (i/2)*n_out + o holds rows i
// (low nibble) and i+1 (high nibble) of column o, nibbles store q+8 with
// q in [-8, 7]. One fp16 scale per (32-row group, column):
// scales[(i/32)*n_out + o]. Dequant: (nibble - 8) * scale. The group scale
// must be applied per group, not per column, so unlike int8 the GEMM/GEMV
// bodies dequantize inline instead of scaling the final accumulator.

#define Q4_GROUP 32

__device__ __forceinline__ float q4_lo(unsigned char b) { return (float)((b & 15) - 8); }
__device__ __forceinline__ float q4_hi(unsigned char b) { return (float)((b >> 4) - 8); }

// Same wide/narrow split as gemv_int8: wide outputs take 4 columns per
// thread from one uchar4 (which now covers two n_in rows at once).
extern "C" __global__ void gemv_int4(float *y, const float *x, const unsigned char *w,
                                     const __half *scales, const float *b,
                                     int n_in, int n_out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    int n_groups = n_in / Q4_GROUP;
    if (n_out % 4 == 0 && n_out >= 4096) {
        const uchar4 *w4 = (const uchar4 *)w;
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int g = 0; g < n_groups; ++g) {
                float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
                for (int i = g * Q4_GROUP; i < (g + 1) * Q4_GROUP; i += 2) {
                    uchar4 c = w4[(size_t)(i / 2) * n4 + o4];
                    float x0 = xs[i], x1 = xs[i + 1];
                    g0 += x0 * q4_lo(c.x) + x1 * q4_hi(c.x);
                    g1 += x0 * q4_lo(c.y) + x1 * q4_hi(c.y);
                    g2 += x0 * q4_lo(c.z) + x1 * q4_hi(c.z);
                    g3 += x0 * q4_lo(c.w) + x1 * q4_hi(c.w);
                }
                const __half2 *s2 = (const __half2 *)(scales + (size_t)g * n_out + 4 * o4);
                float2 sa = __half22float2(s2[0]), sb = __half22float2(s2[1]);
                a0 += g0 * sa.x;
                a1 += g1 * sa.y;
                a2 += g2 * sb.x;
                a3 += g3 * sb.y;
            }
            int o = 4 * o4;
            y[o + 0] = a0 + (b ? b[o + 0] : 0.0f);
            y[o + 1] = a1 + (b ? b[o + 1] : 0.0f);
            y[o + 2] = a2 + (b ? b[o + 2] : 0.0f);
            y[o + 3] = a3 + (b ? b[o + 3] : 0.0f);
        }
        return;
    }
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < n_groups; ++g) {
            float gs = 0.0f;
            for (int i = g * Q4_GROUP; i < (g + 1) * Q4_GROUP; i += 2) {
                unsigned char c = w[(size_t)(i / 2) * n_out + o];
                gs += xs[i] * q4_lo(c) + xs[i + 1] * q4_hi(c);
            }
            acc += gs * __half2float(scales[(size_t)g * n_out + o]);
        }
        y[o] = acc + (b ? b[o] : 0.0f);
    }
}

__device__ __forceinline__ float embed_int4_at(const unsigned char *wte_t,
                                               const __half *scales, int i, int tok,
                                               int n_vocab) {
    unsigned char c = wte_t[(size_t)(i / 2) * n_vocab + tok];
    float q = (i & 1) ? q4_hi(c) : q4_lo(c);
    return q * __half2float(scales[(size_t)(i / Q4_GROUP) * n_vocab + tok]);
}

extern "C" __global__ void embed_int4(float *out, const unsigned char *wte_t,
                                      const __half *scales, const float *wpe,
                                      int tok, int pos, int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int4_at(wte_t, scales, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int4_dyn(float *out, const unsigned char *wte_t,
                                          const __half *scales, const float *wpe,
                                          const int *tok_ptr, const int *pos_ptr,
                                          int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int4_at(wte_t, scales, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int4_batch(float *out, const unsigned char *wte_t,
                                            const __half *scales, const float *wpe,
                                            const int *toks, int pos0, int n_tok,
                                            int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = embed_int4_at(wte_t, scales, i, toks[t], n_vocab) +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

extern "C" __global__ void copy_kv_dyn(float *kcache, float *vcache, const float *qkv,
                                       const int *pos_ptr, int q_dim, int kv_dim) {
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kcache[(size_t)pos * kv_dim + i] = qkv[q_dim + i];
        vcache[(size_t)pos * kv_dim + i] = qkv[q_dim + kv_dim + i];
    }
}

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * g. One block.
extern "C" __global__ void rmsnorm(float *out, const float *x, const float *g,
                                   int n, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i] * x[i];
    red[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / n + eps);
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inv * g[i];
    }
}

extern "C" __global__ void layernorm_batch(float *out, const float *x, const float *g,
                                           const float *b, int rows, int n) {
    __shared__ float red[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float *xr = x + (size_t)row * n;
    float *orow = out + (size_t)row * n;

    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += xr[i];
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
        float d = xr[i] - mean;
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
        orow[i] = (xr[i] - mean) * inv * g[i] + b[i];
    }
}

extern "C" __global__ void rmsnorm_batch(float *out, const float *x, const float *g,
                                         int rows, int n, float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float *xr = x + (size_t)row * n;
    float *orow = out + (size_t)row * n;

    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += xr[i] * xr[i];
    red[tid] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / n + eps);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        orow[i] = xr[i] * inv * g[i];
    }
}

// Rotary position embedding over the Q and K sections of the qkv buffer
// (HF rotate_half convention: pairs are (d, d + head_dim/2)). K heads follow
// Q heads directly in memory, so one flat index covers both.
__device__ void rope_impl(float *qkv, int pos, int n_head, int n_kv_head,
                          int head_dim, float theta) {
    int half = head_dim / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n_head + n_kv_head) * half) return;
    int h = i / half;
    int d = i % half;
    float *base = qkv + h * head_dim;
    float freq = __powf(theta, -2.0f * d / head_dim);
    float c, s;
    __sincosf(pos * freq, &s, &c);
    float x1 = base[d], x2 = base[d + half];
    base[d] = x1 * c - x2 * s;
    base[d + half] = x1 * s + x2 * c;
}

extern "C" __global__ void rope(float *qkv, int pos, int n_head, int n_kv_head,
                                int head_dim, float theta) {
    rope_impl(qkv, pos, n_head, n_kv_head, head_dim, theta);
}

extern "C" __global__ void rope_dyn(float *qkv, const int *pos_ptr, int n_head,
                                    int n_kv_head, int head_dim, float theta) {
    rope_impl(qkv, *pos_ptr, n_head, n_kv_head, head_dim, theta);
}

extern "C" __global__ void rope_batch(float *qkv, int pos0, int n_tok, int n_head,
                                      int n_kv_head, int head_dim, int stride,
                                      float theta) {
    int half = head_dim / 2;
    int per_row = (n_head + n_kv_head) * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tok * per_row;
    if (idx >= total) return;
    int t = idx / per_row;
    int i = idx - t * per_row;
    int h = i / half;
    int d = i % half;
    float *base = qkv + (size_t)t * stride + h * head_dim;
    float freq = __powf(theta, -2.0f * d / head_dim);
    float c, s;
    __sincosf((pos0 + t) * freq, &s, &c);
    float x1 = base[d], x2 = base[d + half];
    base[d] = x1 * c - x2 * s;
    base[d + half] = x1 * s + x2 * c;
}

// SwiGLU combine: x = silu(x) * y.
extern "C" __global__ void silu_mul(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + __expf(-v)) * y[i];
    }
}

// Quantizes the new K/V rows into int8 caches, one fp32 absmax scale per
// (position, kv head). One block per kv head, one thread per head dim.
__device__ void quantize_kv_impl(signed char *kq, signed char *vq,
                                 float *ks, float *vs, const float *qkv,
                                 int pos, int q_dim, int n_kv_head, int head_dim) {
    __shared__ float red[128];
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *k = qkv + q_dim + h * head_dim;
    const float *v = qkv + q_dim + kv_dim + h * head_dim;

    red[d] = fabsf(k[d]);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) red[d] = fmaxf(red[d], red[d + s]);
        __syncthreads();
    }
    float kscale = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;
    __syncthreads();

    red[d] = fabsf(v[d]);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) red[d] = fmaxf(red[d], red[d + s]);
        __syncthreads();
    }
    float vscale = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;

    size_t row = (size_t)pos * kv_dim + h * head_dim;
    kq[row + d] = (signed char)lrintf(k[d] / kscale);
    vq[row + d] = (signed char)lrintf(v[d] / vscale);
    if (d == 0) {
        ks[pos * n_kv_head + h] = kscale;
        vs[pos * n_kv_head + h] = vscale;
    }
}

extern "C" __global__ void quantize_kv(signed char *kq, signed char *vq,
                                       float *ks, float *vs, const float *qkv,
                                       int pos, int q_dim, int n_kv_head, int head_dim) {
    quantize_kv_impl(kq, vq, ks, vs, qkv, pos, q_dim, n_kv_head, head_dim);
}

extern "C" __global__ void quantize_kv_dyn(signed char *kq, signed char *vq,
                                           float *ks, float *vs, const float *qkv,
                                           const int *pos_ptr, int q_dim, int n_kv_head,
                                           int head_dim) {
    quantize_kv_impl(kq, vq, ks, vs, qkv, *pos_ptr, q_dim, n_kv_head, head_dim);
}

extern "C" __global__ void quantize_kv_batch(signed char *kq, signed char *vq,
                                             float *ks, float *vs, const float *qkv,
                                             int pos0, int q_dim, int n_kv_head,
                                             int head_dim, int stride) {
    __shared__ float red[128];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *row = qkv + (size_t)t * stride;
    const float *k = row + q_dim + h * head_dim;
    const float *v = row + q_dim + kv_dim + h * head_dim;
    int pos = pos0 + t;

    red[d] = fabsf(k[d]);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) red[d] = fmaxf(red[d], red[d + s]);
        __syncthreads();
    }
    float kscale = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;
    __syncthreads();

    red[d] = fabsf(v[d]);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) red[d] = fmaxf(red[d], red[d + s]);
        __syncthreads();
    }
    float vscale = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;

    size_t out = (size_t)pos * kv_dim + h * head_dim;
    kq[out + d] = (signed char)lrintf(k[d] / kscale);
    vq[out + d] = (signed char)lrintf(v[d] / vscale);
    if (d == 0) {
        ks[pos * n_kv_head + h] = kscale;
        vs[pos * n_kv_head + h] = vscale;
    }
}

// Causal attention for one new token over the KV cache (one block per query
// head). Cache layout per layer: [t][n_kv_head * head_dim]; with grouped-query
// attention (n_kv_head < n_head) several query heads share one kv head.
// Scores for up to n_ctx cached positions live in shared memory.
__device__ void attn_decode_impl(float *out, const float *qkv,
                                 const float *kcache, const float *vcache,
                                 int t_cur, int n_head, int n_kv_head, int head_dim) {
    __shared__ float s[1024]; // n_ctx max
    __shared__ float red[128];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int kvd = n_kv_head * head_dim;
    int kvh = h / (n_head / n_kv_head);
    const float *q = qkv + h * head_dim;
    float scale = rsqrtf((float)head_dim);

    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        const float *k = kcache + (size_t)t * kvd + kvh * head_dim;
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
            acc += s[t] * vcache[(size_t)t * kvd + kvh * head_dim + d];
        }
        out[h * head_dim + d] = acc * inv;
    }
}

extern "C" __global__ void attn_decode(float *out, const float *qkv,
                                       const float *kcache, const float *vcache,
                                       int t_cur, int n_head, int n_kv_head, int head_dim) {
    attn_decode_impl(out, qkv, kcache, vcache, t_cur, n_head, n_kv_head, head_dim);
}

extern "C" __global__ void attn_decode_dyn(float *out, const float *qkv,
                                           const float *kcache, const float *vcache,
                                           const int *pos_ptr, int n_head, int n_kv_head,
                                           int head_dim) {
    attn_decode_impl(out, qkv, kcache, vcache, *pos_ptr, n_head, n_kv_head, head_dim);
}

// Same attention over an int8 KV cache: scores and the V accumulation
// dequantize on the fly with the per-(position, head) scales, so the cache
// traffic — the part that grows with context length — shrinks 4x.
__device__ void attn_decode_q8_impl(float *out, const float *qkv,
                                    const signed char *kq, const signed char *vq,
                                    const float *ks, const float *vs,
                                    int t_cur, int n_head, int n_kv_head, int head_dim) {
    __shared__ float s[1024]; // n_ctx max
    __shared__ float red[128];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int kvd = n_kv_head * head_dim;
    int kvh = h / (n_head / n_kv_head);
    const float *q = qkv + h * head_dim;
    float scale = rsqrtf((float)head_dim);

    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        // head rows are head_dim-byte aligned, so char4 loads are safe and
        // cut the byte-load instruction count 4x
        const char4 *k4 = (const char4 *)(kq + (size_t)t * kvd + kvh * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < head_dim / 4; ++d) {
            char4 c = k4[d];
            dot += q[4 * d] * (float)c.x + q[4 * d + 1] * (float)c.y +
                   q[4 * d + 2] * (float)c.z + q[4 * d + 3] * (float)c.w;
        }
        s[t] = dot * ks[t * n_kv_head + kvh] * scale;
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
            acc += s[t] * vs[t * n_kv_head + kvh] *
                   (float)vq[(size_t)t * kvd + kvh * head_dim + d];
        }
        out[h * head_dim + d] = acc * inv;
    }
}

extern "C" __global__ void attn_decode_q8(float *out, const float *qkv,
                                          const signed char *kq, const signed char *vq,
                                          const float *ks, const float *vs,
                                          int t_cur, int n_head, int n_kv_head,
                                          int head_dim) {
    attn_decode_q8_impl(out, qkv, kq, vq, ks, vs, t_cur, n_head, n_kv_head, head_dim);
}

extern "C" __global__ void attn_decode_q8_dyn(float *out, const float *qkv,
                                              const signed char *kq, const signed char *vq,
                                              const float *ks, const float *vs,
                                              const int *pos_ptr, int n_head, int n_kv_head,
                                              int head_dim) {
    attn_decode_q8_impl(out, qkv, kq, vq, ks, vs, *pos_ptr, n_head, n_kv_head, head_dim);
}

// ---- batched prefill / speculative-verify path ----------------------------
// Decode is one token at a time (GEMV, memory-bound); prefill and draft
// verification process T tokens at once, so the matmuls become GEMMs that
// read each weight once for all T rows — the stage-1 playbook (smem tiles,
// register micro-tiles) applied inside the engine.

__device__ __forceinline__ float tof(float x) { return x; }
__device__ __forceinline__ float tof(__half x) { return __half2float(x); }
__device__ __forceinline__ float tof(signed char x) { return (float)x; }

// Four consecutive B elements widened to fp32 in one (or two) wide loads —
// the char4 GEMV lesson applied to the GEMM tile loads: scalar byte/half
// loads are instruction-bound and leave the bus idle.
__device__ __forceinline__ void load4(const float *p, float *o) {
    float4 v = *reinterpret_cast<const float4 *>(p);
    o[0] = v.x, o[1] = v.y, o[2] = v.z, o[3] = v.w;
}
__device__ __forceinline__ void load4(const __half *p, float *o) {
    __half2 a = *reinterpret_cast<const __half2 *>(p);
    __half2 b = *reinterpret_cast<const __half2 *>(p + 2);
    o[0] = __low2float(a), o[1] = __high2float(a);
    o[2] = __low2float(b), o[3] = __high2float(b);
}
__device__ __forceinline__ void load4(const signed char *p, float *o) {
    char4 c = *reinterpret_cast<const char4 *>(p);
    o[0] = (float)c.x, o[1] = (float)c.y, o[2] = (float)c.z, o[3] = (float)c.w;
}

// C[M,N] = A[M,K] @ B[K,N] (+ scales per column for int8) + bias.
// BM x 64 tiles, BK = 16, 256 threads each computing a (BM/16) x 4 micro-tile.
// Two tile heights: BM=64 for prefill-sized M, BM=16 for speculative verify —
// a 64-row tile burns 8x the FMAs on an 8-row draft batch, and that compute
// waste (not bandwidth) was the floor of the verify pass. K is a multiple of
// 16 for every matrix in the engine; N tiles are bounds-checked, and B loads
// vectorize whenever the tile sits fully inside an N divisible by 4.
template <typename BT, bool SCALED, int BM>
__device__ void gemm_body(float *C, const float *A, const BT *B, const float *scales,
                          const float *bias, int M, int N, int K) {
    constexpr int RM = BM / 16; // micro-tile rows per thread
    __shared__ float As[16][BM];
    __shared__ float Bs[16][64];
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    float acc[RM][4] = {};
    bool vec = (N % 4 == 0) && (bn + 64 <= N);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = tid; i < BM * 16; i += 256) {
            int m = i / 16, k = i % 16;
            As[k][m] = (bm + m < M) ? A[(size_t)(bm + m) * K + k0 + k] : 0.0f;
        }
        if (vec) {
            int k = tid / 16, n = (tid % 16) * 4;
            float t[4];
            load4(&B[(size_t)(k0 + k) * N + bn + n], t);
            *reinterpret_cast<float4 *>(&Bs[k][n]) = make_float4(t[0], t[1], t[2], t[3]);
        } else {
            for (int i = tid; i < 16 * 64; i += 256) {
                int k = i / 64, n = i % 64;
                Bs[k][n] = (bn + n < N) ? tof(B[(size_t)(k0 + k) * N + bn + n]) : 0.0f;
            }
        }
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            float a[RM], b[4];
            for (int i = 0; i < RM; ++i) a[i] = As[k][threadIdx.y * RM + i];
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][threadIdx.x * 4 + j];
            for (int i = 0; i < RM; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < RM; ++i) {
        int row = bm + threadIdx.y * RM + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + threadIdx.x * 4 + j;
            if (col >= N) continue;
            float v = acc[i][j];
            if (SCALED) v *= scales[col];
            C[(size_t)row * N + col] = v + bias[col];
        }
    }
}

extern "C" __global__ void gemm_f32(float *C, const float *A, const float *B,
                                    const float *bias, int M, int N, int K) {
    gemm_body<float, false, 64>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_half(float *C, const float *A, const __half *B,
                                     const float *bias, int M, int N, int K) {
    gemm_body<__half, false, 64>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_int8(float *C, const float *A, const signed char *B,
                                     const float *scales, const float *bias,
                                     int M, int N, int K) {
    gemm_body<signed char, true, 64>(C, A, B, scales, bias, M, N, K);
}

extern "C" __global__ void gemm_f32_skinny(float *C, const float *A, const float *B,
                                           const float *bias, int M, int N, int K) {
    gemm_body<float, false, 16>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_half_skinny(float *C, const float *A, const __half *B,
                                            const float *bias, int M, int N, int K) {
    gemm_body<__half, false, 16>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_int8_skinny(float *C, const float *A, const signed char *B,
                                            const float *scales, const float *bias,
                                            int M, int N, int K) {
    gemm_body<signed char, true, 16>(C, A, B, scales, bias, M, N, K);
}

// Draft-verify GEMM (M <= 8) as a multi-row GEMV: square tiles waste compute
// when M is a handful of rows (a 16x64 tile burns 2-16x the FMAs and its 1x4
// micro-tile is shared-load-bound), so instead each thread owns output
// columns gemv-style — B streams through exactly once with zero wasted
// compute, the 8-row accumulator lives in registers, and A is staged through
// shared memory where reads are warp-broadcast (all threads read the same
// As[m][kk]). Column ownership copies the gemv_int8 heuristic: wide matrices
// (N % 4 == 0, N >= 4096) take 4 columns per thread via one vectorized load,
// narrow ones keep 1 column per thread — fewer threads starve the SMs of
// latency-hiding warps. Rows past M are zero-padded so the per-thread loops
// fully unroll and the accumulators never spill.
#define ROWS_M 8
#define ROWS_KT 128

__device__ __forceinline__ bool gemm_rows_wide(int N) { return N % 4 == 0 && N >= 4096; }

template <typename BT, bool SCALED>
__device__ void gemm_rows_body(float *C, const float *A, const BT *B,
                               const float *scales, const float *bias,
                               int M, int N, int K) {
    __shared__ float As[ROWS_M][ROWS_KT];
    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;
    bool wide = gemm_rows_wide(N);
    bool active = o < (wide ? N / 4 : N);
    float acc[ROWS_M][4] = {};

    for (int k0 = 0; k0 < K; k0 += ROWS_KT) {
        int kt = min(ROWS_KT, K - k0);
        for (int i = tid; i < ROWS_M * ROWS_KT; i += blockDim.x) {
            int m = i / ROWS_KT, kk = i % ROWS_KT;
            As[m][kk] = (m < M && kk < kt) ? A[(size_t)m * K + k0 + kk] : 0.0f;
        }
        __syncthreads();
        if (active && wide) {
            for (int kk = 0; kk < kt; ++kk) {
                float b[4];
                load4(&B[(size_t)(k0 + kk) * N + 4 * o], b);
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) {
                    float a = As[m][kk];
                    acc[m][0] += a * b[0];
                    acc[m][1] += a * b[1];
                    acc[m][2] += a * b[2];
                    acc[m][3] += a * b[3];
                }
            }
        } else if (active) {
            for (int kk = 0; kk < kt; ++kk) {
                float b = tof(B[(size_t)(k0 + kk) * N + o]);
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) acc[m][0] += As[m][kk] * b;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int ncols = wide ? 4 : 1;
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < ncols; ++j) {
            int col = wide ? 4 * o + j : o;
            float v = acc[m][j];
            if (SCALED) v *= scales[col];
            C[(size_t)m * N + col] = v + bias[col];
        }
    }
}

extern "C" __global__ void gemm_rows_f32(float *C, const float *A, const float *B,
                                         const float *bias, int M, int N, int K) {
    gemm_rows_body<float, false>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_rows_half(float *C, const float *A, const __half *B,
                                          const float *bias, int M, int N, int K) {
    gemm_rows_body<__half, false>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_rows_int8(float *C, const float *A, const signed char *B,
                                          const float *scales, const float *bias,
                                          int M, int N, int K) {
    gemm_rows_body<signed char, true>(C, A, B, scales, bias, M, N, K);
}

// int4 GEMM: same tiling as gemm_body, but the group scale depends on the k
// row, so B dequantizes during the shared-tile fill instead of scaling the
// final accumulator. A 16-row k-tile always sits inside one 32-row quant
// group (k0 is a multiple of 16), so the scale row is constant per tile and
// one byte fills two k-rows of Bs.
template <int BM>
__device__ void gemm_int4_body(float *C, const float *A, const unsigned char *B,
                               const __half *scales, const float *bias,
                               int M, int N, int K) {
    constexpr int RM = BM / 16;
    __shared__ float As[16][BM];
    __shared__ float Bs[16][64];
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    float acc[RM][4] = {};

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = tid; i < BM * 16; i += 256) {
            int m = i / 16, k = i % 16;
            As[k][m] = (bm + m < M) ? A[(size_t)(bm + m) * K + k0 + k] : 0.0f;
        }
        int gk = k0 / Q4_GROUP;
        for (int i = tid; i < 8 * 64; i += 256) {
            int kb = i / 64, n = i % 64; // byte row kb covers k rows 2kb, 2kb+1
            if (bn + n < N) {
                unsigned char c = B[(size_t)(k0 / 2 + kb) * N + bn + n];
                float sc = __half2float(scales[(size_t)gk * N + bn + n]);
                Bs[2 * kb][n] = q4_lo(c) * sc;
                Bs[2 * kb + 1][n] = q4_hi(c) * sc;
            } else {
                Bs[2 * kb][n] = 0.0f;
                Bs[2 * kb + 1][n] = 0.0f;
            }
        }
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            float a[RM], b[4];
            for (int i = 0; i < RM; ++i) a[i] = As[k][threadIdx.y * RM + i];
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][threadIdx.x * 4 + j];
            for (int i = 0; i < RM; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < RM; ++i) {
        int row = bm + threadIdx.y * RM + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + threadIdx.x * 4 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = acc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int4(float *C, const float *A, const unsigned char *B,
                                     const __half *scales, const float *bias,
                                     int M, int N, int K) {
    gemm_int4_body<64>(C, A, B, scales, bias, M, N, K);
}

extern "C" __global__ void gemm_int4_skinny(float *C, const float *A, const unsigned char *B,
                                            const __half *scales, const float *bias,
                                            int M, int N, int K) {
    gemm_int4_body<16>(C, A, B, scales, bias, M, N, K);
}

// int4 draft-verify GEMM (M <= 8): gemm_rows with inline dequant. Scales
// reload every 32 k-rows (K is a multiple of 32, so groups never straddle
// the 128-row k-tile), one uchar4 covers 4 columns x 2 k-rows.
extern "C" __global__ void gemm_rows_int4(float *C, const float *A, const unsigned char *B,
                                          const __half *scales, const float *bias,
                                          int M, int N, int K) {
    __shared__ float As[ROWS_M][ROWS_KT];
    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;
    bool wide = gemm_rows_wide(N);
    bool active = o < (wide ? N / 4 : N);
    float acc[ROWS_M][4] = {};
    float s[4];

    for (int k0 = 0; k0 < K; k0 += ROWS_KT) {
        int kt = min(ROWS_KT, K - k0);
        for (int i = tid; i < ROWS_M * ROWS_KT; i += blockDim.x) {
            int m = i / ROWS_KT, kk = i % ROWS_KT;
            As[m][kk] = (m < M && kk < kt) ? A[(size_t)m * K + k0 + kk] : 0.0f;
        }
        __syncthreads();
        if (active && wide) {
            for (int kk = 0; kk < kt; kk += 2) {
                if ((k0 + kk) % Q4_GROUP == 0) {
                    const __half2 *s2 =
                        (const __half2 *)(scales + (size_t)((k0 + kk) / Q4_GROUP) * N + 4 * o);
                    float2 sa = __half22float2(s2[0]), sb = __half22float2(s2[1]);
                    s[0] = sa.x, s[1] = sa.y, s[2] = sb.x, s[3] = sb.y;
                }
                uchar4 c = *(const uchar4 *)(B + (size_t)((k0 + kk) / 2) * N + 4 * o);
                float b0[4] = {q4_lo(c.x) * s[0], q4_lo(c.y) * s[1],
                               q4_lo(c.z) * s[2], q4_lo(c.w) * s[3]};
                float b1[4] = {q4_hi(c.x) * s[0], q4_hi(c.y) * s[1],
                               q4_hi(c.z) * s[2], q4_hi(c.w) * s[3]};
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) {
                    float a0 = As[m][kk], a1 = As[m][kk + 1];
                    acc[m][0] += a0 * b0[0] + a1 * b1[0];
                    acc[m][1] += a0 * b0[1] + a1 * b1[1];
                    acc[m][2] += a0 * b0[2] + a1 * b1[2];
                    acc[m][3] += a0 * b0[3] + a1 * b1[3];
                }
            }
        } else if (active) {
            for (int kk = 0; kk < kt; kk += 2) {
                if ((k0 + kk) % Q4_GROUP == 0) {
                    s[0] = __half2float(scales[(size_t)((k0 + kk) / Q4_GROUP) * N + o]);
                }
                unsigned char c = B[(size_t)((k0 + kk) / 2) * N + o];
                float b0 = q4_lo(c) * s[0], b1 = q4_hi(c) * s[0];
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m)
                    acc[m][0] += As[m][kk] * b0 + As[m][kk + 1] * b1;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int ncols = wide ? 4 : 1;
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < ncols; ++j) {
            int col = wide ? 4 * o + j : o;
            C[(size_t)m * N + col] = acc[m][j] + bias[col];
        }
    }
}

extern "C" __global__ void embed_batch(float *out, const float *wte_t, const float *wpe,
                                       const int *toks, int pos0, int n_tok,
                                       int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = wte_t[(size_t)i * n_vocab + toks[t]] + wpe[(pos0 + t) * n_embd + i];
    }
}

extern "C" __global__ void embed_half_batch(float *out, const __half *wte_t,
                                            const float *wpe, const int *toks, int pos0,
                                            int n_tok, int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = __half2float(wte_t[(size_t)i * n_vocab + toks[t]]) +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

extern "C" __global__ void embed_int8_batch(float *out, const signed char *wte_t,
                                            const float *scales, const float *wpe,
                                            const int *toks, int pos0, int n_tok,
                                            int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        int tok = toks[t];
        out[idx] = (float)wte_t[(size_t)i * n_vocab + tok] * scales[tok] +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

extern "C" __global__ void copy_kv_batch(float *kcache, float *vcache, const float *qkv,
                                         int pos0, int q_dim, int kv_dim, int stride) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kcache[(size_t)(pos0 + t) * kv_dim + i] = qkv[(size_t)t * stride + q_dim + i];
        vcache[(size_t)(pos0 + t) * kv_dim + i] = qkv[(size_t)t * stride + q_dim + kv_dim + i];
    }
}

// Flash-style batched causal attention over the KV cache (the stage-2
// algorithm adapted to GQA and cache layout): one block of 64 threads per
// (64-query tile, head); K/V tiles staged through shared memory, online
// softmax with running max/sum, no materialized score matrix. The query at
// row qi sits at absolute position pos0 + qi and attends to keys 0..pos0+qi.
// head_dim is fixed at 64 (q and acc live in registers).
template <bool Q8>
__device__ void attn_prefill_body(float *out, const float *qkv,
                                  const float *kcache, const float *vcache,
                                  const signed char *kq, const signed char *vq,
                                  const float *ks, const float *vs,
                                  int pos0, int n_tok, int n_head, int n_kv_head,
                                  int qkv_stride, int out_stride) {
    __shared__ float Kt[64][64];
    __shared__ float Vt[64][64];
    int h = blockIdx.x;
    int tile0 = blockIdx.y * 64;
    int tid = threadIdx.x;
    int kvd = n_kv_head * 64;
    int kvh = h / (n_head / n_kv_head);
    int qi = tile0 + tid;
    bool active = qi < n_tok;
    int pq = pos0 + (active ? qi : 0);

    float q[64], acc[64] = {};
    float m = -CUDART_INF_F, l = 0.0f;
    if (active) {
        for (int d = 0; d < 64; ++d) q[d] = qkv[(size_t)qi * qkv_stride + h * 64 + d];
    }
    float scale = rsqrtf(64.0f);

    int max_key = pos0 + min(tile0 + 63, n_tok - 1);
    for (int kt = 0; kt <= max_key; kt += 64) {
        int tile_n = min(64, max_key - kt + 1);
        for (int x = tid; x < tile_n * 64; x += 64) {
            int r = x / 64, d = x % 64;
            if (Q8) {
                Kt[r][d] = (float)kq[(size_t)(kt + r) * kvd + kvh * 64 + d] *
                           ks[(kt + r) * n_kv_head + kvh];
                Vt[r][d] = (float)vq[(size_t)(kt + r) * kvd + kvh * 64 + d] *
                           vs[(kt + r) * n_kv_head + kvh];
            } else {
                Kt[r][d] = kcache[(size_t)(kt + r) * kvd + kvh * 64 + d];
                Vt[r][d] = vcache[(size_t)(kt + r) * kvd + kvh * 64 + d];
            }
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < tile_n; ++j) {
                int kp = kt + j;
                if (kp > pq) break;
                float dot = 0.0f;
                for (int d = 0; d < 64; ++d) dot += q[d] * Kt[j][d];
                float s = dot * scale;
                float mn = fmaxf(m, s);
                float corr = __expf(m - mn);
                float p = __expf(s - mn);
                l = l * corr + p;
                for (int d = 0; d < 64; ++d) acc[d] = acc[d] * corr + p * Vt[j][d];
                m = mn;
            }
        }
        __syncthreads();
    }
    if (active) {
        float inv = 1.0f / l;
        for (int d = 0; d < 64; ++d) {
            out[(size_t)qi * out_stride + h * 64 + d] = acc[d] * inv;
        }
    }
}

extern "C" __global__ void attn_prefill(float *out, const float *qkv,
                                        const float *kcache, const float *vcache,
                                        int pos0, int n_tok, int n_head, int n_kv_head,
                                        int qkv_stride, int out_stride) {
    attn_prefill_body<false>(out, qkv, kcache, vcache, nullptr, nullptr, nullptr, nullptr,
                             pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride);
}

extern "C" __global__ void attn_prefill_q8(float *out, const float *qkv,
                                           const signed char *kq, const signed char *vq,
                                           const float *ks, const float *vs,
                                           int pos0, int n_tok, int n_head, int n_kv_head,
                                           int qkv_stride, int out_stride) {
    attn_prefill_body<true>(out, qkv, nullptr, nullptr, kq, vq, ks, vs,
                            pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride);
}

// Per-row greedy argmax for the speculative verify step (one block per row).
extern "C" __global__ void argmax_rows(int *out, const float *logits, int n_vocab) {
    __shared__ float vals[256];
    __shared__ int idxs[256];
    const float *row = logits + (size_t)blockIdx.x * n_vocab;
    int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = row[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_i;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) {
            float other = vals[tid + k];
            int other_i = idxs[tid + k];
            if (other > vals[tid] || (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = idxs[0];
}

extern "C" __global__ void copy_row(float *dst, const float *src, int row, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) dst[i] = src[(size_t)row * cols + i];
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

extern "C" __global__ void argmax_advance(int *tok_ptr, int *pos_ptr,
                                          const float *logits, int n_vocab) {
    __shared__ float vals[256];
    __shared__ int idxs[256];
    int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_i;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) {
            float other = vals[tid + k];
            int other_i = idxs[tid + k];
            if (other > vals[tid] || (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *tok_ptr = idxs[0];
        *pos_ptr += 1;
    }
}
