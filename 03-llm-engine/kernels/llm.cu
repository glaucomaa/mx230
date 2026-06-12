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

// int8 wte_t is packed 4-along-n_in like every int8 weight (see the dp4a
// section): element (i, tok) lives at byte ((i/4)*n_vocab + tok)*4 + (i&3).
__device__ __forceinline__ float embed_int8_at(const signed char *wte_t,
                                               const float *scales, int i, int tok,
                                               int n_vocab) {
    signed char q = wte_t[((size_t)(i / 4) * n_vocab + tok) * 4 + (i & 3)];
    return (float)q * scales[tok];
}

extern "C" __global__ void embed_int8(float *out, const signed char *wte_t,
                                      const float *scales, const float *wpe,
                                      int tok, int pos, int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int8_at(wte_t, scales, i, tok, n_vocab) + wpe[pos * n_embd + i];
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
        out[i] = embed_int8_at(wte_t, scales, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

// One block; mean/var over n via shared-memory reduction.
// ---- warp-shuffle block reductions -----------------------------------------
// A block-wide reduction is two shuffle sweeps and a 32-slot smem bounce
// (3 __syncthreads) instead of a log2(blockDim) smem tree with 8+. The
// leading sync makes back-to-back reductions safe on one shared buffer.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    return v;
}
template <bool MAX>
__device__ __forceinline__ float block_red(float v, float *red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    v = MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = (lane < nw) ? red[lane] : (MAX ? -CUDART_INF_F : 0.0f);
        v = MAX ? warp_max(v) : warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}
__device__ __forceinline__ float block_sum(float v, float *red) {
    return block_red<false>(v, red);
}
__device__ __forceinline__ float block_max(float v, float *red) {
    return block_red<true>(v, red);
}

extern "C" __global__ void layernorm(float *out, const float *x, const float *g,
                                     const float *b, int n) {
    __shared__ float red[32];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i];
    float mean = block_sum(s, red) / n;

    s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = x[i] - mean;
        s += d * d;
    }
    float inv = rsqrtf(block_sum(s, red) / n + LN_EPS);

    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

// y[o] (+)= sum_i x[i] * w[i*n_out+o] + b[o]; x staged through shared
// memory. accum != 0 adds into y instead of overwriting — the residual
// add fused into the projection that produced it, one launch instead of two.
extern "C" __global__ void gemv(float *y, const float *x, const float *w,
                                const float *b, int n_in, int n_out, int accum) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * w[(size_t)i * n_out + o];
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

// fp16 storage, fp32 math: weights are loaded as half and immediately widened.
extern "C" __global__ void gemv_half(float *y, const float *x, const __half *w,
                                     const float *b, int n_in, int n_out, int accum) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; ++i) {
            acc += xs[i] * __half2float(w[(size_t)i * n_out + o]);
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

// ---- dp4a int8 path --------------------------------------------------------
// sm_61's dp4a does 4 int8 MACs per instruction (measured: 2941 GOPS vs
// 735 fp32 GFLOPS — see common/examples/isa), so the int8 matmuls quantize
// activations on the fly (absmax per 32-value group, llama.cpp Q8-style)
// and multiply in integers, paying one float op per group instead of one
// per weight. Weights are repacked at load into int32 words of 4
// consecutive n_in rows: w32[(i/4)*n_out + o] holds w[i..i+3, o] —
// consecutive columns stay consecutive in memory, coalescing unchanged.

// Activation groups are 8 values (2 dp4a words): GPT-2's activation
// outliers wreck a 32-wide absmax group (one outlier costs 31 neighbours
// their precision, ppl 25.6 -> 26.3); at 8 the damage is contained and ppl
// recovers, while the float work is still one scale per two dp4a.
#define AG 4

// One thread per AG-value group of a row-major activation block: absmax
// scale, AG/4 packed int32 words, and the group sum (the int4 path
// subtracts 8 * sum * d to fold its nibble offset away analytically).
extern "C" __global__ void quantize_act(int *xq, float *xs, int *xsum,
                                        const float *x, int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const float *xg = x + (size_t)g * AG;
    float amax = 0.0f;
    for (int j = 0; j < AG; ++j) amax = fmaxf(amax, fabsf(xg[j]));
    float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int sum = 0;
    for (int q = 0; q < AG / 4; ++q) {
        int packed = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(xg[4 * q + j] * id)));
            sum += v;
            packed |= (v & 0xFF) << (8 * j);
        }
        xq[(size_t)g * (AG / 4) + q] = packed;
    }
    xs[g] = amax > 0.0f ? amax / 127.0f : 1.0f;
    xsum[g] = sum;
}

// int8 GEMV via dp4a. The activation row is quantized inside the kernel
// (it already passes through shared memory); the dot then runs 2 dp4a per
// 8-value group instead of 8 convert+FMA pairs. Same wide/narrow split as
// before: wide outputs take 4 columns per thread via one int4 load.
extern "C" __global__ void gemv_int8(float *y, const float *x, const signed char *w,
                                     const float *scales, const float *b,
                                     int n_in, int n_out, int accum) {
    extern __shared__ char smem_raw[];
    int n_groups = n_in / AG;
    int *xq = (int *)smem_raw;              // n_in bytes as n_in/4 ints
    float *xs = (float *)(smem_raw + n_in); // n_groups scales
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        const float *xg = x + g * AG;
        float amax = 0.0f;
        for (int j = 0; j < AG; ++j) amax = fmaxf(amax, fabsf(xg[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int q = 0; q < AG / 4; ++q) {
            int packed = 0;
            for (int j = 0; j < 4; ++j) {
                int v = max(-127, min(127, __float2int_rn(xg[4 * q + j] * id)));
                packed |= (v & 0xFF) << (8 * j);
            }
            xq[g * (AG / 4) + q] = packed;
        }
        xs[g] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    if (n_out % 4 == 0 && n_out >= 4096) {
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int g = 0; g < n_groups; ++g) {
                int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
                for (int q = 0; q < AG / 4; ++q) {
                    int4 wv =
                        *(const int4 *)(w32 + (size_t)((AG / 4) * g + q) * n_out + 4 * o4);
                    int xv = xq[(AG / 4) * g + q];
                    i0 = __dp4a(wv.x, xv, i0);
                    i1 = __dp4a(wv.y, xv, i1);
                    i2 = __dp4a(wv.z, xv, i2);
                    i3 = __dp4a(wv.w, xv, i3);
                }
                float sx = xs[g];
                a0 += (float)i0 * sx;
                a1 += (float)i1 * sx;
                a2 += (float)i2 * sx;
                a3 += (float)i3 * sx;
            }
            int o = 4 * o4;
            float r0 = a0 * scales[o + 0] + (b ? b[o + 0] : 0.0f);
            float r1 = a1 * scales[o + 1] + (b ? b[o + 1] : 0.0f);
            float r2 = a2 * scales[o + 2] + (b ? b[o + 2] : 0.0f);
            float r3 = a3 * scales[o + 3] + (b ? b[o + 3] : 0.0f);
            y[o + 0] = accum ? y[o + 0] + r0 : r0;
            y[o + 1] = accum ? y[o + 1] + r1 : r1;
            y[o + 2] = accum ? y[o + 2] + r2 : r2;
            y[o + 3] = accum ? y[o + 3] + r3 : r3;
        }
        return;
    }
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < n_groups; ++g) {
            int ig = 0;
            for (int q = 0; q < AG / 4; ++q) {
                ig = __dp4a(w32[(size_t)((AG / 4) * g + q) * n_out + o],
                            xq[(AG / 4) * g + q], ig);
            }
            acc += (float)ig * xs[g];
        }
        float r = acc * scales[o] + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

// ---- int4 weights (Q4_0-style), dp4a math ----------------------------------
// Eight weights per int32 word packed along n_in: word (i/8)*n_out + o
// holds rows i..i+7 of column o, byte j carrying rows i+j (low nibble) and
// i+4+j (high nibble), nibbles store q+8 with q in [-8, 7]. That byte order
// lines both nibble planes up with the activation dp4a words:
// (w & 0x0F0F0F0F) pairs with x[i..i+3], (w >> 4 & ...) with x[i+4..i+7].
// One fp16 scale per (32-row group, column): scales[(i/32)*n_out + o].
//
// The GEMV keeps weights packed and folds the +8 nibble bias away
// analytically (8 * sum of dequantized activations per weight group); the
// GEMMs unpack nibbles to signed bytes once per shared tile (__vsubss4)
// and reuse the int8 micro-kernel shape.

#define Q4_GROUP 32

__device__ __forceinline__ int q4_lo8(int w) { return w & 0x0F0F0F0F; }
__device__ __forceinline__ int q4_hi8(int w) { return (w >> 4) & 0x0F0F0F0F; }

extern "C" __global__ void gemv_int4(float *y, const float *x, const unsigned char *w,
                                     const __half *scales, const float *b,
                                     int n_in, int n_out, int accum) {
    extern __shared__ char smem_raw[];
    int nq = n_in / 4;        // activation dp4a words
    int nw = n_in / Q4_GROUP; // 32-row weight groups
    int *xq = (int *)smem_raw;                   // nq words
    float *xs = (float *)(smem_raw + n_in);      // nq scales
    int *xsum = (int *)(smem_raw + 2 * n_in);    // nq group sums
    float *s32 = (float *)(smem_raw + 3 * n_in); // nw correction sums
    for (int g = threadIdx.x; g < nq; g += blockDim.x) {
        const float *xg = x + g * 4;
        float amax = 0.0f;
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(xg[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int packed = 0, sum = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(xg[j] * id)));
            sum += v;
            packed |= (v & 0xFF) << (8 * j);
        }
        xq[g] = packed;
        xs[g] = amax > 0.0f ? amax / 127.0f : 1.0f;
        xsum[g] = sum;
    }
    __syncthreads();
    for (int wg = threadIdx.x; wg < nw; wg += blockDim.x) {
        float s = 0.0f;
        for (int g = 8 * wg; g < 8 * wg + 8; ++g) s += xs[g] * (float)xsum[g];
        s32[wg] = s;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    if (n_out % 4 == 0 && n_out >= 4096) {
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int wg = 0; wg < nw; ++wg) {
                float i0 = 0.0f, i1 = 0.0f, i2 = 0.0f, i3 = 0.0f;
                for (int r = 0; r < 4; ++r) { // packed word = 8 rows
                    int wr = wg * 4 + r;
                    int4 wv = *(const int4 *)(w32 + (size_t)wr * n_out + 4 * o4);
                    int xa = xq[2 * wr], xb = xq[2 * wr + 1];
                    float sa = xs[2 * wr], sb = xs[2 * wr + 1];
                    i0 += sa * (float)__dp4a(q4_lo8(wv.x), xa, 0) +
                          sb * (float)__dp4a(q4_hi8(wv.x), xb, 0);
                    i1 += sa * (float)__dp4a(q4_lo8(wv.y), xa, 0) +
                          sb * (float)__dp4a(q4_hi8(wv.y), xb, 0);
                    i2 += sa * (float)__dp4a(q4_lo8(wv.z), xa, 0) +
                          sb * (float)__dp4a(q4_hi8(wv.z), xb, 0);
                    i3 += sa * (float)__dp4a(q4_lo8(wv.w), xa, 0) +
                          sb * (float)__dp4a(q4_hi8(wv.w), xb, 0);
                }
                float corr = 8.0f * s32[wg];
                const __half2 *d2 = (const __half2 *)(scales + (size_t)wg * n_out + 4 * o4);
                float2 da = __half22float2(d2[0]), db = __half22float2(d2[1]);
                a0 += (i0 - corr) * da.x;
                a1 += (i1 - corr) * da.y;
                a2 += (i2 - corr) * db.x;
                a3 += (i3 - corr) * db.y;
            }
            int o = 4 * o4;
            float r0 = a0 + (b ? b[o + 0] : 0.0f);
            float r1 = a1 + (b ? b[o + 1] : 0.0f);
            float r2 = a2 + (b ? b[o + 2] : 0.0f);
            float r3 = a3 + (b ? b[o + 3] : 0.0f);
            y[o + 0] = accum ? y[o + 0] + r0 : r0;
            y[o + 1] = accum ? y[o + 1] + r1 : r1;
            y[o + 2] = accum ? y[o + 2] + r2 : r2;
            y[o + 3] = accum ? y[o + 3] + r3 : r3;
        }
        return;
    }
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n_out;
         o += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int wg = 0; wg < nw; ++wg) {
            float inner = 0.0f;
            for (int r = 0; r < 4; ++r) {
                int wr = wg * 4 + r;
                int wv = w32[(size_t)wr * n_out + o];
                inner += xs[2 * wr] * (float)__dp4a(q4_lo8(wv), xq[2 * wr], 0) +
                         xs[2 * wr + 1] * (float)__dp4a(q4_hi8(wv), xq[2 * wr + 1], 0);
            }
            acc += (inner - 8.0f * s32[wg]) *
                   __half2float(scales[(size_t)wg * n_out + o]);
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

__device__ __forceinline__ float embed_int4_at(const unsigned char *wte_t,
                                               const __half *scales, int i, int tok,
                                               int n_vocab) {
    // word (i/8)*n_vocab + tok, byte i%4, low nibble for i%8 < 4
    unsigned char c = wte_t[((size_t)(i / 8) * n_vocab + tok) * 4 + (i % 4)];
    int q = (i & 4) ? (c >> 4) : (c & 15);
    return (float)(q - 8) * __half2float(scales[(size_t)(i / Q4_GROUP) * n_vocab + tok]);
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
    __shared__ float red[32];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i] * x[i];
    float inv = rsqrtf(block_sum(s, red) / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inv * g[i];
    }
}

// The batch norms keep the smem-tree reduction on purpose: in prefill they
// are a rounding error next to the GEMMs, and the tree preserves the exact
// summation order the batch==decode argmax gate was calibrated against
// (int4/kv8 logits sit close enough that reordering flips near-ties).
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
    __shared__ float red[32];
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *k = qkv + q_dim + h * head_dim;
    const float *v = qkv + q_dim + kv_dim + h * head_dim;

    float kmax = block_max(fabsf(k[d]), red);
    float kscale = kmax > 0.0f ? kmax / 127.0f : 1.0f;
    float vmax = block_max(fabsf(v[d]), red);
    float vscale = vmax > 0.0f ? vmax / 127.0f : 1.0f;

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
    __shared__ float red[32];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *row = qkv + (size_t)t * stride;
    const float *k = row + q_dim + h * head_dim;
    const float *v = row + q_dim + kv_dim + h * head_dim;
    int pos = pos0 + t;

    float kmax = block_max(fabsf(k[d]), red);
    float kscale = kmax > 0.0f ? kmax / 127.0f : 1.0f;
    float vmax = block_max(fabsf(v[d]), red);
    float vscale = vmax > 0.0f ? vmax / 127.0f : 1.0f;

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
    __shared__ float red[32];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int kvd = n_kv_head * head_dim;
    int kvh = h / (n_head / n_kv_head);
    float scale = rsqrtf((float)head_dim);

    // head rows are head_dim-float aligned, so float4 loads are safe and
    // cut the K/V load instruction count 4x (the char4 lesson, fp32 edition)
    int hd4 = head_dim / 4;
    const float4 *q4 = (const float4 *)(qkv + h * head_dim);
    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        const float4 *k4 = (const float4 *)(kcache + (size_t)t * kvd + kvh * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < hd4; ++d) {
            float4 qv = q4[d], kv = k4[d];
            dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        s[t] = dot * scale;
        m = fmaxf(m, s[t]);
    }
    m = block_max(m, red);

    float l = 0.0f;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        s[t] = __expf(s[t] - m);
        l += s[t];
    }
    float inv = 1.0f / block_sum(l, red);

    for (int d4 = tid; d4 < hd4; d4 += blockDim.x) {
        const float *vbase = vcache + kvh * head_dim + 4 * d4;
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        for (int t = 0; t <= t_cur; ++t) {
            float4 v = *(const float4 *)(vbase + (size_t)t * kvd);
            float st = s[t];
            ax += st * v.x, ay += st * v.y, az += st * v.z, aw += st * v.w;
        }
        *(float4 *)(out + h * head_dim + 4 * d4) =
            make_float4(ax * inv, ay * inv, az * inv, aw * inv);
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
    __shared__ float red[32];
    __shared__ int qq[32]; // q quantized to int8: head_dim/4 packed words
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int kvd = n_kv_head * head_dim;
    int kvh = h / (n_head / n_kv_head);
    const float *q = qkv + h * head_dim;
    float scale = rsqrtf((float)head_dim);

    // Quantize this head's q on the fly (one absmax scale per head): with K
    // already int8 the whole score dot runs on dp4a — 4 MACs per issue
    // instead of one convert+FMA per byte, the same cure the GEMVs got.
    int hd4 = head_dim / 4;
    float qmax = block_max(tid < head_dim ? fabsf(q[tid]) : 0.0f, red);
    float qid = qmax > 0.0f ? 127.0f / qmax : 0.0f;
    for (int d = tid; d < hd4; d += blockDim.x) {
        int packed = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(q[4 * d + j] * qid)));
            packed |= (v & 0xFF) << (8 * j);
        }
        qq[d] = packed;
    }
    __syncthreads();
    float qs = (qmax > 0.0f ? qmax / 127.0f : 0.0f) * scale;

    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        // head rows are head_dim-byte aligned: int4 vector loads, dp4a dot
        const int4 *k4 = (const int4 *)(kq + (size_t)t * kvd + kvh * head_dim);
        int dot = 0;
        for (int d = 0; d < hd4 / 4; ++d) {
            int4 kw = k4[d];
            dot = __dp4a(kw.x, qq[4 * d + 0], dot);
            dot = __dp4a(kw.y, qq[4 * d + 1], dot);
            dot = __dp4a(kw.z, qq[4 * d + 2], dot);
            dot = __dp4a(kw.w, qq[4 * d + 3], dot);
        }
        s[t] = (float)dot * ks[t * n_kv_head + kvh] * qs;
        m = fmaxf(m, s[t]);
    }
    m = block_max(m, red);

    float l = 0.0f;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        s[t] = __expf(s[t] - m);
        l += s[t];
    }
    float inv = 1.0f / block_sum(l, red);

    for (int d4 = tid; d4 < head_dim / 4; d4 += blockDim.x) {
        const signed char *vbase = vq + kvh * head_dim + 4 * d4;
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        for (int t = 0; t <= t_cur; ++t) {
            char4 v = *(const char4 *)(vbase + (size_t)t * kvd);
            float st = s[t] * vs[t * n_kv_head + kvh];
            ax += st * (float)v.x, ay += st * (float)v.y;
            az += st * (float)v.z, aw += st * (float)v.w;
        }
        *(float4 *)(out + h * head_dim + 4 * d4) =
            make_float4(ax * inv, ay * inv, az * inv, aw * inv);
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

// Wide tier for real prefill (M > 64): the 01-gemm ladder's endgame
// (gemm_05/06) grafted onto engine shapes. 128x128 tile, BK = 8, 256
// threads each owning an 8x8 register micro-tile — 64 FMAs per 16 smem
// reads vs the 64-tile's 16 per 8, so the kernel stops being smem-issue
// bound. A is staged with float4 loads and stored transposed so the inner
// loop reads both tiles contiguously; smem is double-buffered with the
// next tile's global loads issued before the compute loop (one
// __syncthreads per k-step, latency hidden behind FMAs). M/N edges are
// bounds-checked: OOB rows load zeros, the N edge falls back to scalar
// loads and the epilogue clips.
template <typename BT>
__device__ void gemm_wide_body(float *C, const float *A, const BT *B,
                               const float *bias, int M, int N, int K) {
    __shared__ float As[2][8][128]; // transposed: As[buf][k][m]
    __shared__ float Bs[2][8][128];
    int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    int tid = threadIdx.x;
    int arow = tid >> 1, acol = (tid & 1) * 4;  // A tile: one float4 each
    int brow = tid >> 5, bcol = (tid & 31) * 4; // B tile: one 4-vec each
    int trow = tid >> 4, tcol = tid & 15;
    bool vec = (N % 4 == 0) && (bn + 128 <= N);

    float acc[8][8] = {};
    float a4[4], b4[4];
    // K % 16 == 0 engine-wide, so the float4 A load never crosses K.
    auto stage = [&](int k0) {
        if (bm + arow < M) {
            load4(&A[(size_t)(bm + arow) * K + k0 + acol], a4);
        } else {
            a4[0] = a4[1] = a4[2] = a4[3] = 0.0f;
        }
        if (vec) {
            load4(&B[(size_t)(k0 + brow) * N + bn + bcol], b4);
        } else {
            for (int j = 0; j < 4; ++j)
                b4[j] = (bn + bcol + j < N)
                            ? tof(B[(size_t)(k0 + brow) * N + bn + bcol + j])
                            : 0.0f;
        }
    };
    auto store = [&](int buf) {
        As[buf][acol + 0][arow] = a4[0];
        As[buf][acol + 1][arow] = a4[1];
        As[buf][acol + 2][arow] = a4[2];
        As[buf][acol + 3][arow] = a4[3];
        *reinterpret_cast<float4 *>(&Bs[buf][brow][bcol]) =
            make_float4(b4[0], b4[1], b4[2], b4[3]);
    };

    stage(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += 8) {
        if (k0 + 8 < K) stage(k0 + 8);
        for (int k = 0; k < 8; ++k) {
            float rm[8], rn[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) rm[i] = As[buf][k][trow * 8 + i];
#pragma unroll
            for (int j = 0; j < 8; ++j) rn[j] = Bs[buf][k][tcol * 8 + j];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += rm[i] * rn[j];
        }
        if (k0 + 8 < K) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }

    for (int i = 0; i < 8; ++i) {
        int row = bm + trow * 8 + i;
        if (row >= M) continue;
        for (int j = 0; j < 8; ++j) {
            int col = bn + tcol * 8 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = acc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_f32_wide(float *C, const float *A, const float *B,
                                         const float *bias, int M, int N, int K) {
    gemm_wide_body<float>(C, A, B, bias, M, N, K);
}

extern "C" __global__ void gemm_half_wide(float *C, const float *A, const __half *B,
                                          const float *bias, int M, int N, int K) {
    gemm_wide_body<__half>(C, A, B, bias, M, N, K);
}

// int8 GEMM via dp4a: A arrives pre-quantized by quantize_act (packed int32
// + one scale per 32-k group per row), B is the repacked int32 weight
// layout. BK = 32 so each k-tile covers exactly one activation-scale group:
// the micro-kernel accumulates in int (8 dp4a per micro-tile element) and
// pays one float multiply per tile, per-column weight scales fold into the
// epilogue as before.
template <int BM>
__device__ void gemm_i8_body(float *C, const int *Aq, const float *ascale,
                             const int *B32, const float *wscale, const float *bias,
                             int M, int N, int K) {
    constexpr int RM = BM / 16;
    __shared__ int As[8][BM];
    __shared__ int Bs[8][64];
    __shared__ float Ss[32 / AG][BM]; // per (activation group, row) scale
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    int kq = K / 4, kg = K / AG;
    float facc[RM][4] = {};

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int i = tid; i < BM * 8; i += 256) {
            int m = i / 8, q = i % 8;
            As[q][m] = (bm + m < M) ? Aq[(size_t)(bm + m) * kq + k0 / 4 + q] : 0;
        }
        for (int i = tid; i < 8 * 64; i += 256) {
            int q = i / 64, n = i % 64;
            Bs[q][n] = (bn + n < N) ? B32[(size_t)(k0 / 4 + q) * N + bn + n] : 0;
        }
        for (int i = tid; i < (32 / AG) * BM; i += 256) {
            int gg = i / BM, m = i % BM;
            Ss[gg][m] =
                (bm + m < M) ? ascale[(size_t)(bm + m) * kg + k0 / AG + gg] : 0.0f;
        }
        __syncthreads();
        for (int gg = 0; gg < 32 / AG; ++gg) {
            int iacc[RM][4] = {};
            for (int q = (AG / 4) * gg; q < (AG / 4) * (gg + 1); ++q) {
                int a[RM], b[4];
                for (int i = 0; i < RM; ++i) a[i] = As[q][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) b[j] = Bs[q][threadIdx.x * 4 + j];
                for (int i = 0; i < RM; ++i)
                    for (int j = 0; j < 4; ++j) iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }
            for (int i = 0; i < RM; ++i) {
                float sa = Ss[gg][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) facc[i][j] += (float)iacc[i][j] * sa;
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < RM; ++i) {
        int row = bm + threadIdx.y * RM + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + threadIdx.x * 4 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = facc[i][j] * wscale[col] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int8(float *C, const int *Aq, const float *ascale,
                                     const int *B32, const float *wscale,
                                     const float *bias, int M, int N, int K) {
    gemm_i8_body<64>(C, Aq, ascale, B32, wscale, bias, M, N, K);
}

extern "C" __global__ void gemm_f32_skinny(float *C, const float *A, const float *B,
                                           const float *bias, int M, int N, int K) {
    gemm_body<float, false, 16>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_half_skinny(float *C, const float *A, const __half *B,
                                            const float *bias, int M, int N, int K) {
    gemm_body<__half, false, 16>(C, A, B, nullptr, bias, M, N, K);
}

extern "C" __global__ void gemm_int8_skinny(float *C, const int *Aq, const float *ascale,
                                            const int *B32, const float *wscale,
                                            const float *bias, int M, int N, int K) {
    gemm_i8_body<16>(C, Aq, ascale, B32, wscale, bias, M, N, K);
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

// Draft-verify int8 GEMM (M <= 8) via dp4a: A pre-quantized like the tiled
// version; per 8-value group the 8x4 accumulator runs in int (2 dp4a), then
// one float multiply per row by that group's activation scale. Wide columns
// take one int4 load = 4 columns x 4 k-rows per dp4a quad.
extern "C" __global__ void gemm_rows_int8(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const float *wscale,
                                          const float *bias, int M, int N, int K) {
    __shared__ int As[ROWS_M][ROWS_KT / 4];
    __shared__ float Ss[ROWS_M][ROWS_KT / AG];
    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;
    bool wide = gemm_rows_wide(N);
    bool active = o < (wide ? N / 4 : N);
    int kq = K / 4, kg = K / AG;
    float facc[ROWS_M][4] = {};

    for (int k0 = 0; k0 < K; k0 += ROWS_KT) {
        int kt = min(ROWS_KT, K - k0); // K % 32 == 0, so kt is too
        for (int i = tid; i < ROWS_M * ROWS_KT / 4; i += blockDim.x) {
            int m = i / (ROWS_KT / 4), q = i % (ROWS_KT / 4);
            As[m][q] = (m < M && 4 * q < kt) ? Aq[(size_t)m * kq + k0 / 4 + q] : 0;
        }
        for (int i = tid; i < ROWS_M * ROWS_KT / AG; i += blockDim.x) {
            int m = i / (ROWS_KT / AG), g = i % (ROWS_KT / AG);
            Ss[m][g] = (m < M && AG * g < kt) ? ascale[(size_t)m * kg + k0 / AG + g] : 0.0f;
        }
        __syncthreads();
        if (active && wide) {
            for (int g = 0; g < kt / AG; ++g) {
                int iacc[ROWS_M][4] = {};
                for (int q = (AG / 4) * g; q < (AG / 4) * (g + 1); ++q) {
                    int4 wv = *(const int4 *)(B32 + (size_t)(k0 / 4 + q) * N + 4 * o);
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        int a = As[m][q];
                        iacc[m][0] = __dp4a(a, wv.x, iacc[m][0]);
                        iacc[m][1] = __dp4a(a, wv.y, iacc[m][1]);
                        iacc[m][2] = __dp4a(a, wv.z, iacc[m][2]);
                        iacc[m][3] = __dp4a(a, wv.w, iacc[m][3]);
                    }
                }
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) {
                    float sa = Ss[m][g];
                    facc[m][0] += (float)iacc[m][0] * sa;
                    facc[m][1] += (float)iacc[m][1] * sa;
                    facc[m][2] += (float)iacc[m][2] * sa;
                    facc[m][3] += (float)iacc[m][3] * sa;
                }
            }
        } else if (active) {
            for (int g = 0; g < kt / AG; ++g) {
                int iacc[ROWS_M] = {};
                for (int q = (AG / 4) * g; q < (AG / 4) * (g + 1); ++q) {
                    int wv = B32[(size_t)(k0 / 4 + q) * N + o];
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m)
                        iacc[m] = __dp4a(As[m][q], wv, iacc[m]);
                }
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) facc[m][0] += (float)iacc[m] * Ss[m][g];
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int ncols = wide ? 4 : 1;
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < ncols; ++j) {
            int col = wide ? 4 * o + j : o;
            C[(size_t)m * N + col] = facc[m][j] * wscale[col] + bias[col];
        }
    }
}

// int4 GEMM via dp4a: same shape as gemm_i8_body (A pre-quantized by
// quantize_act), but the B tile unpacks packed nibble words into signed
// int8 dp4a words during the shared-tile fill (__vsubss4 folds the +8
// nibble bias away, so no correction terms anywhere), and the per-tile
// accumulator is scaled by that 32-row group's fp16 weight scale at the
// tile boundary instead of a per-column scale in the epilogue.
template <int BM>
__device__ void gemm_i4_body(float *C, const int *Aq, const float *ascale,
                             const int *B32, const __half *wscale, const float *bias,
                             int M, int N, int K) {
    constexpr int RM = BM / 16;
    __shared__ int As[8][BM];
    __shared__ int Bs[8][64];
    __shared__ float Ss[32 / AG][BM]; // per (activation group, row) scale
    __shared__ float Sw[64];          // per-column weight scale of this tile
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    int kq = K / 4, kg = K / AG, kw = K / 8; // packed-word rows of B
    float facc[RM][4] = {};

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int i = tid; i < BM * 8; i += 256) {
            int m = i / 8, q = i % 8;
            As[q][m] = (bm + m < M) ? Aq[(size_t)(bm + m) * kq + k0 / 4 + q] : 0;
        }
        for (int i = tid; i < 4 * 64; i += 256) {
            int wr = i / 64, n = i % 64; // packed word wr covers k rows 8wr..8wr+7
            int wv = (bn + n < N && k0 / 8 + wr < kw)
                         ? B32[(size_t)(k0 / 8 + wr) * N + bn + n]
                         : 0x88888888; // nibbles of 8 unpack to 0
            Bs[2 * wr][n] = __vsubss4(q4_lo8(wv), 0x08080808);
            Bs[2 * wr + 1][n] = __vsubss4(q4_hi8(wv), 0x08080808);
        }
        for (int i = tid; i < (32 / AG) * BM; i += 256) {
            int gg = i / BM, m = i % BM;
            Ss[gg][m] =
                (bm + m < M) ? ascale[(size_t)(bm + m) * kg + k0 / AG + gg] : 0.0f;
        }
        if (tid < 64) {
            Sw[tid] = (bn + tid < N)
                          ? __half2float(wscale[(size_t)(k0 / 32) * N + bn + tid])
                          : 0.0f;
        }
        __syncthreads();
        float tacc[RM][4] = {};
        for (int gg = 0; gg < 32 / AG; ++gg) {
            int iacc[RM][4] = {};
            for (int q = (AG / 4) * gg; q < (AG / 4) * (gg + 1); ++q) {
                int a[RM], b[4];
                for (int i = 0; i < RM; ++i) a[i] = As[q][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) b[j] = Bs[q][threadIdx.x * 4 + j];
                for (int i = 0; i < RM; ++i)
                    for (int j = 0; j < 4; ++j) iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }
            for (int i = 0; i < RM; ++i) {
                float sa = Ss[gg][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) tacc[i][j] += (float)iacc[i][j] * sa;
            }
        }
        for (int i = 0; i < RM; ++i)
            for (int j = 0; j < 4; ++j)
                facc[i][j] += tacc[i][j] * Sw[threadIdx.x * 4 + j];
        __syncthreads();
    }
    for (int i = 0; i < RM; ++i) {
        int row = bm + threadIdx.y * RM + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + threadIdx.x * 4 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int4(float *C, const int *Aq, const float *ascale,
                                     const int *B32, const __half *wscale,
                                     const float *bias, int M, int N, int K) {
    gemm_i4_body<64>(C, Aq, ascale, B32, wscale, bias, M, N, K);
}

extern "C" __global__ void gemm_int4_skinny(float *C, const int *Aq, const float *ascale,
                                            const int *B32, const __half *wscale,
                                            const float *bias, int M, int N, int K) {
    gemm_i4_body<16>(C, Aq, ascale, B32, wscale, bias, M, N, K);
}

// int4 draft-verify GEMM (M <= 8) via dp4a: B words unpack in-register
// (__vsubss4, no bias correction); the float accumulator is built per
// 32-row weight group, then scaled by that group's fp16 weight scale.
extern "C" __global__ void gemm_rows_int4(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const __half *wscale,
                                          const float *bias, int M, int N, int K) {
    __shared__ int As[ROWS_M][ROWS_KT / 4];
    __shared__ float Ss[ROWS_M][ROWS_KT / AG];
    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;
    bool wide = gemm_rows_wide(N);
    bool active = o < (wide ? N / 4 : N);
    int kq = K / 4, kg = K / AG;
    float facc[ROWS_M][4] = {};

    for (int k0 = 0; k0 < K; k0 += ROWS_KT) {
        int kt = min(ROWS_KT, K - k0); // K % 32 == 0, so kt is too
        for (int i = tid; i < ROWS_M * ROWS_KT / 4; i += blockDim.x) {
            int m = i / (ROWS_KT / 4), q = i % (ROWS_KT / 4);
            As[m][q] = (m < M && 4 * q < kt) ? Aq[(size_t)m * kq + k0 / 4 + q] : 0;
        }
        for (int i = tid; i < ROWS_M * ROWS_KT / AG; i += blockDim.x) {
            int m = i / (ROWS_KT / AG), g = i % (ROWS_KT / AG);
            Ss[m][g] = (m < M && AG * g < kt) ? ascale[(size_t)m * kg + k0 / AG + g] : 0.0f;
        }
        __syncthreads();
        if (active && wide) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                float gacc[ROWS_M][4] = {};
                for (int r = 0; r < 4; ++r) { // 4 packed words per weight group
                    int wr = (k0 + wg * 32) / 8 + r;
                    int4 wv = *(const int4 *)(B32 + (size_t)wr * N + 4 * o);
                    int blo[4] = {__vsubss4(q4_lo8(wv.x), 0x08080808),
                                  __vsubss4(q4_lo8(wv.y), 0x08080808),
                                  __vsubss4(q4_lo8(wv.z), 0x08080808),
                                  __vsubss4(q4_lo8(wv.w), 0x08080808)};
                    int bhi[4] = {__vsubss4(q4_hi8(wv.x), 0x08080808),
                                  __vsubss4(q4_hi8(wv.y), 0x08080808),
                                  __vsubss4(q4_hi8(wv.z), 0x08080808),
                                  __vsubss4(q4_hi8(wv.w), 0x08080808)};
                    int qa = (wg * 32) / 4 + 2 * r, qb = qa + 1;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        int a0 = As[m][qa], a1 = As[m][qb];
                        float s0 = Ss[m][qa], s1 = Ss[m][qb];
                        gacc[m][0] += s0 * (float)__dp4a(blo[0], a0, 0) +
                                      s1 * (float)__dp4a(bhi[0], a1, 0);
                        gacc[m][1] += s0 * (float)__dp4a(blo[1], a0, 0) +
                                      s1 * (float)__dp4a(bhi[1], a1, 0);
                        gacc[m][2] += s0 * (float)__dp4a(blo[2], a0, 0) +
                                      s1 * (float)__dp4a(bhi[2], a1, 0);
                        gacc[m][3] += s0 * (float)__dp4a(blo[3], a0, 0) +
                                      s1 * (float)__dp4a(bhi[3], a1, 0);
                    }
                }
                int gw = (k0 + wg * 32) / Q4_GROUP;
                const __half2 *d2 = (const __half2 *)(wscale + (size_t)gw * N + 4 * o);
                float2 da = __half22float2(d2[0]), db = __half22float2(d2[1]);
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) {
                    facc[m][0] += gacc[m][0] * da.x;
                    facc[m][1] += gacc[m][1] * da.y;
                    facc[m][2] += gacc[m][2] * db.x;
                    facc[m][3] += gacc[m][3] * db.y;
                }
            }
        } else if (active) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                float gacc[ROWS_M] = {};
                for (int r = 0; r < 4; ++r) {
                    int wr = (k0 + wg * 32) / 8 + r;
                    int wv = B32[(size_t)wr * N + o];
                    int blo = __vsubss4(q4_lo8(wv), 0x08080808);
                    int bhi = __vsubss4(q4_hi8(wv), 0x08080808);
                    int qa = (wg * 32) / 4 + 2 * r, qb = qa + 1;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        gacc[m] += Ss[m][qa] * (float)__dp4a(blo, As[m][qa], 0) +
                                   Ss[m][qb] * (float)__dp4a(bhi, As[m][qb], 0);
                    }
                }
                int gw = (k0 + wg * 32) / Q4_GROUP;
                float d = __half2float(wscale[(size_t)gw * N + o]);
#pragma unroll
                for (int m = 0; m < ROWS_M; ++m) facc[m][0] += gacc[m] * d;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int ncols = wide ? 4 : 1;
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < ncols; ++j) {
            int col = wide ? 4 * o + j : o;
            C[(size_t)m * N + col] = facc[m][j] + bias[col];
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
        out[idx] = embed_int8_at(wte_t, scales, i, toks[t], n_vocab) +
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

// Block-wide (value, index) argmax with ties to the lowest index: one warp
// shuffle sweep, a 32-slot bounce, one more sweep in warp 0. Result is valid
// in thread 0 only.
__device__ __forceinline__ void block_argmax(float &best, int &best_i,
                                             float *vals, int *idxs) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFFu, best, o);
        int oi = __shfl_down_sync(0xFFFFFFFFu, best_i, o);
        if (ov > best || (ov == best && oi < best_i)) {
            best = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        vals[w] = best;
        idxs[w] = best_i;
    }
    __syncthreads();
    if (w == 0) {
        best = (lane < nw) ? vals[lane] : -CUDART_INF_F;
        best_i = (lane < nw) ? idxs[lane] : 0x7FFFFFFF;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xFFFFFFFFu, best, o);
            int oi = __shfl_down_sync(0xFFFFFFFFu, best_i, o);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
    }
}

// Per-row greedy argmax for the speculative verify step (one block per row).
extern "C" __global__ void argmax_rows(int *out, const float *logits, int n_vocab) {
    __shared__ float vals[32];
    __shared__ int idxs[32];
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
    block_argmax(best, best_i, vals, idxs);
    if (tid == 0) out[blockIdx.x] = best_i;
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
    __shared__ float vals[32];
    __shared__ int idxs[32];
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
    block_argmax(best, best_i, vals, idxs);
    if (tid == 0) {
        *tok_ptr = best_i;
        *pos_ptr += 1;
    }
}
