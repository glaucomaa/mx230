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
                                     const float *b, int n, float eps) {
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
    float inv = rsqrtf(block_sum(s, red) / n + eps);

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

// Activation-group width is a compile-time knob (build.rs compiles this
// file twice): GPT-2's activation outliers wreck wide absmax groups (a
// 32-wide group costs ppl 25.6 -> 26.3), so it runs AG=4 — one scale per
// dp4a word, zero ppl cost. The RoPE models have no such outliers and run
// the AG=8 variant, halving the scale-FMAs in the dp4a GEMMs.
#ifndef AG
#define AG 4
#endif

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

// ---- int4 weights (--int4k): k-quants two-level scales, dp4a math ----------
// Eight weights per int32 word packed along n_in: word (i/8)*n_out + o
// holds rows i..i+7 of column o, byte j carrying rows i+j (low nibble) and
// i+4+j (high nibble). That byte order lines both nibble planes up with the
// activation dp4a words: (w & 0x0F0F0F0F) pairs with x[i..i+3], (w >> 4 & ...)
// with x[i+4..i+7]. Values store unsigned q in [0, 15] with the same two-level
// w = d*q - m scheme as int3/int2: d = d_super * sd, m = m_super * sm, where
// sub[(i/16)*n_out + o] packs sd (lo nibble) and sm (hi nibble) per 16-row
// sub-block (= two packed words), and dm[(i/128)*n_out + o] is the
// (d_super, m_super) half2 of the 128-row super-block. The -m term folds
// analytically: m * (sum of dequantized activations over the sub-block).

// 32-k tile granularity shared by the draft-verify (rows) GEMMs of every
// int tier; the two-level weight scales live on 16-row sub-blocks within it.
#define Q4_GROUP 32

__device__ __forceinline__ int q4_lo8(int w) { return w & 0x0F0F0F0F; }
__device__ __forceinline__ int q4_hi8(int w) { return (w >> 4) & 0x0F0F0F0F; }

extern "C" __global__ void gemv_int4k(float *y, const float *x, const unsigned char *w,
                                     const unsigned char *sub, const __half2 *dm,
                                     const float *b, int n_in, int n_out, int accum) {
    extern __shared__ char smem_raw[];
    int nq = n_in / 4;   // activation dp4a words
    int ns = n_in / 16;  // 16-row sub-blocks (= two packed words)
    int *xq = (int *)smem_raw;                   // nq words
    float *xs = (float *)(smem_raw + n_in);      // nq scales
    int *xsum = (int *)(smem_raw + 2 * n_in);    // nq group sums
    float *s16 = (float *)(smem_raw + 3 * n_in); // ns correction sums
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
    for (int wr = threadIdx.x; wr < ns; wr += blockDim.x) {
        float s = 0.0f;
        for (int g = 4 * wr; g < 4 * wr + 4; ++g) s += xs[g] * (float)xsum[g];
        s16[wr] = s;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    if (n_out % 4 == 0 && n_out >= 4096) {
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int s = 0; s < n_in / 128; ++s) {
                int4 dmi = *(const int4 *)(dm + (size_t)s * n_out + 4 * o4);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
                for (int sb = 0; sb < 8; ++sb) { // 8 sub-blocks per super
                    int wri = 8 * s + sb; // sub-block index (16 rows)
                    float i0 = 0.0f, i1 = 0.0f, i2 = 0.0f, i3 = 0.0f;
                    for (int pw = 0; pw < 2; ++pw) { // packed word = 8 rows
                        int wr = 2 * wri + pw;
                        int4 wv = *(const int4 *)(w32 + (size_t)wr * n_out + 4 * o4);
                        int xa = xq[2 * wr], xbw = xq[2 * wr + 1];
                        float sa = xs[2 * wr], sbw = xs[2 * wr + 1];
                        i0 += sa * (float)__dp4a(q4_lo8(wv.x), xa, 0) +
                              sbw * (float)__dp4a(q4_hi8(wv.x), xbw, 0);
                        i1 += sa * (float)__dp4a(q4_lo8(wv.y), xa, 0) +
                              sbw * (float)__dp4a(q4_hi8(wv.y), xbw, 0);
                        i2 += sa * (float)__dp4a(q4_lo8(wv.z), xa, 0) +
                              sbw * (float)__dp4a(q4_hi8(wv.z), xbw, 0);
                        i3 += sa * (float)__dp4a(q4_lo8(wv.w), xa, 0) +
                              sbw * (float)__dp4a(q4_hi8(wv.w), xbw, 0);
                    }
                    uchar4 sbq = *(const uchar4 *)(sub + (size_t)wri * n_out + 4 * o4);
                    float sx = s16[wri];
                    a0 += dm0.x * (sbq.x & 15) * i0 - dm0.y * (sbq.x >> 4) * sx;
                    a1 += dm1.x * (sbq.y & 15) * i1 - dm1.y * (sbq.y >> 4) * sx;
                    a2 += dm2.x * (sbq.z & 15) * i2 - dm2.y * (sbq.z >> 4) * sx;
                    a3 += dm3.x * (sbq.w & 15) * i3 - dm3.y * (sbq.w >> 4) * sx;
                }
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
        for (int s = 0; s < n_in / 128; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * n_out + o]);
            for (int sb = 0; sb < 8; ++sb) {
                int wri = 8 * s + sb;
                float inner = 0.0f;
                for (int pw = 0; pw < 2; ++pw) {
                    int wr = 2 * wri + pw;
                    int wv = w32[(size_t)wr * n_out + o];
                    inner += xs[2 * wr] * (float)__dp4a(q4_lo8(wv), xq[2 * wr], 0) +
                             xs[2 * wr + 1] * (float)__dp4a(q4_hi8(wv), xq[2 * wr + 1], 0);
                }
                unsigned char sbq = sub[(size_t)wri * n_out + o];
                acc += dmv.x * (sbq & 15) * inner - dmv.y * (sbq >> 4) * s16[wri];
            }
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

__device__ __forceinline__ float embed_int4k_at(const unsigned char *wte_t,
                                               const unsigned char *sub,
                                               const __half2 *dm, int i, int tok,
                                               int n_vocab) {
    // word (i/8)*n_vocab + tok, byte i%4, low nibble for i%8 < 4
    unsigned char c = wte_t[((size_t)(i / 8) * n_vocab + tok) * 4 + (i % 4)];
    int q = (i & 4) ? (c >> 4) : (c & 15);
    unsigned char sb = sub[(size_t)(i / 16) * n_vocab + tok];
    float2 dmv = __half22float2(dm[(size_t)(i / 128) * n_vocab + tok]);
    return dmv.x * (sb & 15) * q - dmv.y * (sb >> 4);
}

extern "C" __global__ void embed_int4k(float *out, const unsigned char *wte_t,
                                      const unsigned char *sub, const __half2 *dm,
                                      const float *wpe, int tok, int pos,
                                      int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int4k_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int4k_dyn(float *out, const unsigned char *wte_t,
                                          const unsigned char *sub, const __half2 *dm,
                                          const float *wpe, const int *tok_ptr,
                                          const int *pos_ptr, int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int4k_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int4k_batch(float *out, const unsigned char *wte_t,
                                            const unsigned char *sub, const __half2 *dm,
                                            const float *wpe, const int *toks, int pos0,
                                            int n_tok, int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = embed_int4k_at(wte_t, sub, dm, i, toks[t], n_vocab) +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

// ---- int4 weights (--int4, Q4_0-style, fast path), dp4a math --------------
// `perm` (or null): GPTQ act-order. When non-null the activation is gathered as
// x[perm[i]] before quantizing, so it lines up with weights stored in the same
// descending-Hessian channel order (scales stay per contiguous 32-group). The
// dot is permutation-invariant, so the result equals the unpermuted GEMV.
extern "C" __global__ void gemv_int4(float *y, const float *x, const unsigned char *w,
                                     const __half *scales, const float *b,
                                     int n_in, int n_out, int accum, const int *perm) {
    extern __shared__ char smem_raw[];
    int nq = n_in / 4;        // activation dp4a words
    int nw = n_in / Q4_GROUP; // 32-row weight groups
    int *xq = (int *)smem_raw;                   // nq words
    float *xs = (float *)(smem_raw + n_in);      // nq scales
    int *xsum = (int *)(smem_raw + 2 * n_in);    // nq group sums
    float *s32 = (float *)(smem_raw + 3 * n_in); // nw correction sums
    for (int g = threadIdx.x; g < nq; g += blockDim.x) {
        float xv[4];
        for (int j = 0; j < 4; ++j) xv[j] = perm ? x[perm[g * 4 + j]] : x[g * 4 + j];
        float amax = 0.0f;
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(xv[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int packed = 0, sum = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(xv[j] * id)));
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

// ---- int2 weights, k-quants two-level scales, dp4a math --------------------
// 16 weights per int32 word along n_in: word (i/16)*n_out + o holds rows
// i..i+15 of column o; byte j carries rows i+j, i+4+j, i+8+j, i+12+j in bit
// pairs (1:0), (3:2), (5:4), (7:6). Plane p = (w >> 2p) & 0x03030303 lines
// up with activation dp4a word p of the span. Values store unsigned q in
// [0, 3] with w = d*q - m per 16-row sub-block (= one packed word):
// d = d_super * sd, m = m_super * sm, where sub[(i/16)*n_out + o] packs
// sd (lo nibble) and sm (hi nibble), and dm[(i/128)*n_out + o] is the
// (d_super, m_super) half2 of the 128-row super-block. The -m term folds
// analytically: m * (sum of dequantized activations over the sub-block).
__device__ __forceinline__ int q2_plane(int w, int p) {
    return (w >> (2 * p)) & 0x03030303;
}

extern "C" __global__ void gemv_int2(float *y, const float *x, const unsigned char *w,
                                     const unsigned char *sub, const __half2 *dm,
                                     const float *b, int n_in, int n_out, int accum) {
    extern __shared__ char smem_raw[];
    int nq = n_in / 4;    // activation dp4a words
    int ns = n_in / 16;   // packed words = 16-row sub-blocks
    int nsup = n_in / 128;
    int *xq = (int *)smem_raw;                   // nq words
    float *xs = (float *)(smem_raw + n_in);      // nq scales
    int *xsum = (int *)(smem_raw + 2 * n_in);    // nq group sums
    float *s16 = (float *)(smem_raw + 3 * n_in); // ns correction sums
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
    for (int wr = threadIdx.x; wr < ns; wr += blockDim.x) {
        float s = 0.0f;
        for (int g = 4 * wr; g < 4 * wr + 4; ++g) s += xs[g] * (float)xsum[g];
        s16[wr] = s;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    if (n_out % 4 == 0 && n_out >= 4096) {
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int s = 0; s < nsup; ++s) {
                int4 dmi = *(const int4 *)(dm + (size_t)s * n_out + 4 * o4);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
                for (int r = 0; r < 8; ++r) { // packed word = 16 rows
                    int wr = 8 * s + r;
                    int4 wv = *(const int4 *)(w32 + (size_t)wr * n_out + 4 * o4);
                    float i0 = 0.0f, i1 = 0.0f, i2 = 0.0f, i3 = 0.0f;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int xa = xq[4 * wr + p];
                        float sa = xs[4 * wr + p];
                        i0 += sa * (float)__dp4a(q2_plane(wv.x, p), xa, 0);
                        i1 += sa * (float)__dp4a(q2_plane(wv.y, p), xa, 0);
                        i2 += sa * (float)__dp4a(q2_plane(wv.z, p), xa, 0);
                        i3 += sa * (float)__dp4a(q2_plane(wv.w, p), xa, 0);
                    }
                    uchar4 sb = *(const uchar4 *)(sub + (size_t)wr * n_out + 4 * o4);
                    float sx = s16[wr];
                    a0 += dm0.x * (sb.x & 15) * i0 - dm0.y * (sb.x >> 4) * sx;
                    a1 += dm1.x * (sb.y & 15) * i1 - dm1.y * (sb.y >> 4) * sx;
                    a2 += dm2.x * (sb.z & 15) * i2 - dm2.y * (sb.z >> 4) * sx;
                    a3 += dm3.x * (sb.w & 15) * i3 - dm3.y * (sb.w >> 4) * sx;
                }
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
        for (int s = 0; s < nsup; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * n_out + o]);
            for (int r = 0; r < 8; ++r) {
                int wr = 8 * s + r;
                int wv = w32[(size_t)wr * n_out + o];
                float inner = 0.0f;
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    inner += xs[4 * wr + p] *
                             (float)__dp4a(q2_plane(wv, p), xq[4 * wr + p], 0);
                unsigned char sb = sub[(size_t)wr * n_out + o];
                acc += dmv.x * (sb & 15) * inner - dmv.y * (sb >> 4) * s16[wr];
            }
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

__device__ __forceinline__ float embed_int2_at(const unsigned char *wte_t,
                                               const unsigned char *sub,
                                               const __half2 *dm, int i, int tok,
                                               int n_vocab) {
    // word (i/16)*n_vocab + tok, byte i%4, bit pair (i%16)/4
    unsigned char c = wte_t[((size_t)(i / 16) * n_vocab + tok) * 4 + (i % 4)];
    int q = (c >> (2 * ((i % 16) / 4))) & 3;
    unsigned char sb = sub[(size_t)(i / 16) * n_vocab + tok];
    float2 dmv = __half22float2(dm[(size_t)(i / 128) * n_vocab + tok]);
    return dmv.x * (sb & 15) * q - dmv.y * (sb >> 4);
}

extern "C" __global__ void embed_int2(float *out, const unsigned char *wte_t,
                                      const unsigned char *sub, const __half2 *dm,
                                      const float *wpe, int tok, int pos,
                                      int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int2_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int2_dyn(float *out, const unsigned char *wte_t,
                                          const unsigned char *sub, const __half2 *dm,
                                          const float *wpe, const int *tok_ptr,
                                          const int *pos_ptr, int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int2_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int2_batch(float *out, const unsigned char *wte_t,
                                            const unsigned char *sub, const __half2 *dm,
                                            const float *wpe, const int *toks, int pos0,
                                            int n_tok, int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = embed_int2_at(wte_t, sub, dm, i, toks[t], n_vocab) +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

// ---- int3 weights, k-quants two-level scales, dp4a math --------------------
// Three int32 words per (32-row group, column), word-rows interleaved as
// (i/32)*3 + j: j = 0/1 are int2-style lo planes (bit pairs of the stored
// value's low 2 bits, 16 rows each), j = 2 is the hi word — byte k, bit b
// carries the high bit of row i + 4b + k, so plane (r, p) extracts as
// (whi >> (4r + p)) & 0x01010101. Values store unsigned q in [0, 7]
// (plane = lo2 | hi << 2) with the same two-level w = d*q - m scheme as
// int2: each lo word is one 16-row sub-block, sub/dm laid out identically.
__device__ __forceinline__ int q3_plane(int wlo, int whi, int r, int p) {
    return q2_plane(wlo, p) | (((whi >> (4 * r + p)) & 0x01010101) << 2);
}

extern "C" __global__ void gemv_int3(float *y, const float *x, const unsigned char *w,
                                     const unsigned char *sub, const __half2 *dm,
                                     const float *b, int n_in, int n_out, int accum) {
    extern __shared__ char smem_raw[];
    int nq = n_in / 4;  // activation dp4a words
    int ns = n_in / 16; // 16-row sub-blocks
    int nsup = n_in / 128;
    int *xq = (int *)smem_raw;                   // nq words
    float *xs = (float *)(smem_raw + n_in);      // nq scales
    int *xsum = (int *)(smem_raw + 2 * n_in);    // nq group sums
    float *s16 = (float *)(smem_raw + 3 * n_in); // ns correction sums
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
    for (int wr = threadIdx.x; wr < ns; wr += blockDim.x) {
        float s = 0.0f;
        for (int g = 4 * wr; g < 4 * wr + 4; ++g) s += xs[g] * (float)xsum[g];
        s16[wr] = s;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    if (n_out % 4 == 0 && n_out >= 4096) {
        int n4 = n_out / 4;
        for (int o4 = blockIdx.x * blockDim.x + threadIdx.x; o4 < n4;
             o4 += gridDim.x * blockDim.x) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int s = 0; s < nsup; ++s) { // 4 weight groups per super-block
                int4 dmi = *(const int4 *)(dm + (size_t)s * n_out + 4 * o4);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
                for (int g = 0; g < 4; ++g) {
                    int wg = 4 * s + g;
                    int4 lo0 = *(const int4 *)(w32 + (size_t)(wg * 3 + 0) * n_out + 4 * o4);
                    int4 lo1 = *(const int4 *)(w32 + (size_t)(wg * 3 + 1) * n_out + 4 * o4);
                    int4 hi = *(const int4 *)(w32 + (size_t)(wg * 3 + 2) * n_out + 4 * o4);
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        int4 lo = r == 0 ? lo0 : lo1;
                        float i0 = 0.0f, i1 = 0.0f, i2 = 0.0f, i3 = 0.0f;
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            int gq = 8 * wg + 4 * r + p;
                            int xa = xq[gq];
                            float sa = xs[gq];
                            i0 += sa * (float)__dp4a(q3_plane(lo.x, hi.x, r, p), xa, 0);
                            i1 += sa * (float)__dp4a(q3_plane(lo.y, hi.y, r, p), xa, 0);
                            i2 += sa * (float)__dp4a(q3_plane(lo.z, hi.z, r, p), xa, 0);
                            i3 += sa * (float)__dp4a(q3_plane(lo.w, hi.w, r, p), xa, 0);
                        }
                        int wr = 2 * wg + r;
                        uchar4 sb = *(const uchar4 *)(sub + (size_t)wr * n_out + 4 * o4);
                        float sx = s16[wr];
                        a0 += dm0.x * (sb.x & 15) * i0 - dm0.y * (sb.x >> 4) * sx;
                        a1 += dm1.x * (sb.y & 15) * i1 - dm1.y * (sb.y >> 4) * sx;
                        a2 += dm2.x * (sb.z & 15) * i2 - dm2.y * (sb.z >> 4) * sx;
                        a3 += dm3.x * (sb.w & 15) * i3 - dm3.y * (sb.w >> 4) * sx;
                    }
                }
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
        for (int s = 0; s < nsup; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * n_out + o]);
            for (int g = 0; g < 4; ++g) {
                int wg = 4 * s + g;
                int lo[2] = {w32[(size_t)(wg * 3 + 0) * n_out + o],
                             w32[(size_t)(wg * 3 + 1) * n_out + o]};
                int hi = w32[(size_t)(wg * 3 + 2) * n_out + o];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float inner = 0.0f;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int gq = 8 * wg + 4 * r + p;
                        inner +=
                            xs[gq] * (float)__dp4a(q3_plane(lo[r], hi, r, p), xq[gq], 0);
                    }
                    int wr = 2 * wg + r;
                    unsigned char sb = sub[(size_t)wr * n_out + o];
                    acc += dmv.x * (sb & 15) * inner - dmv.y * (sb >> 4) * s16[wr];
                }
            }
        }
        float r = acc + (b ? b[o] : 0.0f);
        y[o] = accum ? y[o] + r : r;
    }
}

__device__ __forceinline__ float embed_int3_at(const unsigned char *wte_t,
                                               const unsigned char *sub,
                                               const __half2 *dm, int i, int tok,
                                               int n_vocab) {
    size_t base = (size_t)(i / 32) * 3;
    unsigned char clo =
        wte_t[((base + (i % 32) / 16) * n_vocab + tok) * 4 + (i % 4)];
    unsigned char chi = wte_t[((base + 2) * n_vocab + tok) * 4 + (i % 4)];
    int lo2 = (clo >> (2 * ((i % 16) / 4))) & 3;
    int hi = (chi >> ((i % 32) / 4)) & 1;
    unsigned char sb = sub[(size_t)(i / 16) * n_vocab + tok];
    float2 dmv = __half22float2(dm[(size_t)(i / 128) * n_vocab + tok]);
    return dmv.x * (sb & 15) * (hi << 2 | lo2) - dmv.y * (sb >> 4);
}

extern "C" __global__ void embed_int3(float *out, const unsigned char *wte_t,
                                      const unsigned char *sub, const __half2 *dm,
                                      const float *wpe, int tok, int pos,
                                      int n_embd, int n_vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int3_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int3_dyn(float *out, const unsigned char *wte_t,
                                          const unsigned char *sub, const __half2 *dm,
                                          const float *wpe, const int *tok_ptr,
                                          const int *pos_ptr, int n_embd, int n_vocab) {
    int tok = *tok_ptr;
    int pos = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embed_int3_at(wte_t, sub, dm, i, tok, n_vocab) + wpe[pos * n_embd + i];
    }
}

extern "C" __global__ void embed_int3_batch(float *out, const unsigned char *wte_t,
                                            const unsigned char *sub, const __half2 *dm,
                                            const float *wpe, const int *toks, int pos0,
                                            int n_tok, int n_embd, int n_vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_tok * n_embd) {
        int t = idx / n_embd, i = idx % n_embd;
        out[idx] = embed_int3_at(wte_t, sub, dm, i, toks[t], n_vocab) +
                   wpe[(pos0 + t) * n_embd + i];
    }
}

// PagedAttention address translation: logical position t -> physical cache row.
// The KV cache is a pool of fixed-size blocks and a per-sequence block_table
// maps logical block (t / block_size) to a physical block, so logically
// contiguous positions can live in scattered physical blocks (the groundwork
// for shared prefixes in the next stage). PAGED=false collapses to the identity,
// so the linear kernels keep their exact addressing and codegen — paging is
// opt-in and bit-identical to the contiguous cache.
template <bool PAGED>
__device__ __forceinline__ size_t kv_row(const int *block_table, int block_size, int t) {
    if (PAGED) return (size_t)(block_table[t / block_size] * block_size + t % block_size);
    return (size_t)t;
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

// Graph-path fp32 KV write through the block_table (decode keeps tok/pos on the
// device, so the physical row is resolved here rather than on the host).
extern "C" __global__ void copy_kv_paged_dyn(float *kcache, float *vcache, const float *qkv,
                                             const int *pos_ptr, int q_dim, int kv_dim,
                                             const int *bt, int bs) {
    size_t row = kv_row<true>(bt, bs, *pos_ptr) * kv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kcache[row + i] = qkv[q_dim + i];
        vcache[row + i] = qkv[q_dim + kv_dim + i];
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
                                           const float *b, int rows, int n, float eps) {
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
    float inv = rsqrtf(red[0] / n + eps);
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

// Affine-quantizes one head row (head_dim threads, one per dim) into signed
// int8 spanning the row's actual [min, max] rather than a symmetric absmax:
// scale = (max - min) / 255, beta = min + 128 * scale, q in [-128, 127] with
// x ~ scale * q + beta. Two things matter versus symmetric absmax — the range
// tightens on skewed rows, and all 256 codes are used (255-code midpoint
// variants that keep the same scale only lose symmetric's exact-zero code and
// measure worse). The attention kernels fold the offset back analytically
// (K via the q-byte sum, V via the softmax weights), so q stays signed int8
// and the dp4a score path is untouched.
__device__ __forceinline__ void quant_affine(const float *src, signed char *dst,
                                             float *scale_out, float *beta_out,
                                             int d, float *red) {
    float hi = block_max(src[d], red);
    float lo = -block_max(-src[d], red);
    float scale = (hi > lo) ? (hi - lo) / 255.0f : 1.0f;
    float beta = lo + 128.0f * scale;
    dst[d] = (signed char)max(-128, min(127, __float2int_rn((src[d] - beta) / scale)));
    if (d == 0) {
        *scale_out = scale;
        *beta_out = beta;
    }
}

// Quantizes the new K/V rows into int8 caches, one fp32 (scale, beta) affine
// pair per (position, kv head). One block per kv head, one thread per head dim.
template <bool PAGED>
__device__ void quantize_kv_impl(signed char *kq, signed char *vq,
                                 float *ks, float *vs, float *kb, float *vb,
                                 const float *qkv,
                                 int pos, int q_dim, int n_kv_head, int head_dim,
                                 const int *bt, int bs) {
    __shared__ float red[32];
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *k = qkv + q_dim + h * head_dim;
    const float *v = qkv + q_dim + kv_dim + h * head_dim;

    size_t pr = kv_row<PAGED>(bt, bs, pos);
    size_t row = pr * kv_dim + h * head_dim;
    size_t sd = pr * n_kv_head + h;
    quant_affine(k, kq + row, &ks[sd], &kb[sd], d, red);
    quant_affine(v, vq + row, &vs[sd], &vb[sd], d, red);
}

// The non-graph decode write resolves the physical row on the host and passes it
// in as `pos`, so this linear kernel covers both linear and paged non-graph
// writes; only the graph path (device-resident pos) needs the paged variant.
extern "C" __global__ void quantize_kv(signed char *kq, signed char *vq,
                                       float *ks, float *vs, float *kb, float *vb,
                                       const float *qkv,
                                       int pos, int q_dim, int n_kv_head, int head_dim) {
    quantize_kv_impl<false>(kq, vq, ks, vs, kb, vb, qkv, pos, q_dim, n_kv_head, head_dim,
                            nullptr, 0);
}

extern "C" __global__ void quantize_kv_dyn(signed char *kq, signed char *vq,
                                           float *ks, float *vs, float *kb, float *vb,
                                           const float *qkv,
                                           const int *pos_ptr, int q_dim, int n_kv_head,
                                           int head_dim) {
    quantize_kv_impl<false>(kq, vq, ks, vs, kb, vb, qkv, *pos_ptr, q_dim, n_kv_head, head_dim,
                            nullptr, 0);
}

extern "C" __global__ void quantize_kv_paged_dyn(signed char *kq, signed char *vq,
                                                 float *ks, float *vs, float *kb, float *vb,
                                                 const float *qkv,
                                                 const int *pos_ptr, int q_dim, int n_kv_head,
                                                 int head_dim, const int *bt, int bs) {
    quantize_kv_impl<true>(kq, vq, ks, vs, kb, vb, qkv, *pos_ptr, q_dim, n_kv_head, head_dim,
                           bt, bs);
}

template <bool PAGED>
__device__ void quantize_kv_batch_impl(signed char *kq, signed char *vq,
                                       float *ks, float *vs, float *kb, float *vb,
                                       const float *qkv, int pos0, int q_dim, int n_kv_head,
                                       int head_dim, int stride, const int *bt, int bs) {
    __shared__ float red[32];
    int t = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;
    int kv_dim = n_kv_head * head_dim;
    const float *row = qkv + (size_t)t * stride;
    const float *k = row + q_dim + h * head_dim;
    const float *v = row + q_dim + kv_dim + h * head_dim;

    size_t pr = kv_row<PAGED>(bt, bs, pos0 + t);
    size_t out = pr * kv_dim + h * head_dim;
    size_t sd = pr * n_kv_head + h;
    quant_affine(k, kq + out, &ks[sd], &kb[sd], d, red);
    quant_affine(v, vq + out, &vs[sd], &vb[sd], d, red);
}

extern "C" __global__ void quantize_kv_batch(signed char *kq, signed char *vq,
                                             float *ks, float *vs, float *kb, float *vb,
                                             const float *qkv,
                                             int pos0, int q_dim, int n_kv_head,
                                             int head_dim, int stride) {
    quantize_kv_batch_impl<false>(kq, vq, ks, vs, kb, vb, qkv, pos0, q_dim, n_kv_head,
                                  head_dim, stride, nullptr, 0);
}

extern "C" __global__ void quantize_kv_batch_paged(signed char *kq, signed char *vq,
                                                   float *ks, float *vs, float *kb, float *vb,
                                                   const float *qkv,
                                                   int pos0, int q_dim, int n_kv_head,
                                                   int head_dim, int stride,
                                                   const int *bt, int bs) {
    quantize_kv_batch_impl<true>(kq, vq, ks, vs, kb, vb, qkv, pos0, q_dim, n_kv_head,
                                 head_dim, stride, bt, bs);
}

// Causal attention for one new token over the KV cache (one block per query
// head). Cache layout per layer: [t][n_kv_head * head_dim]; with grouped-query
// attention (n_kv_head < n_head) several query heads share one kv head.
// Scores for up to n_ctx cached positions live in shared memory.
template <bool PAGED>
__device__ void attn_decode_impl(float *out, const float *qkv,
                                 const float *kcache, const float *vcache,
                                 int t_cur, int n_head, int n_kv_head, int head_dim,
                                 const int *bt, int bs) {
    __shared__ float s[2048]; // n_ctx max (8 KB of the 48 KB block budget)
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
        const float4 *k4 = (const float4 *)(kcache + kv_row<PAGED>(bt, bs, t) * kvd + kvh * head_dim);
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
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        for (int t = 0; t <= t_cur; ++t) {
            float4 v = *(const float4 *)(vcache + kv_row<PAGED>(bt, bs, t) * kvd +
                                         kvh * head_dim + 4 * d4);
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
    attn_decode_impl<false>(out, qkv, kcache, vcache, t_cur, n_head, n_kv_head, head_dim,
                            nullptr, 0);
}

extern "C" __global__ void attn_decode_dyn(float *out, const float *qkv,
                                           const float *kcache, const float *vcache,
                                           const int *pos_ptr, int n_head, int n_kv_head,
                                           int head_dim) {
    attn_decode_impl<false>(out, qkv, kcache, vcache, *pos_ptr, n_head, n_kv_head, head_dim,
                            nullptr, 0);
}

// Paged variants: same math, K/V gathered through the block_table (bt, block
// size bs) instead of the contiguous t*kvd stride.
extern "C" __global__ void attn_decode_paged(float *out, const float *qkv,
                                             const float *kcache, const float *vcache,
                                             int t_cur, int n_head, int n_kv_head, int head_dim,
                                             const int *bt, int bs) {
    attn_decode_impl<true>(out, qkv, kcache, vcache, t_cur, n_head, n_kv_head, head_dim, bt, bs);
}

extern "C" __global__ void attn_decode_paged_dyn(float *out, const float *qkv,
                                                 const float *kcache, const float *vcache,
                                                 const int *pos_ptr, int n_head, int n_kv_head,
                                                 int head_dim, const int *bt, int bs) {
    attn_decode_impl<true>(out, qkv, kcache, vcache, *pos_ptr, n_head, n_kv_head, head_dim, bt, bs);
}

// Same attention over an int8 KV cache: scores and the V accumulation
// dequantize on the fly with the per-(position, head) affine (scale, beta)
// pairs, so the cache traffic — the part that grows with context length —
// shrinks 4x. K's beta folds into the score via the q-byte sum; V's beta is
// the softmax-weighted mean of the betas (added uniformly to every output dim).
template <bool PAGED>
__device__ void attn_decode_q8_impl(float *out, const float *qkv,
                                    const signed char *kq, const signed char *vq,
                                    const float *ks, const float *vs,
                                    const float *kb, const float *vb,
                                    int t_cur, int n_head, int n_kv_head, int head_dim,
                                    const int *bt, int bs) {
    __shared__ float s[2048]; // n_ctx max (8 KB of the 48 KB block budget)
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
    // Σ of this head's quantized q bytes: dp4a against all-ones sums four per
    // issue. This is the factor K's affine offset multiplies in every score.
    int qsum_p = 0;
    for (int d = tid; d < hd4; d += blockDim.x) qsum_p = __dp4a(qq[d], 0x01010101, qsum_p);
    float qsum = block_sum((float)qsum_p, red);

    float m = -CUDART_INF_F;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        size_t kr = kv_row<PAGED>(bt, bs, t);
        // head rows are head_dim-byte aligned: int4 vector loads, dp4a dot
        const int4 *k4 = (const int4 *)(kq + kr * kvd + kvh * head_dim);
        int dot = 0;
        for (int d = 0; d < hd4 / 4; ++d) {
            int4 kw = k4[d];
            dot = __dp4a(kw.x, qq[4 * d + 0], dot);
            dot = __dp4a(kw.y, qq[4 * d + 1], dot);
            dot = __dp4a(kw.z, qq[4 * d + 2], dot);
            dot = __dp4a(kw.w, qq[4 * d + 3], dot);
        }
        // q·k = qs·(ksᵀ·dot + kβ·Σq): the dp4a handles the scale·scale term,
        // the q-sum handles K's recovered offset.
        s[t] = qs * (ks[kr * n_kv_head + kvh] * (float)dot + kb[kr * n_kv_head + kvh] * qsum);
        m = fmaxf(m, s[t]);
    }
    m = block_max(m, red);

    float l = 0.0f;
    for (int t = tid; t <= t_cur; t += blockDim.x) {
        s[t] = __expf(s[t] - m);
        l += s[t];
    }
    float inv = 1.0f / block_sum(l, red);

    // V's affine offset: Σ_t s[t]·vβ_t (un-normalized; the ·inv below turns it
    // into the softmax-weighted mean). Same scalar for every output dim.
    float vb_p = 0.0f;
    for (int t = tid; t <= t_cur; t += blockDim.x)
        vb_p += s[t] * vb[kv_row<PAGED>(bt, bs, t) * n_kv_head + kvh];
    float vbsum = block_sum(vb_p, red);

    for (int d4 = tid; d4 < head_dim / 4; d4 += blockDim.x) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        for (int t = 0; t <= t_cur; ++t) {
            size_t vr = kv_row<PAGED>(bt, bs, t);
            char4 v = *(const char4 *)(vq + vr * kvd + kvh * head_dim + 4 * d4);
            float st = s[t] * vs[vr * n_kv_head + kvh];
            ax += st * (float)v.x, ay += st * (float)v.y;
            az += st * (float)v.z, aw += st * (float)v.w;
        }
        *(float4 *)(out + h * head_dim + 4 * d4) = make_float4(
            (ax + vbsum) * inv, (ay + vbsum) * inv,
            (az + vbsum) * inv, (aw + vbsum) * inv);
    }
}

extern "C" __global__ void attn_decode_q8(float *out, const float *qkv,
                                          const signed char *kq, const signed char *vq,
                                          const float *ks, const float *vs,
                                          const float *kb, const float *vb,
                                          int t_cur, int n_head, int n_kv_head,
                                          int head_dim) {
    attn_decode_q8_impl<false>(out, qkv, kq, vq, ks, vs, kb, vb, t_cur, n_head, n_kv_head,
                               head_dim, nullptr, 0);
}

extern "C" __global__ void attn_decode_q8_dyn(float *out, const float *qkv,
                                              const signed char *kq, const signed char *vq,
                                              const float *ks, const float *vs,
                                              const float *kb, const float *vb,
                                              const int *pos_ptr, int n_head, int n_kv_head,
                                              int head_dim) {
    attn_decode_q8_impl<false>(out, qkv, kq, vq, ks, vs, kb, vb, *pos_ptr, n_head, n_kv_head,
                               head_dim, nullptr, 0);
}

extern "C" __global__ void attn_decode_q8_paged(float *out, const float *qkv,
                                                const signed char *kq, const signed char *vq,
                                                const float *ks, const float *vs,
                                                const float *kb, const float *vb,
                                                int t_cur, int n_head, int n_kv_head,
                                                int head_dim, const int *bt, int bs) {
    attn_decode_q8_impl<true>(out, qkv, kq, vq, ks, vs, kb, vb, t_cur, n_head, n_kv_head,
                              head_dim, bt, bs);
}

extern "C" __global__ void attn_decode_q8_paged_dyn(float *out, const float *qkv,
                                                    const signed char *kq, const signed char *vq,
                                                    const float *ks, const float *vs,
                                                    const float *kb, const float *vb,
                                                    const int *pos_ptr, int n_head, int n_kv_head,
                                                    int head_dim, const int *bt, int bs) {
    attn_decode_q8_impl<true>(out, qkv, kq, vq, ks, vs, kb, vb, *pos_ptr, n_head, n_kv_head,
                              head_dim, bt, bs);
}

// ---- continuous-batch decode (Stage 5c) ------------------------------------
// One token for each of n_seq sequences in one forward. The weight read is
// shared across the whole batch (the point on a memory-bound card); each
// sequence keeps its own position pos[s] and its own block table (row s of
// `tables`, n_log entries each), attending only to its own cached KV. The
// per-(head, sequence) attention and KV write reuse the single-sequence paged
// device bodies with per-sequence base pointers — grid.y selects the sequence.

extern "C" __global__ void attn_decode_batched(
        float *out, const float *qkv, const float *kcache, const float *vcache,
        const int *pos, int n_head, int n_kv_head, int head_dim,
        const int *tables, int block_size, int n_log, int qkv_stride, int out_stride) {
    int s = blockIdx.y;
    attn_decode_impl<true>(out + (size_t)s * out_stride, qkv + (size_t)s * qkv_stride,
                           kcache, vcache, pos[s], n_head, n_kv_head, head_dim,
                           tables + (size_t)s * n_log, block_size);
}

extern "C" __global__ void attn_decode_q8_batched(
        float *out, const float *qkv, const signed char *kq, const signed char *vq,
        const float *ks, const float *vs, const float *kb, const float *vb,
        const int *pos, int n_head, int n_kv_head, int head_dim,
        const int *tables, int block_size, int n_log, int qkv_stride, int out_stride) {
    int s = blockIdx.y;
    attn_decode_q8_impl<true>(out + (size_t)s * out_stride, qkv + (size_t)s * qkv_stride,
                              kq, vq, ks, vs, kb, vb, pos[s], n_head, n_kv_head, head_dim,
                              tables + (size_t)s * n_log, block_size);
}

// Per-sequence fp32 KV write: sequence s (grid.y) writes its new token's K/V to
// the physical block that holds its position pos[s].
extern "C" __global__ void copy_kv_seqpos(
        float *kcache, float *vcache, const float *qkv, const int *pos,
        int q_dim, int kv_dim, int qkv_stride, const int *tables, int block_size, int n_log) {
    int s = blockIdx.y;
    size_t row = kv_row<true>(tables + (size_t)s * n_log, block_size, pos[s]) * kv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kcache[row + i] = qkv[(size_t)s * qkv_stride + q_dim + i];
        vcache[row + i] = qkv[(size_t)s * qkv_stride + q_dim + kv_dim + i];
    }
}

extern "C" __global__ void quantize_kv_seqpos(
        signed char *kq, signed char *vq, float *ks, float *vs, float *kb, float *vb,
        const float *qkv, const int *pos, int q_dim, int n_kv_head, int head_dim,
        int qkv_stride, const int *tables, int block_size, int n_log) {
    int s = blockIdx.y;
    quantize_kv_impl<true>(kq, vq, ks, vs, kb, vb, qkv + (size_t)s * qkv_stride, pos[s],
                           q_dim, n_kv_head, head_dim, tables + (size_t)s * n_log, block_size);
}

// RoPE with a per-sequence position (vs rope_batch's pos0 + row).
extern "C" __global__ void rope_seqpos(float *qkv, const int *pos, int n_seq, int n_head,
                                       int n_kv_head, int head_dim, int stride, float theta) {
    int half = head_dim / 2;
    int per_row = (n_head + n_kv_head) * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_seq * per_row) return;
    int s = idx / per_row;
    int i = idx - s * per_row;
    int h = i / half;
    int d = i % half;
    float *base = qkv + (size_t)s * stride + h * head_dim;
    float freq = __powf(theta, -2.0f * d / head_dim);
    float c, sn;
    __sincosf(pos[s] * freq, &sn, &c);
    float x1 = base[d], x2 = base[d + half];
    base[d] = x1 * c - x2 * sn;
    base[d + half] = x1 * sn + x2 * c;
}

// GPT-2 learned positions for continuous-batch decode: the batched embed runs
// with a zero wpe (token-only), so each row s holds wte[tok_s] alone — a
// row-independent value. Add wpe[pos[s]] exactly once here. This single add
// makes a sequence's embedding bit-identical whether it is row 0 (alone) or row
// s of a batch; an earlier add-then-subtract fixup was non-bit-identical because
// the cancelled wpe[row] term took a row-dependent floating-point path, which a
// kv8 near-tie could amplify into a token flip. (RoPE models zero wpe → skip.)
extern "C" __global__ void add_wpe_seqpos(float *x, const float *wpe, const int *pos,
                                          int n_seq, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_seq * n_embd) return;
    int s = idx / n_embd, i = idx % n_embd;
    x[idx] += wpe[(size_t)pos[s] * n_embd + i];
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

// Wide dp4a tier for real prefill (M > 64), the int edition of the fp32
// wide tier: 128x128 tile over 32 k-values (8 dp4a words), 256 threads
// each owning an 8x8 micro-tile — 64 dp4a + 64 scale-FMAs per 24 smem
// reads, double the old 64-tile's compute-to-smem ratio. With AG == 4
// each packed word carries its own activation scale, so the scale tile
// Ss mirrors As word-for-word and the micro-kernel pays one float FMA
// per dp4a (the price of outlier-proof 4-wide activation groups).
extern "C" __global__ void gemm_int8_wide(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const float *wscale,
                                          const float *bias, int M, int N, int K) {
    constexpr int WG = AG / 4; // dp4a words per activation-scale group
    constexpr int SC = 4 / WG; // scale loads per staging thread (4 words)
    __shared__ int As[2][8][128];
    __shared__ float Ss[2][8 / WG][128];
    __shared__ int Bs[2][8][128];
    int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    int tid = threadIdx.x;
    int arow = tid >> 1, acol = (tid & 1) * 4;  // A: one int4 each
    int brow = tid >> 5, bcol = (tid & 31) * 4; // B: one 4-vec each
    int trow = tid >> 4, tcol = tid & 15;
    int kq = K / 4;
    bool vec = (N % 4 == 0) && (bn + 128 <= N);
    float facc[8][8] = {};
    int a4[4], b4[4];
    float sg[SC];

    auto stage = [&](int kw0) {
        if (bm + arow < M) {
            *reinterpret_cast<int4 *>(a4) =
                *reinterpret_cast<const int4 *>(&Aq[(size_t)(bm + arow) * kq + kw0 + acol]);
#pragma unroll
            for (int j = 0; j < SC; ++j)
                sg[j] = ascale[(size_t)(bm + arow) * (kq / WG) + (kw0 + acol) / WG + j];
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = 0;
#pragma unroll
            for (int j = 0; j < SC; ++j) sg[j] = 0.0f;
        }
        if (vec) {
            *reinterpret_cast<int4 *>(b4) =
                *reinterpret_cast<const int4 *>(&B32[(size_t)(kw0 + brow) * N + bn + bcol]);
        } else {
            for (int j = 0; j < 4; ++j)
                b4[j] = (bn + bcol + j < N) ? B32[(size_t)(kw0 + brow) * N + bn + bcol + j]
                                            : 0;
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) As[buf][acol + j][arow] = a4[j];
#pragma unroll
        for (int j = 0; j < SC; ++j) Ss[buf][acol / WG + j][arow] = sg[j];
        *reinterpret_cast<int4 *>(&Bs[buf][brow][bcol]) = *reinterpret_cast<int4 *>(b4);
    };

    stage(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int kw0 = 0; kw0 < kq; kw0 += 8) {
        if (kw0 + 8 < kq) stage(kw0 + 8);
        for (int g = 0; g < 8 / WG; ++g) {
            int rm[WG][8], rn[WG][8];
            float rs[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) rs[i] = Ss[buf][g][trow * 8 + i];
#pragma unroll
            for (int w = 0; w < WG; ++w) {
#pragma unroll
                for (int i = 0; i < 8; ++i) rm[w][i] = As[buf][g * WG + w][trow * 8 + i];
#pragma unroll
                for (int j = 0; j < 8; ++j) rn[w][j] = Bs[buf][g * WG + w][tcol * 8 + j];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int acc = 0;
#pragma unroll
                    for (int w = 0; w < WG; ++w) acc = __dp4a(rm[w][i], rn[w][j], acc);
                    facc[i][j] += (float)acc * rs[i];
                }
        }
        if (kw0 + 8 < kq) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    for (int i = 0; i < 8; ++i) {
        int row = bm + trow * 8 + i;
        if (row >= M) continue;
        for (int j = 0; j < 8; ++j) {
            int col = bn + tcol * 8 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = facc[i][j] * wscale[col] + bias[col];
        }
    }
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

// Fused small-batch int8 GEMM via dp4a, modeled on gemv_int8: the M activation
// rows are quantized in shared memory (no separate quantize_act pass, no Aq
// round-trip), then each thread streams its weight column ONCE, accumulating all
// M outputs — bandwidth-bound like the decode GEMV instead of the K-tiled
// gemm_rows. Templated on MAXM so the accumulator arrays size to the actual row
// count (size-8 arrays at M=2 spill registers and wreck occupancy). The caller
// guarantees the shared budget (M*K + 4*M*K/AG bytes) fits.
template <int MAXM>
__device__ __forceinline__ void gemv_rows_int8_body(float *C, const float *A,
                                                    const signed char *w,
                                                    const float *scales, const float *bias,
                                                    int M, int N, int K) {
    extern __shared__ char smem_raw[];
    int n_groups = K / AG;
    int nq = K / 4;                                       // dp4a words per row
    int *xq = (int *)smem_raw;                            // M*nq ints
    float *xs = (float *)(smem_raw + (size_t)M * nq * 4); // M*n_groups floats
    for (int idx = threadIdx.x; idx < M * n_groups; idx += blockDim.x) {
        int m = idx / n_groups, g = idx % n_groups;
        const float *xg = A + (size_t)m * K + g * AG;
        float amax = 0.0f;
        for (int j = 0; j < AG; ++j) amax = fmaxf(amax, fabsf(xg[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int q = 0; q < AG / 4; ++q) {
            int packed = 0;
            for (int j = 0; j < 4; ++j) {
                int v = max(-127, min(127, __float2int_rn(xg[4 * q + j] * id)));
                packed |= (v & 0xFF) << (8 * j);
            }
            xq[(size_t)m * nq + (AG / 4) * g + q] = packed;
        }
        xs[(size_t)m * n_groups + g] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < N;
         o += gridDim.x * blockDim.x) {
        float acc[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
        for (int g = 0; g < n_groups; ++g) {
            int ig[MAXM];
#pragma unroll
            for (int m = 0; m < MAXM; ++m) ig[m] = 0;
            for (int q = 0; q < AG / 4; ++q) {
                int wv = w32[(size_t)((AG / 4) * g + q) * N + o];
#pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M)
                        ig[m] = __dp4a(xq[(size_t)m * nq + (AG / 4) * g + q], wv, ig[m]);
            }
#pragma unroll
            for (int m = 0; m < MAXM; ++m)
                if (m < M)
                    acc[m] += (float)ig[m] * xs[(size_t)m * n_groups + g];
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) {
                float r = acc[m] * scales[o] + (bias ? bias[o] : 0.0f);
                C[(size_t)m * N + o] = r;
            }
    }
}

extern "C" __global__ void gemv_rows_int8_m2(float *C, const float *A, const signed char *w,
                                             const float *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int8_body<2>(C, A, w, scales, bias, M, N, K);
}
extern "C" __global__ void gemv_rows_int8_m4(float *C, const float *A, const signed char *w,
                                             const float *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int8_body<4>(C, A, w, scales, bias, M, N, K);
}
extern "C" __global__ void gemv_rows_int8_m8(float *C, const float *A, const signed char *w,
                                             const float *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int8_body<8>(C, A, w, scales, bias, M, N, K);
}

// Fused tier-0 int4 GEMM: the row-wise twin of gemv_int4. The M activation rows
// are quantized in shared memory (per-4-element dp4a words + group sums for the
// zero-point correction), then each thread streams its int4 weight column once,
// unpacking nibbles with q4_lo8/q4_hi8. No separate quantize_act launch, no Aq
// round-trip, and bit-identical to the decode GEMV (same reduction order).
// Shared budget M*(3*nq + nw)*4 bytes (nq=K/4, nw=K/Q4_GROUP); caller checks it.
template <int MAXM>
__device__ __forceinline__ void gemv_rows_int4_body(float *C, const float *A,
                                                    const unsigned char *w,
                                                    const __half *scales, const float *bias,
                                                    int M, int N, int K) {
    extern __shared__ char smem_raw[];
    int nq = K / 4;          // activation dp4a words per row
    int nw = K / Q4_GROUP;   // 32-row weight groups per row
    int *xq = (int *)smem_raw;                     // M*nq ints
    float *xs = (float *)(xq + (size_t)M * nq);    // M*nq scales
    int *xsum = (int *)(xs + (size_t)M * nq);      // M*nq group sums
    float *s32 = (float *)(xsum + (size_t)M * nq); // M*nw correction sums
    for (int idx = threadIdx.x; idx < M * nq; idx += blockDim.x) {
        int m = idx / nq, g = idx % nq;
        const float *xg = A + (size_t)m * K + g * 4;
        float amax = 0.0f;
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(xg[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int packed = 0, sum = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(xg[j] * id)));
            sum += v;
            packed |= (v & 0xFF) << (8 * j);
        }
        xq[(size_t)m * nq + g] = packed;
        xs[(size_t)m * nq + g] = amax > 0.0f ? amax / 127.0f : 1.0f;
        xsum[(size_t)m * nq + g] = sum;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * nw; idx += blockDim.x) {
        int m = idx / nw, wg = idx % nw;
        float s = 0.0f;
        for (int g = 8 * wg; g < 8 * wg + 8; ++g)
            s += xs[(size_t)m * nq + g] * (float)xsum[(size_t)m * nq + g];
        s32[(size_t)m * nw + wg] = s;
    }
    __syncthreads();

    const int *w32 = (const int *)w;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < N;
         o += gridDim.x * blockDim.x) {
        float acc[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
        for (int wg = 0; wg < nw; ++wg) {
            float inner[MAXM];
#pragma unroll
            for (int m = 0; m < MAXM; ++m) inner[m] = 0.0f;
            for (int r = 0; r < 4; ++r) { // packed word = 8 rows of column o
                int wr = wg * 4 + r;
                int wv = w32[(size_t)wr * N + o];
                int lo = q4_lo8(wv), hi = q4_hi8(wv);
#pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M) {
                        const int *xqm = xq + (size_t)m * nq;
                        const float *xsm = xs + (size_t)m * nq;
                        inner[m] += xsm[2 * wr] * (float)__dp4a(lo, xqm[2 * wr], 0) +
                                    xsm[2 * wr + 1] * (float)__dp4a(hi, xqm[2 * wr + 1], 0);
                    }
            }
            float sc = __half2float(scales[(size_t)wg * N + o]);
#pragma unroll
            for (int m = 0; m < MAXM; ++m)
                if (m < M)
                    acc[m] += (inner[m] - 8.0f * s32[(size_t)m * nw + wg]) * sc;
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) {
                float r = acc[m] + (bias ? bias[o] : 0.0f);
                C[(size_t)m * N + o] = r;
            }
    }
}

extern "C" __global__ void gemv_rows_int4_m2(float *C, const float *A, const unsigned char *w,
                                             const __half *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int4_body<2>(C, A, w, scales, bias, M, N, K);
}
extern "C" __global__ void gemv_rows_int4_m4(float *C, const float *A, const unsigned char *w,
                                             const __half *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int4_body<4>(C, A, w, scales, bias, M, N, K);
}
extern "C" __global__ void gemv_rows_int4_m8(float *C, const float *A, const unsigned char *w,
                                             const __half *scales, const float *bias,
                                             int M, int N, int K) {
    gemv_rows_int4_body<8>(C, A, w, scales, bias, M, N, K);
}

// Shared staging for the fused k-quant row GEMMs (int4k/int3/int2): quantize the
// M activation rows into shared (per-4-element dp4a words + the 16-row sub-block
// sums s16 that fold the two-level -m term). Identical to the decode GEMVs'
// prologue; only the weight unpack in the output loop differs per mode.
__device__ __forceinline__ void stage_kquant_rows(const float *A, int M, int K,
                                                  int *xq, float *xs, int *xsum, float *s16) {
    int nq = K / 4, ns = K / 16;
    for (int idx = threadIdx.x; idx < M * nq; idx += blockDim.x) {
        int m = idx / nq, g = idx % nq;
        const float *xg = A + (size_t)m * K + g * 4;
        float amax = 0.0f;
        for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(xg[j]));
        float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int packed = 0, sum = 0;
        for (int j = 0; j < 4; ++j) {
            int v = max(-127, min(127, __float2int_rn(xg[j] * id)));
            sum += v;
            packed |= (v & 0xFF) << (8 * j);
        }
        xq[(size_t)m * nq + g] = packed;
        xs[(size_t)m * nq + g] = amax > 0.0f ? amax / 127.0f : 1.0f;
        xsum[(size_t)m * nq + g] = sum;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * ns; idx += blockDim.x) {
        int m = idx / ns, wr = idx % ns;
        float s = 0.0f;
        for (int g = 4 * wr; g < 4 * wr + 4; ++g)
            s += xs[(size_t)m * nq + g] * (float)xsum[(size_t)m * nq + g];
        s16[(size_t)m * ns + wr] = s;
    }
    __syncthreads();
}

// Fused tier-0 int4k GEMM: row-wise twin of gemv_int4k (two-level k-quant
// scales, 16-row sub-blocks). Shared = M*(3*nq + ns)*4 (nq=K/4, ns=K/16).
template <int MAXM>
__device__ __forceinline__ void gemv_rows_int4k_body(float *C, const float *A,
                                                     const unsigned char *w,
                                                     const unsigned char *sub, const __half2 *dm,
                                                     const float *bias, int M, int N, int K) {
    extern __shared__ char smem_raw[];
    int nq = K / 4, ns = K / 16;
    int *xq = (int *)smem_raw;
    float *xs = (float *)(xq + (size_t)M * nq);
    int *xsum = (int *)(xs + (size_t)M * nq);
    float *s16 = (float *)(xsum + (size_t)M * nq);
    stage_kquant_rows(A, M, K, xq, xs, xsum, s16);

    const int *w32 = (const int *)w;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < N;
         o += gridDim.x * blockDim.x) {
        float acc[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
        for (int s = 0; s < K / 128; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * N + o]);
            for (int sb = 0; sb < 8; ++sb) {
                int wri = 8 * s + sb;
                float inner[MAXM];
#pragma unroll
                for (int m = 0; m < MAXM; ++m) inner[m] = 0.0f;
                for (int pw = 0; pw < 2; ++pw) { // packed word = 8 rows
                    int wr = 2 * wri + pw;
                    int wv = w32[(size_t)wr * N + o];
                    int lo = q4_lo8(wv), hi = q4_hi8(wv);
#pragma unroll
                    for (int m = 0; m < MAXM; ++m)
                        if (m < M) {
                            const int *xqm = xq + (size_t)m * nq;
                            const float *xsm = xs + (size_t)m * nq;
                            inner[m] += xsm[2 * wr] * (float)__dp4a(lo, xqm[2 * wr], 0) +
                                        xsm[2 * wr + 1] * (float)__dp4a(hi, xqm[2 * wr + 1], 0);
                        }
                }
                unsigned char sbq = sub[(size_t)wri * N + o];
                float sd = (float)(sbq & 15), sm = (float)(sbq >> 4);
#pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M)
                        acc[m] += dmv.x * sd * inner[m] - dmv.y * sm * s16[(size_t)m * ns + wri];
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) C[(size_t)m * N + o] = acc[m] + (bias ? bias[o] : 0.0f);
    }
}

// Fused tier-0 int2 GEMM: row-wise twin of gemv_int2 (16 rows per packed word,
// 4 bit-plane dp4a words each).
template <int MAXM>
__device__ __forceinline__ void gemv_rows_int2_body(float *C, const float *A,
                                                    const unsigned char *w,
                                                    const unsigned char *sub, const __half2 *dm,
                                                    const float *bias, int M, int N, int K) {
    extern __shared__ char smem_raw[];
    int nq = K / 4, ns = K / 16;
    int *xq = (int *)smem_raw;
    float *xs = (float *)(xq + (size_t)M * nq);
    int *xsum = (int *)(xs + (size_t)M * nq);
    float *s16 = (float *)(xsum + (size_t)M * nq);
    stage_kquant_rows(A, M, K, xq, xs, xsum, s16);

    const int *w32 = (const int *)w;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < N;
         o += gridDim.x * blockDim.x) {
        float acc[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
        for (int s = 0; s < K / 128; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * N + o]);
            for (int r = 0; r < 8; ++r) { // packed word = 16 rows
                int wr = 8 * s + r;
                int wv = w32[(size_t)wr * N + o];
                float inner[MAXM];
#pragma unroll
                for (int m = 0; m < MAXM; ++m) inner[m] = 0.0f;
                for (int p = 0; p < 4; ++p) {
                    int plane = q2_plane(wv, p);
#pragma unroll
                    for (int m = 0; m < MAXM; ++m)
                        if (m < M)
                            inner[m] += xs[(size_t)m * nq + 4 * wr + p] *
                                        (float)__dp4a(plane, xq[(size_t)m * nq + 4 * wr + p], 0);
                }
                unsigned char sbq = sub[(size_t)wr * N + o];
                float sd = (float)(sbq & 15), sm = (float)(sbq >> 4);
#pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M)
                        acc[m] += dmv.x * sd * inner[m] - dmv.y * sm * s16[(size_t)m * ns + wr];
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) C[(size_t)m * N + o] = acc[m] + (bias ? bias[o] : 0.0f);
    }
}

// Fused tier-0 int3 GEMM: row-wise twin of gemv_int3 (32-row group = 3 words;
// q3_plane splices the two lo words with the hi bit word).
template <int MAXM>
__device__ __forceinline__ void gemv_rows_int3_body(float *C, const float *A,
                                                    const unsigned char *w,
                                                    const unsigned char *sub, const __half2 *dm,
                                                    const float *bias, int M, int N, int K) {
    extern __shared__ char smem_raw[];
    int nq = K / 4, ns = K / 16;
    int *xq = (int *)smem_raw;
    float *xs = (float *)(xq + (size_t)M * nq);
    int *xsum = (int *)(xs + (size_t)M * nq);
    float *s16 = (float *)(xsum + (size_t)M * nq);
    stage_kquant_rows(A, M, K, xq, xs, xsum, s16);

    const int *w32 = (const int *)w;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < N;
         o += gridDim.x * blockDim.x) {
        float acc[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
        for (int s = 0; s < K / 128; ++s) {
            float2 dmv = __half22float2(dm[(size_t)s * N + o]);
            for (int g = 0; g < 4; ++g) {
                int wg = 4 * s + g;
                int lo[2] = {w32[(size_t)(wg * 3 + 0) * N + o],
                             w32[(size_t)(wg * 3 + 1) * N + o]};
                int hi = w32[(size_t)(wg * 3 + 2) * N + o];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float inner[MAXM];
#pragma unroll
                    for (int m = 0; m < MAXM; ++m) inner[m] = 0.0f;
                    for (int p = 0; p < 4; ++p) {
                        int gq = 8 * wg + 4 * r + p;
                        int plane = q3_plane(lo[r], hi, r, p);
#pragma unroll
                        for (int m = 0; m < MAXM; ++m)
                            if (m < M)
                                inner[m] += xs[(size_t)m * nq + gq] *
                                            (float)__dp4a(plane, xq[(size_t)m * nq + gq], 0);
                    }
                    int wr = 2 * wg + r;
                    unsigned char sbq = sub[(size_t)wr * N + o];
                    float sd = (float)(sbq & 15), sm = (float)(sbq >> 4);
#pragma unroll
                    for (int m = 0; m < MAXM; ++m)
                        if (m < M)
                            acc[m] +=
                                dmv.x * sd * inner[m] - dmv.y * sm * s16[(size_t)m * ns + wr];
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) C[(size_t)m * N + o] = acc[m] + (bias ? bias[o] : 0.0f);
    }
}

#define KQUANT_ROWS_WRAPPERS(NAME)                                                             \
    extern "C" __global__ void gemv_rows_##NAME##_m2(                                           \
        float *C, const float *A, const unsigned char *w, const unsigned char *sub,            \
        const __half2 *dm, const float *bias, int M, int N, int K) {                           \
        gemv_rows_##NAME##_body<2>(C, A, w, sub, dm, bias, M, N, K);                            \
    }                                                                                          \
    extern "C" __global__ void gemv_rows_##NAME##_m4(                                           \
        float *C, const float *A, const unsigned char *w, const unsigned char *sub,            \
        const __half2 *dm, const float *bias, int M, int N, int K) {                           \
        gemv_rows_##NAME##_body<4>(C, A, w, sub, dm, bias, M, N, K);                            \
    }                                                                                          \
    extern "C" __global__ void gemv_rows_##NAME##_m8(                                           \
        float *C, const float *A, const unsigned char *w, const unsigned char *sub,            \
        const __half2 *dm, const float *bias, int M, int N, int K) {                           \
        gemv_rows_##NAME##_body<8>(C, A, w, sub, dm, bias, M, N, K);                            \
    }
KQUANT_ROWS_WRAPPERS(int4k)
KQUANT_ROWS_WRAPPERS(int2)
KQUANT_ROWS_WRAPPERS(int3)
#undef KQUANT_ROWS_WRAPPERS

// int4 GEMM via dp4a: same shape as gemm_i8_body (A pre-quantized by
// quantize_act), but the B tile unpacks packed nibble words into unsigned
// dp4a planes during the shared-tile fill (q4_lo8/q4_hi8 — no bias subtract).
// Two-level k-quants scales like the int2/int3 bodies: per 16-row sub-block,
// C += d_sb * (q·x) - m_sb * sum(x), where the per-row activation sums come
// from one dp4a against 0x01010101 per word. A 32-k tile holds two sub-blocks
// (Bs[0..3] / Bs[4..7]), each two packed words wide.
template <int BM>
__device__ void gemm_i4k_body(float *C, const int *Aq, const float *ascale,
                             const int *B32, const unsigned char *wsub,
                             const __half2 *wdm, const float *bias,
                             int M, int N, int K) {
    constexpr int RM = BM / 16;
    constexpr int WGq = AG / 4;
    __shared__ int As[8][BM];
    __shared__ int Bs[8][64];
    __shared__ float Ss[32 / AG][BM];        // per (activation group, row) scale
    __shared__ float Sx[2][BM];              // per (sub-block, row) act sum
    __shared__ float Swd[2][64], Swm[2][64]; // per (sub-block, column) d/m
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
                         : 0; // nibbles of 0 contribute nothing
            Bs[2 * wr][n] = q4_lo8(wv);
            Bs[2 * wr + 1][n] = q4_hi8(wv);
        }
        for (int i = tid; i < (32 / AG) * BM; i += 256) {
            int gg = i / BM, m = i % BM;
            Ss[gg][m] =
                (bm + m < M) ? ascale[(size_t)(bm + m) * kg + k0 / AG + gg] : 0.0f;
        }
        if (tid < 128) {
            int r = tid / 64, n = tid % 64;
            bool in = bn + n < N;
            float2 dmv = in ? __half22float2(wdm[(size_t)(k0 / 128) * N + bn + n])
                            : make_float2(0.0f, 0.0f);
            unsigned char sb = in ? wsub[(size_t)(k0 / 16 + r) * N + bn + n] : 0;
            Swd[r][n] = dmv.x * (sb & 15);
            Swm[r][n] = dmv.y * (sb >> 4);
        }
        __syncthreads();
        if (tid < 2 * BM) { // per-row activation sums need As/Ss in smem
            int r = tid / BM, m = tid % BM;
            float sx = 0.0f;
            for (int q = 4 * r; q < 4 * r + 4; ++q)
                sx += Ss[q / WGq][m] * (float)__dp4a(As[q][m], 0x01010101, 0);
            Sx[r][m] = sx;
        }
        __syncthreads();
        float tacc[2][RM][4] = {};
        for (int gg = 0; gg < 32 / AG; ++gg) {
            int sb = gg * AG / 16;
            int iacc[RM][4] = {};
            for (int q = WGq * gg; q < WGq * (gg + 1); ++q) {
                int a[RM], b[4];
                for (int i = 0; i < RM; ++i) a[i] = As[q][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) b[j] = Bs[q][threadIdx.x * 4 + j];
                for (int i = 0; i < RM; ++i)
                    for (int j = 0; j < 4; ++j) iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }
            for (int i = 0; i < RM; ++i) {
                float sa = Ss[gg][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) tacc[sb][i][j] += (float)iacc[i][j] * sa;
            }
        }
        for (int i = 0; i < RM; ++i) {
            int row = threadIdx.y * RM + i;
            for (int j = 0; j < 4; ++j) {
                int col = threadIdx.x * 4 + j;
                facc[i][j] += tacc[0][i][j] * Swd[0][col] - Sx[0][row] * Swm[0][col] +
                              tacc[1][i][j] * Swd[1][col] - Sx[1][row] * Swm[1][col];
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
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int4k(float *C, const int *Aq, const float *ascale,
                                     const int *B32, const unsigned char *wsub,
                                     const __half2 *wdm, const float *bias,
                                     int M, int N, int K) {
    gemm_i4k_body<64>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

// Wide int4 tier: 128x64 tile over 32 k-values, 256 threads with an 8x4
// micro-tile. Full row height (matches the int8 wide tier), so prefill reads
// the weight matrix M/128 times, not M/64 — the old half-height tile doubled
// weight traffic, which dominates a memory-bound prefill. The k-quants two-
// level scales change per 16-row sub-block (two per 32-k tile), so the micro-
// kernel keeps one activation-scaled partial (tacc) and folds the sub-block's
// (d, m) at its boundary — NOT a doubled tacc[2][...] (that's what spilled the
// int3/int2 wide attempts on 3 SMs). Per-row activation sums for the -m term
// live in 16 registers (sx[2][8]), no extra smem or sync. Packed nibbles
// unpack to unsigned planes once per tile during the Bs fill.
extern "C" __global__ __launch_bounds__(256, 2) void gemm_int4k_wide(
    float *C, const int *Aq, const float *ascale, const int *B32,
    const unsigned char *wsub, const __half2 *wdm, const float *bias,
    int M, int N, int K) {
    constexpr int WG = AG / 4;     // dp4a words per activation-scale group
    constexpr int SC = 4 / WG;     // scale loads per staging thread (4 A words)
    constexpr int GPS = 16 / AG;   // g-iterations per 16-row sub-block
    __shared__ int As[2][8][128];
    __shared__ float Ss[2][8 / WG][128];
    __shared__ int Bs[2][8][64];
    __shared__ float Swd[2][2][64], Swm[2][2][64]; // [buf][sub-block][col]
    int bm = blockIdx.y * 128, bn = blockIdx.x * 64;
    int tid = threadIdx.x;
    int arow = tid >> 1, acol = (tid & 1) * 4; // A: one int4 (4 words) each
    int bwr = tid >> 6, bcol = tid & 63;       // B: packed word bwr, column bcol
    int trow = tid >> 4, tcol = tid & 15;
    int kq = K / 4, kw = K / 8;
    float facc[8][4] = {};
    int a4[4], pb;
    float sg[SC], swd[2], swm[2];

    auto stage = [&](int k0) {
        if (bm + arow < M) {
            *reinterpret_cast<int4 *>(a4) = *reinterpret_cast<const int4 *>(
                &Aq[(size_t)(bm + arow) * kq + k0 / 4 + acol]);
#pragma unroll
            for (int j = 0; j < SC; ++j)
                sg[j] = ascale[(size_t)(bm + arow) * (kq / WG) + (k0 / 4 + acol) / WG + j];
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = 0;
#pragma unroll
            for (int j = 0; j < SC; ++j) sg[j] = 0.0f;
        }
        // packed word bwr covers k rows 8*bwr..8*bwr+7 of this 32-k tile
        pb = (bn + bcol < N && k0 / 8 + bwr < kw)
                 ? B32[(size_t)(k0 / 8 + bwr) * N + bn + bcol]
                 : 0; // nibbles of 0 contribute nothing
        if (tid < 64) {
            bool in = bn + tid < N;
            float2 dmv = in ? __half22float2(wdm[(size_t)(k0 / 128) * N + bn + tid])
                            : make_float2(0.0f, 0.0f);
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                unsigned char s = in ? wsub[(size_t)(k0 / 16 + sb) * N + bn + tid] : 0;
                swd[sb] = dmv.x * (s & 15);
                swm[sb] = dmv.y * (s >> 4);
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) As[buf][acol + j][arow] = a4[j];
#pragma unroll
        for (int j = 0; j < SC; ++j) Ss[buf][acol / WG + j][arow] = sg[j];
        Bs[buf][2 * bwr][bcol] = q4_lo8(pb);
        Bs[buf][2 * bwr + 1][bcol] = q4_hi8(pb);
        if (tid < 64)
#pragma unroll
            for (int sb = 0; sb < 2; ++sb) {
                Swd[buf][sb][tid] = swd[sb];
                Swm[buf][sb][tid] = swm[sb];
            }
    };

    stage(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
        if (k0 + 32 < K) stage(k0 + 32);
        // per-row activation sums for this tile's two sub-blocks (dp4a vs 1s)
        float sx[2][8];
#pragma unroll
        for (int sb = 0; sb < 2; ++sb)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float s = 0.0f;
#pragma unroll
                for (int q = 4 * sb; q < 4 * sb + 4; ++q)
                    s += Ss[buf][q / WG][trow * 8 + i] *
                         (float)__dp4a(As[buf][q][trow * 8 + i], 0x01010101, 0);
                sx[sb][i] = s;
            }
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            float tacc[8][4] = {};
#pragma unroll
            for (int gi = 0; gi < GPS; ++gi) {
                int g = sb * GPS + gi;
                int rm[WG][8], rn[WG][4];
                float rs[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) rs[i] = Ss[buf][g][trow * 8 + i];
#pragma unroll
                for (int w = 0; w < WG; ++w) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) rm[w][i] = As[buf][g * WG + w][trow * 8 + i];
#pragma unroll
                    for (int j = 0; j < 4; ++j) rn[w][j] = Bs[buf][g * WG + w][tcol * 4 + j];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int acc = 0;
#pragma unroll
                        for (int w = 0; w < WG; ++w) acc = __dp4a(rm[w][i], rn[w][j], acc);
                        tacc[i][j] += (float)acc * rs[i];
                    }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    facc[i][j] += tacc[i][j] * Swd[buf][sb][tcol * 4 + j] -
                                  sx[sb][i] * Swm[buf][sb][tcol * 4 + j];
        }
        if (k0 + 32 < K) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    for (int i = 0; i < 8; ++i) {
        int row = bm + trow * 8 + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + tcol * 4 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int4k_skinny(float *C, const int *Aq, const float *ascale,
                                            const int *B32, const unsigned char *wsub,
                                            const __half2 *wdm, const float *bias,
                                            int M, int N, int K) {
    gemm_i4k_body<16>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

// int4 draft-verify GEMM (M <= 8) via dp4a: nibble planes unpack in-register
// (unsigned q), two-level k-quants scales applied per 16-row sub-block (two
// packed words wide) with per-row activation sums precomputed in shared memory.
extern "C" __global__ void gemm_rows_int4k(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const unsigned char *wsub,
                                          const __half2 *wdm, const float *bias,
                                          int M, int N, int K) {
    __shared__ int As[ROWS_M][ROWS_KT / 4];
    __shared__ float Ss[ROWS_M][ROWS_KT / AG];
    __shared__ float Sx[ROWS_M][ROWS_KT / 16]; // per (row, sub-block) act sum
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
        for (int i = tid; i < ROWS_M * ROWS_KT / 16; i += blockDim.x) {
            int m = i / (ROWS_KT / 16), t = i % (ROWS_KT / 16);
            float sx = 0.0f;
            for (int q = 4 * t; q < 4 * t + 4; ++q)
                sx += Ss[m][q / (AG / 4)] * (float)__dp4a(As[m][q], 0x01010101, 0);
            Sx[m][t] = sx;
        }
        __syncthreads();
        if (active && wide) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                int4 dmi = *(const int4 *)(wdm + (size_t)((k0 + wg * 32) / 128) * N + 4 * o);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
                for (int sb = 0; sb < 2; ++sb) { // 2 sub-blocks per weight group
                    float gacc[ROWS_M][4] = {};
                    for (int pw = 0; pw < 2; ++pw) { // 2 packed words per sub-block
                        int wr = (k0 + wg * 32) / 8 + sb * 2 + pw;
                        int4 wv = *(const int4 *)(B32 + (size_t)wr * N + 4 * o);
                        int blo[4] = {q4_lo8(wv.x), q4_lo8(wv.y), q4_lo8(wv.z),
                                      q4_lo8(wv.w)};
                        int bhi[4] = {q4_hi8(wv.x), q4_hi8(wv.y), q4_hi8(wv.z),
                                      q4_hi8(wv.w)};
                        int qa = wg * 8 + sb * 4 + pw * 2, qb = qa + 1;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m) {
                            int a0 = As[m][qa], a1 = As[m][qb];
                            float s0 = Ss[m][qa / (AG / 4)], s1 = Ss[m][qb / (AG / 4)];
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
                    uchar4 sbq = *(const uchar4 *)(wsub +
                                                   (size_t)((k0 + wg * 32) / 16 + sb) * N + 4 * o);
                    float d0 = dm0.x * (sbq.x & 15), m0 = dm0.y * (sbq.x >> 4);
                    float d1 = dm1.x * (sbq.y & 15), m1 = dm1.y * (sbq.y >> 4);
                    float d2 = dm2.x * (sbq.z & 15), m2 = dm2.y * (sbq.z >> 4);
                    float d3 = dm3.x * (sbq.w & 15), m3 = dm3.y * (sbq.w >> 4);
                    int t = wg * 2 + sb;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        float sx = Sx[m][t];
                        facc[m][0] += gacc[m][0] * d0 - sx * m0;
                        facc[m][1] += gacc[m][1] * d1 - sx * m1;
                        facc[m][2] += gacc[m][2] * d2 - sx * m2;
                        facc[m][3] += gacc[m][3] * d3 - sx * m3;
                    }
                }
            }
        } else if (active) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                float2 dmv = __half22float2(wdm[(size_t)((k0 + wg * 32) / 128) * N + o]);
                for (int sb = 0; sb < 2; ++sb) {
                    float gacc[ROWS_M] = {};
                    for (int pw = 0; pw < 2; ++pw) {
                        int wr = (k0 + wg * 32) / 8 + sb * 2 + pw;
                        int wv = B32[(size_t)wr * N + o];
                        int blo = q4_lo8(wv), bhi = q4_hi8(wv);
                        int qa = wg * 8 + sb * 4 + pw * 2, qb = qa + 1;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m) {
                            gacc[m] +=
                                Ss[m][qa / (AG / 4)] * (float)__dp4a(blo, As[m][qa], 0) +
                                Ss[m][qb / (AG / 4)] * (float)__dp4a(bhi, As[m][qb], 0);
                        }
                    }
                    unsigned char sbq = wsub[(size_t)((k0 + wg * 32) / 16 + sb) * N + o];
                    float d = dmv.x * (sbq & 15), mm = dmv.y * (sbq >> 4);
                    int t = wg * 2 + sb;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) facc[m][0] += gacc[m] * d - Sx[m][t] * mm;
                }
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

// Wide int4 tier: 128x64 tile over 32 k-values, 256 threads with an 8x4
// micro-tile. Full row height (matches the int8 wide tier), so prefill reads
// the weight matrix M/128 times, not M/64 — the old half-height tile doubled
// weight traffic, which dominates a memory-bound prefill. int4 still needs a
// second accumulator (the per-32-row fp16 weight scale changes every k-tile),
// so the micro-kernel keeps an activation-scaled partial (tacc) and folds the
// weight scale once per tile; an 8x4 micro-tile holds facc + tacc in the same
// 64 registers the old 64x128 4x8 shape used — taller tile, free of cost.
// Packed nibbles unpack to signed bytes once per tile during the Bs fill
// (__vsubss4), exactly like the 64-tile body.
extern "C" __global__ void gemm_int4_wide(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const __half *wscale,
                                          const float *bias, int M, int N, int K) {
    constexpr int WG = AG / 4;   // dp4a words per activation-scale group
    constexpr int SC = 4 / WG;   // scale loads per staging thread (4 A words)
    __shared__ int As[2][8][128];
    __shared__ float Ss[2][8 / WG][128];
    __shared__ int Bs[2][8][64];
    __shared__ float Sw[2][64];
    int bm = blockIdx.y * 128, bn = blockIdx.x * 64;
    int tid = threadIdx.x;
    int arow = tid >> 1, acol = (tid & 1) * 4; // A: one int4 (4 words) each
    int bwr = tid >> 6, bcol = tid & 63;       // B: packed word bwr, column bcol
    int trow = tid >> 4, tcol = tid & 15;
    int kq = K / 4, kw = K / 8;
    float facc[8][4] = {};
    int a4[4], pb;
    float sg[SC], swv;

    auto stage = [&](int k0) {
        if (bm + arow < M) {
            *reinterpret_cast<int4 *>(a4) = *reinterpret_cast<const int4 *>(
                &Aq[(size_t)(bm + arow) * kq + k0 / 4 + acol]);
#pragma unroll
            for (int j = 0; j < SC; ++j)
                sg[j] = ascale[(size_t)(bm + arow) * (kq / WG) + (k0 / 4 + acol) / WG + j];
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = 0;
#pragma unroll
            for (int j = 0; j < SC; ++j) sg[j] = 0.0f;
        }
        // packed word bwr covers k rows 8*bwr..8*bwr+7 of this 32-k tile
        pb = (bn + bcol < N && k0 / 8 + bwr < kw)
                 ? B32[(size_t)(k0 / 8 + bwr) * N + bn + bcol]
                 : 0x88888888; // nibbles of 8 unpack to 0
        if (tid < 64)
            swv = (bn + tid < N)
                      ? __half2float(wscale[(size_t)(k0 / 32) * N + bn + tid])
                      : 0.0f;
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) As[buf][acol + j][arow] = a4[j];
#pragma unroll
        for (int j = 0; j < SC; ++j) Ss[buf][acol / WG + j][arow] = sg[j];
        Bs[buf][2 * bwr][bcol] = __vsubss4(q4_lo8(pb), 0x08080808);
        Bs[buf][2 * bwr + 1][bcol] = __vsubss4(q4_hi8(pb), 0x08080808);
        if (tid < 64) Sw[buf][tid] = swv;
    };

    stage(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
        if (k0 + 32 < K) stage(k0 + 32);
        float tacc[8][4] = {};
        for (int g = 0; g < 8 / WG; ++g) {
            int rm[WG][8], rn[WG][4];
            float rs[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) rs[i] = Ss[buf][g][trow * 8 + i];
#pragma unroll
            for (int w = 0; w < WG; ++w) {
#pragma unroll
                for (int i = 0; i < 8; ++i) rm[w][i] = As[buf][g * WG + w][trow * 8 + i];
#pragma unroll
                for (int j = 0; j < 4; ++j) rn[w][j] = Bs[buf][g * WG + w][tcol * 4 + j];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int acc = 0;
#pragma unroll
                    for (int w = 0; w < WG; ++w) acc = __dp4a(rm[w][i], rn[w][j], acc);
                    tacc[i][j] += (float)acc * rs[i];
                }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                facc[i][j] += tacc[i][j] * Sw[buf][tcol * 4 + j];
        if (k0 + 32 < K) store(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
    for (int i = 0; i < 8; ++i) {
        int row = bm + trow * 8 + i;
        if (row >= M) continue;
        for (int j = 0; j < 4; ++j) {
            int col = bn + tcol * 4 + j;
            if (col >= N) continue;
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
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
                        // Ss is per AG-group, not per word
                        float s0 = Ss[m][qa / (AG / 4)], s1 = Ss[m][qb / (AG / 4)];
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
                        gacc[m] +=
                            Ss[m][qa / (AG / 4)] * (float)__dp4a(blo, As[m][qa], 0) +
                            Ss[m][qb / (AG / 4)] * (float)__dp4a(bhi, As[m][qb], 0);
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

// int2 GEMM: the int4 body with a different tile fill — each packed word
// covers 16 k-rows and unpacks into four byte-plane words (unsigned q in
// [0, 3]; OOB fills with 0 so padding contributes nothing). Two-level
// scales: per 16-row sub-block, C += d_sb * (q·x) - m_sb * sum(x), where
// the activation sums come from one dp4a against 0x01010101 per word.
template <int BM>
__device__ void gemm_i2_body(float *C, const int *Aq, const float *ascale,
                             const int *B32, const unsigned char *wsub,
                             const __half2 *wdm, const float *bias,
                             int M, int N, int K) {
    constexpr int RM = BM / 16;
    constexpr int WGq = AG / 4;
    __shared__ int As[8][BM];
    __shared__ int Bs[8][64];
    __shared__ float Ss[32 / AG][BM];     // per (activation group, row) scale
    __shared__ float Sx[2][BM];           // per (sub-block, row) act sum
    __shared__ float Swd[2][64], Swm[2][64]; // per (sub-block, column) d/m
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    int kq = K / 4, kg = K / AG, kp = K / 16; // packed-word rows of B
    float facc[RM][4] = {};

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int i = tid; i < BM * 8; i += 256) {
            int m = i / 8, q = i % 8;
            As[q][m] = (bm + m < M) ? Aq[(size_t)(bm + m) * kq + k0 / 4 + q] : 0;
        }
        for (int i = tid; i < 2 * 64; i += 256) {
            int wr = i / 64, n = i % 64; // packed word wr covers k rows 16wr..16wr+15
            int wv = (bn + n < N && k0 / 16 + wr < kp)
                         ? B32[(size_t)(k0 / 16 + wr) * N + bn + n]
                         : 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) Bs[4 * wr + p][n] = q2_plane(wv, p);
        }
        for (int i = tid; i < (32 / AG) * BM; i += 256) {
            int gg = i / BM, m = i % BM;
            Ss[gg][m] =
                (bm + m < M) ? ascale[(size_t)(bm + m) * kg + k0 / AG + gg] : 0.0f;
        }
        if (tid < 128) {
            int r = tid / 64, n = tid % 64;
            bool in = bn + n < N;
            float2 dmv = in ? __half22float2(wdm[(size_t)(k0 / 128) * N + bn + n])
                            : make_float2(0.0f, 0.0f);
            unsigned char sb = in ? wsub[(size_t)(k0 / 16 + r) * N + bn + n] : 0;
            Swd[r][n] = dmv.x * (sb & 15);
            Swm[r][n] = dmv.y * (sb >> 4);
        }
        __syncthreads();
        if (tid < 2 * BM) { // per-row activation sums need As/Ss in smem
            int r = tid / BM, m = tid % BM;
            float sx = 0.0f;
            for (int q = 4 * r; q < 4 * r + 4; ++q)
                sx += Ss[q / WGq][m] * (float)__dp4a(As[q][m], 0x01010101, 0);
            Sx[r][m] = sx;
        }
        __syncthreads();
        float tacc[2][RM][4] = {};
        for (int gg = 0; gg < 32 / AG; ++gg) {
            int sb = gg * AG / 16;
            int iacc[RM][4] = {};
            for (int q = WGq * gg; q < WGq * (gg + 1); ++q) {
                int a[RM], b[4];
                for (int i = 0; i < RM; ++i) a[i] = As[q][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) b[j] = Bs[q][threadIdx.x * 4 + j];
                for (int i = 0; i < RM; ++i)
                    for (int j = 0; j < 4; ++j) iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }
            for (int i = 0; i < RM; ++i) {
                float sa = Ss[gg][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) tacc[sb][i][j] += (float)iacc[i][j] * sa;
            }
        }
        for (int i = 0; i < RM; ++i) {
            int row = threadIdx.y * RM + i;
            for (int j = 0; j < 4; ++j) {
                int col = threadIdx.x * 4 + j;
                facc[i][j] += tacc[0][i][j] * Swd[0][col] - Sx[0][row] * Swm[0][col] +
                              tacc[1][i][j] * Swd[1][col] - Sx[1][row] * Swm[1][col];
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
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int2(float *C, const int *Aq, const float *ascale,
                                     const int *B32, const unsigned char *wsub,
                                     const __half2 *wdm, const float *bias,
                                     int M, int N, int K) {
    gemm_i2_body<64>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

extern "C" __global__ void gemm_int2_skinny(float *C, const int *Aq, const float *ascale,
                                            const int *B32, const unsigned char *wsub,
                                            const __half2 *wdm, const float *bias,
                                            int M, int N, int K) {
    gemm_i2_body<16>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

// int2 draft-verify GEMM (M <= 8): plane unpack in-register (unsigned q),
// two-level scales applied per 16-row sub-block with per-row activation
// sums precomputed in shared memory.
extern "C" __global__ void gemm_rows_int2(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const unsigned char *wsub,
                                          const __half2 *wdm, const float *bias,
                                          int M, int N, int K) {
    __shared__ int As[ROWS_M][ROWS_KT / 4];
    __shared__ float Ss[ROWS_M][ROWS_KT / AG];
    __shared__ float Sx[ROWS_M][ROWS_KT / 16]; // per (row, sub-block) act sum
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
        for (int i = tid; i < ROWS_M * ROWS_KT / 16; i += blockDim.x) {
            int m = i / (ROWS_KT / 16), t = i % (ROWS_KT / 16);
            float sx = 0.0f;
            for (int q = 4 * t; q < 4 * t + 4; ++q)
                sx += Ss[m][q / (AG / 4)] * (float)__dp4a(As[m][q], 0x01010101, 0);
            Sx[m][t] = sx;
        }
        __syncthreads();
        if (active && wide) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                int4 dmi = *(const int4 *)(wdm + (size_t)((k0 + wg * 32) / 128) * N + 4 * o);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
                for (int r = 0; r < 2; ++r) { // 2 packed words per weight group
                    int wr = (k0 + wg * 32) / 16 + r;
                    int4 wv = *(const int4 *)(B32 + (size_t)wr * N + 4 * o);
                    float gacc[ROWS_M][4] = {};
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int bp[4] = {q2_plane(wv.x, p), q2_plane(wv.y, p),
                                     q2_plane(wv.z, p), q2_plane(wv.w, p)};
                        int q = (wg * 32) / 4 + 4 * r + p;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m) {
                            int a = As[m][q];
                            float s = Ss[m][q / (AG / 4)];
                            gacc[m][0] += s * (float)__dp4a(bp[0], a, 0);
                            gacc[m][1] += s * (float)__dp4a(bp[1], a, 0);
                            gacc[m][2] += s * (float)__dp4a(bp[2], a, 0);
                            gacc[m][3] += s * (float)__dp4a(bp[3], a, 0);
                        }
                    }
                    uchar4 sb = *(const uchar4 *)(wsub + (size_t)wr * N + 4 * o);
                    float d0 = dm0.x * (sb.x & 15), m0 = dm0.y * (sb.x >> 4);
                    float d1 = dm1.x * (sb.y & 15), m1 = dm1.y * (sb.y >> 4);
                    float d2 = dm2.x * (sb.z & 15), m2 = dm2.y * (sb.z >> 4);
                    float d3 = dm3.x * (sb.w & 15), m3 = dm3.y * (sb.w >> 4);
                    int t = (wg * 32) / 16 + r;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        float sx = Sx[m][t];
                        facc[m][0] += gacc[m][0] * d0 - sx * m0;
                        facc[m][1] += gacc[m][1] * d1 - sx * m1;
                        facc[m][2] += gacc[m][2] * d2 - sx * m2;
                        facc[m][3] += gacc[m][3] * d3 - sx * m3;
                    }
                }
            }
        } else if (active) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                float2 dmv =
                    __half22float2(wdm[(size_t)((k0 + wg * 32) / 128) * N + o]);
                for (int r = 0; r < 2; ++r) {
                    int wr = (k0 + wg * 32) / 16 + r;
                    int wv = B32[(size_t)wr * N + o];
                    float gacc[ROWS_M] = {};
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int bp = q2_plane(wv, p);
                        int q = (wg * 32) / 4 + 4 * r + p;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m)
                            gacc[m] += Ss[m][q / (AG / 4)] * (float)__dp4a(bp, As[m][q], 0);
                    }
                    unsigned char sb = wsub[(size_t)wr * N + o];
                    float d = dmv.x * (sb & 15), mm = dmv.y * (sb >> 4);
                    int t = (wg * 32) / 16 + r;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m)
                        facc[m][0] += gacc[m] * d - Sx[m][t] * mm;
                }
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

// int3 GEMM: the int2 body with a triple-word tile fill — one (lo0, lo1,
// hi) word set per column covers the whole 32-k tile. Unsigned q in
// [0, 7]; OOB fills with 0 (q = 0, scales 0). Same two-level sub-block
// scales as the int2 body.
template <int BM>
__device__ void gemm_i3_body(float *C, const int *Aq, const float *ascale,
                             const int *B32, const unsigned char *wsub,
                             const __half2 *wdm, const float *bias,
                             int M, int N, int K) {
    constexpr int RM = BM / 16;
    constexpr int WGq = AG / 4;
    __shared__ int As[8][BM];
    __shared__ int Bs[8][64];
    __shared__ float Ss[32 / AG][BM];     // per (activation group, row) scale
    __shared__ float Sx[2][BM];           // per (sub-block, row) act sum
    __shared__ float Swd[2][64], Swm[2][64]; // per (sub-block, column) d/m
    int bm = blockIdx.y * BM, bn = blockIdx.x * 64;
    int tid = threadIdx.y * 16 + threadIdx.x;
    int kq = K / 4, kg = K / AG;
    float facc[RM][4] = {};

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int i = tid; i < BM * 8; i += 256) {
            int m = i / 8, q = i % 8;
            As[q][m] = (bm + m < M) ? Aq[(size_t)(bm + m) * kq + k0 / 4 + q] : 0;
        }
        if (tid < 64) {
            int n = tid;
            size_t base = (size_t)(k0 / 32) * 3;
            bool in = bn + n < N;
            int lo0 = in ? B32[(base + 0) * N + bn + n] : 0;
            int lo1 = in ? B32[(base + 1) * N + bn + n] : 0;
            int hi = in ? B32[(base + 2) * N + bn + n] : 0;
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    Bs[4 * r + p][n] = q3_plane(r == 0 ? lo0 : lo1, hi, r, p);
        }
        for (int i = tid; i < (32 / AG) * BM; i += 256) {
            int gg = i / BM, m = i % BM;
            Ss[gg][m] =
                (bm + m < M) ? ascale[(size_t)(bm + m) * kg + k0 / AG + gg] : 0.0f;
        }
        if (tid < 128) {
            int r = tid / 64, n = tid % 64;
            bool in = bn + n < N;
            float2 dmv = in ? __half22float2(wdm[(size_t)(k0 / 128) * N + bn + n])
                            : make_float2(0.0f, 0.0f);
            unsigned char sb = in ? wsub[(size_t)(k0 / 16 + r) * N + bn + n] : 0;
            Swd[r][n] = dmv.x * (sb & 15);
            Swm[r][n] = dmv.y * (sb >> 4);
        }
        __syncthreads();
        if (tid < 2 * BM) { // per-row activation sums need As/Ss in smem
            int r = tid / BM, m = tid % BM;
            float sx = 0.0f;
            for (int q = 4 * r; q < 4 * r + 4; ++q)
                sx += Ss[q / WGq][m] * (float)__dp4a(As[q][m], 0x01010101, 0);
            Sx[r][m] = sx;
        }
        __syncthreads();
        float tacc[2][RM][4] = {};
        for (int gg = 0; gg < 32 / AG; ++gg) {
            int sb = gg * AG / 16;
            int iacc[RM][4] = {};
            for (int q = WGq * gg; q < WGq * (gg + 1); ++q) {
                int a[RM], b[4];
                for (int i = 0; i < RM; ++i) a[i] = As[q][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) b[j] = Bs[q][threadIdx.x * 4 + j];
                for (int i = 0; i < RM; ++i)
                    for (int j = 0; j < 4; ++j) iacc[i][j] = __dp4a(a[i], b[j], iacc[i][j]);
            }
            for (int i = 0; i < RM; ++i) {
                float sa = Ss[gg][threadIdx.y * RM + i];
                for (int j = 0; j < 4; ++j) tacc[sb][i][j] += (float)iacc[i][j] * sa;
            }
        }
        for (int i = 0; i < RM; ++i) {
            int row = threadIdx.y * RM + i;
            for (int j = 0; j < 4; ++j) {
                int col = threadIdx.x * 4 + j;
                facc[i][j] += tacc[0][i][j] * Swd[0][col] - Sx[0][row] * Swm[0][col] +
                              tacc[1][i][j] * Swd[1][col] - Sx[1][row] * Swm[1][col];
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
            C[(size_t)row * N + col] = facc[i][j] + bias[col];
        }
    }
}

extern "C" __global__ void gemm_int3(float *C, const int *Aq, const float *ascale,
                                     const int *B32, const unsigned char *wsub,
                                     const __half2 *wdm, const float *bias,
                                     int M, int N, int K) {
    gemm_i3_body<64>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

extern "C" __global__ void gemm_int3_skinny(float *C, const int *Aq, const float *ascale,
                                            const int *B32, const unsigned char *wsub,
                                            const __half2 *wdm, const float *bias,
                                            int M, int N, int K) {
    gemm_i3_body<16>(C, Aq, ascale, B32, wsub, wdm, bias, M, N, K);
}

// int3 draft-verify GEMM (M <= 8): triple-word loads, plane assembly
// in-register, accumulator per 32-row weight group scaled by its fp16
// weight scale.
extern "C" __global__ void gemm_rows_int3(float *C, const int *Aq, const float *ascale,
                                          const int *B32, const unsigned char *wsub,
                                          const __half2 *wdm, const float *bias,
                                          int M, int N, int K) {
    __shared__ int As[ROWS_M][ROWS_KT / 4];
    __shared__ float Ss[ROWS_M][ROWS_KT / AG];
    __shared__ float Sx[ROWS_M][ROWS_KT / 16]; // per (row, sub-block) act sum
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
        for (int i = tid; i < ROWS_M * ROWS_KT / 16; i += blockDim.x) {
            int m = i / (ROWS_KT / 16), t = i % (ROWS_KT / 16);
            float sx = 0.0f;
            for (int q = 4 * t; q < 4 * t + 4; ++q)
                sx += Ss[m][q / (AG / 4)] * (float)__dp4a(As[m][q], 0x01010101, 0);
            Sx[m][t] = sx;
        }
        __syncthreads();
        if (active && wide) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                size_t base = (size_t)((k0 + wg * 32) / 32) * 3;
                int4 lo0 = *(const int4 *)(B32 + (base + 0) * N + 4 * o);
                int4 lo1 = *(const int4 *)(B32 + (base + 1) * N + 4 * o);
                int4 hi = *(const int4 *)(B32 + (base + 2) * N + 4 * o);
                int4 dmi = *(const int4 *)(wdm + (size_t)((k0 + wg * 32) / 128) * N + 4 * o);
                float2 dm0 = __half22float2(*(const __half2 *)&dmi.x);
                float2 dm1 = __half22float2(*(const __half2 *)&dmi.y);
                float2 dm2 = __half22float2(*(const __half2 *)&dmi.z);
                float2 dm3 = __half22float2(*(const __half2 *)&dmi.w);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    int4 lo = r == 0 ? lo0 : lo1;
                    float gacc[ROWS_M][4] = {};
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int bp[4] = {q3_plane(lo.x, hi.x, r, p), q3_plane(lo.y, hi.y, r, p),
                                     q3_plane(lo.z, hi.z, r, p), q3_plane(lo.w, hi.w, r, p)};
                        int q = (wg * 32) / 4 + 4 * r + p;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m) {
                            int a = As[m][q];
                            float s = Ss[m][q / (AG / 4)];
                            gacc[m][0] += s * (float)__dp4a(bp[0], a, 0);
                            gacc[m][1] += s * (float)__dp4a(bp[1], a, 0);
                            gacc[m][2] += s * (float)__dp4a(bp[2], a, 0);
                            gacc[m][3] += s * (float)__dp4a(bp[3], a, 0);
                        }
                    }
                    int wr = (k0 + wg * 32) / 16 + r;
                    uchar4 sb = *(const uchar4 *)(wsub + (size_t)wr * N + 4 * o);
                    float d0 = dm0.x * (sb.x & 15), m0 = dm0.y * (sb.x >> 4);
                    float d1 = dm1.x * (sb.y & 15), m1 = dm1.y * (sb.y >> 4);
                    float d2 = dm2.x * (sb.z & 15), m2 = dm2.y * (sb.z >> 4);
                    float d3 = dm3.x * (sb.w & 15), m3 = dm3.y * (sb.w >> 4);
                    int t = (wg * 32) / 16 + r;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m) {
                        float sx = Sx[m][t];
                        facc[m][0] += gacc[m][0] * d0 - sx * m0;
                        facc[m][1] += gacc[m][1] * d1 - sx * m1;
                        facc[m][2] += gacc[m][2] * d2 - sx * m2;
                        facc[m][3] += gacc[m][3] * d3 - sx * m3;
                    }
                }
            }
        } else if (active) {
            for (int wg = 0; wg < kt / Q4_GROUP; ++wg) {
                size_t base = (size_t)((k0 + wg * 32) / 32) * 3;
                int lo[2] = {B32[(base + 0) * N + o], B32[(base + 1) * N + o]};
                int hi = B32[(base + 2) * N + o];
                float2 dmv =
                    __half22float2(wdm[(size_t)((k0 + wg * 32) / 128) * N + o]);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float gacc[ROWS_M] = {};
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int bp = q3_plane(lo[r], hi, r, p);
                        int q = (wg * 32) / 4 + 4 * r + p;
#pragma unroll
                        for (int m = 0; m < ROWS_M; ++m)
                            gacc[m] += Ss[m][q / (AG / 4)] * (float)__dp4a(bp, As[m][q], 0);
                    }
                    int wr = (k0 + wg * 32) / 16 + r;
                    unsigned char sb = wsub[(size_t)wr * N + o];
                    float d = dmv.x * (sb & 15), mm = dmv.y * (sb >> 4);
                    int t = (wg * 32) / 16 + r;
#pragma unroll
                    for (int m = 0; m < ROWS_M; ++m)
                        facc[m][0] += gacc[m] * d - Sx[m][t] * mm;
                }
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

extern "C" __global__ void copy_kv_batch_paged(float *kcache, float *vcache, const float *qkv,
                                               int pos0, int q_dim, int kv_dim, int stride,
                                               const int *bt, int bs) {
    int t = blockIdx.y;
    size_t row = kv_row<true>(bt, bs, pos0 + t) * kv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        kcache[row + i] = qkv[(size_t)t * stride + q_dim + i];
        vcache[row + i] = qkv[(size_t)t * stride + q_dim + kv_dim + i];
    }
}

// Flash-style batched causal attention over the KV cache (the stage-2
// algorithm adapted to GQA and cache layout): one block of 64 threads per
// (64-query tile, head); K/V tiles staged through shared memory, online
// softmax with running max/sum, no materialized score matrix. The query at
// row qi sits at absolute position pos0 + qi and attends to keys 0..pos0+qi.
// head_dim is fixed at 64 (q and acc live in registers).
//
// KIND selects how QKᵀ scores are computed:
//   0  fp32 cache, fp32 dot — the exact default (attn_prefill)
//   1  int8 cache, dp4a     — kv8: K stays int8 with its affine (scale, β)
//   2  fp32 cache, dp4a     — opt-in: K quantized on the fly, symmetric
// Both dp4a variants quantize q in registers (one absmax scale per (row, head))
// and reduce the 64-dim dot to 16 dp4a over an int8 K tile in shared memory
// (16 words/row vs 64 floats). V always stays fp32 in the tile, so the
// softmax-weighted accumulation is fp32 for every KIND. KIND 2 keeps the cache
// exact (decode still reads fp32 K) and only approximates the scores, paying a
// small per-tile K-requantization amortized over the 64 queries in the tile.
template <int KIND, bool PAGED>
__device__ void attn_prefill_body(float *out, const float *qkv,
                                  const float *kcache, const float *vcache,
                                  const signed char *kq, const signed char *vq,
                                  const float *ks, const float *vs,
                                  const float *kb, const float *vb,
                                  int pos0, int n_tok, int n_head, int n_kv_head,
                                  int qkv_stride, int out_stride,
                                  const int *bt, int bs) {
    constexpr bool DP4A = KIND != 0;     // QKᵀ via dp4a (else the exact fp32 dot)
    constexpr bool I8_CACHE = KIND == 1; // K/V sourced from the int8 cache
    __shared__ float Kt[KIND == 0 ? 64 : 1][64];
    __shared__ int Kq[KIND == 0 ? 1 : 64][16];
    __shared__ float Ks[KIND == 0 ? 1 : 64];
    __shared__ float Kb[KIND == 0 ? 1 : 64];
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
    int qw[16];
    float qs = 0.0f; // dequant scale of this thread's quantized q row
    int qsum = 0;    // Σ of this row's quantized q bytes (folds K's affine offset)
    float m = -CUDART_INF_F, l = 0.0f;
    if (active) {
        for (int d = 0; d < 64; ++d) q[d] = qkv[(size_t)qi * qkv_stride + h * 64 + d];
        if (DP4A) {
            float amax = 0.0f;
            for (int d = 0; d < 64; ++d) amax = fmaxf(amax, fabsf(q[d]));
            float id = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int w = 0; w < 16; ++w) {
                int packed = 0;
                for (int j = 0; j < 4; ++j) {
                    int v = max(-127, min(127, __float2int_rn(q[4 * w + j] * id)));
                    packed |= (v & 0xFF) << (8 * j);
                    qsum += v;
                }
                qw[w] = packed;
            }
            qs = amax > 0.0f ? amax / 127.0f : 0.0f;
        }
    }
    float scale = rsqrtf(64.0f);

    int max_key = pos0 + min(tile0 + 63, n_tok - 1);
    for (int kt = 0; kt <= max_key; kt += 64) {
        int tile_n = min(64, max_key - kt + 1);
        if (I8_CACHE) {
            const int *k32 = (const int *)kq;
            for (int x = tid; x < tile_n * 16; x += 64) {
                int r = x / 16, w = x % 16;
                Kq[r][w] = k32[(kv_row<PAGED>(bt, bs, kt + r) * kvd + kvh * 64) / 4 + w];
            }
            for (int r = tid; r < tile_n; r += 64) {
                Ks[r] = ks[kv_row<PAGED>(bt, bs, kt + r) * n_kv_head + kvh];
                Kb[r] = kb[kv_row<PAGED>(bt, bs, kt + r) * n_kv_head + kvh];
            }
        } else if (DP4A && tid < tile_n) {
            // KIND 2: quantize this K row to int8 on the fly (symmetric absmax).
            // Two light passes over the fp32 row so we never hold all 64 values
            // in registers next to q[]/acc[]; cost amortizes over 64 queries.
            const float *krow = kcache + kv_row<PAGED>(bt, bs, kt + tid) * kvd + kvh * 64;
            float amax = 0.0f;
            for (int d = 0; d < 64; ++d) amax = fmaxf(amax, fabsf(krow[d]));
            float id = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int w = 0; w < 16; ++w) {
                int packed = 0;
                for (int j = 0; j < 4; ++j) {
                    int v = max(-127, min(127, __float2int_rn(krow[4 * w + j] * id)));
                    packed |= (v & 0xFF) << (8 * j);
                }
                Kq[tid][w] = packed;
            }
            Ks[tid] = amax > 0.0f ? amax / 127.0f : 0.0f;
            Kb[tid] = 0.0f; // symmetric: K's affine offset is zero
        }
        for (int x = tid; x < tile_n * 64; x += 64) {
            int r = x / 64, d = x % 64;
            size_t pr = kv_row<PAGED>(bt, bs, kt + r);
            if (I8_CACHE) {
                // dequant + recover V's affine offset directly into the tile:
                // the softmax-weighted sum below then carries it for free.
                Vt[r][d] = (float)vq[pr * kvd + kvh * 64 + d] *
                               vs[pr * n_kv_head + kvh] +
                           vb[pr * n_kv_head + kvh];
            } else {
                if (KIND == 0) Kt[r][d] = kcache[pr * kvd + kvh * 64 + d];
                Vt[r][d] = vcache[pr * kvd + kvh * 64 + d];
            }
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < tile_n; ++j) {
                int kp = kt + j;
                if (kp > pq) break;
                float s;
                if (DP4A) {
                    int dot = 0;
#pragma unroll
                    for (int w = 0; w < 16; ++w) dot = __dp4a(qw[w], Kq[j][w], dot);
                    // q·k = qs·scale·(Ksⱼ·dot + Kβⱼ·Σq): dp4a term plus K's affine
                    // offset (Kβ = 0 for the symmetric KIND-2 path)
                    s = qs * scale * ((float)dot * Ks[j] + (float)qsum * Kb[j]);
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < 64; ++d) dot += q[d] * Kt[j][d];
                    s = dot * scale;
                }
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
    attn_prefill_body<0, false>(out, qkv, kcache, vcache, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr,
                                pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride,
                                nullptr, 0);
}

extern "C" __global__ void attn_prefill_q8(float *out, const float *qkv,
                                           const signed char *kq, const signed char *vq,
                                           const float *ks, const float *vs,
                                           const float *kb, const float *vb,
                                           int pos0, int n_tok, int n_head, int n_kv_head,
                                           int qkv_stride, int out_stride) {
    attn_prefill_body<1, false>(out, qkv, nullptr, nullptr, kq, vq, ks, vs, kb, vb,
                                pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride,
                                nullptr, 0);
}

// Paged prefill variants: K/V gathered through the block_table (bt, block size
// bs). Same KIND semantics (0 exact fp32, 1 int8 cache, 2 on-the-fly dp4a).
extern "C" __global__ void attn_prefill_paged(float *out, const float *qkv,
                                              const float *kcache, const float *vcache,
                                              int pos0, int n_tok, int n_head, int n_kv_head,
                                              int qkv_stride, int out_stride,
                                              const int *bt, int bs) {
    attn_prefill_body<0, true>(out, qkv, kcache, vcache, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr,
                               pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride, bt, bs);
}

extern "C" __global__ void attn_prefill_q8_paged(float *out, const float *qkv,
                                                 const signed char *kq, const signed char *vq,
                                                 const float *ks, const float *vs,
                                                 const float *kb, const float *vb,
                                                 int pos0, int n_tok, int n_head, int n_kv_head,
                                                 int qkv_stride, int out_stride,
                                                 const int *bt, int bs) {
    attn_prefill_body<1, true>(out, qkv, nullptr, nullptr, kq, vq, ks, vs, kb, vb,
                               pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride, bt, bs);
}

// Opt-in (--prefill-dp4a): same fp32 cache as attn_prefill, but QKᵀ scores run
// on dp4a — q and each K tile row are quantized to int8 on the fly. ~6-8%
// faster prefill at the cost of int8-approximate scores, so the exact default
// stays KIND 0 (decode also keeps reading fp32 K, so the cache is untouched).
extern "C" __global__ void attn_prefill_dp4a(float *out, const float *qkv,
                                             const float *kcache, const float *vcache,
                                             int pos0, int n_tok, int n_head, int n_kv_head,
                                             int qkv_stride, int out_stride) {
    attn_prefill_body<2, false>(out, qkv, kcache, vcache, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr,
                                pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride,
                                nullptr, 0);
}

extern "C" __global__ void attn_prefill_dp4a_paged(float *out, const float *qkv,
                                                   const float *kcache, const float *vcache,
                                                   int pos0, int n_tok, int n_head, int n_kv_head,
                                                   int qkv_stride, int out_stride,
                                                   const int *bt, int bs) {
    attn_prefill_body<2, true>(out, qkv, kcache, vcache, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr,
                               pos0, n_tok, n_head, n_kv_head, qkv_stride, out_stride, bt, bs);
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
