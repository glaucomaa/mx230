//! CUDA inference engine: decode runs one token at a time through GEMV
//! kernels with a per-layer KV cache; prompt prefill and speculative draft
//! verification batch tokens through GEMM + flash-style attention. Weights
//! are stored fp32, fp16 (fp32 math), or int8 per-output-channel.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use half::f16;

use crate::gptq;
use crate::model::{Arch, Config, Model};

const LLM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/llm.ptx"));
const LLM_AG8_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/llm_ag8.ptx"));
const MAX_SPEC_TOKENS: usize = 32;

/// True when the kernels were compiled (nvcc present at build time). Lets tests
/// skip gracefully instead of panicking inside `Engine::new` -> `load_ptx`.
#[allow(dead_code)] // used only by the integration test in main.rs
pub fn ptx_available() -> bool {
    !LLM_PTX.trim().is_empty() && !LLM_AG8_PTX.trim().is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    Fp32,
    Fp16,
    Int8,
    Int4,
    /// int4 with k-quants two-level scales (Q4_K-style): better quality than
    /// the Q4_0 `Int4` path, but the richer dequant costs ~20-30% decode and
    /// ~25% prefill on this 3-SM card — a separate, opt-in mode (`--int4k`).
    Int4K,
    Int3,
    Int2,
}

impl WeightMode {
    pub fn parse(args: &[String]) -> Self {
        let picked: Vec<WeightMode> = [
            ("--fp16", WeightMode::Fp16),
            ("--int8", WeightMode::Int8),
            ("--int4", WeightMode::Int4),
            ("--int4k", WeightMode::Int4K),
            ("--int3", WeightMode::Int3),
            ("--int2", WeightMode::Int2),
        ]
        .iter()
        .filter(|(f, _)| args.iter().any(|a| a == f))
        .map(|&(_, m)| m)
        .collect();
        assert!(
            picked.len() <= 1,
            "choose only one of --fp16, --int8, --int4, --int4k, --int3, --int2"
        );
        picked.first().copied().unwrap_or(WeightMode::Fp32)
    }

    pub fn label(self) -> &'static str {
        match self {
            WeightMode::Fp32 => "fp32",
            WeightMode::Fp16 => "fp16",
            WeightMode::Int8 => "int8",
            WeightMode::Int4 => "int4",
            WeightMode::Int4K => "int4k",
            WeightMode::Int3 => "int3",
            WeightMode::Int2 => "int2",
        }
    }

    fn bytes_per_param(self) -> f64 {
        match self {
            WeightMode::Fp32 => 4.0,
            WeightMode::Fp16 => 2.0,
            WeightMode::Int8 => 1.0,
            // packed nibble + one fp16 scale per 32-weight group
            WeightMode::Int4 => 0.5 + 2.0 / Q4_GROUP as f64,
            // packed nibble + k-quants two-level scales (one (sd, sm) byte
            // per 16-row sub-block, one fp16 (d, m) pair per 128-row super)
            WeightMode::Int4K => 0.5 + 1.0 / Q_SUB as f64 + 4.0 / Q_SUPER as f64,
            // three bits per weight (2-bit planes + hi-bit word) + k-quants
            // scales: one (sd, sm) byte per 16, one fp16 (d, m) per 128
            WeightMode::Int3 => 0.375 + 1.0 / Q_SUB as f64 + 4.0 / Q_SUPER as f64,
            // two bits per weight + the same two-level scales
            WeightMode::Int2 => 0.25 + 1.0 / Q_SUB as f64 + 4.0 / Q_SUPER as f64,
        }
    }

    /// Storage tier for embeddings and lm_head. Below int4 these tensors
    /// destroy quality far out of proportion to their bytes, so the low
    /// rungs keep them one tier up — the same move llama.cpp's Q2_K/Q3_K
    /// presets make (token_embd/output stay at Q4_K/Q6_K).
    fn embed_mode(self) -> WeightMode {
        match self {
            WeightMode::Int3 | WeightMode::Int2 => WeightMode::Int4,
            m => m,
        }
    }

    /// Storage tier for ffn_down (fc2): the second most damage-prone
    /// tensor in the Q2_K playbook, bumped one tier on the bottom rung.
    fn ffn_down_mode(self) -> WeightMode {
        match self {
            WeightMode::Int2 => WeightMode::Int4,
            m => m,
        }
    }
}

const Q4_GROUP: usize = 32;
/// k-quants-style two-level blocks for int3/int2 (see quantize_kq)
const Q_SUB: usize = 16;
const Q_SUPER: usize = 128;

/// Activation-quantization group size; must match `AG` of the kernel module
/// the engine loads. GPT-2 needs 4-wide groups (activation outliers wreck
/// wider absmax groups); the RoPE models run the AG=8 module and pay half
/// the scale-FMAs in the dp4a GEMMs.
fn act_group(arch: Arch) -> usize {
    match arch {
        Arch::Gpt2 => 4,
        Arch::Qwen2 | Arch::Llama => 8,
    }
}

/// Effective storage tiers for the embedding/lm_head and ffn-down tensors:
/// the usual `embed_mode`/`ffn_down_mode` policy, but bumped to int8 by the
/// mixed-precision flags when the body `mode` is sub-int8 (never a downgrade).
fn mixed_tiers(mode: WeightMode, embed_int8: bool, ffn_down_int8: bool) -> (WeightMode, WeightMode) {
    let sub_int8 = matches!(
        mode,
        WeightMode::Int4 | WeightMode::Int4K | WeightMode::Int3 | WeightMode::Int2
    );
    let embed = if embed_int8 && sub_int8 {
        WeightMode::Int8
    } else {
        mode.embed_mode()
    };
    let ffn = if ffn_down_int8 && sub_int8 {
        WeightMode::Int8
    } else {
        mode.ffn_down_mode()
    };
    (embed, ffn)
}

/// Approximate weight footprint on device for a given storage mode.
/// Embeddings/lm_head count at embed_mode and ffn_down at ffn_down_mode
/// (the low rungs keep those tensors one tier up), with the mixed-precision
/// flags optionally bumping them to int8.
pub fn weight_mb(c: &Config, mode: WeightMode, embed_int8: bool, ffn_down_int8: bool) -> f64 {
    let (e, inter) = (c.n_embd, c.n_inter);
    let (embed_tier, ffn_tier) = mixed_tiers(mode, embed_int8, ffn_down_int8);
    let mlp_up = match c.arch {
        Arch::Gpt2 => e * inter,
        Arch::Qwen2 | Arch::Llama => 2 * e * inter,
    };
    let per_layer = e * c.qkv_dim() + c.q_dim() * e + mlp_up;
    let wpe = match c.arch {
        Arch::Gpt2 => c.n_ctx * e,
        _ => 0,
    };
    let heads = match c.arch {
        Arch::Llama => 2, // untied lm_head
        _ => 1,
    };
    let embed = heads * c.n_vocab * e;
    let ffn_down = c.n_layer * inter * e;
    let body = wpe + c.n_layer * per_layer;
    (embed as f64 * embed_tier.bytes_per_param()
        + ffn_down as f64 * ffn_tier.bytes_per_param()
        + body as f64 * mode.bytes_per_param())
        / 1e6
}

impl fmt::Display for WeightMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Per-layer KV cache, either fp32 or int8 with a per-(position, head) affine
/// (scale, beta) pair — `x ~ scale * q + beta`, q signed so dp4a is unchanged.
/// Quantization happens on write (quantize_kv kernel), dequantization inside
/// the attention kernel (K's beta via the q-sum, V's via the softmax weights).
enum KvCache {
    F32 {
        k: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_embd]
        v: Vec<CudaSlice<f32>>,
    },
    Q8 {
        k: Vec<CudaSlice<i8>>,
        v: Vec<CudaSlice<i8>>,
        ks: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_kv_head] scales
        vs: Vec<CudaSlice<f32>>,
        kb: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_kv_head] offsets (betas)
        vb: Vec<CudaSlice<f32>>,
    },
}

pub enum Weights {
    F32(CudaSlice<f32>),
    F16(CudaSlice<f16>),
    Int8 {
        q: CudaSlice<i8>,
        scales: CudaSlice<f32>,
    },
    Int4 {
        q: CudaSlice<u8>,       // int32 words of 8 rows per column (see quantize_q4)
        scales: CudaSlice<f16>, // [(n_in/32), n_out]
        // GPTQ act-order: original input channel per stored position; the decode
        // GEMV gathers the activation by this before the dp4a. None => identity.
        perm: Option<CudaSlice<i32>>,
    },
    Int4K {
        q: CudaSlice<u8>,   // same nibble layout as Int4 (see quantize_q4k)
        sub: CudaSlice<u8>, // 4-bit (sd, sm) per (16-row sub-block, column)
        dm: CudaSlice<f16>, // fp16 (d, m) pair per (128-row super-block, column)
    },
    Int3 {
        q: CudaSlice<u8>,   // 3 int32 words per 32 rows per column (quantize_q3)
        sub: CudaSlice<u8>, // 4-bit (sd, sm) per (16-row sub-block, column)
        dm: CudaSlice<f16>, // fp16 (d, m) pair per (128-row super-block, column)
    },
    Int2 {
        q: CudaSlice<u8>,   // int32 words of 16 rows per column (see quantize_q2)
        sub: CudaSlice<u8>, // 4-bit (sd, sm) per (16-row sub-block, column)
        dm: CudaSlice<f16>, // fp16 (d, m) pair per (128-row super-block, column)
    },
}

/// Q4_0-style group quantization of a [n_in, n_out] matrix: per (32-row
/// group, column), scale = signed absmax / -8, nibbles store q+8 in [0, 15].
/// Packing matches the dp4a kernels: int32 word (i/8)*n_out + o holds rows
/// i..i+7 of column o, byte j carrying rows i+j (low nibble) and i+4+j
/// (high nibble) — both nibble planes line up with activation dp4a words.
fn quantize_q4(w: &[f32], n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f16>) {
    assert!(
        n_in.is_multiple_of(Q4_GROUP),
        "int4 needs n_in divisible by {Q4_GROUP}"
    );
    let n_groups = n_in / Q4_GROUP;
    let mut scales = vec![f16::ZERO; n_groups * n_out];
    let mut q = vec![0u8; n_in / 2 * n_out];
    for o in 0..n_out {
        for g in 0..n_groups {
            let mut m = 0.0f32; // value with the largest magnitude, sign kept
            for i in g * Q4_GROUP..(g + 1) * Q4_GROUP {
                let v = w[i * n_out + o];
                if v.abs() > m.abs() {
                    m = v;
                }
            }
            let d = m / -8.0;
            // fp16 rounding must match what the kernel dequantizes with
            let dh = f16::from_f32(d);
            scales[g * n_out + o] = dh;
            let id = if dh.to_f32() != 0.0 {
                1.0 / dh.to_f32()
            } else {
                0.0
            };
            for i in g * Q4_GROUP..(g + 1) * Q4_GROUP {
                let nib = ((w[i * n_out + o] * id).round() + 8.0).clamp(0.0, 15.0) as u8;
                q[((i / 8) * n_out + o) * 4 + (i % 4)] |= nib << (4 * ((i % 8) / 4));
            }
        }
    }
    (q, scales)
}

/// Q4_K-style two-level quantization (`--int4k`): the shared k-quants fit
/// (quantize_kq, qmax=15) gives asymmetric `w ~ d*q - m` per 16-row sub-block
/// with q unsigned in [0, 15], a 4-bit (sd, sm) byte per sub-block and one
/// fp16 (d_super, m_super) pair per 128-row super-block. The nibbles repack
/// into the SAME dp4a layout as quantize_q4 (no +8 bias — the kernel folds the
/// -m term via the activation sums), so the int4k kernels reuse q4_lo8/q4_hi8.
fn quantize_q4k(w: &[f32], n_in: usize, n_out: usize) -> (Vec<u8>, Vec<u8>, Vec<f16>) {
    let (qv, sub, dm) = quantize_kq(w, n_in, n_out, 15);
    let mut q = vec![0u8; n_in / 2 * n_out];
    for i in 0..n_in {
        for o in 0..n_out {
            let nib = qv[o * n_in + i]; // column-major from quantize_kq
            q[((i / 8) * n_out + o) * 4 + (i % 4)] |= nib << (4 * ((i % 8) / 4));
        }
    }
    (q, sub, dm)
}

/// Least-squares fit of w ~ d*q - m over one sub-block (q unsigned in
/// [0, qmax], m >= 0), k-quants style: try a grid of scale candidates,
/// round to q, then solve the 2x2 normal equations for (d, m) and keep
/// the candidate with the lowest squared error. The fit is uniform:
/// importance-weighting by |w| (llama.cpp's Q2_K choice) was tried and
/// cost TinyLlama int2 a 24x perplexity regression on this scheme.
fn fit_sub(x: &[f32], qmax: f32) -> (f32, f32) {
    let lo = x.iter().copied().fold(0.0f32, f32::min);
    let hi = x.iter().copied().fold(0.0f32, f32::max);
    if hi == lo {
        return (0.0, 0.0);
    }
    let n = x.len() as f32;
    let sumx: f32 = x.iter().sum();
    let mut best_dm = ((hi - lo) / qmax, -lo);
    let mut best_err = f32::INFINITY;
    for is in -9..=9 {
        let iscale = (qmax + 0.1 * is as f32) / (hi - lo);
        let (mut suml, mut suml2, mut sumlx) = (0.0f32, 0.0f32, 0.0f32);
        let qs: Vec<f32> = x
            .iter()
            .map(|&v| ((v - lo) * iscale).round().clamp(0.0, qmax))
            .collect();
        for (&v, &q) in x.iter().zip(&qs) {
            suml += q;
            suml2 += q * q;
            sumlx += q * v;
        }
        let det = n * suml2 - suml * suml;
        let (mut d, mut m) = if det > 0.0 {
            let d = (n * sumlx - sumx * suml) / det;
            (d, (d * suml - sumx) / n)
        } else {
            ((hi - lo) / qmax, -lo)
        };
        if m < 0.0 {
            // negative min can't be stored (4-bit unsigned): refit d alone
            m = 0.0;
            d = if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
        }
        let err: f32 = x
            .iter()
            .zip(&qs)
            .map(|(&v, &q)| (d * q - m - v) * (d * q - m - v))
            .sum();
        if err < best_err {
            best_err = err;
            best_dm = (d, m);
        }
    }
    best_dm
}

/// Two-level k-quants-style quantization shared by int3/int2: asymmetric
/// w ~ d*q - m per 16-row sub-block (q unsigned in [0, qmax]), with
/// d = d_super * sd and m = m_super * sm, sd/sm 4-bit packed in one byte
/// per (sub-block, column) (lo nibble = sd), and (d_super, m_super) one
/// fp16 pair per (128-row super-block, column). Returns the per-element
/// q values (column-major [n_out][n_in]) plus both scale planes.
fn quantize_kq(
    w: &[f32],
    n_in: usize,
    n_out: usize,
    qmax: u8,
) -> (Vec<u8>, Vec<u8>, Vec<f16>) {
    assert!(
        n_in.is_multiple_of(Q_SUPER),
        "int3/int2 need n_in divisible by {Q_SUPER}"
    );
    let (n_subs, n_supers) = (n_in / Q_SUB, n_in / Q_SUPER);
    // column-major worker outputs so the grid-search fit (the expensive
    // part) parallelizes over disjoint column chunks
    let mut qv_t = vec![0u8; n_in * n_out];
    let mut sub_t = vec![0u8; n_subs * n_out];
    let mut dm_t = vec![f16::ZERO; 2 * n_supers * n_out];
    let qmaxf = qmax as f32;
    let nthreads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cpc = n_out.div_ceil(nthreads); // columns per chunk
    std::thread::scope(|sc| {
        let chunks = qv_t
            .chunks_mut(cpc * n_in)
            .zip(sub_t.chunks_mut(cpc * n_subs))
            .zip(dm_t.chunks_mut(cpc * 2 * n_supers));
        for (ci, ((qc, sc_), dc)) in chunks.enumerate() {
            sc.spawn(move || {
                let mut col = vec![0.0f32; Q_SUB];
                for oc in 0..qc.len() / n_in {
                    let o = ci * cpc + oc;
                    for s in 0..n_supers {
                        let mut fits = [(0.0f32, 0.0f32); Q_SUPER / Q_SUB];
                        for (t, fit) in fits.iter_mut().enumerate() {
                            let row0 = s * Q_SUPER + t * Q_SUB;
                            for (j, c) in col.iter_mut().enumerate() {
                                *c = w[(row0 + j) * n_out + o];
                            }
                            *fit = fit_sub(&col, qmaxf);
                        }
                        let dmax = fits.iter().fold(0.0f32, |a, f| a.max(f.0));
                        let mmax = fits.iter().fold(0.0f32, |a, f| a.max(f.1));
                        let dsup = f16::from_f32(dmax / 15.0);
                        let msup = f16::from_f32(mmax / 15.0);
                        dc[(oc * n_supers + s) * 2] = dsup;
                        dc[(oc * n_supers + s) * 2 + 1] = msup;
                        let (dsf, msf) = (dsup.to_f32(), msup.to_f32());
                        for (t, fit) in fits.iter().enumerate() {
                            let sd = if dsf > 0.0 {
                                (fit.0 / dsf).round().clamp(0.0, 15.0) as u8
                            } else {
                                0
                            };
                            let sm = if msf > 0.0 {
                                (fit.1 / msf).round().clamp(0.0, 15.0) as u8
                            } else {
                                0
                            };
                            let row0 = s * Q_SUPER + t * Q_SUB;
                            sc_[oc * n_subs + row0 / Q_SUB] = sd | (sm << 4);
                            // re-quantize against the scales the kernel
                            // will actually use
                            let (d, m) = (dsf * sd as f32, msf * sm as f32);
                            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                            for j in 0..Q_SUB {
                                let i = row0 + j;
                                qc[oc * n_in + i] = ((w[i * n_out + o] + m) * id)
                                    .round()
                                    .clamp(0.0, qmaxf)
                                    as u8;
                            }
                        }
                    }
                }
            });
        }
    });
    // transpose back to the row-major [block, n_out] device layouts
    let mut sub = vec![0u8; n_subs * n_out];
    let mut dm = vec![f16::ZERO; 2 * n_supers * n_out];
    for o in 0..n_out {
        for t in 0..n_subs {
            sub[t * n_out + o] = sub_t[o * n_subs + t];
        }
        for s in 0..n_supers {
            dm[(s * n_out + o) * 2] = dm_t[(o * n_supers + s) * 2];
            dm[(s * n_out + o) * 2 + 1] = dm_t[(o * n_supers + s) * 2 + 1];
        }
    }
    (qv_t, sub, dm)
}


/// int3 packing: per-element q in [0, 7] from quantize_kq split into two
/// int2-style lo plane words plus one hi-bit word (byte i%4, bit (i%32)/4)
/// — word rows interleave as (i/32)*3 + j, matching the dp4a kernels'
/// q3_plane.
fn quantize_q3(w: &[f32], n_in: usize, n_out: usize) -> (Vec<u8>, Vec<u8>, Vec<f16>) {
    let (qv, sub, dm) = quantize_kq(w, n_in, n_out, 7);
    let mut q = vec![0u8; n_in / 32 * 3 * 4 * n_out];
    for i in 0..n_in {
        for o in 0..n_out {
            let v = qv[o * n_in + i];
            let base = (i / 32) * 3;
            q[((base + (i % 32) / 16) * n_out + o) * 4 + (i % 4)] |=
                (v & 3) << (2 * ((i % 16) / 4));
            q[((base + 2) * n_out + o) * 4 + (i % 4)] |= (v >> 2) << ((i % 32) / 4);
        }
    }
    (q, sub, dm)
}

/// int2 packing: per-element q in [0, 3] from quantize_kq into bit pairs.
/// int32 word (i/16)*n_out + o holds rows i..i+15 of column o, byte j
/// carrying rows i+j, i+4+j, i+8+j, i+12+j in successive bit pairs — each
/// 2-bit plane lines up with one activation dp4a word.
fn quantize_q2(w: &[f32], n_in: usize, n_out: usize) -> (Vec<u8>, Vec<u8>, Vec<f16>) {
    let (qv, sub, dm) = quantize_kq(w, n_in, n_out, 3);
    let mut q = vec![0u8; n_in / 4 * n_out];
    for i in 0..n_in {
        for o in 0..n_out {
            q[((i / 16) * n_out + o) * 4 + (i % 4)] |=
                qv[o * n_in + i] << (2 * ((i % 16) / 4));
        }
    }
    (q, sub, dm)
}

/// Per-output-channel absmax quantization of a [n_in, n_out] matrix, packed
/// for dp4a: int32 words of 4 consecutive n_in rows per column — byte
/// ((i/4)*n_out + o)*4 + (i%4) holds q[i, o]. Consecutive columns stay
/// consecutive, so coalescing matches the old row-major byte layout.
fn quantize(w: &[f32], n_in: usize, n_out: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(n_in.is_multiple_of(4), "int8 packing needs n_in % 4 == 0");
    let mut scales = vec![0.0f32; n_out];
    for o in 0..n_out {
        let mut amax = 0.0f32;
        for i in 0..n_in {
            amax = amax.max(w[i * n_out + o].abs());
        }
        scales[o] = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    }
    let mut q = vec![0i8; n_in * n_out];
    for i in 0..n_in {
        for o in 0..n_out {
            let v = (w[i * n_out + o] / scales[o]).round().clamp(-127.0, 127.0) as i8;
            q[((i / 4) * n_out + o) * 4 + (i % 4)] = v;
        }
    }
    (q, scales)
}

fn to_half(w: &[f32]) -> Vec<f16> {
    w.iter().copied().map(f16::from_f32).collect()
}

struct LayerG {
    ln1_g: CudaSlice<f32>,
    ln1_b: Option<CudaSlice<f32>>, // None for RMSNorm (Qwen2)
    qkv_w: Weights,
    qkv_b: CudaSlice<f32>,
    proj_w: Weights,
    proj_b: CudaSlice<f32>,
    ln2_g: CudaSlice<f32>,
    ln2_b: Option<CudaSlice<f32>>,
    fc_w: Weights, // GPT-2 fc | Qwen2 SwiGLU gate
    fc_b: Option<CudaSlice<f32>>,
    up_w: Option<Weights>, // Qwen2 SwiGLU up
    fc2_w: Weights,        // GPT-2 fc2 | Qwen2 SwiGLU down
    fc2_b: Option<CudaSlice<f32>>,
}

struct Kernels {
    /// activation-group width of the loaded module (AG in llm.cu)
    ag: usize,
    embed: CudaFunction,
    embed_half: CudaFunction,
    embed_int8: CudaFunction,
    embed_dyn: CudaFunction,
    embed_half_dyn: CudaFunction,
    embed_int8_dyn: CudaFunction,
    embed_int4: CudaFunction,
    embed_int4_dyn: CudaFunction,
    embed_batch: CudaFunction,
    embed_half_batch: CudaFunction,
    embed_int8_batch: CudaFunction,
    embed_int4_batch: CudaFunction,
    embed_int2: CudaFunction,
    embed_int2_dyn: CudaFunction,
    embed_int2_batch: CudaFunction,
    embed_int3: CudaFunction,
    embed_int3_dyn: CudaFunction,
    embed_int3_batch: CudaFunction,
    layernorm: CudaFunction,
    rmsnorm: CudaFunction,
    rope: CudaFunction,
    rope_dyn: CudaFunction,
    rope_batch: CudaFunction,
    silu_mul: CudaFunction,
    gemv: CudaFunction,
    gemv_half: CudaFunction,
    gemv_int8: CudaFunction,
    gemv_int4: CudaFunction,
    gemv_int2: CudaFunction,
    gemv_int3: CudaFunction,
    gemm_f32: CudaFunction,
    gemm_half: CudaFunction,
    gemm_f32_wide: CudaFunction,
    gemm_half_wide: CudaFunction,
    gemm_int8_wide: CudaFunction,
    gemm_int4_wide: CudaFunction,
    gemm_int8: CudaFunction,
    gemm_int4: CudaFunction,
    gemm_f32_skinny: CudaFunction,
    gemm_half_skinny: CudaFunction,
    gemm_int8_skinny: CudaFunction,
    gemm_int4_skinny: CudaFunction,
    gemm_rows_f32: CudaFunction,
    gemm_rows_half: CudaFunction,
    gemm_rows_int8: CudaFunction,
    gemm_rows_int4: CudaFunction,
    gemm_int2: CudaFunction,
    gemm_int2_skinny: CudaFunction,
    gemm_rows_int2: CudaFunction,
    gemm_int3: CudaFunction,
    gemm_int3_skinny: CudaFunction,
    gemm_rows_int3: CudaFunction,
    // int4k (Q4_K-style two-level scales) — parallel set to the int4 path
    embed_int4k: CudaFunction,
    embed_int4k_dyn: CudaFunction,
    embed_int4k_batch: CudaFunction,
    gemv_int4k: CudaFunction,
    gemm_int4k_wide: CudaFunction,
    gemm_int4k: CudaFunction,
    gemm_int4k_skinny: CudaFunction,
    gemm_rows_int4k: CudaFunction,
    quantize_act: CudaFunction,
    copy_kv_dyn: CudaFunction,
    copy_kv_batch: CudaFunction,
    quantize_kv: CudaFunction,
    quantize_kv_dyn: CudaFunction,
    quantize_kv_batch: CudaFunction,
    attn_decode: CudaFunction,
    attn_decode_dyn: CudaFunction,
    attn_decode_q8: CudaFunction,
    attn_decode_q8_dyn: CudaFunction,
    attn_prefill: CudaFunction,
    attn_prefill_q8: CudaFunction,
    layernorm_batch: CudaFunction,
    rmsnorm_batch: CudaFunction,
    add_inplace: CudaFunction,
    gelu_inplace: CudaFunction,
    argmax_advance: CudaFunction,
    argmax_rows: CudaFunction,
    copy_row: CudaFunction,
}

/// Load every kernel from a compiled module into a `Kernels` table. `ag` is
/// the module's activation-group width (must match `act_group` for the arch
/// whose PTX was loaded). Shared by `Engine::new_quant` and `kbench`.
fn load_kernels(module: &Arc<CudaModule>, ag: usize) -> Kernels {
    let f = |name: &str| module.load_function(name).unwrap();
    Kernels {
        ag,
        embed: f("embed"),
        embed_half: f("embed_half"),
        embed_int8: f("embed_int8"),
        embed_dyn: f("embed_dyn"),
        embed_half_dyn: f("embed_half_dyn"),
        embed_int8_dyn: f("embed_int8_dyn"),
        embed_int4: f("embed_int4"),
        embed_int4_dyn: f("embed_int4_dyn"),
        embed_batch: f("embed_batch"),
        embed_half_batch: f("embed_half_batch"),
        embed_int8_batch: f("embed_int8_batch"),
        embed_int4_batch: f("embed_int4_batch"),
        embed_int2: f("embed_int2"),
        embed_int2_dyn: f("embed_int2_dyn"),
        embed_int2_batch: f("embed_int2_batch"),
        embed_int3: f("embed_int3"),
        embed_int3_dyn: f("embed_int3_dyn"),
        embed_int3_batch: f("embed_int3_batch"),
        layernorm: f("layernorm"),
        rmsnorm: f("rmsnorm"),
        rope: f("rope"),
        rope_dyn: f("rope_dyn"),
        rope_batch: f("rope_batch"),
        silu_mul: f("silu_mul"),
        gemv: f("gemv"),
        gemv_half: f("gemv_half"),
        gemv_int8: f("gemv_int8"),
        gemv_int4: f("gemv_int4"),
        gemv_int2: f("gemv_int2"),
        gemv_int3: f("gemv_int3"),
        gemm_f32: f("gemm_f32"),
        gemm_half: f("gemm_half"),
        gemm_f32_wide: f("gemm_f32_wide"),
        gemm_half_wide: f("gemm_half_wide"),
        gemm_int8_wide: f("gemm_int8_wide"),
        gemm_int4_wide: f("gemm_int4_wide"),
        gemm_int8: f("gemm_int8"),
        gemm_int4: f("gemm_int4"),
        gemm_f32_skinny: f("gemm_f32_skinny"),
        gemm_half_skinny: f("gemm_half_skinny"),
        gemm_int8_skinny: f("gemm_int8_skinny"),
        gemm_int4_skinny: f("gemm_int4_skinny"),
        gemm_rows_f32: f("gemm_rows_f32"),
        gemm_rows_half: f("gemm_rows_half"),
        gemm_rows_int8: f("gemm_rows_int8"),
        gemm_rows_int4: f("gemm_rows_int4"),
        gemm_int2: f("gemm_int2"),
        gemm_int2_skinny: f("gemm_int2_skinny"),
        gemm_rows_int2: f("gemm_rows_int2"),
        gemm_int3: f("gemm_int3"),
        gemm_int3_skinny: f("gemm_int3_skinny"),
        gemm_rows_int3: f("gemm_rows_int3"),
        embed_int4k: f("embed_int4k"),
        embed_int4k_dyn: f("embed_int4k_dyn"),
        embed_int4k_batch: f("embed_int4k_batch"),
        gemv_int4k: f("gemv_int4k"),
        gemm_int4k_wide: f("gemm_int4k_wide"),
        gemm_int4k: f("gemm_int4k"),
        gemm_int4k_skinny: f("gemm_int4k_skinny"),
        gemm_rows_int4k: f("gemm_rows_int4k"),
        quantize_act: f("quantize_act"),
        copy_kv_dyn: f("copy_kv_dyn"),
        copy_kv_batch: f("copy_kv_batch"),
        quantize_kv: f("quantize_kv"),
        quantize_kv_dyn: f("quantize_kv_dyn"),
        quantize_kv_batch: f("quantize_kv_batch"),
        attn_decode: f("attn_decode"),
        attn_decode_dyn: f("attn_decode_dyn"),
        attn_decode_q8: f("attn_decode_q8"),
        attn_decode_q8_dyn: f("attn_decode_q8_dyn"),
        attn_prefill: f("attn_prefill"),
        attn_prefill_q8: f("attn_prefill_q8"),
        layernorm_batch: f("layernorm_batch"),
        rmsnorm_batch: f("rmsnorm_batch"),
        add_inplace: f("add_inplace"),
        gelu_inplace: f("gelu_inplace"),
        argmax_advance: f("argmax_advance"),
        argmax_rows: f("argmax_rows"),
        copy_row: f("copy_row"),
    }
}

fn cfg1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn cfg_gemm(m: usize, n: usize, bm: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(64) as u32, m.div_ceil(bm) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

fn cfg_gemm_wide(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// LayerNorm (bias present) or RMSNorm (bias None), one block.
#[allow(clippy::too_many_arguments)]
fn norm(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    out: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    b: Option<&CudaSlice<f32>>,
    n: usize,
    eps: f32,
) {
    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    match b {
        Some(b) => {
            let mut lb = stream.launch_builder(&k.layernorm);
            lb.arg(out).arg(x).arg(g).arg(b).arg(&n_i).arg(&eps);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        None => {
            let mut lb = stream.launch_builder(&k.rmsnorm);
            lb.arg(out).arg(x).arg(g).arg(&n_i).arg(&eps);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemv(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    y: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    w: &Weights,
    b: &CudaSlice<f32>,
    n_in: usize,
    n_out: usize,
    accum: bool,
) {
    let (ni, no) = (n_in as i32, n_out as i32);
    let acc = accum as i32;
    // fp32/fp16 stage x as floats; int8 quantizes x in-kernel into packed
    // int32 plus one scale per 8-value group (AG in llm.cu)
    // int8/int4 quantize x in-kernel: packed words + per-group scales
    // (+ group sums and per-32 correction sums for int4's nibble bias)
    let smem = match w {
        Weights::Int8 { .. } => n_in + n_in / k.ag * 4,
        Weights::Int4 { .. } => 3 * n_in + n_in / 8,
        // + one fp32 correction sum per 16-row sub-block
        Weights::Int4K { .. } | Weights::Int3 { .. } | Weights::Int2 { .. } => {
            3 * n_in + n_in / 4
        }
        _ => n_in * 4,
    };
    let cfg = LaunchConfig {
        grid_dim: (n_out.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    match w {
        Weights::F32(w) => {
            let mut lb = stream.launch_builder(&k.gemv);
            lb.arg(y).arg(x).arg(w).arg(b).arg(&ni).arg(&no).arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::F16(w) => {
            let mut lb = stream.launch_builder(&k.gemv_half);
            lb.arg(y).arg(x).arg(w).arg(b).arg(&ni).arg(&no).arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int8 { q, scales } => {
            let mut lb = stream.launch_builder(&k.gemv_int8);
            lb.arg(y)
                .arg(x)
                .arg(q)
                .arg(scales)
                .arg(b)
                .arg(&ni)
                .arg(&no)
                .arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int4 { q, scales, perm } => {
            let mut lb = stream.launch_builder(&k.gemv_int4);
            lb.arg(y)
                .arg(x)
                .arg(q)
                .arg(scales)
                .arg(b)
                .arg(&ni)
                .arg(&no)
                .arg(&acc);
            // GPTQ act-order permutation pointer, or a null pointer (a
            // pointer-width zero scalar) when there is none — gemv_int4 guards
            // on `perm != nullptr`.
            let null_perm = 0u64;
            match perm {
                Some(p) => lb.arg(p),
                None => lb.arg(&null_perm),
            };
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int4K { q, sub, dm } => {
            let mut lb = stream.launch_builder(&k.gemv_int4k);
            lb.arg(y)
                .arg(x)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(b)
                .arg(&ni)
                .arg(&no)
                .arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int3 { q, sub, dm } => {
            let mut lb = stream.launch_builder(&k.gemv_int3);
            lb.arg(y)
                .arg(x)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(b)
                .arg(&ni)
                .arg(&no)
                .arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int2 { q, sub, dm } => {
            let mut lb = stream.launch_builder(&k.gemv_int2);
            lb.arg(y)
                .arg(x)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(b)
                .arg(&ni)
                .arg(&no)
                .arg(&acc);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
}

/// Scratch for on-the-fly activation quantization (the dp4a GEMM path):
/// packed int32 rows, per-32-group absmax scales and group sums.
pub struct ActQuant {
    q: CudaSlice<i32>,
    scale: CudaSlice<f32>,
    sum: CudaSlice<i32>,
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    c: &mut CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &Weights,
    bias: &CudaSlice<f32>,
    m: usize,
    n: usize,
    kk: usize,
    act: &mut ActQuant,
) {
    debug_assert!(kk.is_multiple_of(16), "gemm kernels assume K % 16 == 0");
    let (m_i, n_i, k_i) = (m as i32, n as i32, kk as i32);
    // Four tiers by M. Tiled GEMMs burn whole-tile FMAs regardless of M, so
    // draft-verify batches (M <= 8) go through gemm_rows — a multi-row GEMV
    // with zero wasted compute; 16-row tiles cover the mid range, 64-row
    // tiles medium batches, and real prefill (M > 64) takes the 128x128
    // wide tier (fp32/fp16 only — int GEMMs stay on the 64-tile dp4a path).
    let tier = if m <= 8 {
        0
    } else if m <= 16 {
        1
    } else if m <= 64 {
        2
    } else {
        3
    };
    let cfg = match tier {
        0 => {
            let cols = if n.is_multiple_of(4) && n >= 4096 {
                n / 4
            } else {
                n
            };
            cfg1d(cols)
        }
        3 => cfg_gemm_wide(m, n),
        t => cfg_gemm(m, n, if t == 1 { 16 } else { 64 }),
    };
    match b {
        Weights::F32(w) => {
            let f = match tier {
                0 => &k.gemm_rows_f32,
                1 => &k.gemm_f32_skinny,
                2 => &k.gemm_f32,
                _ => &k.gemm_f32_wide,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(a)
                .arg(w)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::F16(w) => {
            let f = match tier {
                0 => &k.gemm_rows_half,
                1 => &k.gemm_half_skinny,
                2 => &k.gemm_half,
                _ => &k.gemm_half_wide,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(a)
                .arg(w)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int8 { q, scales } => {
            debug_assert!(kk.is_multiple_of(32), "dp4a gemm assumes K % 32 == 0");
            let groups = (m * kk / k.ag) as i32;
            let mut lb = stream.launch_builder(&k.quantize_act);
            lb.arg(&mut act.q)
                .arg(&mut act.scale)
                .arg(&mut act.sum)
                .arg(a)
                .arg(&groups);
            unsafe { lb.launch(cfg1d(groups as usize)) }.unwrap();

            let f = match tier {
                0 => &k.gemm_rows_int8,
                1 => &k.gemm_int8_skinny,
                2 => &k.gemm_int8,
                _ => &k.gemm_int8_wide,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(&act.q)
                .arg(&act.scale)
                .arg(q)
                .arg(scales)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int4 { q, scales, .. } => {
            debug_assert!(kk.is_multiple_of(32), "dp4a gemm assumes K % 32 == 0");
            let groups = (m * kk / k.ag) as i32;
            let mut lb = stream.launch_builder(&k.quantize_act);
            lb.arg(&mut act.q)
                .arg(&mut act.scale)
                .arg(&mut act.sum)
                .arg(a)
                .arg(&groups);
            unsafe { lb.launch(cfg1d(groups as usize)) }.unwrap();

            // int4 wide tile is 128x64 (full row height) — see gemm_int4_wide
            let cfg = if tier == 3 {
                LaunchConfig {
                    grid_dim: (n.div_ceil(64) as u32, m.div_ceil(128) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                cfg
            };
            let f = match tier {
                0 => &k.gemm_rows_int4,
                1 => &k.gemm_int4_skinny,
                2 => &k.gemm_int4,
                _ => &k.gemm_int4_wide,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(&act.q)
                .arg(&act.scale)
                .arg(q)
                .arg(scales)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int4K { q, sub, dm } => {
            debug_assert!(kk.is_multiple_of(32), "dp4a gemm assumes K % 32 == 0");
            let groups = (m * kk / k.ag) as i32;
            let mut lb = stream.launch_builder(&k.quantize_act);
            lb.arg(&mut act.q)
                .arg(&mut act.scale)
                .arg(&mut act.sum)
                .arg(a)
                .arg(&groups);
            unsafe { lb.launch(cfg1d(groups as usize)) }.unwrap();

            // int4k wide tile is 128x64 (full row height) — see gemm_int4k_wide
            let cfg = if tier == 3 {
                LaunchConfig {
                    grid_dim: (n.div_ceil(64) as u32, m.div_ceil(128) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                cfg
            };
            let f = match tier {
                0 => &k.gemm_rows_int4k,
                1 => &k.gemm_int4k_skinny,
                2 => &k.gemm_int4k,
                _ => &k.gemm_int4k_wide,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(&act.q)
                .arg(&act.scale)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int3 { q, sub, dm } => {
            debug_assert!(kk.is_multiple_of(32), "dp4a gemm assumes K % 32 == 0");
            let groups = (m * kk / k.ag) as i32;
            let mut lb = stream.launch_builder(&k.quantize_act);
            lb.arg(&mut act.q)
                .arg(&mut act.scale)
                .arg(&mut act.sum)
                .arg(a)
                .arg(&groups);
            unsafe { lb.launch(cfg1d(groups as usize)) }.unwrap();

            // ladder point like int2: no wide tile, 64-row tier covers tier 3
            let cfg = if tier == 3 { cfg_gemm(m, n, 64) } else { cfg };
            let f = match tier {
                0 => &k.gemm_rows_int3,
                1 => &k.gemm_int3_skinny,
                _ => &k.gemm_int3,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(&act.q)
                .arg(&act.scale)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int2 { q, sub, dm } => {
            debug_assert!(kk.is_multiple_of(32), "dp4a gemm assumes K % 32 == 0");
            let groups = (m * kk / k.ag) as i32;
            let mut lb = stream.launch_builder(&k.quantize_act);
            lb.arg(&mut act.q)
                .arg(&mut act.scale)
                .arg(&mut act.sum)
                .arg(a)
                .arg(&groups);
            unsafe { lb.launch(cfg1d(groups as usize)) }.unwrap();

            // int2 is a quality-ladder point, not a prefill record: no wide
            // tile, the 64-row dp4a tier covers tier 3 too
            let cfg = if tier == 3 { cfg_gemm(m, n, 64) } else { cfg };
            let f = match tier {
                0 => &k.gemm_rows_int2,
                1 => &k.gemm_int2_skinny,
                _ => &k.gemm_int2,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(&act.q)
                .arg(&act.scale)
                .arg(q)
                .arg(sub)
                .arg(dm)
                .arg(bias)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn norm_batch(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    out: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    b: Option<&CudaSlice<f32>>,
    rows: usize,
    n: usize,
    eps: f32,
) {
    let (rows_i, n_i) = (rows as i32, n as i32);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    match b {
        Some(b) => {
            let mut lb = stream.launch_builder(&k.layernorm_batch);
            lb.arg(out).arg(x).arg(g).arg(b).arg(&rows_i).arg(&n_i).arg(&eps);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        None => {
            let mut lb = stream.launch_builder(&k.rmsnorm_batch);
            lb.arg(out).arg(x).arg(g).arg(&rows_i).arg(&n_i).arg(&eps);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
}

fn add(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &mut CudaSlice<f32>,
    y: &CudaSlice<f32>,
    n: usize,
) {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(f);
    lb.arg(x).arg(y).arg(&n_i);
    unsafe { lb.launch(cfg1d(n)) }.unwrap();
}

/// One matmul shape benchmarked in isolation: decode is a one-token GEMV,
/// prefill a GEMM over `m_prefill` tokens. Each row mirrors a single `ggml`
/// `mul_mat`, so the numbers line up with llama.cpp
/// `test-backend-ops perf -o MUL_MAT` (their convention: m = n_out, n = tokens,
/// k = n_in). Decode is memory-bound (report GB/s of weight traffic); prefill
/// is compute-bound (report GFLOP/s).
pub struct KbenchRow {
    pub label: &'static str,
    pub k: usize, // n_in
    pub n: usize, // n_out
    pub decode_us: f32,
    pub decode_gbps: f32,
    pub prefill_us: Option<f32>,
    pub prefill_gflops: Option<f32>,
}

/// The distinct per-matmul shapes of `cfg`, labelled `(name, n_in, n_out,
/// bench_prefill)`. SwiGLU gate and up share a shape so they appear once;
/// `lm_head` is decode-only (real prefill projects logits for one position,
/// not the whole prompt, so a 512-row lm_head GEMM is not representative).
fn matmul_shapes(cfg: &Config) -> Vec<(&'static str, usize, usize, bool)> {
    let e = cfg.n_embd;
    let mut v = vec![
        ("qkv", e, cfg.qkv_dim(), true),
        ("attn_proj", cfg.q_dim(), e, true),
    ];
    let ffn_label = if cfg.arch == Arch::Gpt2 {
        "ffn_up"
    } else {
        "ffn_gate/up"
    };
    v.push((ffn_label, e, cfg.n_inter, true));
    v.push(("ffn_down", cfg.n_inter, e, true));
    v.push(("lm_head", e, cfg.n_vocab, false));
    v
}

/// Public view of `matmul_shapes` so callers can emit the matching llama.cpp
/// `test_mul_mat` perf cases for the same `(n_in, n_out)` set.
pub fn kbench_shapes(cfg: &Config) -> Vec<(&'static str, usize, usize, bool)> {
    matmul_shapes(cfg)
}

/// Bytes of weight storage one decode GEMV streams over a [n_in, n_out] matrix
/// — the memory-bound decode metric. int8: i8 weight + one f32 scale/column;
/// int4: packed nibbles + one f16 scale per 32-row group.
fn decode_weight_bytes(mode: WeightMode, n_in: usize, n_out: usize) -> usize {
    match mode {
        WeightMode::Int8 => n_in * n_out + n_out * 4,
        WeightMode::Int4 => n_in * n_out / 2 + (n_in / Q4_GROUP) * n_out * 2,
        _ => unreachable!("kbench is int8/int4 only"),
    }
}

/// A synthetic `Weights` of shape [n_in, n_out] in `mode`, quantized from
/// random fp32 with the production quantizers. dp4a timing is data-independent,
/// so random weights time identically to a real layer — and we avoid loading
/// (and quantizing) a whole multi-GB checkpoint just to bench one matmul.
fn synth_weights(
    stream: &Arc<CudaStream>,
    mode: WeightMode,
    n_in: usize,
    n_out: usize,
    seed: u64,
) -> Weights {
    let w = common::pseudo_rand(n_in * n_out, seed);
    match mode {
        WeightMode::Int8 => {
            let (q, s) = quantize(&w, n_in, n_out);
            Weights::Int8 {
                q: stream.clone_htod(&q).unwrap(),
                scales: stream.clone_htod(&s).unwrap(),
            }
        }
        WeightMode::Int4 => {
            let (q, s) = quantize_q4(&w, n_in, n_out);
            Weights::Int4 {
                q: stream.clone_htod(&q).unwrap(),
                scales: stream.clone_htod(&s).unwrap(),
                perm: None,
            }
        }
        _ => panic!("kbench supports --int8 and --int4 only"),
    }
}

/// Time each weight matmul of `cfg` in isolation (no tokenizer, sampling, host
/// loop, or kernel fusion) for one weight `mode` — the unit a kernel-vs-MMVQ/MMQ
/// comparison against llama.cpp lives on.
pub fn kbench(
    ctx: &Arc<CudaContext>,
    cfg: &Config,
    mode: WeightMode,
    m_prefill: usize,
) -> Vec<KbenchRow> {
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream().unwrap();
    let ag = act_group(cfg.arch);
    let module = if ag == 8 {
        common::load_ptx(ctx, "llm_ag8", LLM_AG8_PTX).unwrap()
    } else {
        common::load_ptx(ctx, "llm", LLM_PTX).unwrap()
    };
    let k = load_kernels(&module, ag);

    let shapes = matmul_shapes(cfg);
    // activation-quant scratch sized for the widest prefill K (one int32 word
    // per 4 activations, one scale/sum per AG-value group)
    let max_k = shapes
        .iter()
        .filter(|s| s.3)
        .map(|s| s.1)
        .max()
        .unwrap_or(cfg.n_embd);
    let mut act = ActQuant {
        q: stream.alloc_zeros(m_prefill * max_k / 4).unwrap(),
        scale: stream.alloc_zeros(m_prefill * max_k / ag).unwrap(),
        sum: stream.alloc_zeros(m_prefill * max_k / ag).unwrap(),
    };

    type R = Result<(), cudarc::driver::DriverError>;
    let mut rows = Vec::new();
    for (label, ki, ni, do_prefill) in shapes {
        let w = synth_weights(&stream, mode, ki, ni, 0x9E37_79B9 ^ ni as u64);
        let bias = stream.alloc_zeros::<f32>(ni).unwrap();

        // decode: M = 1 GEMV
        let x = stream.clone_htod(&common::pseudo_rand(ki, 7)).unwrap();
        let mut y = stream.alloc_zeros::<f32>(ni).unwrap();
        let ms = common::time_median_ms(&stream, 20, 100, || {
            gemv(&stream, &k, &mut y, &x, &w, &bias, ki, ni, false);
            R::Ok(())
        })
        .unwrap();
        let bytes = decode_weight_bytes(mode, ki, ni);
        let decode_us = ms * 1e3_f32;
        let decode_gbps = bytes as f32 / (ms * 1e-3_f32) / 1e9_f32;

        let (mut prefill_us, mut prefill_gflops) = (None, None);
        if do_prefill {
            let a = stream
                .clone_htod(&common::pseudo_rand(m_prefill * ki, 9))
                .unwrap();
            let mut c = stream.alloc_zeros::<f32>(m_prefill * ni).unwrap();
            let ms = common::time_median_ms(&stream, 5, 30, || {
                gemm(&stream, &k, &mut c, &a, &w, &bias, m_prefill, ni, ki, &mut act);
                R::Ok(())
            })
            .unwrap();
            prefill_us = Some(ms * 1e3_f32);
            let flop = 2.0 * m_prefill as f32 * ni as f32 * ki as f32;
            prefill_gflops = Some(flop / (ms * 1e-3_f32) / 1e9_f32);
        }
        rows.push(KbenchRow {
            label,
            k: ki,
            n: ni,
            decode_us,
            decode_gbps,
            prefill_us,
            prefill_gflops,
        });
    }
    rows
}

pub struct Engine {
    pub config: Config,
    stream: Arc<CudaStream>,
    k: Kernels,
    wte_t: Weights, // [n_embd, n_vocab], transposed token embeddings (tied lm_head)
    lm_head_t: Option<Weights>, // untied lm_head (Llama), same layout as wte_t
    wpe: CudaSlice<f32>,
    layers: Vec<LayerG>,
    lnf_g: CudaSlice<f32>,
    lnf_b: Option<CudaSlice<f32>>,
    kv: KvCache,
    // scratch buffers
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    h: CudaSlice<f32>,
    h2: CudaSlice<f32>,        // SwiGLU up-branch scratch (Qwen2)
    zero_bias: CudaSlice<f32>, // for the bias-free lm_head GEMV
    logits: CudaSlice<f32>,
    batch_tok: CudaSlice<i32>,
    batch_x: CudaSlice<f32>,
    batch_xb: CudaSlice<f32>,
    batch_qkv: CudaSlice<f32>,
    batch_attn: CudaSlice<f32>,
    batch_h: CudaSlice<f32>,
    batch_h2: CudaSlice<f32>,
    batch_logits: CudaSlice<f32>,
    batch_argmax: CudaSlice<i32>,
    act: ActQuant,
    graph_tok: CudaSlice<i32>,
    graph_pos: CudaSlice<i32>,
    decode_graph: Option<CudaGraph>,
    // GPTQ act-order stores weights in a permuted channel order that only the
    // decode GEMV (gemv_int4) gathers correctly; the batch-prefill GEMM does
    // not, so prefill falls back to the per-token decode loop when this is set.
    gptq_act_order: bool,
}

impl Engine {
    pub fn new(ctx: &Arc<CudaContext>, model: &Model, mode: WeightMode, kv8: bool) -> Self {
        Self::new_quant(ctx, model, mode, kv8, None, false, false)
    }

    /// Like `new`, but with the quantization knobs:
    /// - `gptq`: when supplied, the covered linears are uploaded from the
    ///   sidecar's pre-quantized Q4_0 blobs instead of round-to-nearest (Int4
    ///   mode only; uncovered tensors take the normal path).
    /// - `embed_int8` / `ffn_down_int8`: mixed precision — keep the embedding /
    ///   lm_head, resp. the per-layer ffn-down, at int8 while the body stays at
    ///   a sub-int8 `mode`. The int4 logits projection through `wte_t` is where
    ///   most of int4's perplexity damage lives, so an int8 `wte_t` recovers it.
    #[allow(clippy::too_many_arguments)]
    pub fn new_quant(
        ctx: &Arc<CudaContext>,
        model: &Model,
        mode: WeightMode,
        kv8: bool,
        gptq: Option<&gptq::Sidecar>,
        embed_int8: bool,
        ffn_down_int8: bool,
    ) -> Self {
        let c = model.config;
        let (embed_tier, ffn_tier) = mixed_tiers(mode, embed_int8, ffn_down_int8);
        // The decode-attention kernels keep the per-position score row in a
        // fixed `__shared__ float s[2048]`; n_ctx beyond that silently corrupts
        // neighbouring shared memory, so refuse it up front.
        assert!(
            c.n_ctx <= 2048,
            "decode-attn scratch s[2048] overflow: n_ctx={}",
            c.n_ctx
        );
        let (e, v) = (c.n_embd, c.n_vocab);
        // This engine schedules all work on one stream. Disabling cudarc's
        // cross-stream event tracking keeps CUDA stream capture free of
        // external event dependencies.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let ag = act_group(c.arch);
        let module = if ag == 8 {
            common::load_ptx(ctx, "llm_ag8", LLM_AG8_PTX).unwrap()
        } else {
            common::load_ptx(ctx, "llm", LLM_PTX).unwrap()
        };
        let k = load_kernels(&module, ag);

        let up = |t: &[f32]| stream.clone_htod(t).unwrap();
        let upw_as = |t: &[f32], n_in: usize, n_out: usize, mode: WeightMode| -> Weights {
            match mode {
                WeightMode::Fp32 => Weights::F32(up(t)),
                WeightMode::Fp16 => Weights::F16(stream.clone_htod(&to_half(t)).unwrap()),
                WeightMode::Int8 => {
                    let (q, s) = quantize(t, n_in, n_out);
                    Weights::Int8 {
                        q: stream.clone_htod(&q).unwrap(),
                        scales: up(&s),
                    }
                }
                WeightMode::Int4 => {
                    let (q, s) = quantize_q4(t, n_in, n_out);
                    Weights::Int4 {
                        q: stream.clone_htod(&q).unwrap(),
                        scales: stream.clone_htod(&s).unwrap(),
                        perm: None,
                    }
                }
                WeightMode::Int4K => {
                    let (q, sub, dm) = quantize_q4k(t, n_in, n_out);
                    Weights::Int4K {
                        q: stream.clone_htod(&q).unwrap(),
                        sub: stream.clone_htod(&sub).unwrap(),
                        dm: stream.clone_htod(&dm).unwrap(),
                    }
                }
                WeightMode::Int3 => {
                    let (q, sub, dm) = quantize_q3(t, n_in, n_out);
                    Weights::Int3 {
                        q: stream.clone_htod(&q).unwrap(),
                        sub: stream.clone_htod(&sub).unwrap(),
                        dm: stream.clone_htod(&dm).unwrap(),
                    }
                }
                WeightMode::Int2 => {
                    let (q, sub, dm) = quantize_q2(t, n_in, n_out);
                    Weights::Int2 {
                        q: stream.clone_htod(&q).unwrap(),
                        sub: stream.clone_htod(&sub).unwrap(),
                        dm: stream.clone_htod(&dm).unwrap(),
                    }
                }
            }
        };
        // GPTQ override: use the sidecar's pre-quantized Q4_0 blob for a covered
        // (layer, role) tensor, else fall back to the normal upload at `fb_mode`.
        let gptq_or = |li: usize,
                       role: gptq::Role,
                       t: &[f32],
                       n_in: usize,
                       n_out: usize,
                       fb_mode: WeightMode|
         -> Weights {
            if let Some(e) = gptq.and_then(|sc| sc.get(li, role)) {
                // act-order perm is identity when the sidecar was built without
                // it; upload it only when it actually reorders (Some => the
                // gemv gathers the activation, None => plain contiguous path)
                let is_identity = e.perm.iter().enumerate().all(|(k, &p)| p as usize == k);
                Weights::Int4 {
                    q: stream.clone_htod(&e.q).unwrap(),
                    scales: stream.clone_htod(&e.scales).unwrap(),
                    perm: if is_identity {
                        None
                    } else {
                        Some(stream.clone_htod(&e.perm).unwrap())
                    },
                }
            } else {
                upw_as(t, n_in, n_out, fb_mode)
            }
        };

        // transpose [v, e] -> [e, v] so the lm_head GEMV is coalesced
        let transpose_ve = |w: &[f32]| {
            let mut t = vec![0.0f32; e * v];
            for tok in 0..v {
                for i in 0..e {
                    t[i * v + tok] = w[tok * e + i];
                }
            }
            t
        };
        let wte_t = transpose_ve(&model.wte);

        let opt = |t: &[f32]| -> Option<CudaSlice<f32>> {
            if t.is_empty() {
                None
            } else {
                Some(up(t))
            }
        };
        let (qd, qkvd, inter) = (c.q_dim(), c.qkv_dim(), c.n_inter);
        let layers = model
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| LayerG {
                ln1_g: up(&l.ln1_g),
                ln1_b: opt(&l.ln1_b),
                qkv_w: gptq_or(li, gptq::Role::Qkv, &l.qkv_w, e, qkvd, mode),
                qkv_b: up(&l.qkv_b),
                proj_w: gptq_or(li, gptq::Role::Proj, &l.proj_w, qd, e, mode),
                proj_b: up(&l.proj_b),
                ln2_g: up(&l.ln2_g),
                ln2_b: opt(&l.ln2_b),
                fc_w: gptq_or(li, gptq::Role::Fc, &l.fc_w, e, inter, mode),
                fc_b: opt(&l.fc_b),
                up_w: if l.up_w.is_empty() {
                    None
                } else {
                    Some(gptq_or(li, gptq::Role::Up, &l.up_w, e, inter, mode))
                },
                fc2_w: gptq_or(li, gptq::Role::Fc2, &l.fc2_w, inter, e, ffn_tier),
                fc2_b: opt(&l.fc2_b),
            })
            .collect();

        let gptq_act_order = gptq.is_some_and(|sc| sc.has_act_order());

        Engine {
            config: c,
            k,
            gptq_act_order,
            // embeddings/lm_head stay one tier up on the low rungs (embed_mode),
            // or int8 under --embed-int8 — see embed_tier above
            wte_t: upw_as(&wte_t, e, v, embed_tier),
            lm_head_t: if model.lm_head.is_empty() {
                None
            } else {
                Some(upw_as(&transpose_ve(&model.lm_head), e, v, embed_tier))
            },
            // RoPE models have no learned positions; a zero table keeps the
            // embed kernels uniform across archs
            wpe: if model.wpe.is_empty() {
                stream.alloc_zeros(c.n_ctx * e).unwrap()
            } else {
                up(&model.wpe)
            },
            layers,
            lnf_g: up(&model.lnf_g),
            lnf_b: opt(&model.lnf_b),
            kv: if kv8 {
                KvCache::Q8 {
                    k: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.kv_dim()).unwrap())
                        .collect(),
                    v: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.kv_dim()).unwrap())
                        .collect(),
                    ks: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.n_kv_head).unwrap())
                        .collect(),
                    vs: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.n_kv_head).unwrap())
                        .collect(),
                    kb: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.n_kv_head).unwrap())
                        .collect(),
                    vb: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.n_kv_head).unwrap())
                        .collect(),
                }
            } else {
                KvCache::F32 {
                    k: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.kv_dim()).unwrap())
                        .collect(),
                    v: (0..c.n_layer)
                        .map(|_| stream.alloc_zeros(c.n_ctx * c.kv_dim()).unwrap())
                        .collect(),
                }
            },
            x: stream.alloc_zeros(e).unwrap(),
            xb: stream.alloc_zeros(e).unwrap(),
            qkv: stream.alloc_zeros(qkvd).unwrap(),
            attn: stream.alloc_zeros(qd).unwrap(),
            h: stream.alloc_zeros(inter).unwrap(),
            h2: stream.alloc_zeros(inter).unwrap(),
            zero_bias: stream.alloc_zeros(v).unwrap(),
            logits: stream.alloc_zeros(v).unwrap(),
            batch_tok: stream.alloc_zeros(c.n_ctx).unwrap(),
            batch_x: stream.alloc_zeros(c.n_ctx * e).unwrap(),
            batch_xb: stream.alloc_zeros(c.n_ctx * e).unwrap(),
            batch_qkv: stream.alloc_zeros(c.n_ctx * qkvd).unwrap(),
            batch_attn: stream.alloc_zeros(c.n_ctx * qd).unwrap(),
            batch_h: stream.alloc_zeros(c.n_ctx * inter).unwrap(),
            batch_h2: stream.alloc_zeros(c.n_ctx * inter).unwrap(),
            batch_logits: stream.alloc_zeros(MAX_SPEC_TOKENS * v).unwrap(),
            batch_argmax: stream.alloc_zeros(MAX_SPEC_TOKENS).unwrap(),
            // activation-quant scratch sized for the widest K (n_inter),
            // one scale/sum per AG-value group of the loaded module
            act: ActQuant {
                q: stream.alloc_zeros(c.n_ctx * inter / 4).unwrap(),
                scale: stream.alloc_zeros(c.n_ctx * inter / ag).unwrap(),
                sum: stream.alloc_zeros(c.n_ctx * inter / ag).unwrap(),
            },
            graph_tok: stream.alloc_zeros(1).unwrap(),
            graph_pos: stream.alloc_zeros(1).unwrap(),
            decode_graph: None,
            stream,
        }
    }

    fn launch_embed(&mut self, tok: i32, pos: i32) {
        let c = self.config;
        let (e_i, v_i) = (c.n_embd as i32, c.n_vocab as i32);
        match &self.wte_t {
            Weights::F32(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed);
                lb.arg(&mut self.x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::F16(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed_half);
                lb.arg(&mut self.x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int8 { q, scales } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int8);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int4 { q, scales, .. } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int4K { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4k);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int3 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int3);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int2 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int2);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&tok)
                    .arg(&pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
        }
    }

    fn launch_embed_dyn(&mut self) {
        let c = self.config;
        let (e_i, v_i) = (c.n_embd as i32, c.n_vocab as i32);
        match &self.wte_t {
            Weights::F32(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed_dyn);
                lb.arg(&mut self.x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::F16(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed_half_dyn);
                lb.arg(&mut self.x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int8 { q, scales } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int8_dyn);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int4 { q, scales, .. } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4_dyn);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int4K { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4k_dyn);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int3 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int3_dyn);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
            Weights::Int2 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int2_dyn);
                lb.arg(&mut self.x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.graph_tok)
                    .arg(&self.graph_pos)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(c.n_embd)) }.unwrap();
            }
        }
    }

    fn forward_body(&mut self, pos: usize) {
        let c = self.config;
        let e = c.n_embd;
        let (qd, kvd, qkvd, inter) = (c.q_dim(), c.kv_dim(), c.qkv_dim(), c.n_inter);
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let eps = c.norm_eps;
        for l in 0..c.n_layer {
            let layer = &self.layers[l];

            norm(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.x,
                &layer.ln1_g,
                layer.ln1_b.as_ref(),
                e,
                eps,
            );
            gemv(
                &self.stream,
                &self.k,
                &mut self.qkv,
                &self.xb,
                &layer.qkv_w,
                &layer.qkv_b,
                e,
                qkvd,
                false,
            );

            let (t_i, nh_i, nkv_i, hd_i) = (pos as i32, nh as i32, nkv as i32, hd as i32);
            if c.arch != Arch::Gpt2 {
                let mut lb = self.stream.launch_builder(&self.k.rope);
                lb.arg(&mut self.qkv)
                    .arg(&t_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&hd_i)
                    .arg(&c.rope_theta);
                unsafe { lb.launch(cfg1d((nh + nkv) * hd / 2)) }.unwrap();
            }

            let attn_cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            match &mut self.kv {
                KvCache::F32 { k, v } => {
                    self.stream
                        .memcpy_dtod(
                            &self.qkv.slice(qd..qd + kvd),
                            &mut k[l].slice_mut(pos * kvd..(pos + 1) * kvd),
                        )
                        .unwrap();
                    self.stream
                        .memcpy_dtod(
                            &self.qkv.slice(qd + kvd..qkvd),
                            &mut v[l].slice_mut(pos * kvd..(pos + 1) * kvd),
                        )
                        .unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_decode);
                    lb.arg(&mut self.attn)
                        .arg(&self.qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&t_i)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
                KvCache::Q8 { k, v, ks, vs, kb, vb } => {
                    let qd_i = qd as i32;
                    let q_cfg = LaunchConfig {
                        grid_dim: (nkv as u32, 1, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut lb = self.stream.launch_builder(&self.k.quantize_kv);
                    lb.arg(&mut k[l])
                        .arg(&mut v[l])
                        .arg(&mut ks[l])
                        .arg(&mut vs[l])
                        .arg(&mut kb[l])
                        .arg(&mut vb[l])
                        .arg(&self.qkv)
                        .arg(&t_i)
                        .arg(&qd_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(q_cfg) }.unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_decode_q8);
                    lb.arg(&mut self.attn)
                        .arg(&self.qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&ks[l])
                        .arg(&vs[l])
                        .arg(&kb[l])
                        .arg(&vb[l])
                        .arg(&t_i)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
            }

            // residual add fused into the projection epilogue: one launch
            // and no xb round-trip
            gemv(
                &self.stream,
                &self.k,
                &mut self.x,
                &self.attn,
                &layer.proj_w,
                &layer.proj_b,
                qd,
                e,
                true,
            );

            norm(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.x,
                &layer.ln2_g,
                layer.ln2_b.as_ref(),
                e,
                eps,
            );
            gemv(
                &self.stream,
                &self.k,
                &mut self.h,
                &self.xb,
                &layer.fc_w,
                layer.fc_b.as_ref().unwrap_or(&self.zero_bias),
                e,
                inter,
                false,
            );
            let n_i = inter as i32;
            match &layer.up_w {
                None => {
                    let mut lb = self.stream.launch_builder(&self.k.gelu_inplace);
                    lb.arg(&mut self.h).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
                Some(up_w) => {
                    gemv(
                        &self.stream,
                        &self.k,
                        &mut self.h2,
                        &self.xb,
                        up_w,
                        &self.zero_bias,
                        e,
                        inter,
                        false,
                    );
                    let mut lb = self.stream.launch_builder(&self.k.silu_mul);
                    lb.arg(&mut self.h).arg(&self.h2).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
            }
            gemv(
                &self.stream,
                &self.k,
                &mut self.x,
                &self.h,
                &layer.fc2_w,
                layer.fc2_b.as_ref().unwrap_or(&self.zero_bias),
                inter,
                e,
                true,
            );
        }

        norm(
            &self.stream,
            &self.k,
            &mut self.xb,
            &self.x,
            &self.lnf_g,
            self.lnf_b.as_ref(),
            e,
            eps,
        );
        gemv(
            &self.stream,
            &self.k,
            &mut self.logits,
            &self.xb,
            self.lm_head_t.as_ref().unwrap_or(&self.wte_t),
            &self.zero_bias,
            e,
            c.n_vocab,
            false,
        );
    }

    fn forward_body_dyn(&mut self) {
        let c = self.config;
        let e = c.n_embd;
        let (qd, kvd, qkvd, inter) = (c.q_dim(), c.kv_dim(), c.qkv_dim(), c.n_inter);
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let eps = c.norm_eps;
        for l in 0..c.n_layer {
            let layer = &self.layers[l];

            norm(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.x,
                &layer.ln1_g,
                layer.ln1_b.as_ref(),
                e,
                eps,
            );
            gemv(
                &self.stream,
                &self.k,
                &mut self.qkv,
                &self.xb,
                &layer.qkv_w,
                &layer.qkv_b,
                e,
                qkvd,
                false,
            );

            let (nh_i, nkv_i, hd_i) = (nh as i32, nkv as i32, hd as i32);
            if c.arch != Arch::Gpt2 {
                let mut lb = self.stream.launch_builder(&self.k.rope_dyn);
                lb.arg(&mut self.qkv)
                    .arg(&self.graph_pos)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&hd_i)
                    .arg(&c.rope_theta);
                unsafe { lb.launch(cfg1d((nh + nkv) * hd / 2)) }.unwrap();
            }

            let attn_cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            match &mut self.kv {
                KvCache::F32 { k, v } => {
                    let (qd_i, kvd_i) = (qd as i32, kvd as i32);
                    let mut lb = self.stream.launch_builder(&self.k.copy_kv_dyn);
                    lb.arg(&mut k[l])
                        .arg(&mut v[l])
                        .arg(&self.qkv)
                        .arg(&self.graph_pos)
                        .arg(&qd_i)
                        .arg(&kvd_i);
                    unsafe { lb.launch(cfg1d(kvd)) }.unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_decode_dyn);
                    lb.arg(&mut self.attn)
                        .arg(&self.qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&self.graph_pos)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
                KvCache::Q8 { k, v, ks, vs, kb, vb } => {
                    let qd_i = qd as i32;
                    let q_cfg = LaunchConfig {
                        grid_dim: (nkv as u32, 1, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut lb = self.stream.launch_builder(&self.k.quantize_kv_dyn);
                    lb.arg(&mut k[l])
                        .arg(&mut v[l])
                        .arg(&mut ks[l])
                        .arg(&mut vs[l])
                        .arg(&mut kb[l])
                        .arg(&mut vb[l])
                        .arg(&self.qkv)
                        .arg(&self.graph_pos)
                        .arg(&qd_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(q_cfg) }.unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_decode_q8_dyn);
                    lb.arg(&mut self.attn)
                        .arg(&self.qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&ks[l])
                        .arg(&vs[l])
                        .arg(&kb[l])
                        .arg(&vb[l])
                        .arg(&self.graph_pos)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
            }

            // residual add fused into the projection epilogue: one launch
            // and no xb round-trip
            gemv(
                &self.stream,
                &self.k,
                &mut self.x,
                &self.attn,
                &layer.proj_w,
                &layer.proj_b,
                qd,
                e,
                true,
            );

            norm(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.x,
                &layer.ln2_g,
                layer.ln2_b.as_ref(),
                e,
                eps,
            );
            gemv(
                &self.stream,
                &self.k,
                &mut self.h,
                &self.xb,
                &layer.fc_w,
                layer.fc_b.as_ref().unwrap_or(&self.zero_bias),
                e,
                inter,
                false,
            );
            let n_i = inter as i32;
            match &layer.up_w {
                None => {
                    let mut lb = self.stream.launch_builder(&self.k.gelu_inplace);
                    lb.arg(&mut self.h).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
                Some(up_w) => {
                    gemv(
                        &self.stream,
                        &self.k,
                        &mut self.h2,
                        &self.xb,
                        up_w,
                        &self.zero_bias,
                        e,
                        inter,
                        false,
                    );
                    let mut lb = self.stream.launch_builder(&self.k.silu_mul);
                    lb.arg(&mut self.h).arg(&self.h2).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
            }
            gemv(
                &self.stream,
                &self.k,
                &mut self.x,
                &self.h,
                &layer.fc2_w,
                layer.fc2_b.as_ref().unwrap_or(&self.zero_bias),
                inter,
                e,
                true,
            );
        }

        norm(
            &self.stream,
            &self.k,
            &mut self.xb,
            &self.x,
            &self.lnf_g,
            self.lnf_b.as_ref(),
            e,
            eps,
        );
        gemv(
            &self.stream,
            &self.k,
            &mut self.logits,
            &self.xb,
            self.lm_head_t.as_ref().unwrap_or(&self.wte_t),
            &self.zero_bias,
            e,
            c.n_vocab,
            false,
        );
    }

    /// Runs one token through the model, returns logits on the host.
    pub fn forward(&mut self, tok: u32, pos: usize) -> Vec<f32> {
        assert!(pos < self.config.n_ctx, "context overflow");
        self.launch_embed(tok as i32, pos as i32);
        self.forward_body(pos);
        self.stream.clone_dtoh(&self.logits).unwrap()
    }

    fn launch_embed_batch(&mut self, tokens: &[u32], pos0: usize) {
        let c = self.config;
        let n = tokens.len();
        let host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.stream
            .memcpy_htod(&host, &mut self.batch_tok.slice_mut(0..n))
            .unwrap();
        let (pos_i, n_i, e_i, v_i) = (pos0 as i32, n as i32, c.n_embd as i32, c.n_vocab as i32);
        match &self.wte_t {
            Weights::F32(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed_batch);
                lb.arg(&mut self.batch_x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::F16(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed_half_batch);
                lb.arg(&mut self.batch_x)
                    .arg(w)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::Int8 { q, scales } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int8_batch);
                lb.arg(&mut self.batch_x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::Int4 { q, scales, .. } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4_batch);
                lb.arg(&mut self.batch_x)
                    .arg(q)
                    .arg(scales)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::Int4K { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int4k_batch);
                lb.arg(&mut self.batch_x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::Int3 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int3_batch);
                lb.arg(&mut self.batch_x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
            Weights::Int2 { q, sub, dm } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int2_batch);
                lb.arg(&mut self.batch_x)
                    .arg(q)
                    .arg(sub)
                    .arg(dm)
                    .arg(&self.wpe)
                    .arg(&self.batch_tok)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&e_i)
                    .arg(&v_i);
                unsafe { lb.launch(cfg1d(n * c.n_embd)) }.unwrap();
            }
        }
    }

    /// Batched causal trunk shared by prefill and speculative verification:
    /// embed -> layers -> final norm. Leaves the normalized hidden states for
    /// all rows in `batch_xb`.
    fn batch_body(&mut self, tokens: &[u32], pos0: usize) {
        assert!(!tokens.is_empty());
        assert!(pos0 + tokens.len() <= self.config.n_ctx, "context overflow");
        assert!(
            self.config.head_dim == 64,
            "prefill kernels assume head_dim=64"
        );

        let c = self.config;
        let n = tokens.len();
        let e = c.n_embd;
        let (qd, kvd, qkvd, inter) = (c.q_dim(), c.kv_dim(), c.qkv_dim(), c.n_inter);
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let eps = c.norm_eps;
        self.launch_embed_batch(tokens, pos0);

        for l in 0..c.n_layer {
            let layer = &self.layers[l];
            norm_batch(
                &self.stream,
                &self.k,
                &mut self.batch_xb,
                &self.batch_x,
                &layer.ln1_g,
                layer.ln1_b.as_ref(),
                n,
                e,
                eps,
            );
            gemm(
                &self.stream,
                &self.k,
                &mut self.batch_qkv,
                &self.batch_xb,
                &layer.qkv_w,
                &layer.qkv_b,
                n,
                qkvd,
                e,
                &mut self.act,
            );

            if c.arch != Arch::Gpt2 {
                let (pos_i, n_i, nh_i, nkv_i, hd_i, stride_i) = (
                    pos0 as i32,
                    n as i32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    qkvd as i32,
                );
                let mut lb = self.stream.launch_builder(&self.k.rope_batch);
                lb.arg(&mut self.batch_qkv)
                    .arg(&pos_i)
                    .arg(&n_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&hd_i)
                    .arg(&stride_i)
                    .arg(&c.rope_theta);
                unsafe { lb.launch(cfg1d(n * (nh + nkv) * hd / 2)) }.unwrap();
            }

            let (pos_i, n_i, nh_i, nkv_i, qd_i, kvd_i, qkvd_i) = (
                pos0 as i32,
                n as i32,
                nh as i32,
                nkv as i32,
                qd as i32,
                kvd as i32,
                qkvd as i32,
            );
            let attn_cfg = LaunchConfig {
                grid_dim: (nh as u32, n.div_ceil(64) as u32, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            match &mut self.kv {
                KvCache::F32 { k, v } => {
                    let mut lb = self.stream.launch_builder(&self.k.copy_kv_batch);
                    lb.arg(&mut k[l])
                        .arg(&mut v[l])
                        .arg(&self.batch_qkv)
                        .arg(&pos_i)
                        .arg(&qd_i)
                        .arg(&kvd_i)
                        .arg(&qkvd_i);
                    let copy_cfg = LaunchConfig {
                        grid_dim: (kvd.div_ceil(256) as u32, n as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { lb.launch(copy_cfg) }.unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_prefill);
                    lb.arg(&mut self.batch_attn)
                        .arg(&self.batch_qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&pos_i)
                        .arg(&n_i)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&qkvd_i)
                        .arg(&qd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
                KvCache::Q8 { k, v, ks, vs, kb, vb } => {
                    let q_cfg = LaunchConfig {
                        grid_dim: (nkv as u32, n as u32, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let hd_i = hd as i32;
                    let mut lb = self.stream.launch_builder(&self.k.quantize_kv_batch);
                    lb.arg(&mut k[l])
                        .arg(&mut v[l])
                        .arg(&mut ks[l])
                        .arg(&mut vs[l])
                        .arg(&mut kb[l])
                        .arg(&mut vb[l])
                        .arg(&self.batch_qkv)
                        .arg(&pos_i)
                        .arg(&qd_i)
                        .arg(&nkv_i)
                        .arg(&hd_i)
                        .arg(&qkvd_i);
                    unsafe { lb.launch(q_cfg) }.unwrap();

                    let mut lb = self.stream.launch_builder(&self.k.attn_prefill_q8);
                    lb.arg(&mut self.batch_attn)
                        .arg(&self.batch_qkv)
                        .arg(&k[l])
                        .arg(&v[l])
                        .arg(&ks[l])
                        .arg(&vs[l])
                        .arg(&kb[l])
                        .arg(&vb[l])
                        .arg(&pos_i)
                        .arg(&n_i)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&qkvd_i)
                        .arg(&qd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
            }

            gemm(
                &self.stream,
                &self.k,
                &mut self.batch_xb,
                &self.batch_attn,
                &layer.proj_w,
                &layer.proj_b,
                n,
                e,
                qd,
                &mut self.act,
            );
            add(
                &self.stream,
                &self.k.add_inplace,
                &mut self.batch_x,
                &self.batch_xb,
                n * e,
            );

            norm_batch(
                &self.stream,
                &self.k,
                &mut self.batch_xb,
                &self.batch_x,
                &layer.ln2_g,
                layer.ln2_b.as_ref(),
                n,
                e,
                eps,
            );
            gemm(
                &self.stream,
                &self.k,
                &mut self.batch_h,
                &self.batch_xb,
                &layer.fc_w,
                layer.fc_b.as_ref().unwrap_or(&self.zero_bias),
                n,
                inter,
                e,
                &mut self.act,
            );
            let total_i = (n * inter) as i32;
            match &layer.up_w {
                None => {
                    let mut lb = self.stream.launch_builder(&self.k.gelu_inplace);
                    lb.arg(&mut self.batch_h).arg(&total_i);
                    unsafe { lb.launch(cfg1d(n * inter)) }.unwrap();
                }
                Some(up_w) => {
                    gemm(
                        &self.stream,
                        &self.k,
                        &mut self.batch_h2,
                        &self.batch_xb,
                        up_w,
                        &self.zero_bias,
                        n,
                        inter,
                        e,
                        &mut self.act,
                    );
                    let mut lb = self.stream.launch_builder(&self.k.silu_mul);
                    lb.arg(&mut self.batch_h).arg(&self.batch_h2).arg(&total_i);
                    unsafe { lb.launch(cfg1d(n * inter)) }.unwrap();
                }
            }
            gemm(
                &self.stream,
                &self.k,
                &mut self.batch_xb,
                &self.batch_h,
                &layer.fc2_w,
                layer.fc2_b.as_ref().unwrap_or(&self.zero_bias),
                n,
                e,
                inter,
                &mut self.act,
            );
            add(
                &self.stream,
                &self.k.add_inplace,
                &mut self.batch_x,
                &self.batch_xb,
                n * e,
            );
        }

        norm_batch(
            &self.stream,
            &self.k,
            &mut self.batch_xb,
            &self.batch_x,
            &self.lnf_g,
            self.lnf_b.as_ref(),
            n,
            e,
            eps,
        );
    }

    /// Batched prompt prefill; returns the final row's logits on the host.
    pub fn prefill(&mut self, tokens: &[u32], pos0: usize) -> Vec<f32> {
        if tokens.len() == 1 {
            return self.forward(tokens[0], pos0);
        }
        // GPTQ act-order weights are permuted in a way only the decode GEMV
        // gathers correctly; the batch GEMM would mis-multiply, so prefill
        // token-by-token through the (correct) decode path instead.
        if self.gptq_act_order {
            let mut logits = Vec::new();
            for (i, &t) in tokens.iter().enumerate() {
                logits = self.forward(t, pos0 + i);
            }
            return logits;
        }
        self.batch_body(tokens, pos0);
        let c = self.config;
        let (n, e) = (tokens.len(), c.n_embd);

        let row_i = (n - 1) as i32;
        let e_i = e as i32;
        let mut lb = self.stream.launch_builder(&self.k.copy_row);
        lb.arg(&mut self.xb)
            .arg(&self.batch_xb)
            .arg(&row_i)
            .arg(&e_i);
        unsafe { lb.launch(cfg1d(e)) }.unwrap();
        gemv(
            &self.stream,
            &self.k,
            &mut self.logits,
            &self.xb,
            self.lm_head_t.as_ref().unwrap_or(&self.wte_t),
            &self.zero_bias,
            e,
            c.n_vocab,
            false,
        );
        self.stream.clone_dtoh(&self.logits).unwrap()
    }

    /// Speculative verification: per-row greedy argmax over the lm_head
    /// logits, computed on device. Only the n token ids cross the bus; the
    /// n x n_vocab logits never leave the GPU.
    pub fn verify_argmax(&mut self, tokens: &[u32], pos0: usize) -> Vec<u32> {
        assert!(
            tokens.len() <= MAX_SPEC_TOKENS,
            "speculative verify supports at most {MAX_SPEC_TOKENS} tokens"
        );
        self.batch_body(tokens, pos0);
        let c = self.config;
        let n = tokens.len();

        gemm(
            &self.stream,
            &self.k,
            &mut self.batch_logits,
            &self.batch_xb,
            self.lm_head_t.as_ref().unwrap_or(&self.wte_t),
            &self.zero_bias,
            n,
            c.n_vocab,
            c.n_embd,
            &mut self.act,
        );
        let v_i = c.n_vocab as i32;
        let mut lb = self.stream.launch_builder(&self.k.argmax_rows);
        lb.arg(&mut self.batch_argmax)
            .arg(&self.batch_logits)
            .arg(&v_i);
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.unwrap();
        self.stream
            .clone_dtoh(&self.batch_argmax.slice(0..n))
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect()
    }

    fn capture_decode_graph(&mut self) {
        if self.decode_graph.is_some() {
            return;
        }
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        self.launch_embed_dyn();
        self.forward_body_dyn();
        let v_i = self.config.n_vocab as i32;
        let mut lb = self.stream.launch_builder(&self.k.argmax_advance);
        lb.arg(&mut self.graph_tok)
            .arg(&mut self.graph_pos)
            .arg(&self.logits)
            .arg(&v_i);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.unwrap();
        let graph = self
            .stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .expect("stream capture produced no graph");
        graph.upload().unwrap();
        self.decode_graph = Some(graph);
    }

    pub fn prepare_decode_graph(&mut self) {
        self.capture_decode_graph();
    }

    /// Replays a captured one-token decode graph. The graph keeps token and
    /// position on device, so the host submits one graph launch per token and
    /// does not copy logits back between steps.
    pub fn graph_decode(&mut self, first_tok: u32, pos: usize, n_steps: usize) -> u32 {
        assert!(pos + n_steps <= self.config.n_ctx, "context overflow");
        self.stream
            .memcpy_htod(&[first_tok as i32], &mut self.graph_tok)
            .unwrap();
        self.stream
            .memcpy_htod(&[pos as i32], &mut self.graph_pos)
            .unwrap();
        self.capture_decode_graph();
        for _ in 0..n_steps {
            self.decode_graph.as_ref().unwrap().launch().unwrap();
        }
        self.stream.synchronize().unwrap();
        self.stream.clone_dtoh(&self.graph_tok).unwrap()[0] as u32
    }

    /// Greedy generation; returns only the newly generated token ids.
    /// Autoregressive decode. `sampler` selects each token from the logits;
    /// pass `Sampler::greedy()` for argmax (bit-identical to the old behaviour).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        sampler: &mut crate::sample::Sampler,
        mut on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        assert!(!prompt.is_empty());
        let mut logits = self.prefill(prompt, 0);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        for _ in 0..n_new {
            let next = sampler.pick(&logits);
            out.push(next);
            on_token(next);
            logits = self.forward(next, pos);
            pos += 1;
        }
        out
    }

    /// Lossless prompt-lookup speculative decoding. Candidate tokens are copied
    /// from earlier n-gram matches in the prompt/generated history and accepted
    /// only when the full model's greedy verification agrees.
    pub fn generate_speculative(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        spec_k: usize,
        on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        assert!(!prompt.is_empty());
        let logits = self.prefill(prompt, 0);
        self.speculative_loop(prompt, argmax(&logits), n_new, spec_k, on_token)
    }

    /// Decode part of speculative generation, for callers that have already
    /// prefilled `prompt` into the KV cache: `first` is the greedy token the
    /// prefill predicted. Logits stay on the GPU throughout — the host only
    /// ever sees argmax token ids (one per verified row).
    pub fn speculative_loop(
        &mut self,
        prompt: &[u32],
        first: u32,
        n_new: usize,
        spec_k: usize,
        mut on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        assert!(spec_k > 0);
        assert!(
            prompt.len() + n_new <= self.config.n_ctx,
            "context overflow"
        );
        let mut history = prompt.to_vec();
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        let mut greedy = first;
        while out.len() < n_new {
            let room = (n_new - out.len()).min(MAX_SPEC_TOKENS);
            let draft = prompt_lookup(&history, spec_k.min(room));
            if draft.first().copied() != Some(greedy) {
                out.push(greedy);
                history.push(greedy);
                on_token(greedy);
                let logits = self.forward(greedy, pos);
                pos += 1;
                greedy = argmax(&logits);
                continue;
            }
            let row_argmax = self.verify_argmax(&draft, pos);
            let mut accepted = 1usize;
            while accepted < draft.len() && row_argmax[accepted - 1] == draft[accepted] {
                accepted += 1;
            }
            for &tok in &draft[..accepted] {
                out.push(tok);
                history.push(tok);
                on_token(tok);
            }
            pos += accepted;
            // row `accepted-1` predicts the token after the last accepted one:
            // the rejection's correction (or the bonus token when all passed)
            greedy = row_argmax[accepted - 1];
        }
        out
    }
}

/// First-index argmax — ties break the same way as the device argmax kernels
/// (`argmax_rows`, `argmax_advance`), so host and GPU greedy paths agree.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn prompt_lookup(history: &[u32], max_tokens: usize) -> Vec<u32> {
    if max_tokens == 0 || history.len() < 3 {
        return Vec::new();
    }
    let max_ngram = 8.min(history.len() / 2);
    for ngram in (2..=max_ngram).rev() {
        let cur = history.len() - ngram;
        let suffix = &history[cur..];
        for i in (0..cur).rev() {
            if &history[i..i + ngram] == suffix {
                let start = i + ngram;
                let end = (start + max_tokens).min(history.len());
                if start < end {
                    return history[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}
