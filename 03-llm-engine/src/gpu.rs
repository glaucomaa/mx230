//! CUDA inference engine: decode runs one token at a time through GEMV
//! kernels with a per-layer KV cache; prompt prefill and speculative draft
//! verification batch tokens through GEMM + flash-style attention. Weights
//! are stored fp32, fp16 (fp32 math), or int8 per-output-channel.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::f16;

use crate::model::{Arch, Config, Model};

const LLM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/llm.ptx"));
const MAX_SPEC_TOKENS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMode {
    Fp32,
    Fp16,
    Int8,
}

impl WeightMode {
    pub fn parse(args: &[String]) -> Self {
        let fp16 = args.iter().any(|a| a == "--fp16");
        let int8 = args.iter().any(|a| a == "--int8");
        assert!(!(fp16 && int8), "choose only one of --fp16 or --int8");
        if int8 {
            WeightMode::Int8
        } else if fp16 {
            WeightMode::Fp16
        } else {
            WeightMode::Fp32
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WeightMode::Fp32 => "fp32",
            WeightMode::Fp16 => "fp16",
            WeightMode::Int8 => "int8",
        }
    }

    fn bytes_per_param(self) -> f64 {
        match self {
            WeightMode::Fp32 => 4.0,
            WeightMode::Fp16 => 2.0,
            WeightMode::Int8 => 1.0,
        }
    }
}

/// Approximate weight footprint on device for a given storage mode.
pub fn weight_mb(c: &Config, mode: WeightMode) -> f64 {
    let (e, inter) = (c.n_embd, c.n_inter);
    let mlp = match c.arch {
        Arch::Gpt2 => 2 * e * inter,
        Arch::Qwen2 => 3 * e * inter,
    };
    let per_layer = e * c.qkv_dim() + c.q_dim() * e + mlp;
    let wpe = match c.arch {
        Arch::Gpt2 => c.n_ctx * e,
        Arch::Qwen2 => 0,
    };
    let params = c.n_vocab * e + wpe + c.n_layer * per_layer;
    params as f64 * mode.bytes_per_param() / 1e6
}

impl fmt::Display for WeightMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Per-layer KV cache, either fp32 or int8 with one absmax scale per
/// (position, head). Quantization happens on write (quantize_kv kernel),
/// dequantization inside the attention kernel.
enum KvCache {
    F32 {
        k: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_embd]
        v: Vec<CudaSlice<f32>>,
    },
    Q8 {
        k: Vec<CudaSlice<i8>>,
        v: Vec<CudaSlice<i8>>,
        ks: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_head]
        vs: Vec<CudaSlice<f32>>,
    },
}

pub enum Weights {
    F32(CudaSlice<f32>),
    F16(CudaSlice<f16>),
    Int8 {
        q: CudaSlice<i8>,
        scales: CudaSlice<f32>,
    },
}

/// Per-output-channel absmax quantization of a [n_in, n_out] matrix.
fn quantize(w: &[f32], n_in: usize, n_out: usize) -> (Vec<i8>, Vec<f32>) {
    let mut scales = vec![0.0f32; n_out];
    for o in 0..n_out {
        let mut amax = 0.0f32;
        for i in 0..n_in {
            amax = amax.max(w[i * n_out + o].abs());
        }
        scales[o] = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    }
    let q = (0..n_in * n_out)
        .map(|idx| (w[idx] / scales[idx % n_out]).round().clamp(-127.0, 127.0) as i8)
        .collect();
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
    embed: CudaFunction,
    embed_half: CudaFunction,
    embed_int8: CudaFunction,
    embed_dyn: CudaFunction,
    embed_half_dyn: CudaFunction,
    embed_int8_dyn: CudaFunction,
    embed_batch: CudaFunction,
    embed_half_batch: CudaFunction,
    embed_int8_batch: CudaFunction,
    layernorm: CudaFunction,
    rmsnorm: CudaFunction,
    rope: CudaFunction,
    rope_dyn: CudaFunction,
    rope_batch: CudaFunction,
    silu_mul: CudaFunction,
    gemv: CudaFunction,
    gemv_half: CudaFunction,
    gemv_int8: CudaFunction,
    gemm_f32: CudaFunction,
    gemm_half: CudaFunction,
    gemm_int8: CudaFunction,
    gemm_f32_skinny: CudaFunction,
    gemm_half_skinny: CudaFunction,
    gemm_int8_skinny: CudaFunction,
    gemm_rows_f32: CudaFunction,
    gemm_rows_half: CudaFunction,
    gemm_rows_int8: CudaFunction,
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
            lb.arg(out).arg(x).arg(g).arg(b).arg(&n_i);
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
) {
    let (ni, no) = (n_in as i32, n_out as i32);
    let cfg = LaunchConfig {
        grid_dim: (n_out.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (n_in * 4) as u32,
    };
    match w {
        Weights::F32(w) => {
            let mut lb = stream.launch_builder(&k.gemv);
            lb.arg(y).arg(x).arg(w).arg(b).arg(&ni).arg(&no);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::F16(w) => {
            let mut lb = stream.launch_builder(&k.gemv_half);
            lb.arg(y).arg(x).arg(w).arg(b).arg(&ni).arg(&no);
            unsafe { lb.launch(cfg) }.unwrap();
        }
        Weights::Int8 { q, scales } => {
            let mut lb = stream.launch_builder(&k.gemv_int8);
            lb.arg(y).arg(x).arg(q).arg(scales).arg(b).arg(&ni).arg(&no);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
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
) {
    debug_assert!(kk.is_multiple_of(16), "gemm kernels assume K % 16 == 0");
    let (m_i, n_i, k_i) = (m as i32, n as i32, kk as i32);
    // Three tiers by M. Tiled GEMMs burn whole-tile FMAs regardless of M, so
    // draft-verify batches (M <= 8) go through gemm_rows — a multi-row GEMV
    // with zero wasted compute; 16-row tiles cover the mid range, 64-row
    // tiles the real prefill.
    let tier = if m <= 8 {
        0
    } else if m <= 16 {
        1
    } else {
        2
    };
    let cfg = if tier == 0 {
        let cols = if n.is_multiple_of(4) && n >= 4096 {
            n / 4
        } else {
            n
        };
        cfg1d(cols)
    } else {
        cfg_gemm(m, n, if tier == 1 { 16 } else { 64 })
    };
    match b {
        Weights::F32(w) => {
            let f = match tier {
                0 => &k.gemm_rows_f32,
                1 => &k.gemm_f32_skinny,
                _ => &k.gemm_f32,
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
                _ => &k.gemm_half,
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
            let f = match tier {
                0 => &k.gemm_rows_int8,
                1 => &k.gemm_int8_skinny,
                _ => &k.gemm_int8,
            };
            let mut lb = stream.launch_builder(f);
            lb.arg(c)
                .arg(a)
                .arg(q)
                .arg(scales)
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
            lb.arg(out).arg(x).arg(g).arg(b).arg(&rows_i).arg(&n_i);
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

pub struct Engine {
    pub config: Config,
    stream: Arc<CudaStream>,
    k: Kernels,
    wte_t: Weights, // [n_embd, n_vocab], transposed token embeddings (tied lm_head)
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
    graph_tok: CudaSlice<i32>,
    graph_pos: CudaSlice<i32>,
    decode_graph: Option<CudaGraph>,
}

impl Engine {
    pub fn new(ctx: &Arc<CudaContext>, model: &Model, mode: WeightMode, kv8: bool) -> Self {
        let c = model.config;
        let (e, v) = (c.n_embd, c.n_vocab);
        // This engine schedules all work on one stream. Disabling cudarc's
        // cross-stream event tracking keeps CUDA stream capture free of
        // external event dependencies.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let module = common::load_ptx(ctx, "llm", LLM_PTX).unwrap();
        let f = |name: &str| module.load_function(name).unwrap();
        let k = Kernels {
            embed: f("embed"),
            embed_half: f("embed_half"),
            embed_int8: f("embed_int8"),
            embed_dyn: f("embed_dyn"),
            embed_half_dyn: f("embed_half_dyn"),
            embed_int8_dyn: f("embed_int8_dyn"),
            embed_batch: f("embed_batch"),
            embed_half_batch: f("embed_half_batch"),
            embed_int8_batch: f("embed_int8_batch"),
            layernorm: f("layernorm"),
            rmsnorm: f("rmsnorm"),
            rope: f("rope"),
            rope_dyn: f("rope_dyn"),
            rope_batch: f("rope_batch"),
            silu_mul: f("silu_mul"),
            gemv: f("gemv"),
            gemv_half: f("gemv_half"),
            gemv_int8: f("gemv_int8"),
            gemm_f32: f("gemm_f32"),
            gemm_half: f("gemm_half"),
            gemm_int8: f("gemm_int8"),
            gemm_f32_skinny: f("gemm_f32_skinny"),
            gemm_half_skinny: f("gemm_half_skinny"),
            gemm_int8_skinny: f("gemm_int8_skinny"),
            gemm_rows_f32: f("gemm_rows_f32"),
            gemm_rows_half: f("gemm_rows_half"),
            gemm_rows_int8: f("gemm_rows_int8"),
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
        };

        let up = |t: &[f32]| stream.clone_htod(t).unwrap();
        let upw = |t: &[f32], n_in: usize, n_out: usize| -> Weights {
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
            }
        };

        // transpose wte [v, e] -> wte_t [e, v] so the lm_head GEMV is coalesced
        let mut wte_t = vec![0.0f32; e * v];
        for tok in 0..v {
            for i in 0..e {
                wte_t[i * v + tok] = model.wte[tok * e + i];
            }
        }

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
            .map(|l| LayerG {
                ln1_g: up(&l.ln1_g),
                ln1_b: opt(&l.ln1_b),
                qkv_w: upw(&l.qkv_w, e, qkvd),
                qkv_b: up(&l.qkv_b),
                proj_w: upw(&l.proj_w, qd, e),
                proj_b: up(&l.proj_b),
                ln2_g: up(&l.ln2_g),
                ln2_b: opt(&l.ln2_b),
                fc_w: upw(&l.fc_w, e, inter),
                fc_b: opt(&l.fc_b),
                up_w: if l.up_w.is_empty() {
                    None
                } else {
                    Some(upw(&l.up_w, e, inter))
                },
                fc2_w: upw(&l.fc2_w, inter, e),
                fc2_b: opt(&l.fc2_b),
            })
            .collect();

        Engine {
            config: c,
            k,
            wte_t: upw(&wte_t, e, v),
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
            );

            let (t_i, nh_i, nkv_i, hd_i) = (pos as i32, nh as i32, nkv as i32, hd as i32);
            if c.arch == Arch::Qwen2 {
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
                KvCache::Q8 { k, v, ks, vs } => {
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
                        .arg(&t_i)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
            }

            gemv(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.attn,
                &layer.proj_w,
                &layer.proj_b,
                qd,
                e,
            );
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);

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
                    );
                    let mut lb = self.stream.launch_builder(&self.k.silu_mul);
                    lb.arg(&mut self.h).arg(&self.h2).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
            }
            gemv(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.h,
                &layer.fc2_w,
                layer.fc2_b.as_ref().unwrap_or(&self.zero_bias),
                inter,
                e,
            );
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);
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
            &self.wte_t,
            &self.zero_bias,
            e,
            c.n_vocab,
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
            );

            let (nh_i, nkv_i, hd_i) = (nh as i32, nkv as i32, hd as i32);
            if c.arch == Arch::Qwen2 {
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
                KvCache::Q8 { k, v, ks, vs } => {
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
                        .arg(&self.graph_pos)
                        .arg(&nh_i)
                        .arg(&nkv_i)
                        .arg(&hd_i);
                    unsafe { lb.launch(attn_cfg) }.unwrap();
                }
            }

            gemv(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.attn,
                &layer.proj_w,
                &layer.proj_b,
                qd,
                e,
            );
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);

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
                    );
                    let mut lb = self.stream.launch_builder(&self.k.silu_mul);
                    lb.arg(&mut self.h).arg(&self.h2).arg(&n_i);
                    unsafe { lb.launch(cfg1d(inter)) }.unwrap();
                }
            }
            gemv(
                &self.stream,
                &self.k,
                &mut self.xb,
                &self.h,
                &layer.fc2_w,
                layer.fc2_b.as_ref().unwrap_or(&self.zero_bias),
                inter,
                e,
            );
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);
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
            &self.wte_t,
            &self.zero_bias,
            e,
            c.n_vocab,
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
            );

            if c.arch == Arch::Qwen2 {
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
                KvCache::Q8 { k, v, ks, vs } => {
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
            &self.wte_t,
            &self.zero_bias,
            e,
            c.n_vocab,
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
            &self.wte_t,
            &self.zero_bias,
            n,
            c.n_vocab,
            c.n_embd,
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
    pub fn generate(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        assert!(!prompt.is_empty());
        let mut logits = self.prefill(prompt, 0);
        let mut out = Vec::with_capacity(n_new);
        let mut pos = prompt.len();
        for _ in 0..n_new {
            let next = argmax(&logits);
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
