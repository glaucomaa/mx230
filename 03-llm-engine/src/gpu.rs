//! CUDA inference engine: one token at a time through GEMV kernels with a
//! per-layer KV cache. Weights are fp32 or int8 (per-output-channel absmax),
//! chosen at load time — int8 cuts decode memory traffic ~4x, which on a
//! 40 GB/s bus translates almost directly into tokens/sec.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::model::{Config, Model};

const LLM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/llm.ptx"));

pub enum Weights {
    F32(CudaSlice<f32>),
    Int8 { q: CudaSlice<i8>, scales: CudaSlice<f32> },
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

struct LayerG {
    ln1_g: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    qkv_w: Weights,
    qkv_b: CudaSlice<f32>,
    proj_w: Weights,
    proj_b: CudaSlice<f32>,
    ln2_g: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
    fc_w: Weights,
    fc_b: CudaSlice<f32>,
    fc2_w: Weights,
    fc2_b: CudaSlice<f32>,
}

struct Kernels {
    embed: CudaFunction,
    embed_int8: CudaFunction,
    layernorm: CudaFunction,
    gemv: CudaFunction,
    gemv_int8: CudaFunction,
    attn_decode: CudaFunction,
    add_inplace: CudaFunction,
    gelu_inplace: CudaFunction,
}

fn cfg1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn layernorm(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    out: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    g: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
) {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(f);
    lb.arg(out).arg(x).arg(g).arg(b).arg(&n_i);
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { lb.launch(cfg) }.unwrap();
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
        Weights::Int8 { q, scales } => {
            let mut lb = stream.launch_builder(&k.gemv_int8);
            lb.arg(y).arg(x).arg(q).arg(scales).arg(b).arg(&ni).arg(&no);
            unsafe { lb.launch(cfg) }.unwrap();
        }
    }
}

fn add(stream: &Arc<CudaStream>, f: &CudaFunction, x: &mut CudaSlice<f32>, y: &CudaSlice<f32>, n: usize) {
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
    lnf_b: CudaSlice<f32>,
    kcache: Vec<CudaSlice<f32>>, // per layer: [n_ctx * n_embd]
    vcache: Vec<CudaSlice<f32>>,
    // scratch buffers
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    h: CudaSlice<f32>,
    zero_bias: CudaSlice<f32>, // for the bias-free lm_head GEMV
    logits: CudaSlice<f32>,
}

impl Engine {
    pub fn new(ctx: &Arc<CudaContext>, model: &Model, int8: bool) -> Self {
        let c = model.config;
        let (e, v) = (c.n_embd, c.n_vocab);
        let stream = ctx.default_stream();
        let module = common::load_ptx(ctx, "llm", LLM_PTX).unwrap();
        let f = |name: &str| module.load_function(name).unwrap();
        let k = Kernels {
            embed: f("embed"),
            embed_int8: f("embed_int8"),
            layernorm: f("layernorm"),
            gemv: f("gemv"),
            gemv_int8: f("gemv_int8"),
            attn_decode: f("attn_decode"),
            add_inplace: f("add_inplace"),
            gelu_inplace: f("gelu_inplace"),
        };

        let up = |t: &[f32]| stream.clone_htod(t).unwrap();
        let upw = |t: &[f32], n_in: usize, n_out: usize| -> Weights {
            if int8 {
                let (q, s) = quantize(t, n_in, n_out);
                Weights::Int8 { q: stream.clone_htod(&q).unwrap(), scales: up(&s) }
            } else {
                Weights::F32(up(t))
            }
        };

        // transpose wte [v, e] -> wte_t [e, v] so the lm_head GEMV is coalesced
        let mut wte_t = vec![0.0f32; e * v];
        for tok in 0..v {
            for i in 0..e {
                wte_t[i * v + tok] = model.wte[tok * e + i];
            }
        }

        let layers = model
            .layers
            .iter()
            .map(|l| LayerG {
                ln1_g: up(&l.ln1_g),
                ln1_b: up(&l.ln1_b),
                qkv_w: upw(&l.qkv_w, e, 3 * e),
                qkv_b: up(&l.qkv_b),
                proj_w: upw(&l.proj_w, e, e),
                proj_b: up(&l.proj_b),
                ln2_g: up(&l.ln2_g),
                ln2_b: up(&l.ln2_b),
                fc_w: upw(&l.fc_w, e, 4 * e),
                fc_b: up(&l.fc_b),
                fc2_w: upw(&l.fc2_w, 4 * e, e),
                fc2_b: up(&l.fc2_b),
            })
            .collect();

        Engine {
            config: c,
            k,
            wte_t: upw(&wte_t, e, v),
            wpe: up(&model.wpe),
            layers,
            lnf_g: up(&model.lnf_g),
            lnf_b: up(&model.lnf_b),
            kcache: (0..c.n_layer).map(|_| stream.alloc_zeros(c.n_ctx * e).unwrap()).collect(),
            vcache: (0..c.n_layer).map(|_| stream.alloc_zeros(c.n_ctx * e).unwrap()).collect(),
            x: stream.alloc_zeros(e).unwrap(),
            xb: stream.alloc_zeros(e).unwrap(),
            qkv: stream.alloc_zeros(3 * e).unwrap(),
            attn: stream.alloc_zeros(e).unwrap(),
            h: stream.alloc_zeros(4 * e).unwrap(),
            zero_bias: stream.alloc_zeros(v).unwrap(),
            logits: stream.alloc_zeros(v).unwrap(),
            stream,
        }
    }

    /// Runs one token through the model, returns logits on the host.
    pub fn forward(&mut self, tok: u32, pos: usize) -> Vec<f32> {
        let c = self.config;
        let (e, v, nh, hd) = (c.n_embd, c.n_vocab, c.n_head, c.head_dim());
        let (tok_i, pos_i, e_i, v_i) = (tok as i32, pos as i32, e as i32, v as i32);
        assert!(pos < c.n_ctx, "context overflow");

        match &self.wte_t {
            Weights::F32(w) => {
                let mut lb = self.stream.launch_builder(&self.k.embed);
                lb.arg(&mut self.x).arg(w).arg(&self.wpe).arg(&tok_i).arg(&pos_i).arg(&e_i).arg(&v_i);
                unsafe { lb.launch(cfg1d(e)) }.unwrap();
            }
            Weights::Int8 { q, scales } => {
                let mut lb = self.stream.launch_builder(&self.k.embed_int8);
                lb.arg(&mut self.x).arg(q).arg(scales).arg(&self.wpe).arg(&tok_i).arg(&pos_i).arg(&e_i).arg(&v_i);
                unsafe { lb.launch(cfg1d(e)) }.unwrap();
            }
        }

        for l in 0..c.n_layer {
            let layer = &self.layers[l];

            layernorm(&self.stream, &self.k.layernorm, &mut self.xb, &self.x, &layer.ln1_g, &layer.ln1_b, e);
            gemv(&self.stream, &self.k, &mut self.qkv, &self.xb, &layer.qkv_w, &layer.qkv_b, e, 3 * e);

            // append K and V rows to this layer's cache
            self.stream
                .memcpy_dtod(&self.qkv.slice(e..2 * e), &mut self.kcache[l].slice_mut(pos * e..(pos + 1) * e))
                .unwrap();
            self.stream
                .memcpy_dtod(&self.qkv.slice(2 * e..3 * e), &mut self.vcache[l].slice_mut(pos * e..(pos + 1) * e))
                .unwrap();

            let (t_i, nh_i, hd_i) = (pos as i32, nh as i32, hd as i32);
            let mut lb = self.stream.launch_builder(&self.k.attn_decode);
            lb.arg(&mut self.attn).arg(&self.qkv).arg(&self.kcache[l]).arg(&self.vcache[l]).arg(&t_i).arg(&nh_i).arg(&hd_i);
            let cfg = LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { lb.launch(cfg) }.unwrap();

            gemv(&self.stream, &self.k, &mut self.xb, &self.attn, &layer.proj_w, &layer.proj_b, e, e);
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);

            layernorm(&self.stream, &self.k.layernorm, &mut self.xb, &self.x, &layer.ln2_g, &layer.ln2_b, e);
            gemv(&self.stream, &self.k, &mut self.h, &self.xb, &layer.fc_w, &layer.fc_b, e, 4 * e);
            let n_i = (4 * e) as i32;
            let mut lb = self.stream.launch_builder(&self.k.gelu_inplace);
            lb.arg(&mut self.h).arg(&n_i);
            unsafe { lb.launch(cfg1d(4 * e)) }.unwrap();
            gemv(&self.stream, &self.k, &mut self.xb, &self.h, &layer.fc2_w, &layer.fc2_b, 4 * e, e);
            add(&self.stream, &self.k.add_inplace, &mut self.x, &self.xb, e);
        }

        layernorm(&self.stream, &self.k.layernorm, &mut self.xb, &self.x, &self.lnf_g, &self.lnf_b, e);
        gemv(&self.stream, &self.k, &mut self.logits, &self.xb, &self.wte_t, &self.zero_bias, e, v);

        self.stream.clone_dtoh(&self.logits).unwrap()
    }

    /// Greedy generation; returns only the newly generated token ids.
    pub fn generate(&mut self, prompt: &[u32], n_new: usize, mut on_token: impl FnMut(u32)) -> Vec<u32> {
        assert!(!prompt.is_empty());
        let mut logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits = self.forward(tok, pos);
        }
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
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}
