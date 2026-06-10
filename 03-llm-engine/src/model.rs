//! Model file format and host-side weight storage.
//!
//! `model.bin` layout (little-endian):
//!   magic "MXGP" (u32) | version (u32) | arch (u32) | n_layer | n_head |
//!   n_kv_head | head_dim | n_embd | n_inter | n_ctx | n_vocab (u32 each) |
//!   rope_theta (f32) | norm_eps (f32)
//! followed by fp32 tensors in a fixed per-arch order (see save/load).
//! Linear weights are stored as [n_in, n_out] row-major (y = x @ W + b);
//! HF GPT-2 Conv1D already has that layout, HF Linear (Qwen2) is transposed
//! at export time.

use std::fs;
use std::io::Write as _;
use std::path::Path;

pub const MAGIC: u32 = u32::from_le_bytes(*b"MXGP");
pub const VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Gpt2,
    Qwen2,
    Llama, // TinyLlama-1.1B: Qwen2 layout minus qkv bias, untied lm_head
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize, // < n_head means grouped-query attention
    pub head_dim: usize,
    pub n_embd: usize,
    pub n_inter: usize, // MLP hidden width
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub rope_theta: f32, // 0.0 for learned positional embeddings (GPT-2)
    pub norm_eps: f32,
}

impl Config {
    pub fn gpt2_small() -> Self {
        Config {
            arch: Arch::Gpt2,
            n_layer: 12,
            n_head: 12,
            n_kv_head: 12,
            head_dim: 64,
            n_embd: 768,
            n_inter: 3072,
            n_ctx: 1024,
            n_vocab: 50257,
            rope_theta: 0.0,
            norm_eps: 1e-5,
        }
    }

    /// Qwen2.5-0.5B; n_ctx capped at 1024 (the KV cache window we allocate),
    /// the model itself supports far longer contexts.
    pub fn qwen25_05b() -> Self {
        Config {
            arch: Arch::Qwen2,
            n_layer: 24,
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            n_embd: 896,
            n_inter: 4864,
            n_ctx: 1024,
            n_vocab: 151936,
            rope_theta: 1e6,
            norm_eps: 1e-6,
        }
    }

    /// TinyLlama-1.1B (3T base checkpoint); n_ctx capped at 1024 like Qwen.
    pub fn tinyllama_11b() -> Self {
        Config {
            arch: Arch::Llama,
            n_layer: 22,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 64,
            n_embd: 2048,
            n_inter: 5632,
            n_ctx: 1024,
            n_vocab: 32000,
            rope_theta: 1e4,
            norm_eps: 1e-5,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
}

/// Per-layer weights, all fp32, linear weights as [n_in, n_out].
/// GPT-2 uses every field; Qwen2 leaves the norm biases and MLP biases empty
/// (`fc_w`/`fc2_w` are reused as SwiGLU gate/down, `up_w` is Qwen2-only).
pub struct Layer {
    pub ln1_g: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv_w: Vec<f32>, // [embd, q_dim + 2*kv_dim]
    pub qkv_b: Vec<f32>,
    pub proj_w: Vec<f32>, // [q_dim, embd]
    pub proj_b: Vec<f32>, // zeros for Qwen2 (o_proj has no bias)
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc_w: Vec<f32>, // GPT-2 fc [embd, inter] | Qwen2 gate [embd, inter]
    pub fc_b: Vec<f32>,
    pub up_w: Vec<f32>,  // Qwen2 only: [embd, inter]
    pub fc2_w: Vec<f32>, // GPT-2 fc2 [inter, embd] | Qwen2 down [inter, embd]
    pub fc2_b: Vec<f32>,
}

pub struct Model {
    pub config: Config,
    pub wte: Vec<f32>, // [vocab, embd]; also the lm_head when tied
    pub wpe: Vec<f32>, // [ctx, embd] for GPT-2, empty for RoPE archs
    pub layers: Vec<Layer>,
    pub lnf_g: Vec<f32>,
    pub lnf_b: Vec<f32>,
    pub lm_head: Vec<f32>, // [vocab, embd] when untied (Llama), empty when tied
}

fn write_tensor(out: &mut impl std::io::Write, t: &[f32]) -> std::io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 4) };
    out.write_all(bytes)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn tensor(&mut self, len: usize) -> Vec<f32> {
        let bytes = &self.buf[self.pos..self.pos + len * 4];
        self.pos += len * 4;
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

impl Model {
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let c = &self.config;
        let mut out = std::io::BufWriter::new(fs::File::create(path)?);
        for v in [
            MAGIC,
            VERSION,
            match c.arch {
                Arch::Gpt2 => 0,
                Arch::Qwen2 => 1,
                Arch::Llama => 2,
            },
            c.n_layer as u32,
            c.n_head as u32,
            c.n_kv_head as u32,
            c.head_dim as u32,
            c.n_embd as u32,
            c.n_inter as u32,
            c.n_ctx as u32,
            c.n_vocab as u32,
            c.rope_theta.to_bits(),
            c.norm_eps.to_bits(),
        ] {
            out.write_all(&v.to_le_bytes())?;
        }
        write_tensor(&mut out, &self.wte)?;
        if c.arch == Arch::Gpt2 {
            write_tensor(&mut out, &self.wpe)?;
        }
        for l in &self.layers {
            let tensors: Vec<&Vec<f32>> = match c.arch {
                Arch::Gpt2 => vec![
                    &l.ln1_g, &l.ln1_b, &l.qkv_w, &l.qkv_b, &l.proj_w, &l.proj_b, &l.ln2_g,
                    &l.ln2_b, &l.fc_w, &l.fc_b, &l.fc2_w, &l.fc2_b,
                ],
                Arch::Qwen2 => vec![
                    &l.ln1_g, &l.qkv_w, &l.qkv_b, &l.proj_w, &l.ln2_g, &l.fc_w, &l.up_w, &l.fc2_w,
                ],
                Arch::Llama => vec![
                    &l.ln1_g, &l.qkv_w, &l.proj_w, &l.ln2_g, &l.fc_w, &l.up_w, &l.fc2_w,
                ],
            };
            for t in tensors {
                write_tensor(&mut out, t)?;
            }
        }
        write_tensor(&mut out, &self.lnf_g)?;
        if c.arch == Arch::Gpt2 {
            write_tensor(&mut out, &self.lnf_b)?;
        }
        if c.arch == Arch::Llama {
            write_tensor(&mut out, &self.lm_head)?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        // mmap instead of fs::read: a 4.4 GB model.bin plus its parsed
        // tensors would otherwise double-buffer in RAM
        let file = fs::File::open(path)?;
        let buf = unsafe { memmap2::Mmap::map(&file)? };
        let mut r = Reader { buf: &buf, pos: 0 };
        assert_eq!(r.u32(), MAGIC, "not a model.bin file");
        assert_eq!(
            r.u32(),
            VERSION,
            "unsupported model.bin version — re-run `export`"
        );
        let arch = match r.u32() {
            0 => Arch::Gpt2,
            1 => Arch::Qwen2,
            2 => Arch::Llama,
            a => panic!("unknown arch tag {a}"),
        };
        let config = Config {
            arch,
            n_layer: r.u32() as usize,
            n_head: r.u32() as usize,
            n_kv_head: r.u32() as usize,
            head_dim: r.u32() as usize,
            n_embd: r.u32() as usize,
            n_inter: r.u32() as usize,
            n_ctx: r.u32() as usize,
            n_vocab: r.u32() as usize,
            rope_theta: r.f32(),
            norm_eps: r.f32(),
        };
        let (e, inter) = (config.n_embd, config.n_inter);
        let (qd, qkvd) = (config.q_dim(), config.qkv_dim());
        let wte = r.tensor(config.n_vocab * e);
        let wpe = match arch {
            Arch::Gpt2 => r.tensor(config.n_ctx * e),
            Arch::Qwen2 | Arch::Llama => Vec::new(),
        };
        let layers = (0..config.n_layer)
            .map(|_| match arch {
                Arch::Gpt2 => Layer {
                    ln1_g: r.tensor(e),
                    ln1_b: r.tensor(e),
                    qkv_w: r.tensor(e * qkvd),
                    qkv_b: r.tensor(qkvd),
                    proj_w: r.tensor(qd * e),
                    proj_b: r.tensor(e),
                    ln2_g: r.tensor(e),
                    ln2_b: r.tensor(e),
                    fc_w: r.tensor(e * inter),
                    fc_b: r.tensor(inter),
                    up_w: Vec::new(),
                    fc2_w: r.tensor(inter * e),
                    fc2_b: r.tensor(e),
                },
                Arch::Qwen2 | Arch::Llama => Layer {
                    ln1_g: r.tensor(e),
                    ln1_b: Vec::new(),
                    qkv_w: r.tensor(e * qkvd),
                    qkv_b: if arch == Arch::Qwen2 {
                        r.tensor(qkvd)
                    } else {
                        vec![0.0; qkvd] // Llama attention has no biases
                    },
                    proj_w: r.tensor(qd * e),
                    proj_b: vec![0.0; e],
                    ln2_g: r.tensor(e),
                    ln2_b: Vec::new(),
                    fc_w: r.tensor(e * inter),
                    fc_b: Vec::new(),
                    up_w: r.tensor(e * inter),
                    fc2_w: r.tensor(inter * e),
                    fc2_b: Vec::new(),
                },
            })
            .collect();
        let lnf_g = r.tensor(e);
        let lnf_b = match arch {
            Arch::Gpt2 => r.tensor(e),
            Arch::Qwen2 | Arch::Llama => Vec::new(),
        };
        let lm_head = match arch {
            Arch::Llama => r.tensor(config.n_vocab * e),
            _ => Vec::new(),
        };
        assert_eq!(r.pos, buf.len(), "trailing bytes in model.bin");
        Ok(Model {
            config,
            wte,
            wpe,
            layers,
            lnf_g,
            lnf_b,
            lm_head,
        })
    }
}
