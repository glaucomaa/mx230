//! Model file format and host-side weight storage.
//!
//! `model.bin` layout (little-endian):
//!   magic "MXGP" (u32) | version (u32) | n_layer | n_head | n_embd | n_ctx | n_vocab (u32 each)
//! followed by fp32 tensors in a fixed order (see `TENSOR_ORDER` below).
//! Linear weights are stored as [n_in, n_out] row-major (y = x @ W + b),
//! matching HF GPT-2's Conv1D convention, so export is a straight copy.

use std::fs;
use std::io::Write as _;
use std::path::Path;

pub const MAGIC: u32 = u32::from_le_bytes(*b"MXGP");
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub n_vocab: usize,
}

impl Config {
    pub fn gpt2_small() -> Self {
        Config {
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            n_ctx: 1024,
            n_vocab: 50257,
        }
    }
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// Per-layer weights, all fp32, linear weights as [n_in, n_out].
pub struct Layer {
    pub ln1_g: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv_w: Vec<f32>, // [embd, 3*embd]
    pub qkv_b: Vec<f32>,
    pub proj_w: Vec<f32>, // [embd, embd]
    pub proj_b: Vec<f32>,
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub fc_w: Vec<f32>, // [embd, 4*embd]
    pub fc_b: Vec<f32>,
    pub fc2_w: Vec<f32>, // [4*embd, embd]
    pub fc2_b: Vec<f32>,
}

pub struct Model {
    pub config: Config,
    pub wte: Vec<f32>, // [vocab, embd]; also the (tied) lm_head
    pub wpe: Vec<f32>, // [ctx, embd]
    pub layers: Vec<Layer>,
    pub lnf_g: Vec<f32>,
    pub lnf_b: Vec<f32>,
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
            c.n_layer as u32,
            c.n_head as u32,
            c.n_embd as u32,
            c.n_ctx as u32,
            c.n_vocab as u32,
        ] {
            out.write_all(&v.to_le_bytes())?;
        }
        write_tensor(&mut out, &self.wte)?;
        write_tensor(&mut out, &self.wpe)?;
        for l in &self.layers {
            for t in [
                &l.ln1_g, &l.ln1_b, &l.qkv_w, &l.qkv_b, &l.proj_w, &l.proj_b, &l.ln2_g, &l.ln2_b,
                &l.fc_w, &l.fc_b, &l.fc2_w, &l.fc2_b,
            ] {
                write_tensor(&mut out, t)?;
            }
        }
        write_tensor(&mut out, &self.lnf_g)?;
        write_tensor(&mut out, &self.lnf_b)?;
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let buf = fs::read(path)?;
        let mut r = Reader { buf: &buf, pos: 0 };
        assert_eq!(r.u32(), MAGIC, "not a model.bin file");
        assert_eq!(r.u32(), VERSION, "unsupported model.bin version");
        let config = Config {
            n_layer: r.u32() as usize,
            n_head: r.u32() as usize,
            n_embd: r.u32() as usize,
            n_ctx: r.u32() as usize,
            n_vocab: r.u32() as usize,
        };
        let e = config.n_embd;
        let wte = r.tensor(config.n_vocab * e);
        let wpe = r.tensor(config.n_ctx * e);
        let layers = (0..config.n_layer)
            .map(|_| Layer {
                ln1_g: r.tensor(e),
                ln1_b: r.tensor(e),
                qkv_w: r.tensor(e * 3 * e),
                qkv_b: r.tensor(3 * e),
                proj_w: r.tensor(e * e),
                proj_b: r.tensor(e),
                ln2_g: r.tensor(e),
                ln2_b: r.tensor(e),
                fc_w: r.tensor(e * 4 * e),
                fc_b: r.tensor(4 * e),
                fc2_w: r.tensor(4 * e * e),
                fc2_b: r.tensor(e),
            })
            .collect();
        let lnf_g = r.tensor(e);
        let lnf_b = r.tensor(e);
        assert_eq!(r.pos, buf.len(), "trailing bytes in model.bin");
        Ok(Model {
            config,
            wte,
            wpe,
            layers,
            lnf_g,
            lnf_b,
        })
    }
}
