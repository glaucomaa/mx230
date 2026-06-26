//! Self-speculative (early-exit) decoding support.
//!
//! Decode on this card is memory-bound: single-stream tok/s is bounded by
//! weight-bytes-read per token. The only single-stream lever left is to
//! amortize those weight reads across several tokens per verify — speculative
//! decoding. `prompt_lookup` already covers extractive/repetitive text; this
//! module adds an open-generation drafter that reuses the *same* model's first
//! `k` layers (zero extra VRAM — decisive at 2 GB) and stays bit-identical to
//! greedy because the full model verifies every token.
//!
//! The bare early-exit logits are poorly calibrated (these models are not
//! LayerSkip-trained), so acceptance is lifted by a closed-form linear exit
//! adapter `A_K : h_K -> ĥ_L`, fit offline by ridge regression and stored in a
//! sidecar next to the model. See `calibrate-spec` in `main.rs`.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// How `speculative_loop` produces its candidate tokens.
#[derive(Clone, Copy, Debug)]
pub enum DraftStrategy {
    /// n-gram copy from the running history. Cheap, but only fires on
    /// repetitive/extractive text.
    PromptLookup { k: usize },
    /// Early-exit self-draft: run the first `k` layers of this model to draft
    /// `gamma` tokens (`gamma` includes the already-known greedy token at
    /// index 0, so the drafter predicts `gamma - 1` new tokens).
    SelfSpec { k: usize, gamma: usize },
}

const SPEC_MAGIC: u32 = u32::from_le_bytes(*b"MXSP");
const SPEC_VERSION: u32 = 1;

/// `<model>.bin` -> `<model>.spec.bin`, the exit-adapter sidecar.
pub fn sidecar_path(model_bin: &Path) -> PathBuf {
    let mut s = model_bin.to_path_buf().into_os_string();
    s.push(".spec.bin");
    PathBuf::from(s)
}

/// Closed-form linear exit adapter `A_K`. Maps the layer-`k` hidden h_K to a
/// prediction ĥ_L of the model's final (pre-`lnf`) hidden, so the shared
/// `lnf`+`lm_head` produces better-calibrated early-exit logits.
///
/// `a` is `[n_embd, n_embd]` in the engine's in-major GEMV weight layout —
/// `a[i*n_embd + o]` is the weight from input channel `i` to output channel `o`,
/// i.e. `ĥ_L[o] = Σ_i a[i*e+o] · h_K[i]`.
pub struct ExitAdapter {
    pub k: usize,
    pub n_embd: usize,
    pub a: Vec<f32>,
}

impl ExitAdapter {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = io::BufWriter::new(fs::File::create(path)?);
        for v in [SPEC_MAGIC, SPEC_VERSION, self.k as u32, self.n_embd as u32] {
            out.write_all(&v.to_le_bytes())?;
        }
        let mut bytes = Vec::with_capacity(self.a.len() * 4);
        for &x in &self.a {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        out.write_all(&bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut hdr = [0u8; 16];
        f.read_exact(&mut hdr)?;
        let u = |i: usize| u32::from_le_bytes(hdr[i..i + 4].try_into().unwrap());
        assert_eq!(u(0), SPEC_MAGIC, "not a model.spec.bin file");
        assert_eq!(u(4), SPEC_VERSION, "unsupported model.spec.bin version");
        let k = u(8) as usize;
        let n_embd = u(12) as usize;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        assert_eq!(buf.len(), n_embd * n_embd * 4, "spec adapter size mismatch");
        let a = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(ExitAdapter { k, n_embd, a })
    }
}

/// Solve `X = (G + λI)^{-1} C` for the `e × e` system, via a hand-rolled
/// Cholesky factorization and triangular solves (the project's from-scratch,
/// closed-form ethos — cf. `gptq.rs`). `g` and `c` are row-major `[e*e]` f64
/// accumulators; `g` is overwritten with the ridge-augmented matrix. Returns
/// `X` row-major as `x[i*e + o]` (row `i`, column `o`).
///
/// For the exit adapter, `G = HₖᵀHₖ` and `C = HₖᵀH_L`, so `X = (G+λI)⁻¹C` is the
/// least-squares map with `H_L ≈ Hₖ X`, i.e. column-vector `ĥ_L = Xᵀ h_K`.
pub fn solve_ridge(g: &mut [f64], c: &[f64], e: usize, lambda: f64) -> Vec<f64> {
    assert_eq!(g.len(), e * e);
    assert_eq!(c.len(), e * e);
    // ridge: G += λI
    for i in 0..e {
        g[i * e + i] += lambda;
    }
    // Cholesky G = L Lᵀ (L lower-triangular), in place into `l`
    let mut l = vec![0f64; e * e];
    for i in 0..e {
        for j in 0..=i {
            let mut sum = g[i * e + j];
            for p in 0..j {
                sum -= l[i * e + p] * l[j * e + p];
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "Cholesky: matrix not positive-definite at row {i} (pivot {sum:.3e}); raise --spec-lambda"
                );
                l[i * e + i] = sum.sqrt();
            } else {
                l[i * e + j] = sum / l[j * e + j];
            }
        }
    }
    // For each column o of C: forward-solve L y = C[:,o], back-solve Lᵀ x = y.
    let mut x = vec![0f64; e * e];
    let mut y = vec![0f64; e];
    for o in 0..e {
        for i in 0..e {
            let mut sum = c[i * e + o];
            for p in 0..i {
                sum -= l[i * e + p] * y[p];
            }
            y[i] = sum / l[i * e + i];
        }
        for i in (0..e).rev() {
            let mut sum = y[i];
            for p in (i + 1)..e {
                sum -= l[p * e + i] * x[p * e + o];
            }
            x[i * e + o] = sum / l[i * e + i];
        }
    }
    x
}

/// Narrow the solved `X` to the engine's GEMV weight (just f64 -> f32).
///
/// The GEMV kernel reads weights **in-major** as `w[i*n_out + o]` and computes
/// `y[o] = Σ_i x[i]·w[i*e+o]`. The ridge solution `X` (`x[i*e+o] = X[i,o]`, the
/// least-squares map `H_L ≈ H_K X`) gives exactly `ĥ_L[o] = Σ_i h_K[i]·X[i,o]`,
/// so it already *is* the weight in that layout — no transpose.
pub fn adapter_from_solution(x: &[f64], e: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), e * e);
    x.iter().map(|&v| v as f32).collect()
}
