//! Activation calibration for SmoothQuant.
//!
//! Runs the CPU reference forward over a *separate* corpus (never the ppl test
//! set) and records, per layer, the per-input-channel absmax of the activation
//! entering each norm-fed linear (`ln1 → qkv`, `ln2 → MLP`). SmoothQuant
//! (`smooth.rs`) then migrates that magnitude into the weights.

use crate::cpu::{forward_obs, Site};
use crate::model::Model;

fn site_idx(s: Site) -> usize {
    match s {
        Site::Qkv => 0,
        Site::Proj => 1,
        Site::Fc => 2,
        Site::Fc2 => 3,
    }
}

/// Per-channel activation absmax, `[n_layer][n_embd]` for each of the two
/// norm-fed sites in a transformer block.
pub struct ActStats {
    pub attn: Vec<Vec<f32>>, // input to qkv (ln1 output)
    pub mlp: Vec<Vec<f32>>,  // input to MLP gate/fc (ln2 output)
}

/// Sweep `tokens` in `chunk_len` windows (capped at `max_tokens` total) and
/// accumulate the per-channel activation absmax. The CPU forward is O(T²) per
/// chunk, so we keep windows short — a few hundred tokens already pin the
/// channel magnitudes down for SmoothQuant.
pub fn collect(model: &Model, tokens: &[u32], chunk_len: usize, max_tokens: usize) -> ActStats {
    let c = &model.config;
    let mut stats = ActStats {
        attn: vec![vec![0.0f32; c.n_embd]; c.n_layer],
        mlp: vec![vec![0.0f32; c.n_embd]; c.n_layer],
    };
    let limit = max_tokens.min(tokens.len());
    let mut start = 0usize;
    while start < limit {
        let end = (start + chunk_len).min(limit);
        let chunk = &tokens[start..end];
        if chunk.len() < 2 {
            break;
        }
        forward_obs(model, chunk, &mut |l, site, x| {
            // SmoothQuant only rescales the norm-fed linears (qkv via ln1, MLP
            // gate/fc via ln2); Proj/Fc2 aren't norm-fed, so skip them here.
            let dst = match site {
                Site::Qkv => &mut stats.attn[l],
                Site::Fc => &mut stats.mlp[l],
                Site::Proj | Site::Fc2 => return,
            };
            for (d, &v) in x.iter().enumerate() {
                let a = v.abs();
                if a > dst[d] {
                    dst[d] = a;
                }
            }
        });
        start = end;
    }
    stats
}

/// Per-linear input Hessians for GPTQ: `h[layer][site] = Σ_t x_t x_tᵀ`, the
/// (uncentered) second moment of the activation entering that linear over all
/// calibration positions. Sites index as `cpu::Site`: 0=Qkv, 1=Proj, 2=Fc
/// (also Qwen's `up`), 3=Fc2. The absolute scale is irrelevant to GPTQ (it
/// cancels in the damped inverse), so no normalization is applied.
pub struct Hessians {
    pub count: usize,
    pub h: Vec<[Vec<f32>; 4]>,
    pub n_in: [usize; 4],
}

/// One calibration pass accumulating all per-linear Hessians at once. Memory
/// is `Σ_site n_in² · n_layer · 4 B` (GPT-2 ≈ 0.5 GB; the FFN-down site
/// dominates), and the accumulation is O(Σ n_in²) per token — so keep the
/// window short and the budget modest (GPTQ is robust thanks to damping).
pub fn collect_hessians(
    model: &Model,
    tokens: &[u32],
    chunk_len: usize,
    max_tokens: usize,
) -> Hessians {
    let c = &model.config;
    let n_in = [c.n_embd, c.q_dim(), c.n_embd, c.n_inter];
    let mut h: Vec<[Vec<f32>; 4]> = (0..c.n_layer)
        .map(|_| std::array::from_fn(|s| vec![0.0f32; n_in[s] * n_in[s]]))
        .collect();
    let mut count = 0usize;
    let limit = max_tokens.min(tokens.len());
    let mut start = 0usize;
    while start < limit {
        let end = (start + chunk_len).min(limit);
        let chunk = &tokens[start..end];
        if chunk.len() < 2 {
            break;
        }
        forward_obs(model, chunk, &mut |l, site, x| {
            let s = site_idx(site);
            let n = n_in[s];
            let hl = &mut h[l][s];
            // accumulate the upper triangle only (symmetrized afterwards)
            for i in 0..n {
                let xi = x[i];
                if xi == 0.0 {
                    continue;
                }
                let row = &mut hl[i * n..i * n + n];
                for (j, r) in row.iter_mut().enumerate().take(n).skip(i) {
                    *r += xi * x[j];
                }
            }
            if l == 0 && site == Site::Qkv {
                count += 1;
            }
        });
        start = end;
    }
    // mirror the upper triangle into the lower
    for layer in &mut h {
        for (s, hm) in layer.iter_mut().enumerate() {
            let n = n_in[s];
            for i in 0..n {
                for j in (i + 1)..n {
                    hm[j * n + i] = hm[i * n + j];
                }
            }
        }
    }
    Hessians { count, h, n_in }
}
