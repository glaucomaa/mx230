//! Activation calibration for SmoothQuant.
//!
//! Runs the CPU reference forward over a *separate* corpus (never the ppl test
//! set) and records, per layer, the per-input-channel absmax of the activation
//! entering each norm-fed linear (`ln1 → qkv`, `ln2 → MLP`). SmoothQuant
//! (`smooth.rs`) then migrates that magnitude into the weights.

use crate::cpu::{forward_obs, Site};
use crate::model::Model;

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
            let dst = match site {
                Site::Attn => &mut stats.attn[l],
                Site::Mlp => &mut stats.mlp[l],
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
