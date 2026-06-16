//! SmoothQuant: migrate per-channel activation magnitude into the weights.
//!
//! For a linear `y = x @ W` (W is `[n_in, n_out]`) fed by a preceding norm,
//! pick a per-input-channel factor
//!   `s_j = act_absmax_j^alpha / weight_absmax_j^(1-alpha)`
//! divide the activation channel by `s_j` and multiply that weight *row*
//! (input channel `j`, length `n_out`) by `s_j`. The product `x @ W` is
//! unchanged, so this is **exactly equivalent in fp32** — but the rescaled
//! activations are flatter across channels, which is friendlier to int8/int4
//! quantization downstream.
//!
//! The activation rescale is folded into the preceding norm's `gamma`/`beta`
//! (`out_j = norm_j * gamma_j + beta_j`, so dividing both by `s_j` yields
//! `out_j / s_j` for free — RMSNorm has no `beta`, just `gamma`). Only linears
//! fed directly by a norm are touched: `qkv` (via ln1) and the MLP gate/fc
//! (and Qwen's `up`, which shares the ln2 input) — never `proj`/`down`.

use crate::calib::ActStats;
use crate::model::{Arch, Model};

/// Per-input-channel weight absmax of a `[n_in, n_out]` row-major matrix
/// (the input channel is the row, so absmax over its `n_out` columns).
fn row_absmax(w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    (0..n_in)
        .map(|i| {
            w[i * n_out..(i + 1) * n_out]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()))
        })
        .collect()
}

/// SmoothQuant scale per channel; guarded to 1.0 for dead channels/weights so
/// the fold can never divide a norm parameter by zero or a non-finite value.
fn scales(act_absmax: &[f32], w_absmax: &[f32], alpha: f32) -> Vec<f32> {
    act_absmax
        .iter()
        .zip(w_absmax)
        .map(|(&a, &w)| {
            let s = a.powf(alpha) / w.powf(1.0 - alpha);
            if s.is_finite() && s > 1e-5 {
                s
            } else {
                1.0
            }
        })
        .collect()
}

/// Fold the activation rescale `x_j -> x_j / s_j` into a norm: divide `gamma`
/// (and `beta`, if present) by `s`.
fn fold_norm(gamma: &mut [f32], beta: &mut [f32], s: &[f32]) {
    for (g, &sj) in gamma.iter_mut().zip(s) {
        *g /= sj;
    }
    for (b, &sj) in beta.iter_mut().zip(s) {
        *b /= sj;
    }
}

/// Multiply row `j` (input channel `j`) of a `[n_in, n_out]` weight by `s_j`.
fn scale_rows(w: &mut [f32], n_in: usize, n_out: usize, s: &[f32]) {
    for i in 0..n_in {
        let si = s[i];
        if si != 1.0 {
            for v in &mut w[i * n_out..(i + 1) * n_out] {
                *v *= si;
            }
        }
    }
}

/// Returns a smoothed copy of `model`. `alpha` ∈ [0,1] trades how much
/// magnitude moves from activations into weights (0.5 is the usual default).
pub fn smooth(model: &Model, act: &ActStats, alpha: f32) -> Model {
    let mut m = model.clone();
    let c = m.config;
    let (e, qkvd, inter) = (c.n_embd, c.qkv_dim(), c.n_inter);

    for l in 0..c.n_layer {
        let layer = &mut m.layers[l];

        // attention: ln1 feeds qkv_w [e, qkvd]
        let s = scales(&act.attn[l], &row_absmax(&layer.qkv_w, e, qkvd), alpha);
        fold_norm(&mut layer.ln1_g, &mut layer.ln1_b, &s);
        scale_rows(&mut layer.qkv_w, e, qkvd, &s);

        // mlp: ln2 feeds fc_w [e, inter] (and Qwen's up_w, same input) — so the
        // shared input scale must respect the larger of the two weight magnitudes
        let mut w_absmax = row_absmax(&layer.fc_w, e, inter);
        if c.arch != Arch::Gpt2 {
            for (a, b) in w_absmax.iter_mut().zip(row_absmax(&layer.up_w, e, inter)) {
                *a = a.max(b);
            }
        }
        let s = scales(&act.mlp[l], &w_absmax, alpha);
        fold_norm(&mut layer.ln2_g, &mut layer.ln2_b, &s);
        scale_rows(&mut layer.fc_w, e, inter, &s);
        if c.arch != Arch::Gpt2 {
            scale_rows(&mut layer.up_w, e, inter, &s);
        }
    }
    m
}
