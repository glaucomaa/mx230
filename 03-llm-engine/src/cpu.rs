//! CPU reference forward pass. Slow and simple — exists to pin down
//! correctness of the CUDA path (and to compare against HF logits).

use crate::model::{Arch, Model};

const LN_EPS: f32 = 1e-5;

fn layernorm(x: &[f32], g: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

fn rmsnorm(x: &[f32], g: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * g[i];
    }
}

/// HF rotate_half convention: pairs are (d, d + head_dim/2).
fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for d in 0..half {
            let freq = theta.powf(-2.0 * d as f32 / head_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (x1, x2) = (x[base + d], x[base + d + half]);
            x[base + d] = x1 * cos - x2 * sin;
            x[base + d + half] = x1 * sin + x2 * cos;
        }
    }
}

/// y = x @ W + b, W is [n_in, n_out]
fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize, y: &mut [f32]) {
    y.copy_from_slice(b);
    for i in 0..n_in {
        let xi = x[i];
        let row = &w[i * n_out..(i + 1) * n_out];
        for o in 0..n_out {
            y[o] += xi * row[o];
        }
    }
}

fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (0.7978845608 * (*v + 0.044715 * x3)).tanh());
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Runs the full prompt and returns the logits for the last position.
/// KV "cache" here is just keeping all K/V rows per layer in memory.
pub fn forward(model: &Model, tokens: &[u32]) -> Vec<f32> {
    match model.config.arch {
        Arch::Gpt2 => forward_gpt2(model, tokens),
        Arch::Qwen2 => forward_qwen2(model, tokens),
    }
}

fn forward_gpt2(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let c = &model.config;
    let (e, nh, hd) = (c.n_embd, c.n_head, c.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();

    let mut kcache = vec![vec![0.0f32; tokens.len() * e]; c.n_layer];
    let mut vcache = vec![vec![0.0f32; tokens.len() * e]; c.n_layer];

    let mut x = vec![0.0f32; e];
    let mut xb = vec![0.0f32; e];
    let mut qkv = vec![0.0f32; 3 * e];
    let mut att_out = vec![0.0f32; e];
    let mut proj = vec![0.0f32; e];
    let mut h = vec![0.0f32; 4 * e];
    let mut logits = vec![0.0f32; c.n_vocab];

    for (t, &tok) in tokens.iter().enumerate() {
        for i in 0..e {
            x[i] = model.wte[tok as usize * e + i] + model.wpe[t * e + i];
        }

        for (l, layer) in model.layers.iter().enumerate() {
            layernorm(&x, &layer.ln1_g, &layer.ln1_b, &mut xb);
            linear(&xb, &layer.qkv_w, &layer.qkv_b, e, 3 * e, &mut qkv);
            kcache[l][t * e..(t + 1) * e].copy_from_slice(&qkv[e..2 * e]);
            vcache[l][t * e..(t + 1) * e].copy_from_slice(&qkv[2 * e..3 * e]);

            // causal attention over the cache, head by head
            for head in 0..nh {
                let q = &qkv[head * hd..(head + 1) * hd];
                let mut scores = vec![0.0f32; t + 1];
                let mut m = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let k = &kcache[l][j * e + head * hd..j * e + (head + 1) * hd];
                    *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    m = m.max(*s);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut att_out[head * hd..(head + 1) * hd];
                out.fill(0.0);
                for (j, s) in scores.iter().enumerate() {
                    let p = s / sum;
                    let v = &vcache[l][j * e + head * hd..j * e + (head + 1) * hd];
                    for d in 0..hd {
                        out[d] += p * v[d];
                    }
                }
            }

            linear(&att_out, &layer.proj_w, &layer.proj_b, e, e, &mut proj);
            for i in 0..e {
                x[i] += proj[i];
            }

            layernorm(&x, &layer.ln2_g, &layer.ln2_b, &mut xb);
            linear(&xb, &layer.fc_w, &layer.fc_b, e, 4 * e, &mut h);
            gelu(&mut h);
            linear(&h, &layer.fc2_w, &layer.fc2_b, 4 * e, e, &mut proj);
            for i in 0..e {
                x[i] += proj[i];
            }
        }
    }

    layernorm(&x, &model.lnf_g, &model.lnf_b, &mut xb);
    // tied lm_head: logits[v] = dot(x_final, wte[v])
    for v in 0..c.n_vocab {
        logits[v] = model.wte[v * e..(v + 1) * e]
            .iter()
            .zip(&xb)
            .map(|(a, b)| a * b)
            .sum();
    }
    logits
}

fn forward_qwen2(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let c = &model.config;
    let (e, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv_head, c.head_dim);
    let (qd, kvd, qkvd, inter) = (c.q_dim(), c.kv_dim(), c.qkv_dim(), c.n_inter);
    let group = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut kcache = vec![vec![0.0f32; tokens.len() * kvd]; c.n_layer];
    let mut vcache = vec![vec![0.0f32; tokens.len() * kvd]; c.n_layer];

    let mut x = vec![0.0f32; e];
    let mut xb = vec![0.0f32; e];
    let mut qkv = vec![0.0f32; qkvd];
    let mut att_out = vec![0.0f32; qd];
    let mut proj = vec![0.0f32; e];
    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    let mut logits = vec![0.0f32; c.n_vocab];
    let zero_e = vec![0.0f32; e.max(inter)];

    for (t, &tok) in tokens.iter().enumerate() {
        x.copy_from_slice(&model.wte[tok as usize * e..(tok as usize + 1) * e]);

        for (l, layer) in model.layers.iter().enumerate() {
            rmsnorm(&x, &layer.ln1_g, c.norm_eps, &mut xb);
            linear(&xb, &layer.qkv_w, &layer.qkv_b, e, qkvd, &mut qkv);
            rope(&mut qkv[..qd], nh, hd, t, c.rope_theta);
            rope(&mut qkv[qd..qd + kvd], nkv, hd, t, c.rope_theta);
            kcache[l][t * kvd..(t + 1) * kvd].copy_from_slice(&qkv[qd..qd + kvd]);
            vcache[l][t * kvd..(t + 1) * kvd].copy_from_slice(&qkv[qd + kvd..qkvd]);

            // grouped-query causal attention over the cache
            for head in 0..nh {
                let kvh = head / group;
                let q = &qkv[head * hd..(head + 1) * hd];
                let mut scores = vec![0.0f32; t + 1];
                let mut m = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let k = &kcache[l][j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
                    *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    m = m.max(*s);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut att_out[head * hd..(head + 1) * hd];
                out.fill(0.0);
                for (j, s) in scores.iter().enumerate() {
                    let p = s / sum;
                    let v = &vcache[l][j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
                    for d in 0..hd {
                        out[d] += p * v[d];
                    }
                }
            }

            linear(&att_out, &layer.proj_w, &zero_e[..e], qd, e, &mut proj);
            for i in 0..e {
                x[i] += proj[i];
            }

            rmsnorm(&x, &layer.ln2_g, c.norm_eps, &mut xb);
            linear(&xb, &layer.fc_w, &zero_e[..inter], e, inter, &mut gate);
            linear(&xb, &layer.up_w, &zero_e[..inter], e, inter, &mut up);
            for i in 0..inter {
                gate[i] = silu(gate[i]) * up[i];
            }
            linear(&gate, &layer.fc2_w, &zero_e[..e], inter, e, &mut proj);
            for i in 0..e {
                x[i] += proj[i];
            }
        }
    }

    rmsnorm(&x, &model.lnf_g, c.norm_eps, &mut xb);
    for v in 0..c.n_vocab {
        logits[v] = model.wte[v * e..(v + 1) * e]
            .iter()
            .zip(&xb)
            .map(|(a, b)| a * b)
            .sum();
    }
    logits
}
