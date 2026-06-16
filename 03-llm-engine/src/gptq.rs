//! GPTQ: Hessian-guided 4-bit weight quantization onto the existing Q4_0
//! layout.
//!
//! Round-to-nearest (quantize_q4 in gpu.rs) quantizes each weight in isolation;
//! GPTQ instead quantizes the input channels of a linear in order and, after
//! rounding channel `i`, pushes the rounding error onto the not-yet-quantized
//! channels along the direction the input Hessian `H = Σ_t x_t x_tᵀ` says
//! matters least for the layer output. The output is the **same** `(q, scales)`
//! byte layout `quantize_q4` produces (signed `m/-8` scale per 32-input-channel
//! group, nibbles repacked into the dp4a int32 words), so the int4 kernels and
//! upload path are untouched — only the bits change.
//!
//! The math (Frantar et al. 2022): with `Hinv = chol(H⁻¹)` (upper, so
//! `H⁻¹ = Uᵀ U`), quantizing channel `i` with error `e = (w - q)/U[i,i]` and
//! updating the trailing channels `W[:, j>i] -= e · U[i, j]` is the optimal
//! greedy step under the `‖ΔW·X‖²` metric. We reuse one calibration pass'
//! Hessians (calib.rs) and quantize every layer independently (and in parallel).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use half::f16;

use crate::calib::Hessians;
use crate::model::Model;

const MAGIC: u32 = u32::from_le_bytes(*b"MXGQ");
const VERSION: u32 = 2; // v2 adds the per-linear act-order permutation
const GROUP: usize = 32; // must match Q4_GROUP in gpu.rs

/// Which linear inside a block. Mirrors the four `cpu::Site`s plus `Up` (the
/// Qwen SwiGLU up-projection, which shares the `Fc` input Hessian).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Role {
    Qkv,
    Proj,
    Fc,
    Up,
    Fc2,
}

impl Role {
    fn tag(self) -> u32 {
        match self {
            Role::Qkv => 0,
            Role::Proj => 1,
            Role::Fc => 2,
            Role::Up => 3,
            Role::Fc2 => 4,
        }
    }
    fn from_tag(t: u32) -> Role {
        match t {
            0 => Role::Qkv,
            1 => Role::Proj,
            2 => Role::Fc,
            3 => Role::Up,
            4 => Role::Fc2,
            _ => panic!("bad GPTQ role tag {t}"),
        }
    }
}

/// Pre-quantized Q4_0 blob for one linear: nibbles + per-group scales in the
/// `quantize_q4` layout, plus the act-order permutation. `perm[k]` is the
/// original input channel stored at permuted position `k`; weights are stored
/// in permuted order so the contiguous-32-group scales line up with GPTQ's
/// descending-Hessian quantization order, and the GEMV gathers the activation
/// by `perm` (the dot product is permutation-invariant). Identity perm when
/// act-order is off.
pub struct Entry {
    pub q: Vec<u8>,
    pub scales: Vec<f16>,
    pub perm: Vec<i32>,
}

/// Pre-quantized Q4_0 blobs for the GPTQ-covered linears, keyed by (layer,
/// role). Tensors not present (embeddings, norms, lm_head) fall back to the
/// normal upload path in `Engine::new`.
pub struct Sidecar {
    entries: HashMap<(usize, Role), Entry>,
}

impl Sidecar {
    pub fn get(&self, layer: usize, role: Role) -> Option<&Entry> {
        self.entries.get(&(layer, role))
    }

    /// True if any entry actually reorders channels (non-identity perm) — the
    /// engine then routes prefill through the perm-aware decode GEMV.
    pub fn has_act_order(&self) -> bool {
        self.entries
            .values()
            .any(|e| e.perm.iter().enumerate().any(|(k, &p)| p as usize != k))
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(&MAGIC.to_le_bytes())?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        // deterministic order for reproducible files
        let mut keys: Vec<_> = self.entries.keys().copied().collect();
        keys.sort_by_key(|(l, r)| (*l, r.tag()));
        for key @ (l, r) in keys {
            let e = &self.entries[&key];
            out.write_all(&(l as u32).to_le_bytes())?;
            out.write_all(&r.tag().to_le_bytes())?;
            out.write_all(&(e.q.len() as u32).to_le_bytes())?;
            out.write_all(&e.q)?;
            out.write_all(&(e.scales.len() as u32).to_le_bytes())?;
            let sb =
                unsafe { std::slice::from_raw_parts(e.scales.as_ptr() as *const u8, e.scales.len() * 2) };
            out.write_all(sb)?;
            out.write_all(&(e.perm.len() as u32).to_le_bytes())?;
            let pb =
                unsafe { std::slice::from_raw_parts(e.perm.as_ptr() as *const u8, e.perm.len() * 4) };
            out.write_all(pb)?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let buf = std::fs::read(path)?;
        let mut pos = 0usize;
        let rd = |pos: &mut usize| {
            let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v
        };
        assert_eq!(rd(&mut pos), MAGIC, "not a GPTQ sidecar");
        assert_eq!(rd(&mut pos), VERSION, "unsupported GPTQ sidecar version");
        let n = rd(&mut pos) as usize;
        let mut entries = HashMap::with_capacity(n);
        for _ in 0..n {
            let l = rd(&mut pos) as usize;
            let r = Role::from_tag(rd(&mut pos));
            let qlen = rd(&mut pos) as usize;
            let q = buf[pos..pos + qlen].to_vec();
            pos += qlen;
            let slen = rd(&mut pos) as usize;
            let scales: Vec<f16> = buf[pos..pos + slen * 2]
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            pos += slen * 2;
            let plen = rd(&mut pos) as usize;
            let perm: Vec<i32> = buf[pos..pos + plen * 4]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            pos += plen * 4;
            entries.insert((l, r), Entry { q, scales, perm });
        }
        Ok(Sidecar { entries })
    }
}

/// One linear's quantization job: its weights, dims and the matching Hessian.
struct Job<'a> {
    role: Role,
    w: &'a [f32],
    h: &'a [f32],
    n_in: usize,
    n_out: usize,
}

/// Build a GPTQ sidecar for `model` from calibration `hess`. Layers are
/// quantized in parallel (capped to keep the f64 work matrices in RAM).
/// `act_order` quantizes input channels in descending-Hessian order (the
/// standard fix for the error-feedback instability — essential here, see the
/// GPTQ commit message); off => identity perm == plain order.
pub fn build(model: &Model, hess: &Hessians, damp: f64, act_order: bool) -> Sidecar {
    let c = model.config;
    let (e, qd, qkvd, inter) = (c.n_embd, c.q_dim(), c.qkv_dim(), c.n_inter);
    // the Hessians' per-site input widths must match the linears we feed them
    assert_eq!(hess.n_in, [e, qd, e, inter], "Hessian/model dim mismatch");

    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Vec<((usize, Role), Entry)>>> =
        (0..c.n_layer).map(|_| Mutex::new(Vec::new())).collect();
    // Each worker holds up to two n×n f64 matrices for the widest linear it
    // touches; cap concurrency so peak RAM stays bounded (~0.4 GB/thread at
    // n=4864). The Hessians themselves already cost Σ n_in²·n_layer·4 B.
    let mem_cap = if inter.max(qkvd) > 4096 { 2 } else { 4 };
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(mem_cap)
        .min(c.n_layer.max(1));

    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| loop {
                let l = next.fetch_add(1, Ordering::Relaxed);
                if l >= c.n_layer {
                    break;
                }
                let layer = &model.layers[l];
                let h = &hess.h[l];
                // (role, weights, hessian site, n_in, n_out)
                let mut jobs = vec![
                    Job { role: Role::Qkv, w: &layer.qkv_w, h: &h[0], n_in: e, n_out: qkvd },
                    Job { role: Role::Proj, w: &layer.proj_w, h: &h[1], n_in: qd, n_out: e },
                    Job { role: Role::Fc, w: &layer.fc_w, h: &h[2], n_in: e, n_out: inter },
                    Job { role: Role::Fc2, w: &layer.fc2_w, h: &h[3], n_in: inter, n_out: e },
                ];
                if !layer.up_w.is_empty() {
                    // SwiGLU up shares the ln2 (Fc) input Hessian
                    jobs.push(Job { role: Role::Up, w: &layer.up_w, h: &h[2], n_in: e, n_out: inter });
                }
                let mut done = Vec::with_capacity(jobs.len());
                for j in jobs {
                    let entry = quantize_linear(j.w, j.h, j.n_in, j.n_out, damp, act_order);
                    done.push(((l, j.role), entry));
                }
                *out[l].lock().unwrap() = done;
                eprintln!("GPTQ: layer {l} done");
            });
        }
    });

    let mut entries = HashMap::new();
    for m in out {
        for (k, v) in m.into_inner().unwrap() {
            entries.insert(k, v);
        }
    }
    Sidecar { entries }
}

/// GPTQ-quantize one `[n_in, n_out]` linear (row-major) given its input
/// Hessian `h` (`[n_in, n_in]`). Produces `(q, scales)` byte-identical in
/// layout to `quantize_q4`, plus the act-order permutation. With `act_order`
/// the input channels are processed (and stored) in descending-Hessian order,
/// so the contiguous-32-group scales align with GPTQ's quantization order; the
/// GEMV later gathers the activation by `perm` (the dot is permutation-
/// invariant). Group scales are static (from the original weights).
fn quantize_linear(
    w: &[f32],
    h: &[f32],
    n_in: usize,
    n_out: usize,
    damp: f64,
    act_order: bool,
) -> Entry {
    assert!(n_in.is_multiple_of(GROUP), "GPTQ needs n_in % {GROUP} == 0");

    // perm[k] = original input channel placed at permuted position k
    let mut perm: Vec<i32> = (0..n_in as i32).collect();
    if act_order {
        perm.sort_by(|&a, &b| {
            let (da, db) = (h[a as usize * n_in + a as usize], h[b as usize * n_in + b as usize]);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    // permute the Hessian (rows+cols) and weight rows into that order; when the
    // perm is identity these are plain copies, so the body below is oblivious
    let hp = permute_sym(h, n_in, &perm);
    let wp = permute_rows(w, n_in, n_out, &perm);
    let (w, h) = (wp.as_slice(), hp.as_slice());

    let u = hinv_upper(h, n_in, damp);

    // working weights, column-major [n_out][n_in] so the trailing-channel
    // error update and the group absmax are both sequential in memory
    let mut wc = vec![0.0f32; n_out * n_in];
    for i in 0..n_in {
        let dead = h[i * n_in + i] == 0.0;
        for o in 0..n_out {
            wc[o * n_in + i] = if dead { 0.0 } else { w[i * n_out + o] };
        }
    }

    let n_groups = n_in / GROUP;
    let mut q = vec![0u8; n_in / 2 * n_out];

    // Static per-group scales from the ORIGINAL weights — identical to
    // quantize_q4. Decoupling the grid from the running (error-corrected)
    // weights keeps the quantization grid stable; with a contiguous-group
    // layout (no act-order possible) this is the robust GPTQ choice, and it
    // makes a huge `damp` collapse byte-exactly to round-to-nearest.
    let mut scales = vec![f16::ZERO; n_groups * n_out];
    for o in 0..n_out {
        for g in 0..n_groups {
            let mut m = 0.0f32;
            for i in g * GROUP..(g + 1) * GROUP {
                let v = w[i * n_out + o];
                if v.abs() > m.abs() {
                    m = v;
                }
            }
            scales[g * n_out + o] = f16::from_f32(m / -8.0);
        }
    }

    for i in 0..n_in {
        let g = i / GROUP;
        let uii = u[i * n_in + i];
        let inv_uii = if uii > 1e-12 { 1.0 / uii } else { 0.0 };
        let urow = &u[i * n_in..i * n_in + n_in];
        for o in 0..n_out {
            let woff = o * n_in;
            let d = scales[g * n_out + o].to_f32();
            let wv = wc[woff + i];
            let (nib, dq) = if d != 0.0 {
                let nib = ((wv / d).round() + 8.0).clamp(0.0, 15.0) as u8;
                (nib, (nib as f32 - 8.0) * d)
            } else {
                (8u8, 0.0)
            };
            q[((i / 8) * n_out + o) * 4 + (i % 4)] |= nib << (4 * ((i % 8) / 4));
            let err = (wv - dq) * inv_uii as f32;
            if err != 0.0 {
                let wo = &mut wc[woff..woff + n_in];
                for j in (i + 1)..n_in {
                    wo[j] -= err * urow[j] as f32;
                }
            }
        }
    }
    Entry { q, scales, perm }
}

/// Symmetric permutation `hp[a][b] = h[perm[a]][perm[b]]`.
fn permute_sym(h: &[f32], n: usize, perm: &[i32]) -> Vec<f32> {
    let mut hp = vec![0.0f32; n * n];
    for a in 0..n {
        let pa = perm[a] as usize * n;
        let dst = &mut hp[a * n..a * n + n];
        for b in 0..n {
            dst[b] = h[pa + perm[b] as usize];
        }
    }
    hp
}

/// Row permutation of a `[n_in, n_out]` matrix: row `k` of the result is row
/// `perm[k]` of the input.
fn permute_rows(w: &[f32], n_in: usize, n_out: usize, perm: &[i32]) -> Vec<f32> {
    let mut wp = vec![0.0f32; n_in * n_out];
    for k in 0..n_in {
        let src = perm[k] as usize * n_out;
        wp[k * n_out..k * n_out + n_out].copy_from_slice(&w[src..src + n_out]);
    }
    wp
}

/// Upper-triangular Cholesky factor `U` of the inverse of the damped Hessian:
/// `(H + λI)⁻¹ = Uᵀ U`. Built as `U = L2ᵀ` where `H⁻¹ = L2 L2ᵀ` — the lower
/// Cholesky path keeps every inner product sequential in memory. All in f64.
fn hinv_upper(h: &[f32], n: usize, damp: f64) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut diagsum = 0.0f64;
    for i in 0..n {
        diagsum += h[i * n + i] as f64;
    }
    let lambda = (damp * (diagsum / n as f64)).max(1e-9);
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = h[i * n + j] as f64;
        }
        // dead diagonal -> identity row so the factorization stays defined
        if a[i * n + i] <= 0.0 {
            a[i * n + i] = 1.0;
        }
        a[i * n + i] += lambda;
    }
    // free each big intermediate as soon as it is consumed — at most two
    // n×n f64 matrices are live at once (the widest FFN-down Hessian is ~190 MB
    // each at n=4864, so this roughly halves the per-thread peak)
    let l = cholesky_lower(&a, n); // A = L Lᵀ
    drop(a);
    let linv = invert_lower(&l, n); // L⁻¹ (lower)
    drop(l);
    let hinv = mt_m(&linv, n); // Linvᵀ Linv = A⁻¹ (full symmetric)
    drop(linv);
    let l2 = cholesky_lower(&hinv, n); // A⁻¹ = L2 L2ᵀ
    drop(hinv);
    // U = L2ᵀ (upper): A⁻¹ = Uᵀ U
    let mut u = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = l2[j * n + i];
        }
    }
    u
}

/// Lower Cholesky `A = L Lᵀ` (A symmetric PD, only its lower triangle is read).
fn cholesky_lower(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        let ljj = if s > 1e-12 { s.sqrt() } else { 1e-6 };
        l[j * n + j] = ljj;
        let inv = 1.0 / ljj;
        for i in (j + 1)..n {
            let (li, lj) = (&l[i * n..i * n + j], &l[j * n..j * n + j]);
            let mut s2 = a[i * n + j];
            for k in 0..j {
                s2 -= li[k] * lj[k];
            }
            l[i * n + j] = s2 * inv;
        }
    }
    l
}

/// Inverse of a lower-triangular matrix, column by column (forward
/// substitution into a sequential temp), result lower-triangular.
fn invert_lower(l: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n * n];
    let mut y = vec![0.0f64; n];
    for j in 0..n {
        y[j..].iter_mut().for_each(|v| *v = 0.0);
        y[j] = 1.0 / l[j * n + j];
        for i in (j + 1)..n {
            let lrow = &l[i * n..i * n + i];
            let mut s = 0.0;
            for k in j..i {
                s += lrow[k] * y[k];
            }
            y[i] = -s / l[i * n + i];
        }
        for i in j..n {
            x[i * n + j] = y[i];
        }
    }
    x
}

/// `Mᵀ M` for a lower-triangular `M` (row-`k` rank-1 updates keep it
/// sequential), returned as a full symmetric matrix.
fn mt_m(m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for k in 0..n {
        let mrow = &m[k * n..k * n + n];
        for a in 0..=k {
            let mka = mrow[a];
            if mka == 0.0 {
                continue;
            }
            let hr = &mut out[a * n..a * n + n];
            for b in a..=k {
                hr[b] += mka * mrow[b];
            }
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            out[b * n + a] = out[a * n + b];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::pseudo_rand;

    const DEFAULT_DAMP: f64 = 0.01;

    /// Dequantize a Q4_0 blob (matches the kernel: w ~ (nib-8)*scale).
    fn dequant(q: &[u8], s: &[f16], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n_in * n_out];
        for i in 0..n_in {
            for o in 0..n_out {
                let nib = (q[((i / 8) * n_out + o) * 4 + (i % 4)] >> (4 * ((i % 8) / 4))) & 0xf;
                let d = s[(i / GROUP) * n_out + o].to_f32();
                w[i * n_out + o] = (nib as f32 - 8.0) * d;
            }
        }
        w
    }

    /// Round-to-nearest reference (the quantize_q4 grid), dequantized.
    fn rtn(w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n_in * n_out];
        for o in 0..n_out {
            for g in 0..n_in / GROUP {
                let mut m = 0.0f32;
                for i in g * GROUP..(g + 1) * GROUP {
                    if w[i * n_out + o].abs() > m.abs() {
                        m = w[i * n_out + o];
                    }
                }
                let d = f16::from_f32(m / -8.0).to_f32();
                for i in g * GROUP..(g + 1) * GROUP {
                    let nib = if d != 0.0 {
                        ((w[i * n_out + o] / d).round() + 8.0).clamp(0.0, 15.0)
                    } else {
                        8.0
                    };
                    out[i * n_out + o] = (nib - 8.0) * d;
                }
            }
        }
        out
    }

    /// H-weighted reconstruction error trace((W-Wq)ᵀ H (W-Wq)) — the quantity
    /// GPTQ minimizes. Summed over outputs.
    fn h_err(w: &[f32], wq: &[f32], h: &[f32], n_in: usize, n_out: usize) -> f64 {
        let mut total = 0.0f64;
        for o in 0..n_out {
            let dw: Vec<f64> = (0..n_in).map(|i| (w[i * n_out + o] - wq[i * n_out + o]) as f64).collect();
            for i in 0..n_in {
                let mut hd = 0.0f64;
                for j in 0..n_in {
                    hd += h[i * n_in + j] as f64 * dw[j];
                }
                total += dw[i] * hd;
            }
        }
        total
    }

    /// GPTQ must achieve no worse H-weighted error than round-to-nearest — the
    /// whole point of the Hessian-guided update. A wrong Cholesky/U convention
    /// or propagation sign would break this.
    #[test]
    fn gptq_beats_rtn_on_h_metric() {
        let (n_in, n_out) = (64usize, 12usize); // 2 groups
        let w = pseudo_rand(n_in * n_out, 1);
        // H = BBᵀ + 0.1 I, symmetric PD, with correlated channels
        let b = pseudo_rand(n_in * n_in, 7);
        let mut h = vec![0.0f32; n_in * n_in];
        for i in 0..n_in {
            for j in 0..n_in {
                let mut s = 0.0f32;
                for k in 0..n_in {
                    s += b[i * n_in + k] * b[j * n_in + k];
                }
                h[i * n_in + j] = s + if i == j { 0.1 } else { 0.0 };
            }
        }
        let e = quantize_linear(&w, &h, n_in, n_out, DEFAULT_DAMP, false);
        let wq_gptq = dequant(&e.q, &e.scales, n_in, n_out);
        let wq_rtn = rtn(&w, n_in, n_out);
        let eg = h_err(&w, &wq_gptq, &h, n_in, n_out);
        let er = h_err(&w, &wq_rtn, &h, n_in, n_out);
        assert!(
            eg <= er * 1.02 + 1e-6,
            "GPTQ H-error {eg:.4} should be <= RTN {er:.4}"
        );
        assert_eq!(e.perm, (0..n_in as i32).collect::<Vec<_>>(), "no act-order => identity perm");

        // act-order: weights come back in permuted order; un-permuting by `perm`
        // (the GEMV gathers the activation the same way) must reconstruct a
        // valid quantization that still beats RTN on the H-metric.
        let ea = quantize_linear(&w, &h, n_in, n_out, DEFAULT_DAMP, true);
        let wq_perm = dequant(&ea.q, &ea.scales, n_in, n_out);
        let mut wq_ao = vec![0.0f32; n_in * n_out];
        for k in 0..n_in {
            let orig = ea.perm[k] as usize;
            wq_ao[orig * n_out..orig * n_out + n_out]
                .copy_from_slice(&wq_perm[k * n_out..k * n_out + n_out]);
        }
        let ea_err = h_err(&w, &wq_ao, &h, n_in, n_out);
        assert!(
            ea_err <= er * 1.02 + 1e-6,
            "act-order GPTQ H-error {ea_err:.4} should be <= RTN {er:.4}"
        );
    }
}
