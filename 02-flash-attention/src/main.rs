//! Stage 2: Flash Attention (forward) vs naive attention with a materialized
//! N x N score matrix. Run with: `cargo run -rp flash-attention`
//!
//! Single head, head dim 64, fp32. Both implementations are verified against
//! a CPU reference (causal and non-causal), then timed across sequence
//! lengths until the naive version runs out of VRAM.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

const D: usize = 64;
const VERIFY_N: usize = 512;
const BENCH_NS: &[usize] = &[1024, 2048, 4096, 8192, 16384, 32768];
const WARMUP: usize = 1;
const ITERS: usize = 5;
const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-4;

const NAIVE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_naive.ptx"));
const FLASH_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_flash.ptx"));
const BWD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_bwd.ptx"));

struct Naive {
    scores: CudaFunction,
    softmax: CudaFunction,
    av: CudaFunction,
}

impl Naive {
    /// Enqueues the three naive kernels; `s` is the N x N scratch buffer.
    fn run(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        s: &mut CudaSlice<f32>,
        o: &mut CudaSlice<f32>,
        n: usize,
        causal: bool,
    ) {
        let n_i = n as i32;
        let causal_i = causal as i32;
        let g = n.div_ceil(16) as u32;

        let mut lb = stream.launch_builder(&self.scores);
        lb.arg(q).arg(k).arg(&mut *s).arg(&n_i).arg(&causal_i);
        let cfg = LaunchConfig { grid_dim: (g, g, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.expect("attn_scores");

        let mut lb = stream.launch_builder(&self.softmax);
        lb.arg(&mut *s).arg(&n_i);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.expect("attn_softmax");

        let mut lb = stream.launch_builder(&self.av);
        lb.arg(s).arg(v).arg(o).arg(&n_i);
        let cfg = LaunchConfig {
            grid_dim: (1, n.div_ceil(4) as u32, 1),
            block_dim: (D as u32, 4, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg) }.expect("attn_av");
    }
}

fn run_flash(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    n: usize,
    causal: bool,
) {
    let n_i = n as i32;
    let causal_i = causal as i32;
    let mut lb = stream.launch_builder(f);
    lb.arg(q).arg(k).arg(v).arg(o).arg(&n_i).arg(&causal_i);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("attn_flash");
}

/// Backward kernels: the LSE-emitting forward plus the two backward passes.
struct Bwd {
    flash_lse: CudaFunction,
    preprocess: CudaFunction,
    bwd: CudaFunction,
}

/// Forward that also writes the per-row log-sum-exp `l` (consumed by the bwd).
#[allow(clippy::too_many_arguments)]
fn run_flash_lse(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    l: &mut CudaSlice<f32>,
    n: usize,
    causal: bool,
) {
    let n_i = n as i32;
    let causal_i = causal as i32;
    let mut lb = stream.launch_builder(f);
    lb.arg(q).arg(k).arg(v).arg(&mut *o).arg(l).arg(&n_i).arg(&causal_i);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("attn_flash_lse");
}

/// Backward pass: `Delta = rowsum(dO * O)` then dQ/dK/dV. dq/dk/dv must be
/// zeroed by the caller (dK/dV accumulate with atomics).
#[allow(clippy::too_many_arguments)]
fn run_bwd(
    stream: &Arc<CudaStream>,
    b: &Bwd,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    do_: &CudaSlice<f32>,
    o: &CudaSlice<f32>,
    l: &CudaSlice<f32>,
    delta: &mut CudaSlice<f32>,
    dq: &mut CudaSlice<f32>,
    dk: &mut CudaSlice<f32>,
    dv: &mut CudaSlice<f32>,
    n: usize,
    causal: bool,
) {
    let n_i = n as i32;
    let causal_i = causal as i32;

    let mut lb = stream.launch_builder(&b.preprocess);
    lb.arg(do_).arg(o).arg(&mut *delta).arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("attn_bwd_preprocess");

    let mut lb = stream.launch_builder(&b.bwd);
    lb.arg(q).arg(k).arg(v).arg(do_).arg(l).arg(&*delta)
        .arg(&mut *dq).arg(&mut *dk).arg(&mut *dv).arg(&n_i).arg(&causal_i);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg) }.expect("attn_bwd");
}

/// CPU reference: O = softmax(Q K^T / sqrt(d)) V, row by row.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, causal: bool) -> Vec<f32> {
    let scale = 1.0 / (D as f32).sqrt();
    let mut o = vec![0.0f32; n * D];
    let mut s = vec![0.0f32; n];
    for i in 0..n {
        let cols = if causal { i + 1 } else { n };
        let mut m = f32::NEG_INFINITY;
        for j in 0..cols {
            let mut dot = 0.0;
            for x in 0..D {
                dot += q[i * D + x] * k[j * D + x];
            }
            s[j] = dot * scale;
            m = m.max(s[j]);
        }
        let mut l = 0.0;
        for j in 0..cols {
            s[j] = (s[j] - m).exp();
            l += s[j];
        }
        for j in 0..cols {
            let p = s[j] / l;
            for x in 0..D {
                o[i * D + x] += p * v[j * D + x];
            }
        }
    }
    o
}

/// CPU reference backward: gradients (dQ, dK, dV) of the scalar loss
/// `<O, dO>` w.r.t. Q, K, V. Mirrors `cpu_attention` row by row, then applies
/// the softmax VJP (same recipe the `attn_bwd` kernel implements).
fn cpu_attention_bwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    do_: &[f32],
    n: usize,
    causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (D as f32).sqrt();
    let mut dq = vec![0.0f32; n * D];
    let mut dk = vec![0.0f32; n * D];
    let mut dv = vec![0.0f32; n * D];
    let mut p = vec![0.0f32; n];
    let mut dp = vec![0.0f32; n];
    for i in 0..n {
        let cols = if causal { i + 1 } else { n };
        // forward softmax for row i
        let mut m = f32::NEG_INFINITY;
        for j in 0..cols {
            let mut dot = 0.0;
            for x in 0..D {
                dot += q[i * D + x] * k[j * D + x];
            }
            p[j] = dot * scale;
            m = m.max(p[j]);
        }
        let mut l = 0.0;
        for j in 0..cols {
            p[j] = (p[j] - m).exp();
            l += p[j];
        }
        for j in 0..cols {
            p[j] /= l;
        }
        // dP_ij = dO_i . v_j ; Delta_i = sum_j p_ij dP_ij
        let mut delta = 0.0;
        for j in 0..cols {
            let mut d = 0.0;
            for x in 0..D {
                d += do_[i * D + x] * v[j * D + x];
            }
            dp[j] = d;
            delta += p[j] * d;
        }
        for j in 0..cols {
            let ds = p[j] * (dp[j] - delta) * scale;
            for x in 0..D {
                dv[j * D + x] += p[j] * do_[i * D + x];
                dk[j * D + x] += ds * q[i * D + x];
                dq[i * D + x] += ds * k[j * D + x];
            }
        }
    }
    (dq, dk, dv)
}

/// Scalar loss the backward differentiates: `<O, dO>`, with the forward done
/// in f64 so a central difference resolves the small perturbations cleanly.
fn attn_loss(q: &[f32], k: &[f32], v: &[f32], do_: &[f32], n: usize, causal: bool) -> f64 {
    let scale = 1.0f64 / (D as f64).sqrt();
    let mut loss = 0.0f64;
    let mut s = vec![0.0f64; n];
    for i in 0..n {
        let cols = if causal { i + 1 } else { n };
        let mut m = f64::NEG_INFINITY;
        for j in 0..cols {
            let mut dot = 0.0f64;
            for x in 0..D {
                dot += q[i * D + x] as f64 * k[j * D + x] as f64;
            }
            s[j] = dot * scale;
            m = m.max(s[j]);
        }
        let mut l = 0.0f64;
        for j in 0..cols {
            s[j] = (s[j] - m).exp();
            l += s[j];
        }
        for x in 0..D {
            let mut o = 0.0f64;
            for j in 0..cols {
                o += s[j] / l * v[j * D + x] as f64;
            }
            loss += o * do_[i * D + x] as f64;
        }
    }
    loss
}

/// Finite-difference spot check: the CPU reference gradient must match a
/// central difference of `attn_loss` at a handful of sampled entries. This
/// validates the math the GPU kernel is then checked against. Returns the
/// worst relative error over the samples.
fn fd_check(n: usize, causal: bool) -> f32 {
    let q = common::pseudo_rand(n * D, 21);
    let k = common::pseudo_rand(n * D, 22);
    let v = common::pseudo_rand(n * D, 23);
    let do_ = common::pseudo_rand(n * D, 24);
    let (dq, dk, dv) = cpu_attention_bwd(&q, &k, &v, &do_, n, causal);

    let eps = 1e-3f32;
    // a spread of indices across rows and head dims
    let idxs = [0usize, 1, D + 7, 3 * D + 31, (n / 2) * D + 17, (n - 1) * D + 63];

    // central difference of the loss w.r.t. one entry of input `which`
    // (0 = Q, 1 = K, 2 = V), compared to the analytic gradient at that entry.
    let check = |which: u8, idx: usize, grad: f32| -> f32 {
        let mut qq = q.clone();
        let mut kk = k.clone();
        let mut vv = v.clone();
        let (lp, lm) = match which {
            0 => {
                let orig = qq[idx];
                qq[idx] = orig + eps;
                let lp = attn_loss(&qq, &kk, &vv, &do_, n, causal);
                qq[idx] = orig - eps;
                (lp, attn_loss(&qq, &kk, &vv, &do_, n, causal))
            }
            1 => {
                let orig = kk[idx];
                kk[idx] = orig + eps;
                let lp = attn_loss(&qq, &kk, &vv, &do_, n, causal);
                kk[idx] = orig - eps;
                (lp, attn_loss(&qq, &kk, &vv, &do_, n, causal))
            }
            _ => {
                let orig = vv[idx];
                vv[idx] = orig + eps;
                let lp = attn_loss(&qq, &kk, &vv, &do_, n, causal);
                vv[idx] = orig - eps;
                (lp, attn_loss(&qq, &kk, &vv, &do_, n, causal))
            }
        };
        let fd = ((lp - lm) / (2.0 * eps as f64)) as f32;
        (fd - grad).abs() / (1e-3 + grad.abs())
    };

    let mut worst = 0.0f32;
    for &idx in &idxs {
        worst = worst.max(check(0, idx, dq[idx]));
        worst = worst.max(check(1, idx, dk[idx]));
        worst = worst.max(check(2, idx, dv[idx]));
    }
    worst
}

fn load_kernels(
    ctx: &Arc<CudaContext>,
) -> Result<(Naive, CudaFunction), Box<dyn std::error::Error>> {
    let naive_mod = common::load_ptx(ctx, "attention_naive", NAIVE_PTX)?;
    let naive = Naive {
        scores: naive_mod.load_function("attn_scores")?,
        softmax: naive_mod.load_function("attn_softmax")?,
        av: naive_mod.load_function("attn_av")?,
    };
    let flash = common::load_ptx(ctx, "attention_flash", FLASH_PTX)?.load_function("attn_flash")?;
    Ok((naive, flash))
}

fn load_bwd(ctx: &Arc<CudaContext>) -> Result<Bwd, Box<dyn std::error::Error>> {
    let flash_mod = common::load_ptx(ctx, "attention_flash", FLASH_PTX)?;
    let bwd_mod = common::load_ptx(ctx, "attention_bwd", BWD_PTX)?;
    Ok(Bwd {
        flash_lse: flash_mod.load_function("attn_flash_lse")?,
        preprocess: bwd_mod.load_function("attn_bwd_preprocess")?,
        bwd: bwd_mod.load_function("attn_bwd")?,
    })
}

/// Both implementations match the CPU reference (causal and non-causal) at
/// VERIFY_N, allclose_err < 1.0. Shared by `main` and the `#[test]` below.
fn run_verify(
    stream: &Arc<CudaStream>,
    naive: &Naive,
    flash: &CudaFunction,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = VERIFY_N;
    let q_h = common::pseudo_rand(n * D, 1);
    let k_h = common::pseudo_rand(n * D, 2);
    let v_h = common::pseudo_rand(n * D, 3);
    let q = stream.clone_htod(&q_h)?;
    let k = stream.clone_htod(&k_h)?;
    let v = stream.clone_htod(&v_h)?;

    for causal in [false, true] {
        let want = cpu_attention(&q_h, &k_h, &v_h, n, causal);

        let mut s = stream.alloc_zeros::<f32>(n * n)?;
        let mut o = stream.alloc_zeros::<f32>(n * D)?;
        naive.run(stream, &q, &k, &v, &mut s, &mut o, n, causal);
        let got = stream.clone_dtoh(&o)?;
        let err = common::allclose_err(&got, &want, RTOL, ATOL);
        assert!(err < 1.0, "naive causal={causal}: allclose_err = {err}");
        println!("verify naive  causal={causal:<5} allclose_err = {err:.2e}  OK");

        let mut o = stream.alloc_zeros::<f32>(n * D)?;
        run_flash(stream, flash, &q, &k, &v, &mut o, n, causal);
        let got = stream.clone_dtoh(&o)?;
        let err = common::allclose_err(&got, &want, RTOL, ATOL);
        assert!(err < 1.0, "flash causal={causal}: allclose_err = {err}");
        println!("verify flash  causal={causal:<5} allclose_err = {err:.2e}  OK");
    }
    println!();
    Ok(())
}

/// GPU backward (dQ, dK, dV) matches the CPU reference (causal and non-causal)
/// at VERIFY_N, and the CPU reference matches a finite-difference check. Shared
/// by `main` and the `#[test]`.
fn run_verify_bwd(stream: &Arc<CudaStream>, b: &Bwd) -> Result<(), Box<dyn std::error::Error>> {
    let n = VERIFY_N;
    let q_h = common::pseudo_rand(n * D, 11);
    let k_h = common::pseudo_rand(n * D, 12);
    let v_h = common::pseudo_rand(n * D, 13);
    let do_h = common::pseudo_rand(n * D, 14);
    let q = stream.clone_htod(&q_h)?;
    let k = stream.clone_htod(&k_h)?;
    let v = stream.clone_htod(&v_h)?;
    let do_ = stream.clone_htod(&do_h)?;

    for causal in [false, true] {
        // forward emits O and the log-sum-exp the backward needs
        let mut o = stream.alloc_zeros::<f32>(n * D)?;
        let mut lse = stream.alloc_zeros::<f32>(n)?;
        run_flash_lse(stream, &b.flash_lse, &q, &k, &v, &mut o, &mut lse, n, causal);

        let mut delta = stream.alloc_zeros::<f32>(n)?;
        let mut dq = stream.alloc_zeros::<f32>(n * D)?;
        let mut dk = stream.alloc_zeros::<f32>(n * D)?;
        let mut dv = stream.alloc_zeros::<f32>(n * D)?;
        run_bwd(
            stream, b, &q, &k, &v, &do_, &o, &lse, &mut delta, &mut dq, &mut dk, &mut dv, n, causal,
        );

        let (wq, wk, wv) = cpu_attention_bwd(&q_h, &k_h, &v_h, &do_h, n, causal);
        for (name, got, want) in [
            ("dQ", stream.clone_dtoh(&dq)?, wq),
            ("dK", stream.clone_dtoh(&dk)?, wk),
            ("dV", stream.clone_dtoh(&dv)?, wv),
        ] {
            let err = common::allclose_err(&got, &want, RTOL, ATOL);
            assert!(err < 1.0, "bwd {name} causal={causal}: allclose_err = {err}");
            println!("verify bwd {name} causal={causal:<5} allclose_err = {err:.2e}  OK");
        }
    }

    // the analytic CPU gradient itself is sanity-checked against central
    // differences on a smaller problem (the GPU path is verified against it above)
    for causal in [false, true] {
        let fd = fd_check(128, causal);
        assert!(fd < 5e-2, "finite-difference check causal={causal}: rel err = {fd}");
        println!("verify bwd grad-check causal={causal:<5} fd rel err = {fd:.2e}  OK");
    }
    println!();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("device: {}, head dim = {D}\n", ctx.name()?);

    let (naive, flash) = load_kernels(&ctx)?;
    run_verify(&stream, &naive, &flash)?;

    let bwd = load_bwd(&ctx)?;
    run_verify_bwd(&stream, &bwd)?;

    // --- benchmark (non-causal): time + extra memory for scores ---
    println!(
        "| {:>6} | {:>16} | {:>12} | {:>7} | {:>13} |",
        "N", "naive (S = NxN)", "flash", "speedup", "naive S extra"
    );
    println!("|{}|{}|{}|{}|{}|", "-".repeat(8), "-".repeat(18), "-".repeat(14), "-".repeat(9), "-".repeat(15));
    for &n in BENCH_NS {
        let q = stream.clone_htod(&common::pseudo_rand(n * D, 1))?;
        let k = stream.clone_htod(&common::pseudo_rand(n * D, 2))?;
        let v = stream.clone_htod(&common::pseudo_rand(n * D, 3))?;
        let mut o = stream.alloc_zeros::<f32>(n * D)?;

        let flash_ms =
            common::time_median_ms(&stream, WARMUP, ITERS, || {
                run_flash(&stream, &flash, &q, &k, &v, &mut o, n, false);
                Ok::<(), ()>(())
            })?;

        let s_bytes = n * n * 4;
        let naive_cell = match stream.alloc_zeros::<f32>(n * n) {
            Ok(mut s) => {
                let ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
                    naive.run(&stream, &q, &k, &v, &mut s, &mut o, n, false);
                    Ok::<(), ()>(())
                })?;
                Some(ms)
            }
            Err(_) => None,
        };

        match naive_cell {
            Some(ms) => println!(
                "| {:>6} | {:>13.1} ms | {:>9.1} ms | {:>6.2}x | {:>10.0} MB |",
                n,
                ms,
                flash_ms,
                ms / flash_ms,
                s_bytes as f64 / 1e6
            ),
            None => println!(
                "| {:>6} | {:>16} | {:>9.1} ms | {:>7} | {:>10.0} MB |",
                n,
                "OOM",
                flash_ms,
                "-",
                s_bytes as f64 / 1e6
            ),
        }
    }

    // --- backward benchmark (causal): forward vs backward time ---
    println!("\n| {:>6} | {:>10} | {:>10} | {:>9} |", "N", "fwd", "bwd", "bwd/fwd");
    println!("|{}|{}|{}|{}|", "-".repeat(8), "-".repeat(12), "-".repeat(12), "-".repeat(11));
    for &n in &[1024usize, 2048, 4096] {
        let q = stream.clone_htod(&common::pseudo_rand(n * D, 1))?;
        let k = stream.clone_htod(&common::pseudo_rand(n * D, 2))?;
        let v = stream.clone_htod(&common::pseudo_rand(n * D, 3))?;
        let do_ = stream.clone_htod(&common::pseudo_rand(n * D, 4))?;
        let mut o = stream.alloc_zeros::<f32>(n * D)?;
        let mut lse = stream.alloc_zeros::<f32>(n)?;
        let mut delta = stream.alloc_zeros::<f32>(n)?;
        let mut dq = stream.alloc_zeros::<f32>(n * D)?;
        let mut dk = stream.alloc_zeros::<f32>(n * D)?;
        let mut dv = stream.alloc_zeros::<f32>(n * D)?;

        let fwd_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_flash(&stream, &flash, &q, &k, &v, &mut o, n, true);
            Ok::<(), ()>(())
        })?;
        // forward once for O + log-sum-exp; the backward then reads them. dK/dV
        // accumulate via atomics — harmless for timing, so no per-iter zeroing.
        run_flash_lse(&stream, &bwd.flash_lse, &q, &k, &v, &mut o, &mut lse, n, true);
        let bwd_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_bwd(
                &stream, &bwd, &q, &k, &v, &do_, &o, &lse, &mut delta, &mut dq, &mut dk, &mut dv, n,
                true,
            );
            Ok::<(), ()>(())
        })?;
        println!(
            "| {:>6} | {:>7.2} ms | {:>7.2} ms | {:>8.2}x |",
            n,
            fwd_ms,
            bwd_ms,
            bwd_ms / fwd_ms
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive and flash attention match the CPU reference (causal + non-causal).
    /// Skips gracefully (green) when no CUDA device is present or the PTX is an
    /// empty stub (nvcc missing at build time).
    #[test]
    fn attention_matches_cpu() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip flash test: no CUDA device ({e:?})");
                return;
            }
        };
        if NAIVE_PTX.trim().is_empty() || FLASH_PTX.trim().is_empty() || BWD_PTX.trim().is_empty() {
            eprintln!("skip flash test: PTX is an empty stub (nvcc missing at build time)");
            return;
        }
        let stream = ctx.default_stream();
        let (naive, flash) = load_kernels(&ctx).unwrap();
        run_verify(&stream, &naive, &flash).unwrap();
    }

    /// Backward dQ/dK/dV match the CPU reference (causal + non-causal) and the
    /// CPU reference matches finite differences. Same graceful skip as above.
    #[test]
    fn flash_backward_matches_cpu() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip bwd test: no CUDA device ({e:?})");
                return;
            }
        };
        if FLASH_PTX.trim().is_empty() || BWD_PTX.trim().is_empty() {
            eprintln!("skip bwd test: PTX is an empty stub (nvcc missing at build time)");
            return;
        }
        let stream = ctx.default_stream();
        let bwd = load_bwd(&ctx).unwrap();
        run_verify_bwd(&stream, &bwd).unwrap();
    }
}
