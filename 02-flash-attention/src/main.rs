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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("device: {}, head dim = {D}\n", ctx.name()?);

    let (naive, flash) = load_kernels(&ctx)?;
    run_verify(&stream, &naive, &flash)?;

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
        if NAIVE_PTX.trim().is_empty() || FLASH_PTX.trim().is_empty() {
            eprintln!("skip flash test: PTX is an empty stub (nvcc missing at build time)");
            return;
        }
        let stream = ctx.default_stream();
        let (naive, flash) = load_kernels(&ctx).unwrap();
        run_verify(&stream, &naive, &flash).unwrap();
    }
}
