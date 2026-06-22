//! Stage 4: a fused GEMM + bias + GELU written with CuTe (CUTLASS 3.x) on the
//! SIMT path (sm_61 has no tensor cores), benchmarked against cuBLAS, the
//! hand-rolled stage-1 `gemm_06`, and an unfused two-kernel baseline.
//! Run with: `cargo run -rp cutlass`
//!
//! Shape is the GPT-2 ffn-up: C[M,N] = GELU(x[M,K]·W[K,N] + bias[N]), with
//! K = 768 (n_embd), N = 3072 (4·n_embd), M = prefill tokens. GEMM-only
//! kernels are compared to A·B; the fused/unfused ones to GELU(A·B + bias).
//!
//! Layouts differ per implementation (cuBLAS/gemm_06 row-major, CuTe column-
//! major NT) but the problem and FLOP count (2·M·N·K) are identical, so the
//! GFLOPS comparison is fair; every implementation is verified against the same
//! CPU reference.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

const CUTE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cute_gemm.ptx"));
const GEMM06_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gemm_06_dbuf.ptx"));

// verify on a small divisible shape (fast CPU reference); bench on ffn-up shapes
const VERIFY_MKN: (usize, usize, usize) = (128, 256, 256);
const BENCH_K: usize = 768;
const BENCH_N: usize = 3072;
const BENCH_MS: &[usize] = &[128, 256, 512]; // prefill token counts
const WARMUP: usize = 2;
const ITERS: usize = 7;
// numpy-style allclose: |g - w| <= ATOL + RTOL*|w|; robust on near-zero outputs
const RTOL: f32 = 2e-3;
const ATOL: f32 = 2e-3;

struct Kernels {
    cute_gemm: CudaFunction,
    cute_fused: CudaFunction,
    bias_gelu: CudaFunction,
    gemm06: CudaFunction,
}

fn gflops(m: usize, n: usize, k: usize, ms: f32) -> f32 {
    (2.0 * m as f64 * n as f64 * k as f64 / (ms as f64 * 1e6)) as f32
}

fn gelu_tanh(x: f32) -> f32 {
    let k = 0.797_884_56_f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

/// A (M×K, row-major) reinterpreted as the column-major M×K buffer CuTe wants:
/// a_cm[m + k*M] = a_rm[m*K + k].
fn to_colmajor(a_rm: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            out[i + j * m] = a_rm[i * k + j];
        }
    }
    out
}

/// Row-major reference: C[M,N] = A[M,K]·B[K,N] (no epilogue), accumulated in f64
/// so it is a clean ground truth for the fp32 implementations.
fn cpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p] as f64;
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j] as f64;
            }
        }
    }
    c.iter().map(|&x| x as f32).collect()
}

/// cuBLAS for row-major C[M,N] = A[M,K]·B[K,N]: column-major sgemm with A and B
/// swapped (m=N, n=M, a=B lda=N, b=A ldb=K, ldc=N). Generalizes stage 1.
fn cublas_gemm(blas: &CudaBlas, m: usize, k: usize, n: usize,
               a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>) {
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };
    unsafe { blas.gemm(cfg, b, a, c) }.expect("cublas sgemm");
}

fn cfg_cute(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((m / 128) as u32, (n / 128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
fn cfg_gemm06(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n / 128) as u32, (m / 128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

/// CuTe GEMM (NT, column-major), C_cm = alpha·A_cm·B.
fn run_cute_gemm(stream: &Arc<CudaStream>, f: &CudaFunction, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
                 c: &mut CudaSlice<f32>, m: usize, n: usize, k: usize) {
    let (mi, ni, ki, alpha) = (m as i32, n as i32, k as i32, 1.0f32);
    let mut lb = stream.launch_builder(f);
    lb.arg(a).arg(b).arg(&mut *c).arg(&mi).arg(&ni).arg(&ki).arg(&alpha);
    unsafe { lb.launch(cfg_cute(m, n)) }.expect("cute_gemm");
}

/// Fused CuTe GEMM+bias+GELU, C_cm = GELU(alpha·A_cm·B + bias).
fn run_cute_fused(stream: &Arc<CudaStream>, f: &CudaFunction, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
                  bias: &CudaSlice<f32>, c: &mut CudaSlice<f32>, m: usize, n: usize, k: usize) {
    let (mi, ni, ki, alpha) = (m as i32, n as i32, k as i32, 1.0f32);
    let mut lb = stream.launch_builder(f);
    lb.arg(a).arg(b).arg(bias).arg(&mut *c).arg(&mi).arg(&ni).arg(&ki).arg(&alpha);
    unsafe { lb.launch(cfg_cute(m, n)) }.expect("cute_gemm_bias_gelu");
}

/// Elementwise GELU(C + bias) over column-major C (the unfused epilogue).
fn run_bias_gelu(stream: &Arc<CudaStream>, f: &CudaFunction, c: &mut CudaSlice<f32>,
                 bias: &CudaSlice<f32>, m: usize, n: usize) {
    let (mi, ni) = (m as i32, n as i32);
    let mut lb = stream.launch_builder(f);
    lb.arg(&mut *c).arg(bias).arg(&mi).arg(&ni);
    let cfg = LaunchConfig { grid_dim: ((m * n).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe { lb.launch(cfg) }.expect("bias_gelu");
}

/// Hand-rolled gemm_06 (row-major), C_rm = A_rm·B.
fn run_gemm06(stream: &Arc<CudaStream>, f: &CudaFunction, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
              c: &mut CudaSlice<f32>, m: usize, n: usize, k: usize) {
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    let mut lb = stream.launch_builder(f);
    lb.arg(a).arg(b).arg(&mut *c).arg(&mi).arg(&ni).arg(&ki);
    unsafe { lb.launch(cfg_gemm06(m, n)) }.expect("gemm_dbuf");
}

fn load_kernels(ctx: &Arc<CudaContext>) -> Result<Kernels, Box<dyn std::error::Error>> {
    let cute = common::load_ptx(ctx, "cute_gemm", CUTE_PTX)?;
    let g06 = common::load_ptx(ctx, "gemm_06_dbuf", GEMM06_PTX)?;
    Ok(Kernels {
        cute_gemm: cute.load_function("cute_gemm")?,
        cute_fused: cute.load_function("cute_gemm_bias_gelu")?,
        bias_gelu: cute.load_function("bias_gelu")?,
        gemm06: g06.load_function("gemm_dbuf")?,
    })
}

/// Every implementation matches the CPU reference at VERIFY_MKN (max_rel_err <
/// TOL). GEMM-only kernels are checked against A·B; the fused / unfused paths
/// against GELU(A·B + bias). Shared by `main` and the `#[test]`.
fn run_verify(stream: &Arc<CudaStream>, blas: &CudaBlas, k: &Kernels) -> Result<(), Box<dyn std::error::Error>> {
    let (m, kk, n) = VERIFY_MKN;
    let a_rm = common::pseudo_rand(m * kk, 1);
    let b = common::pseudo_rand(kk * n, 2);
    let bias = common::pseudo_rand(n, 3);
    let a_cm = to_colmajor(&a_rm, m, kk);

    let want_gemm = cpu_gemm(&a_rm, &b, m, kk, n);
    let want_fused: Vec<f32> = want_gemm.iter().enumerate().map(|(idx, &v)| gelu_tanh(v + bias[idx % n])).collect();

    let a_rm_d = stream.clone_htod(&a_rm)?;
    let a_cm_d = stream.clone_htod(&a_cm)?;
    let b_d = stream.clone_htod(&b)?;
    let bias_d = stream.clone_htod(&bias)?;

    // row-major outputs: cuBLAS, gemm_06 vs A·B
    let mut c = stream.alloc_zeros::<f32>(m * n)?;
    cublas_gemm(blas, m, kk, n, &a_rm_d, &b_d, &mut c);
    check(stream, "cuBLAS", &c, &want_gemm, |g| g.to_vec())?;

    let mut c = stream.alloc_zeros::<f32>(m * n)?;
    run_gemm06(stream, &k.gemm06, &a_rm_d, &b_d, &mut c, m, n, kk);
    check(stream, "gemm_06", &c, &want_gemm, |g| g.to_vec())?;

    // column-major outputs: read back as row-major via c_cm[m + n*M]
    let to_rm = |c_cm: &[f32]| (0..m * n).map(|i| c_cm[(i / n) + (i % n) * m]).collect::<Vec<f32>>();

    let mut c = stream.alloc_zeros::<f32>(m * n)?;
    run_cute_gemm(stream, &k.cute_gemm, &a_cm_d, &b_d, &mut c, m, n, kk);
    check(stream, "CuTe gemm", &c, &want_gemm, to_rm)?;

    let mut c = stream.alloc_zeros::<f32>(m * n)?;
    run_cute_fused(stream, &k.cute_fused, &a_cm_d, &b_d, &bias_d, &mut c, m, n, kk);
    check(stream, "CuTe fused", &c, &want_fused, to_rm)?;

    let mut c = stream.alloc_zeros::<f32>(m * n)?;
    run_cute_gemm(stream, &k.cute_gemm, &a_cm_d, &b_d, &mut c, m, n, kk);
    run_bias_gelu(stream, &k.bias_gelu, &mut c, &bias_d, m, n);
    check(stream, "CuTe unfused", &c, &want_fused, to_rm)?;

    println!();
    Ok(())
}

fn check(stream: &Arc<CudaStream>, name: &str, c: &CudaSlice<f32>, want: &[f32],
         to_rm: impl Fn(&[f32]) -> Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let got = to_rm(&stream.clone_dtoh(c)?);
    let err = common::allclose_err(&got, want, RTOL, ATOL);
    assert!(err < 1.0, "{name}: allclose_err = {err}");
    println!("verify {name:<13} allclose_err = {err:.2e}  OK");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    println!("device: {}\n", ctx.name()?);

    let k = load_kernels(&ctx)?;
    run_verify(&stream, &blas, &k)?;

    // --- benchmark (GPT-2 ffn-up: K=768, N=3072, M = prefill tokens) ---
    let names = ["cuBLAS (gemm)", "gemm_06 (gemm)", "CuTe gemm", "CuTe unfused", "CuTe fused"];
    let mut rows: Vec<Vec<f32>> = vec![Vec::new(); names.len()];
    // (M, gemm_ms, biasgelu_alone_ms, unfused_ms, fused_ms)
    let mut fusion: Vec<(usize, f32, f32, f32, f32)> = Vec::new();

    for &m in BENCH_MS {
        let (kk, n) = (BENCH_K, BENCH_N);
        let a_rm = stream.clone_htod(&common::pseudo_rand(m * kk, 1))?;
        let a_cm = stream.clone_htod(&to_colmajor(&common::pseudo_rand(m * kk, 1), m, kk))?;
        let b = stream.clone_htod(&common::pseudo_rand(kk * n, 2))?;
        let bias = stream.clone_htod(&common::pseudo_rand(n, 3))?;
        let mut c = stream.alloc_zeros::<f32>(m * n)?;

        let cublas_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            cublas_gemm(&blas, m, kk, n, &a_rm, &b, &mut c);
            Ok::<(), ()>(())
        })?;
        let g06_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_gemm06(&stream, &k.gemm06, &a_rm, &b, &mut c, m, n, kk);
            Ok::<(), ()>(())
        })?;
        let cute_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_cute_gemm(&stream, &k.cute_gemm, &a_cm, &b, &mut c, m, n, kk);
            Ok::<(), ()>(())
        })?;
        let unfused_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_cute_gemm(&stream, &k.cute_gemm, &a_cm, &b, &mut c, m, n, kk);
            run_bias_gelu(&stream, &k.bias_gelu, &mut c, &bias, m, n);
            Ok::<(), ()>(())
        })?;
        let fused_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_cute_fused(&stream, &k.cute_fused, &a_cm, &b, &bias, &mut c, m, n, kk);
            Ok::<(), ()>(())
        })?;
        let biasgelu_ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            run_bias_gelu(&stream, &k.bias_gelu, &mut c, &bias, m, n);
            Ok::<(), ()>(())
        })?;

        for (row, ms) in rows.iter_mut().zip([cublas_ms, g06_ms, cute_ms, unfused_ms, fused_ms]) {
            row.push(gflops(m, n, kk, ms));
        }
        fusion.push((m, cute_ms, biasgelu_ms, unfused_ms, fused_ms));
    }

    // --- GFLOPS table ---
    print!("| {:<14} |", "impl \\ M");
    for &m in BENCH_MS {
        print!(" {:>12} |", format!("M={m}"));
    }
    println!();
    print!("|{}|", "-".repeat(16));
    for _ in BENCH_MS {
        print!("{}|", "-".repeat(14));
    }
    println!();
    for (name, row) in names.iter().zip(&rows) {
        print!("| {name:<14} |");
        for g in row {
            print!(" {:>10.1} |", g);
        }
        println!();
    }
    println!("\n(GFLOPS = 2·M·N·K / time; ffn-up K={BENCH_K}, N={BENCH_N})");

    // --- fusion analysis ---
    // fused = gemm + in-register GELU before one write; unfused = gemm (writes C)
    // then a separate bias+GELU kernel (reads C, writes C). The standalone
    // bias+GELU time shows whether its memory round-trip is exposed or hidden
    // under the tanh compute.
    println!("\nfusion (ms): gemm | +sep bias+GELU | unfused total | fused | speedup");
    for (m, gemm_ms, biasgelu_ms, unfused_ms, fused_ms) in fusion {
        println!(
            "  M={m:<4} {gemm_ms:>6.3} | {biasgelu_ms:>6.3} | {unfused_ms:>6.3} | {fused_ms:>6.3} | {:.2}x",
            unfused_ms / fused_ms
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every implementation matches the CPU reference. Skips gracefully (green)
    /// when there is no CUDA device or the PTX is an empty stub (nvcc missing).
    #[test]
    fn cutlass_matches_reference() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip cutlass test: no CUDA device ({e:?})");
                return;
            }
        };
        if CUTE_PTX.trim().is_empty() || GEMM06_PTX.trim().is_empty() {
            eprintln!("skip cutlass test: PTX is an empty stub (nvcc missing at build time)");
            return;
        }
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();
        let k = load_kernels(&ctx).unwrap();
        run_verify(&stream, &blas, &k).unwrap();
    }
}
