//! Stage 1: SGEMM optimization ladder vs cuBLAS.
//! Run with: `cargo run -rp gemm`
//!
//! C[M,N] = A[M,K] * B[K,N], row-major, M=N=K (square).
//! Each kernel is verified against cuBLAS, then timed (median over CUDA events).

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

const SIZES: &[usize] = &[256, 512, 1024, 2048];
const VERIFY_SIZE: usize = 512;
const WARMUP: usize = 2;
const ITERS: usize = 7;
const TOL: f32 = 1e-3;

struct Kernel {
    name: &'static str,
    fname: &'static str,
    ptx: &'static str,
    /// (block, grid) for a square n x n problem
    cfg: fn(n: u32) -> LaunchConfig,
    /// problem size must be divisible by this
    div: usize,
}

const KERNELS: &[Kernel] = &[
    Kernel {
        name: "v1 naive",
        fname: "gemm_naive",
        ptx: include_str!(concat!(env!("OUT_DIR"), "/gemm_01_naive.ptx")),
        cfg: |n| LaunchConfig {
            grid_dim: (n / 16, n / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        div: 16,
    },
    Kernel {
        name: "v2 coalesced",
        fname: "gemm_coalesced",
        ptx: include_str!(concat!(env!("OUT_DIR"), "/gemm_02_coalesced.ptx")),
        cfg: |n| LaunchConfig {
            grid_dim: (n / 16, n / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        div: 16,
    },
    Kernel {
        name: "v3 smem tiled",
        fname: "gemm_tiled",
        ptx: include_str!(concat!(env!("OUT_DIR"), "/gemm_03_tiled.ptx")),
        cfg: |n| LaunchConfig {
            grid_dim: (n / 32, n / 32, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        },
        div: 32,
    },
    Kernel {
        name: "v4 blocktiled",
        fname: "gemm_blocktiled",
        ptx: include_str!(concat!(env!("OUT_DIR"), "/gemm_04_blocktiled.ptx")),
        cfg: |n| LaunchConfig {
            grid_dim: (n / 128, n / 128, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        div: 128,
    },
    Kernel {
        name: "v5 vectorized",
        fname: "gemm_vectorized",
        ptx: include_str!(concat!(env!("OUT_DIR"), "/gemm_05_vectorized.ptx")),
        cfg: |n| LaunchConfig {
            grid_dim: (n / 128, n / 128, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        div: 128,
    },
];

fn gflops(n: usize, ms: f32) -> f32 {
    (2.0 * (n as f64).powi(3) / (ms as f64 * 1e6)) as f32
}

/// cuBLAS for row-major data: C_rm = A_rm * B_rm <=> column-major sgemm with
/// A and B swapped: m=N, n=M, a=B (lda=N), b=A (ldb=K), ldc=N.
fn cublas_gemm(
    blas: &CudaBlas,
    n: usize,
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    c: &mut cudarc::driver::CudaSlice<f32>,
) {
    let n_i = n as i32;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n_i,
        n: n_i,
        k: n_i,
        alpha: 1.0f32,
        lda: n_i,
        ldb: n_i,
        beta: 0.0f32,
        ldc: n_i,
    };
    unsafe { blas.gemm(cfg, b, a, c) }.expect("cublas sgemm");
}

fn launch(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    cfg: LaunchConfig,
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    c: &mut cudarc::driver::CudaSlice<f32>,
    n: i32,
) {
    let mut lb = stream.launch_builder(f);
    lb.arg(a).arg(b).arg(c).arg(&n).arg(&n).arg(&n);
    unsafe { lb.launch(cfg) }.expect("launch");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    println!("device: {}\n", ctx.name()?);

    let funcs: Vec<CudaFunction> = KERNELS
        .iter()
        .map(|k| {
            common::load_ptx(&ctx, k.fname, k.ptx)
                .and_then(|m| m.load_function(k.fname))
                .unwrap_or_else(|e| panic!("{}: {e:?}", k.fname))
        })
        .collect();

    // --- correctness at VERIFY_SIZE against cuBLAS ---
    {
        let n = VERIFY_SIZE;
        let a = stream.clone_htod(&common::pseudo_rand(n * n, 1))?;
        let b = stream.clone_htod(&common::pseudo_rand(n * n, 2))?;
        let mut c_ref = stream.alloc_zeros::<f32>(n * n)?;
        cublas_gemm(&blas, n, &a, &b, &mut c_ref);
        let want = stream.clone_dtoh(&c_ref)?;

        for (k, f) in KERNELS.iter().zip(&funcs) {
            let mut c = stream.alloc_zeros::<f32>(n * n)?;
            launch(&stream, f, (k.cfg)(n as u32), &a, &b, &mut c, n as i32);
            let got = stream.clone_dtoh(&c)?;
            let err = common::max_rel_err(&got, &want);
            assert!(err < TOL, "{}: max_rel_err = {err}", k.name);
            println!("verify {:<14} max_rel_err = {err:.2e}  OK", k.name);
        }
        println!();
    }

    // --- benchmark ---
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    let mut cublas_row: Vec<f32> = Vec::new();

    for &n in SIZES {
        let a = stream.clone_htod(&common::pseudo_rand(n * n, 1))?;
        let b = stream.clone_htod(&common::pseudo_rand(n * n, 2))?;
        let mut c = stream.alloc_zeros::<f32>(n * n)?;
        let ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
            cublas_gemm(&blas, n, &a, &b, &mut c);
            Ok::<(), ()>(())
        })?;
        cublas_row.push(gflops(n, ms));
    }

    for (k, f) in KERNELS.iter().zip(&funcs) {
        let mut row = Vec::new();
        for &n in SIZES {
            assert!(n % k.div == 0);
            let a = stream.clone_htod(&common::pseudo_rand(n * n, 1))?;
            let b = stream.clone_htod(&common::pseudo_rand(n * n, 2))?;
            let mut c = stream.alloc_zeros::<f32>(n * n)?;
            let cfg = (k.cfg)(n as u32);
            let ms = common::time_median_ms(&stream, WARMUP, ITERS, || {
                launch(&stream, f, cfg, &a, &b, &mut c, n as i32);
                Ok::<(), ()>(())
            })?;
            row.push(gflops(n, ms));
        }
        rows.push((k.name.to_string(), row));
    }

    // --- table: GFLOPS (% of cuBLAS) ---
    print!("| {:<14} |", "kernel");
    for &n in SIZES {
        print!(" {n:>17} |");
    }
    println!();
    print!("|{}|", "-".repeat(16));
    for _ in SIZES {
        print!("{}|", "-".repeat(19));
    }
    println!();
    for (name, row) in &rows {
        print!("| {name:<14} |");
        for (g, base) in row.iter().zip(&cublas_row) {
            print!(" {:>9.1} ({:>4.0}%) |", g, 100.0 * g / base);
        }
        println!();
    }
    print!("| {:<14} |", "cuBLAS");
    for g in &cublas_row {
        print!(" {:>9.1} (100%) |", g);
    }
    println!();
    Ok(())
}
