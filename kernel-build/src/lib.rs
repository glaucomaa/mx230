//! Compiles kernels/*.cu to OUT_DIR/*.ptx via nvcc (-arch=sm_61).
//! Used from build.rs of every stage crate:
//! ```no_run
//! fn main() { kernel_build::compile_kernels(); }
//! ```
//! If nvcc is missing the build does not fail: empty .ptx stubs are written
//! plus a cargo:warning — the runtime (`common::load_ptx`) reports it clearly.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const ARCH: &str = "sm_61"; // MX230 / GP108, Pascal

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(home) = env::var("CUDA_HOME") {
        let p = Path::new(&home).join("bin/nvcc");
        if p.exists() {
            return Some(p);
        }
    }
    for cand in ["/opt/cuda/bin/nvcc", "/opt/cuda-12.9/bin/nvcc", "/usr/local/cuda-12.9/bin/nvcc"] {
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from("nvcc"))
}

pub fn compile_kernels() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = find_nvcc();
    if nvcc.is_none() {
        println!("cargo:warning=nvcc not found (CUDA_HOME / /opt/cuda / PATH) — writing empty PTX stubs");
    }

    for entry in fs::read_dir("kernels").expect("no kernels/ directory") {
        let cu = entry.unwrap().path();
        if cu.extension().map(|e| e != "cu").unwrap_or(true) {
            continue;
        }
        let ptx = out_dir.join(cu.file_stem().unwrap()).with_extension("ptx");
        run_nvcc(&nvcc, &cu, &ptx, &[]);
    }
}

/// Compiles one .cu into OUT_DIR/<out_name>.ptx with extra nvcc defines —
/// for kernels parameterized at compile time (e.g. -DAG=8).
pub fn compile_kernel_variant(cu: &str, out_name: &str, defines: &[&str]) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx = out_dir.join(out_name).with_extension("ptx");
    run_nvcc(&find_nvcc(), Path::new(cu), &ptx, defines);
}

fn run_nvcc(nvcc: &Option<PathBuf>, cu: &Path, ptx: &Path, defines: &[&str]) {
    let Some(nvcc) = nvcc else {
        fs::write(ptx, "").unwrap();
        return;
    };
    let mut cmd = Command::new(nvcc);
    cmd.args(["-ptx", "-O3", "-arch", ARCH, "-lineinfo", "-Wno-deprecated-gpu-targets"])
        .args(defines)
        .arg(cu)
        .arg("-o")
        .arg(ptx);
    // nvcc 12.x rejects the system gcc 16 headers — use g++-14 from the gcc14 package
    for ccbin in ["/usr/bin/g++-14", "/usr/bin/g++-13"] {
        if Path::new(ccbin).exists() {
            cmd.arg("-ccbin").arg(ccbin);
            break;
        }
    }
    let out = cmd.output().expect("failed to run nvcc");
    if !out.status.success() {
        panic!(
            "nvcc failed on {}:\n{}",
            cu.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
