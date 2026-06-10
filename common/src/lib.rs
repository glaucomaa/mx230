//! Shared utilities: PTX loading, timing, correctness checks.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, DriverError};
use cudarc::nvrtc::Ptx;

/// Loads a PTX compiled by build.rs into the stage crate's OUT_DIR.
/// Panics with a clear message if the PTX is an empty stub (nvcc was missing).
pub fn load_ptx(ctx: &Arc<CudaContext>, name: &str, src: &str) -> Result<Arc<CudaModule>, DriverError> {
    assert!(
        !src.trim().is_empty(),
        "PTX `{name}` is empty: nvcc was not found at build time. \
         Install CUDA 12.x and rebuild (see PLAN.md, stage 0)."
    );
    ctx.load_module(Ptx::from_src(src))
}
