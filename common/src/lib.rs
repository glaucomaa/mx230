//! Shared utilities: PTX loading, CUDA-event timing, correctness checks.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaModule, CudaStream, DriverError};
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

/// Median time of one invocation of `f` in milliseconds (CUDA events).
/// `f` enqueues work on the stream; synchronization happens here.
pub fn time_median_ms<E: std::fmt::Debug>(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<(), E>,
) -> Result<f32, DriverError> {
    for _ in 0..warmup {
        f().unwrap();
    }
    stream.synchronize()?;

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        f().unwrap();
        let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        times.push(start.elapsed_ms(&end)?);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    Ok(times[times.len() / 2])
}

/// Maximum relative error between `got` and `want`.
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-6))
        .fold(0.0f32, f32::max)
}

/// Worst-case allclose ratio: max(|g - w| / (atol + rtol * |w|)).
/// Values <= 1.0 mean every element satisfies |g - w| <= atol + rtol * |w|
/// (same criterion as numpy.allclose). Unlike pure relative error this does
/// not blow up on near-zero outputs.
pub fn allclose_err(got: &[f32], want: &[f32], rtol: f32, atol: f32) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (atol + rtol * w.abs()))
        .fold(0.0f32, f32::max)
}

/// Deterministic pseudo-random f32 values in [-1, 1) with no external deps.
pub fn pseudo_rand(n: usize, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            // xorshift64*
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            let r = seed.wrapping_mul(0x2545F4914F6CDD1D);
            (r >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_rel_err_basics() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(max_rel_err(&a, &a), 0.0);
        // 2.2 vs 2.0 -> 0.1 relative; the worst element of the slice
        let got = [1.0, 2.2, 3.0];
        assert!((max_rel_err(&got, &a) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn allclose_err_threshold() {
        let want = [0.0, 100.0];
        // |0.05 - 0| = 0.05 == atol, |101 - 100| = 1 == rtol*100; both ratios == 1
        let got = [0.05, 101.0];
        assert!(allclose_err(&got, &want, 1e-2, 5e-2) <= 1.0 + 1e-6);
        // double the deviations -> ratio 2.0, fails allclose
        let bad = [0.10, 102.0];
        assert!(allclose_err(&bad, &want, 1e-2, 5e-2) > 1.0);
    }

    #[test]
    fn pseudo_rand_is_deterministic_and_bounded() {
        let a = pseudo_rand(1000, 42);
        let b = pseudo_rand(1000, 42);
        assert_eq!(a, b, "same seed must reproduce the same sequence");
        assert_eq!(a.len(), 1000);
        assert!(a.iter().all(|&x| (-1.0..1.0).contains(&x)), "values in [-1, 1)");
        // a different seed must produce a different sequence
        assert_ne!(pseudo_rand(1000, 43), a);
    }
}
