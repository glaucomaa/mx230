# Stage 4 — CuTe fused GEMM + bias + GELU (SIMT)

A fused **`GELU(x·W + b)`** kernel (the GPT-2 ffn-up) written with **CuTe**
(CUTLASS 3.x), benchmarked against cuBLAS, the hand-rolled stage-1 `gemm_06`,
and an unfused two-kernel baseline. Run:

```
git submodule update --init third_party/cutlass   # one-time, header-only
cargo run -rp cutlass
```

The MX230 (sm_61, Pascal) has **no tensor cores**, so the CuTe `gemm()` lowers to
plain FMAs (`UniversalFMA`), not an MMA atom — none of CUTLASS's Hopper/Ampere
machinery (TMA, warp-group MMA, `cp.async`) is in play. The point is to show
CuTe fluency and a fused epilogue on the only path this card has, staying inside
the repo's architecture: a CuTe `extern "C" __global__` compiled to PTX
(`nvcc -ptx -arch=sm_61`) and launched from Rust via `cudarc`, exactly like every
other kernel here. The de-risk compile lands at **102 registers, 0 spill, 8 KB
smem** — CuTe → sm_61 PTX is clean.

## Results (GFLOPS, ffn-up K=768, N=3072, M = prefill tokens)

| impl \ M       |   M=128 |   M=256 |   M=512 |
|----------------|---------|---------|---------|
| cuBLAS (gemm)  |   603.1 |   613.2 |   669.1 |
| gemm_06 (gemm) |   459.0 |   459.7 |   502.2 |
| CuTe gemm      |   511.6 |   515.8 |   554.1 |
| CuTe unfused   |   473.4 |   476.6 |   510.2 |
| CuTe fused     |   473.0 |   513.6 |   515.5 |

`GFLOPS = 2·M·N·K / time`. Layouts differ per implementation (cuBLAS/`gemm_06`
row-major, CuTe column-major NT) but the problem and FLOP count are identical and
every implementation is verified against the same f64 CPU reference
(`allclose`, rtol/atol 2e-3) — GEMM-only kernels against `A·B`, the
fused/unfused ones against `GELU(A·B + b)`.

**Finding 1 — CuTe beats the hand-rolled kernel.** The CuTe GEMM (≈510–554
GFLOPS) is ~10% faster than stage-1's `gemm_06` (≈459–502) at the same
128×128×8 tile, and reaches ~80% of cuBLAS. CuTe is not just a tensor-core
front-end; on plain SIMT it generates a GEMM competitive with hand-tuning.

**Finding 2 — fusing GELU is ~neutral here, and that's the interesting part.**

```
fusion (ms): gemm | +sep bias+GELU | unfused total | fused | speedup
  M=512   4.360 |  0.367 |  4.735 |  4.687 | 1.01x
```

Fusion should save the M·N output round-trip (~0.29 ms of DRAM at M=512). It
doesn't, because the standalone bias+GELU kernel (0.367 ms) is **compute-bound on
`tanh`**: its memory traffic already overlaps its ~0.33 ms of SFU work, so the
round-trip fusion removes was never exposed. The fused kernel still pays that
same `tanh` serially in its epilogue (≈0.327 ms on top of the 4.360 ms GEMM), so
the net win is ~0.04 ms. The lesson: epilogue fusion pays off when the epilogue
is *memory-bound* (cheap op, e.g. bias-only / ReLU) or the GEMM is memory-bound;
for a transcendental activation on a compute-bound GEMM the saved bytes were
already hidden under the activation's own cost.

## How

`kernels/cute_gemm.cu`, after the CuTe `sgemm_1` tutorial: 128×128×8 CTA tile,
256 threads, each owning an 8×8 output block, mainloop = `copy` (gmem→smem) +
`gemm` (smem→register FMA). The two kernels share a templated `ffn_gemm<FUSE>`
device body; the epilogue is the only difference — `GELU(alpha·acc + bias)` with
the bias broadcast over M via a stride-0 tensor mode, vs `alpha·acc`. The unfused
baseline is `cute_gemm` + a separate elementwise `bias_gelu`.

Build: CuTe is header-only; `build.rs` calls
`kernel_build::compile_kernels_with_args(["-std=c++17", "-I",
"../third_party/cutlass/include"])`, which threads the include path and C++17
through `nvcc` for every `.cu` (harmless for the plain `gemm_06` baseline).
CUTLASS is vendored as the `third_party/cutlass` submodule, pinned at commit
`cf064d2` (header-only; `git submodule update --init third_party/cutlass`).
