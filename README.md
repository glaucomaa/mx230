# mx230-cuda-lab

CUDA kernels + Rust host (`cudarc`) on an NVIDIA MX230 (Pascal sm_61, 2 GB VRAM,
~40 GB/s). The constraint is the point: inference is memory-bound, so every
saved byte of traffic shows up in the benchmarks.

1. `01-gemm/` — SGEMM optimization ladder, benchmarked vs cuBLAS
2. `02-flash-attention/` — Flash Attention (forward) from scratch vs naive
3. `03-llm-engine/` — GPT-2 124M inference engine: KV cache, int8 weights, tokens/sec

Kernels are CUDA C compiled to PTX by `build.rs`; everything host-side is Rust.
