# mx230-cuda-lab

**TL;DR:** hand-written CUDA kernels + Rust host (`cudarc`) on a 2 GB NVIDIA
MX230 with a 40 GB/s bus. Inference is memory-bound — every saved byte shows
up in the numbers:

- **`01-gemm/`** — SGEMM ladder, naive → double-buffered: **11.5 → 509 GFLOPS**
  (48x, up to **82% of cuBLAS**)
- **`02-flash-attention/`** — Flash Attention forward from scratch: **12–19x**
  over naive, zero extra memory, runs where naive OOMs (N=32k needs 4.3 GB)
- **`03-llm-engine/`** — GPT-2 124M and Qwen2.5-0.5B inference in plain
  CUDA: own weight format, BPE tokenizer, KV cache (fp32 or int8,
  quantize-on-write), fp16 storage, int8 (char4-vectorized GEMV), GQA, RoPE,
  SwiGLU, CUDA Graph decode benchmark, WikiText-2 perplexity harness —
  GPT-2: **78 tok/s fp32 (bus saturated), 114 tok/s fp16, 130 tok/s int8**
  vs 45 tok/s PyTorch CPU; Qwen2.5: **52.6 tok/s int8** (fp32 wouldn't even
  fit in VRAM, and PyTorch GPU has no sm_61 kernels at all)

Kernels: CUDA C → PTX (`build.rs`, sm_61). Host, tokenizer, benchmarks: Rust.
Each stage's README has tables and the how/why.
