# mx230-cuda-lab

**TL;DR:** hand-written CUDA kernels + Rust host (`cudarc`) on a 2 GB NVIDIA
MX230 with a 40 GB/s bus. Inference is memory-bound — every saved byte shows
up in the numbers:

- **`01-gemm/`** — SGEMM ladder, naive → double-buffered: **11.5 → 509 GFLOPS**
  (48x, up to **82% of cuBLAS**)
- **`02-flash-attention/`** — Flash Attention forward from scratch: **12–19x**
  over naive, zero extra memory, runs where naive OOMs (N=32k needs 4.3 GB)
- **`03-llm-engine/`** — GPT-2 124M, Qwen2.5-0.5B and TinyLlama-1.1B
  inference in plain CUDA: own weight format, two from-scratch tokenizers
  (byte-level BPE and SentencePiece BPE), KV cache (fp32 or int8,
  quantize-on-write), fp16 storage, int8/int4 quantization with `dp4a`
  integer math (W8A8/W4A8, activations quantized on the fly — sm_61's 4x
  int8 escape hatch from its missing tensor cores), GQA, RoPE, SwiGLU,
  CUDA Graph decode, prompt-lookup speculative decode, WikiText-2
  perplexity harness — GPT-2: **79 tok/s fp32 (bus saturated), 117 tok/s
  fp16, 266 tok/s int8, 371 tok/s int4** vs 45 tok/s PyTorch CPU;
  Qwen2.5: **74 tok/s int8, 104 int4**; TinyLlama-1.1B: **62 tok/s int4**
  on a card its fp16 weights alone wouldn't fit into (and PyTorch GPU has
  no sm_61 kernels at all)

Kernels: CUDA C → PTX (`build.rs`, sm_61). Host, tokenizer, benchmarks: Rust.
Each stage's README has tables and the how/why.
