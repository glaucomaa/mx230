# Stage 3 — GPT-2 inference engine in plain CUDA

GPT-2 124M running on hand-written CUDA kernels with a Rust host: custom
weight format, byte-level BPE tokenizer (no tokenizer crates), KV cache,
int8 weight quantization. One command pipeline:

```
cargo run -rp llm-engine -- export                    # download + convert weights
cargo run -rp llm-engine -- verify                    # GPU logits vs CPU reference
cargo run -rp llm-engine -- generate "Alan Turing was" -n 40 [--int8]
cargo run -rp llm-engine -- bench -n 128
```

## Results (greedy decode, 128 tokens, MX230 / 40 GB/s bus)

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp32**          | 498 MB  | **78.0**   |
| **ours, int8**          | 124 MB  | **122.0**  |
| HF transformers (torch CPU) | 498 MB | 45.1   |

PyTorch GPU is not in the table for a reason worth stating: current torch
wheels ship no sm_61 kernels (`cudaErrorNoKernelImageForDevice`) — Pascal is
simply unsupported, so the hand-written engine is the only way this GPU runs
an LLM at all (`scripts/hf_baseline.py`).

## What the numbers say

Decode is one GEMV per weight matrix per token — pure memory streaming:

- **fp32: 78 tok/s × 498 MB = 38.8 GB/s — the memory bus is saturated.**
  The fp32 engine is provably optimal for this hardware; no further kernel
  cleverness can help, only smaller weights.
- **int8 cuts traffic 4x** but lands at 1.6x, not 4x: at ~3 ms of memory
  traffic per token, the ~115 kernel launches per token (~5 ms of launch
  overhead, fp32 KV attention, sync + logits copy) become the bottleneck.
  Classic Amdahl: kill the big cost and the small ones are the new wall.
  Next steps would be CUDA graphs (one launch per token) and an int8 KV cache.

## Pieces

- `src/export.rs` — pulls `openai-community/gpt2` safetensors (curl) and
  repacks into a flat fp32 `model.bin` (header + tensors in fixed order).
- `src/tokenizer.rs` — GPT-2 byte-level BPE from `vocab.json`/`merges.txt`,
  including a hand-rolled scanner for the GPT-2 pre-tokenization regex
  (the `regex` crate lacks the lookahead it needs).
- `src/cpu.rs` — slow, obvious reference forward; ground truth for the GPU.
- `kernels/llm.cu` — embed, layernorm (block reduction), GEMV (fp32 / int8
  with per-output-channel absmax scales), fused causal KV-cache attention
  (one block per head, online scores in shared memory), GELU, residual add.
- `src/gpu.rs` — engine: weights uploaded fp32 or quantized at load,
  per-layer KV cache, ~115 kernel launches per token, greedy sampling on host.

Verification: fp32 GPU logits match the CPU reference to `8e-5` (allclose);
int8 is checked for argmax agreement. Sample output (greedy, so it loops —
that's GPT-2 124M, not a bug):

> Alan Turing was a brilliant mathematician, and he was a great friend of
> mine. He was a great friend of mine. ...
