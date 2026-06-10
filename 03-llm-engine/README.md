# Stage 3 — GPT-2 inference engine in plain CUDA

GPT-2 124M running on hand-written CUDA kernels with a Rust host: custom
weight format, byte-level BPE tokenizer (no tokenizer crates), KV cache,
fp16 storage and int8 weight quantization. One command pipeline:

```
cargo run -rp llm-engine -- export                    # download + convert weights
cargo run -rp llm-engine -- verify                    # GPU logits vs CPU reference
cargo run -rp llm-engine -- generate "Alan Turing was" -n 40 [--fp16|--int8]
cargo run -rp llm-engine -- bench -n 128 [--graphs]
cargo run -rp llm-engine -- ppl-data                  # download WikiText-2 raw test
cargo run -rp llm-engine -- ppl -n 2048
```

## Results (greedy decode, 128 tokens, MX230 / 40 GB/s bus)

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp32**          | 498 MB  | **78.0**   |
| **ours, fp32 + graph**  | 498 MB  | **78.8**   |
| **ours, fp16 storage**  | 249 MB  | **114.9**  |
| **ours, fp16 + graph**  | 249 MB  | **116.4**  |
| **ours, int8**          | 124 MB  | **122.0**  |
| **ours, int8 + graph**  | 124 MB  | **123.8**  |
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
- **fp16 storage cuts traffic 2x** while still accumulating in fp32. On Pascal
  this avoids slow fp16 arithmetic and tests the pure "smaller weights" axis.
- **CUDA graphs** capture one decode step with token and position kept on the
  device: argmax runs on the GPU, the host submits one graph launch per token
  and never copies logits back. Measured gain is only **~1% in every mode** —
  a negative result worth having: kernel launches are asynchronous, the host
  enqueues ~115 launches/token faster than the GPU drains them, so the GPU
  never goes idle and there is no launch overhead to remove.
- **int8 cuts traffic 4x** but lands at 1.6x, not 4x. With launch overhead
  ruled out by the graph experiment, the remaining wall is inside the decode
  step itself: small GEMVs that can't fully use the bus, attention and KV
  cache traffic still in fp32, and serial reductions (layernorm, softmax)
  that scale with depth, not bytes.

Quality is measured separately with:

```
cargo run -rp llm-engine -- ppl-data
cargo run -rp llm-engine -- ppl -n 2048
```

The harness reports fp32/fp16/int8 perplexity on the same token slice, giving a
quality-vs-traffic table instead of only argmax agreement.

| mode | weights | WikiText-2 raw test tokens | perplexity |
|------|---------|----------------------------|------------|
| fp32 | ~498 MB | 2047                       | 25.388     |
| fp16 | ~249 MB | 2047                       | 25.396     |
| int8 | ~124 MB | 2047                       | 25.601     |

## Pieces

- `src/export.rs` — pulls `openai-community/gpt2` safetensors (curl) and
  repacks into a flat fp32 `model.bin` (header + tensors in fixed order).
- `src/tokenizer.rs` — GPT-2 byte-level BPE from `vocab.json`/`merges.txt`,
  including a hand-rolled scanner for the GPT-2 pre-tokenization regex
  (the `regex` crate lacks the lookahead it needs).
- `src/cpu.rs` — slow, obvious reference forward; ground truth for the GPU.
- `kernels/llm.cu` — embed, layernorm (block reduction), GEMV (fp32 / fp16
  storage / int8 with per-output-channel absmax scales), fused causal KV-cache
  attention (one block per head, online scores in shared memory), GELU,
  residual add, GPU argmax for graph replay.
- `src/gpu.rs` — engine: weights uploaded fp32, converted to fp16, or
  quantized at load; per-layer KV cache; standard host-greedy decode and a
  CUDA-graph benchmark path.

Verification: fp32 GPU logits match the CPU reference to `8e-5` (allclose);
fp16 and int8 report allclose error and are checked for argmax agreement;
the graph decode path must reproduce the host loop's greedy continuation
token-for-token (checked 16 steps deep).
Sample output (greedy, so it loops — that's GPT-2 124M, not a bug):

> Alan Turing was a brilliant mathematician, and he was a great friend of
> mine. He was a great friend of mine. ...
