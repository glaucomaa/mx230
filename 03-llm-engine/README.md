# Stage 3 — LLM inference engine in plain CUDA

GPT-2 124M, Qwen2.5-0.5B and TinyLlama-1.1B running on hand-written CUDA
kernels with a Rust host: custom weight format, two tokenizers written from
scratch (byte-level BPE with both pre-tokenization regexes hand-rolled, and
SentencePiece BPE with byte fallback), KV cache (fp32 or int8), fp16 storage,
int8 and packed int4 weight quantization. The same kernel set serves all
three architectures — LayerNorm/RMSNorm, learned positions/RoPE, GELU/SwiGLU,
full attention/GQA, tied/untied lm_head are per-arch dispatches. One command
pipeline (`--model gpt2|qwen|tinyllama`, default gpt2):

```
cargo run -rp llm-engine -- export [--model qwen]     # download + convert weights
cargo run -rp llm-engine -- verify [--model qwen]     # GPU logits vs CPU reference
cargo run -rp llm-engine -- generate "Alan Turing was" -n 40 [--fp16|--int8|--int4] [--kv8] [--spec]
cargo run -rp llm-engine -- bench -n 128 [--graphs] [--kv8] [--spec]
cargo run -rp llm-engine -- prefill-bench -n 512 [--kv8]
cargo run -rp llm-engine -- ppl-data                  # download WikiText-2 raw test
cargo run -rp llm-engine -- ppl -n 2048 [--model qwen]
cargo run -rp llm-engine -- encode "text" [--model qwen]  # tokenizer debug
```

## Results (greedy decode, 128 tokens, MX230 / 40 GB/s bus)

GPT-2 124M:

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp32**          | 497 MB  | **77.6**   |
| **ours, fp32 + graph**  | 497 MB  | **78.5**   |
| **ours, fp16 storage**  | 249 MB  | **113.8**  |
| **ours, fp16 + graph**  | 249 MB  | **115.6**  |
| **ours, int8**          | 124 MB  | **130.0**  |
| **ours, int8 + graph**  | 124 MB  | **131.9**  |
| **ours, int4**          | 70 MB   | **211.3** (quality collapses — see ppl) |
| HF transformers (torch CPU) | 497 MB | 45.1   |

Qwen2.5-0.5B (24 layers, GQA 14q/2kv, SwiGLU, RoPE, 152k vocab):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp16 storage**  | 988 MB  | **30.2**   |
| **ours, int8**          | 494 MB  | **52.6**   |
| **ours, int4**          | 278 MB  | **59.6**   |

TinyLlama-1.1B (22 layers, GQA 32q/4kv, SwiGLU, RoPE, untied lm_head):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, int8**          | 1100 MB | **28.9**   |
| **ours, int4**          | 619 MB  | **31.3**   |
| **ours, int4 + spec**   | 619 MB  | **41.5**   |

Qwen2.5-0.5B in fp32 is ~1.9 GB of weights — it does not fit in 2 GB VRAM,
so fp16/int8 storage is not an optimization here but the only way the model
runs at all. TinyLlama-1.1B pushes that one step further: even fp16 is
2.2 GB, so the model exists on this card only as int8 (1.1 GB, barely) or
int4 (619 MB, comfortably). And PyTorch still can't touch this GPU (no
sm_61 kernels), so a 1.1B-parameter model generating coherent text at
31 tok/s on a 2019 laptop card is the engine's closing argument.

PyTorch GPU is not in the table for a reason worth stating: current torch
wheels ship no sm_61 kernels (`cudaErrorNoKernelImageForDevice`) — Pascal is
simply unsupported, so the hand-written engine is the only way this GPU runs
an LLM at all (`scripts/hf_baseline.py`).

## llama.cpp baseline

The external baseline is [llama.cpp](https://github.com/ggml-org/llama.cpp)
built from current upstream sources for the same Pascal target:

```
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build-sm61-nofa -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NO_VMM=ON \
  -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FA=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama.cpp/build-sm61-nofa --target llama-bench llama-cli -j 3
```

Models:

- [`lamptablet/gpt2-Q8_0-GGUF`](https://huggingface.co/lamptablet/gpt2-Q8_0-GGUF),
  `gpt2-q8_0.gguf` (GPT-2 base, Q8_0, 167.75 MiB).
- [`neopolita/qwen2.5-0.5b-gguf`](https://huggingface.co/neopolita/qwen2.5-0.5b-gguf),
  `qwen2.5-0.5b_q8_0.gguf` (Qwen2.5-0.5B base, Q8_0, 500.79 MiB).
- [`TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF`](https://huggingface.co/TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF),
  Q4_0 (606.53 MiB) and Q8_0 (1.09 GiB) — the same base checkpoint this
  engine runs.

Benchmark command (build `ac4cdde`):

```
/tmp/llama.cpp/build-sm61-nofa/bin/llama-bench \
  -m 03-llm-engine/models/<model>.gguf \
  -p 512 -n 128 -r 5 -ngl 999 -fa off -o md
```

| model | engine | model storage | prefill / prompt processing | greedy decode |
|-------|--------|---------------|-----------------------------|---------------|
| GPT-2 | llama.cpp CUDA | Q8_0 GGUF, 167.75 MiB | **2756.1 tok/s** (`pp512`) | **144.5 tok/s** (`tg128`) |
| GPT-2 | ours | int8 weights, ~124 MiB | 1080.2 tok/s (`512 / 0.474s`) | 130.0 tok/s |
| Qwen2.5-0.5B | llama.cpp CUDA | Q8_0 GGUF, 500.79 MiB | **871.2 tok/s** (`pp512`) | 45.5 tok/s (`tg128`) |
| Qwen2.5-0.5B | ours | int8 weights, ~494 MiB | 274.4 tok/s (`512 / 1.866s`) | **52.5 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q8_0 GGUF, 1.09 GiB | **384.9 tok/s** (`pp512`) | 22.0 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int8 weights, ~1.1 GB | 106.2 tok/s (`512 / 4.821s`) | **28.9 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q4_0 GGUF, 606.53 MiB | **441.7 tok/s** (`pp512`) | 31.2 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int4 weights, ~619 MB | 79.3 tok/s (`512 / 6.453s`) | **31.3 tok/s** |

This is the honest split. llama.cpp is much faster on prefill everywhere,
and the gap is structural, not just polish. Prefill is compute-bound, and
(1) this engine's GEMM is a deliberately mid-ladder kernel — 64x64 tiles,
4x4 micro-tile, no double buffering — landing at ~230-280 effective GFLOPS
of the 941 fp32 peak, where stage 1 showed 500+ is reachable; and
(2) llama.cpp's MMQ kernels never dequantize at all: they multiply in
integers via `dp4a` (4 int8 MACs per instruction, and sm_61 has it), so
their quantized GEMM ceiling is ~4x the fp32 one this engine's
dequantize-then-FMA design accepts. Decode hides both effects because it is
bandwidth-bound — which is exactly why the decode columns are even. GPT-2 decode also goes to
llama.cpp (144.5 vs 130.0 tok/s). Decode on the two RoPE models goes the
other way: the custom int8 path is 16% faster on Qwen and 31% faster on
TinyLlama (28.9 vs 22.0 tok/s), because the hot loop is narrower and
specialized for one architecture/layout instead of the full GGML execution
model. At 4 bits the engines converge: 31.3 vs 31.2 tok/s on TinyLlama —
both GEMV paths hit the same nibble-unpack instruction wall well short of
the bus, and matching mature Q4_0 kernels exactly is a result in itself.

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
  never goes idle and there is no launch overhead to remove. One exception
  appeared later: GPT-2 int4 moves only ~70 MB/step, and at that weight the
  previously invisible costs finally peek out — graphs alone add +2.4%
  (212.2 → 217.3 tok/s), and graphs + int8 KV reach 226.9 (+7% total). The
  lighter the bytes, the more everything else matters.
- **Batch prefill replaces token-by-token GEMV with GEMM + flash-style causal
  attention.** A 512-token prompt now runs as tiled matmuls over the whole
  prompt and a GQA-aware online-softmax attention pass over the KV cache.
  GEMM tile loads are vectorized (`float4`/`__half2`/`char4` — the same fix
  the int8 GEMV needed). On MX230/GPT-2 the measured time-to-first-token is:

  | mode | token loop | batch prefill | speedup |
  |------|------------|---------------|---------|
  | fp32 | 7.100s     | 0.462s        | 15.4x   |
  | fp16 | 4.993s     | 0.468s        | 10.7x   |
  | int8 | 4.436s     | 0.474s        | 9.4x    |

  The prefill path is checked against the token loop in `verify`: final logits
  may differ at float-rounding scale, but the greedy argmax must match in every
  weight/KV mode.
- **The GEMM dispatches on M, because a square tile wastes compute on skinny
  batches.** A 64x64 tile burns 64 rows of FMAs whether M is 512 or 8, and
  that wasted compute — not bandwidth — was the floor of the speculative
  verify pass: verifying 8 draft tokens cost ~6 decode steps, making spec
  decode a net loss. Three tiers fix it: 64-row tiles for prefill, 16-row
  tiles for mid-size M, and for M <= 8 a multi-row GEMV (`gemm_rows`) where
  each thread owns output columns gemv-style, B streams through once with
  zero wasted FMAs and the 8-row accumulator lives in registers. An 8-token
  verify dropped from 49ms to 15ms (GPT-2 int8) — under 2 decode steps.
- **Prompt-lookup speculative decoding** (`--spec`, optional `--spec-k N`) uses
  repeated n-grams from the prompt/generated history as draft tokens, verifies
  them with one batched forward, and accepts only tokens that match the full
  model's greedy argmax. Logits never leave the GPU: the verify pass argmaxes
  every row on device and ships back token ids, not `n x n_vocab` floats.
  It is lossless by construction — `verify` compares the speculative output
  token-for-token with ordinary greedy decode (host and device argmax break
  ties the same way, first index, so the paths cannot diverge on equal
  logits). Measured on int8 weights, 128-256 new tokens:

  | model | text | greedy | spec | gain |
  |------|------|--------|------|------|
  | GPT-2 | repeated sentence | 130.6 tok/s | 410.7 tok/s | 3.1x |
  | GPT-2 | "Alan Turing was..." | 125.8 tok/s | 255.9 tok/s | 2.0x |
  | Qwen2.5-0.5B | repeated sentence | 56.5 tok/s | 139.4 tok/s | 2.5x |

  Greedy LLM output loops hard, so prompt lookup hits constantly even on
  "normal" text; on text with no repeats spec falls back to one token per
  forward and costs nothing.
- **int8 weights were instruction-bound until the loads got wider.** The
  first int8 GEMV issued one byte load + convert + FMA per weight — the same
  instruction count as fp32 for a quarter of the data, so below bus
  saturation it ran at fp32's pace (and on Qwen actually *lost* to fp16,
  28 vs 30 tok/s). Switching wide matrices (`n_out >= 4096`) to `char4`
  loads with 4 outputs per thread fixed it: Qwen int8 28 → 52.6 tok/s,
  GPT-2 int8 122 → 130. Narrow matrices keep one output per thread — cutting
  the thread count 4x there starves the 3 SMs of latency-hiding warps and
  costs more than the wider loads gain.
- **int8 still lands at 1.7x over fp32, not 4x.** With launch overhead ruled
  out by the graph experiment and load instructions widened, the remaining
  wall is the serial fraction of the decode step: narrow GEMVs, fp32
  attention traffic, and one-block reductions (layernorm, softmax) that
  scale with depth, not bytes.
- **int4 weights pack two per byte** (Q4_0-style: one fp16 scale per 32
  weights of an output column, nibbles store q+8). Memory-wise it is the
  only way TinyLlama-1.1B fits comfortably; speed-wise it beats int8 on
  every model (GPT-2 130 → 211, Qwen 52 → 59.6, TinyLlama 28.9 → 31.3
  tok/s) — bytes win again. But the margins shrink with model width:
  TinyLlama int4 moves only ~20 GB/s of a 40 GB/s bus, while its int8 run
  saturates at ~32 GB/s. The int4 GEMV pays extra instructions per weight
  (nibble unpack, per-group scale loads), and the narrow qkv/proj matrices
  (n_out < 4096) sit on the scalar path — the same instruction wall int8
  hit before `char4`, one level deeper. Unlike int8, the group scale can't
  be folded into the final accumulator (it changes every 32 rows), so the
  GEMM path dequantizes during the shared-tile fill — which is also why
  int4 prefill trails int8 on every model (TinyLlama 6.5s vs 4.8s,
  GPT-2 0.62s vs 0.47s for 512 tokens).
- **int8 KV cache** (`--kv8`): K/V rows are quantized on write with one
  absmax scale per (position, head) and dequantized inside the attention
  kernel. The cache shrinks 75.5 → 19.6 MB and its traffic — the only part
  of decode that grows with context — drops 4x. KV traffic only matters at
  long context (at position 900 it is 66 MB/token fp32, more than half the
  int8 weights), so that is where the gain shows:

  | model | mode | kv | tok/s @128 ctx | tok/s @900 ctx |
  |------|------|------|------|------|
  | GPT-2 | fp32 | fp32 | 77.6 | 66.8 |
  | GPT-2 | fp32 | int8 | 79.0 | 73.9 |
  | GPT-2 | int8 | fp32 | 130.0 | 102.5 |
  | GPT-2 | int8 | int8 | 134.3 | **119.9** |
  | Qwen  | int8 | fp32 | 52.6 | 43.4 |
  | Qwen  | int8 | int8 | 53.3 | **48.9** |

  One implementation detail mattered: a naive byte-at-a-time dequant loop
  made kv8 *slower* than fp32 (92 tok/s at 900 ctx) — the score kernel got
  instruction-bound on byte loads. Vectorizing K loads as `char4` cut the
  load count 4x and flipped the result.

Quality is measured separately with:

```
cargo run -rp llm-engine -- ppl-data
cargo run -rp llm-engine -- ppl -n 2048
```

The harness reports per-mode perplexity on the same token slice, giving a
quality-vs-traffic table instead of only argmax agreement.

| model | mode | kv | WikiText-2 raw test tokens | perplexity |
|------|------|------|----------------------------|------------|
| GPT-2 | fp32 | fp32 | 2047                  | 25.388     |
| GPT-2 | fp16 | fp32 | 2047                  | 25.396     |
| GPT-2 | int8 | fp32 | 2047                  | 25.601     |
| GPT-2 | int4 | fp32 | 2047                  | **261.3**  |
| GPT-2 | fp32 | int8 | 2047                  | 25.378     |
| GPT-2 | fp16 | int8 | 2047                  | 25.367     |
| GPT-2 | int8 | int8 | 2047                  | 25.596     |
| Qwen  | fp16 | fp32 | 2047                  | 12.463     |
| Qwen  | int8 | fp32 | 2047                  | 12.464     |
| Qwen  | int4 | fp32 | 2047                  | 14.262     |
| Qwen  | fp16 | int8 | 2047                  | 12.941     |
| Qwen  | int8 | int8 | 2047                  | 12.944     |
| TinyLlama | int8 | fp32 | 2047              | 7.357      |
| TinyLlama | int4 | fp32 | 2047              | 7.692      |
| TinyLlama | int8 | int8 | 2047              | 7.356      |
| TinyLlama | int4 | int8 | 2047              | 7.695      |

Several quality stories in one table. Int8 *weights* are free on every model
(on Qwen literally so: 12.464 vs 12.463). The int8 *KV cache* depends on GQA
width: free on GPT-2 (12 KV heads, errors average out) and on TinyLlama
(4 KV heads), but costs Qwen +0.48 — with only 2 KV heads each quantized
K/V row is reused by 7 query heads and its error has nowhere to hide.

Int4 *weights* are a clean function of model scale. TinyLlama-1.1B barely
notices (+0.34), Qwen-0.5B pays a real but workable +1.8, and GPT-2 124M
collapses outright (25.6 → 261; greedy output degenerates into "the only,
the only, the only..."), exactly the small-old-model quantization
sensitivity the literature warns about — group-32 absmax has no answer to
GPT-2's weight outliers. The decode speed ladder runs the same direction as
the damage: int4 is most profitable exactly where it is least affordable.

And the model ladder itself: GPT-2 124M at 25.4, Qwen2.5-0.5B at 12.5,
TinyLlama-1.1B at 7.4 — seven years of model progress measured on the same
harness, the biggest model only runnable here because of the quantization
it tolerates best.

## Pieces

- `src/export.rs` — pulls `openai-community/gpt2` / `Qwen/Qwen2.5-0.5B` /
  `TinyLlama-1.1B` safetensors (curl) and repacks into a flat fp32
  `model.bin` (header + tensors in fixed order; bf16 widened, HF Linear
  transposed to [in, out], q/k/v concatenated into one GEMV, untied lm_head
  stored as an extra tensor). The 4.4 GB TinyLlama checkpoint and model.bin
  are mmap'd, not read — otherwise conversion would double-buffer ~9 GB.
- `src/tokenizer.rs` — two from-scratch tokenizers: byte-level BPE from
  `vocab.json`/`merges.txt` with hand-rolled scanners for both the GPT-2
  and Qwen2 pre-tokenization regexes (the `regex` crate lacks the lookahead
  they need), and SentencePiece BPE for TinyLlama parsed out of
  `tokenizer.json` (U+2581 space marker, `<0xXX>` byte fallback, BOS) —
  both verified token-for-token against HF tokenizers.
- `src/cpu.rs` — slow, obvious reference forwards for all archs; ground
  truth for the GPU.
- `kernels/llm.cu` — embed, layernorm/rmsnorm (block reduction), RoPE, GEMV
  (fp32 / fp16 storage / int8 with per-output-channel absmax scales and
  char4 loads on wide outputs / int4 with two weights per byte and
  per-group fp16 scales, uchar4 loads covering 4 columns x 2 rows), fused
  causal KV-cache attention with GQA (one block per query head, online
  scores in shared memory; fp32 and int8-cache variants), batched prefill
  GEMM in three M-tiers (64/16-row tiles + multi-row GEMV, vectorized tile
  loads; int4 variants dequantize during the tile fill), flash-style
  attention, quantize-on-write KV kernels, GELU, SwiGLU combine, residual
  add, GPU argmax for graph replay and per-row argmax for draft
  verification.
- `src/gpu.rs` — engine: weights uploaded fp32, converted to fp16, or
  quantized to int8/int4 at load; tied or untied lm_head; per-layer KV
  cache (fp32 or int8 + scales); standard host-greedy decode, batch
  prefill, prompt-lookup speculative decode (device-side verify, only
  token ids cross the bus) and a CUDA-graph benchmark path.

Verification: fp32 GPU logits match the CPU reference to `8e-5` (allclose);
fp16, int8 and both int8-KV variants report allclose error and are checked
for argmax agreement; batch prefill must match the token loop's greedy argmax;
prompt-lookup speculative decode must reproduce ordinary greedy tokens; the
graph decode path (with fp32 and int8 KV) must reproduce the host loop's
greedy continuation token-for-token (checked 16 steps deep).
Sample output (greedy, so it loops — that's GPT-2 124M, not a bug):

> Alan Turing was a brilliant mathematician, and he was a great friend of
> mine. He was a great friend of mine. ...

Qwen2.5-0.5B on the same kernels:

> Alan Turing was born in 1912 in England. He was the son of a
> mathematician. He was educated at the University of Cambridge, where he
> studied mathematics

TinyLlama-1.1B, int4, on the same kernels:

> Alan Turing was a mathematician and computer scientist who worked on the
> development of the first computer. He was also a pioneer in the field of
> artificial intelligence and was a leading figure in the development of the
