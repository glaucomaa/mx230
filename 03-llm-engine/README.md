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
| **ours, fp32**          | 497 MB  | **79.1**   |
| **ours, fp16 storage**  | 249 MB  | **117.0**  |
| **ours, int8 (dp4a)**   | 124 MB  | **266.5**  |
| **ours, int8 + graph**  | 124 MB  | **276.0**  |
| **ours, int4 (dp4a)**   | 70 MB   | **371.1** (quality collapses — see ppl) |
| **ours, int4 + graph + kv8** | 70 MB | **388.0** |
| HF transformers (torch CPU) | 497 MB | 45.1   |

Qwen2.5-0.5B (24 layers, GQA 14q/2kv, SwiGLU, RoPE, 152k vocab):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp16 storage**  | 988 MB  | **30.5**   |
| **ours, int8 (dp4a)**   | 494 MB  | **74.3**   |
| **ours, int4 (dp4a)**   | 278 MB  | **104.1**  |
| **ours, int4 + graph**  | 278 MB  | **108.6**  |
| **ours, int4 + spec**   | 278 MB  | **129.4**  |

TinyLlama-1.1B (22 layers, GQA 32q/4kv, SwiGLU, RoPE, untied lm_head):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, int8 (dp4a)**   | 1100 MB | **38.9**   |
| **ours, int4 (dp4a)**   | 619 MB  | **61.8**   |
| **ours, int4 + spec**   | 619 MB  | **74.0**   |

Qwen2.5-0.5B in fp32 is ~1.9 GB of weights — it does not fit in 2 GB VRAM,
so fp16/int8 storage is not an optimization here but the only way the model
runs at all. TinyLlama-1.1B pushes that one step further: even fp16 is
2.2 GB, so the model exists on this card only as int8 (1.1 GB, barely) or
int4 (619 MB, comfortably). And PyTorch still can't touch this GPU (no
sm_61 kernels), so a 1.1B-parameter model generating coherent text at
62 tok/s on a 2019 laptop card is the engine's closing argument.

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

Benchmark command (build `1593d56`):

```
/tmp/llama.cpp/build-sm61-nofa/bin/llama-bench \
  -m 03-llm-engine/models/<model>.gguf \
  -p 512 -n 128 -r 5 -ngl 999 -fa off -o md
```

| model | engine | model storage | prefill / prompt processing | greedy decode |
|-------|--------|---------------|-----------------------------|---------------|
| GPT-2 | llama.cpp CUDA | Q8_0 GGUF, 167.75 MiB | **2756.7 tok/s** (`pp512`) | 144.0 tok/s (`tg128`) |
| GPT-2 | ours | int8 weights, ~124 MiB | 2032 tok/s (`512 / 0.252s`) | **266.5 tok/s** |
| Qwen2.5-0.5B | llama.cpp CUDA | Q8_0 GGUF, 500.79 MiB | **866.0 tok/s** (`pp512`) | 45.4 tok/s (`tg128`) |
| Qwen2.5-0.5B | ours | int8 weights, ~494 MiB | 553 tok/s (`512 / 0.926s`) | **74.3 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q8_0 GGUF, 1.09 GiB | **384.3 tok/s** (`pp512`) | 22.0 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int8 weights, ~1.1 GB | 230 tok/s (`512 / 2.225s`) | **38.9 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q4_0 GGUF, 606.53 MiB | **430.5 tok/s** (`pp512`) | 30.8 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int4 weights, ~619 MB | 189 tok/s (`512 / 2.709s`) | **61.8 tok/s** |

Before the dp4a rewrite this table read very differently: llama.cpp won
GPT-2 decode (144 vs 130), tied TinyLlama Q4_0 (31.2 vs 31.3), and its
prefill lead ran up to 5.6x. The diagnosis then was structural — "llama.cpp's
MMQ kernels never dequantize, they multiply in integers via dp4a, so their
quantized GEMM ceiling is ~4x the fp32 one this engine's
dequantize-then-FMA design accepts." Adopting the same weapon settled it:
**decode now goes to this engine on every row** — +85% on GPT-2 (266.5 vs
144.0), +64% on Qwen, +77% on TinyLlama int8, and the int4 "dead heat"
broke open to 2x (61.8 vs 30.8 tok/s), because the specialized hot loop
(one architecture, one layout, activations quantized straight into shared
memory) keeps more of the dp4a ceiling than GGML's general execution model.
Prefill still belongs to llama.cpp, but the gap stopped being structural:
once the dp4a GEMMs got their own wide tiles (int8 128x128, int4 64x128
over 32 k-values) it is 1.36x on GPT-2 (was 2.6x), 1.57x on Qwen (was
3.2x), 1.67–2.3x on TinyLlama (was 3.6–5.6x) — what remains is MMQ's
per-quant-format specialization and years of shape tuning against a
two-shape dp4a tile, plus int4's per-tile weight-scale fold.

## What the numbers say

Decode is one GEMV per weight matrix per token — pure memory streaming:

- **fp32: 79 tok/s × 497 MB = 39.3 GB/s — the memory bus is saturated**
  (the measured streaming roof is 43.8 GB/s, `common/examples/isa`). The
  fp32 engine is provably near-optimal for this hardware; no further kernel
  cleverness can help, only smaller weights.
- **fp16 storage cuts traffic 2x** while still accumulating in fp32. On Pascal
  this avoids slow fp16 arithmetic and tests the pure "smaller weights" axis.
- **dp4a: integer math without tensor cores.** sm_61 has one genuinely
  modern instruction: `dp4a`, 4 int8 MACs per issue. The ISA microbench
  (`common/examples/isa`) measures 2941 GOPS of int8 dot-product against
  735 GFLOPS of fp32 FMA — a real 4.0x — while half2 arithmetic crawls at
  1/64 rate (13 GFLOPS, useless). So the quantized paths stopped
  dequantizing: weights are repacked at load into int32 words along n_in,
  activations are quantized on the fly (absmax per 4-value group,
  llama.cpp Q8-style — small groups because GPT-2's activation outliers
  wreck 32-wide ones), and the inner loop is `__dp4a` on packed words with
  one float multiply per group instead of one per weight. int4 packs both
  nibble planes of 8 rows into one int32 so `w & 0x0F0F0F0F` and
  `(w >> 4) & 0x0F0F0F0F` feed dp4a directly, and the Q4_0 "+8" nibble
  bias folds away analytically (`dot -= 8 * Σx_group` per weight group) —
  no unpack-to-bytes in the GEMV at all. Decode: GPT-2 int8 130 → 266.5,
  int4 211 → 371; Qwen int8 52.6 → 74.3, int4 59.6 → 104.1; TinyLlama
  int8 28.9 → 38.9, int4 31.3 → 61.8 tok/s. TinyLlama int8 now moves
  ~42.8 GB/s — the bus, not instructions, is the wall again, which was
  the whole point.
- **CUDA graphs** capture one decode step with token and position kept on the
  device: argmax runs on the GPU, the host submits one graph launch per token
  and never copies logits back. At fp32/fp16 weights the gain is **~1%** —
  a negative result worth having: kernel launches are asynchronous, the host
  enqueues ~115 launches/token faster than the GPU drains them, so the GPU
  never goes idle and there is no launch overhead to remove. But the lighter
  the step, the more the fixed costs matter: after dp4a, GPT-2 int8 picks up
  +3.6% from graphs (266.5 → 276.0) and int4 +4.4% (371.1 → 387.5, 388.0
  with int8 KV) — at ~70 MB/step the previously invisible overheads finally
  peek out.
- **Batch prefill replaces token-by-token GEMV with GEMM + flash-style causal
  attention.** A 512-token prompt now runs as tiled matmuls over the whole
  prompt and a GQA-aware online-softmax attention pass over the KV cache.
  GEMM tile loads are vectorized (`float4`/`__half2`/`char4` — the same fix
  the int8 GEMV needed). On MX230/GPT-2 the measured time-to-first-token is:

  | mode | token loop | batch prefill | speedup |
  |------|------------|---------------|---------|
  | fp32 | 6.744s     | 0.240s        | 28.1x   |
  | fp16 | 4.659s     | 0.273s        | 17.1x   |
  | int8 | 2.178s     | 0.252s        | 8.6x    |
  | int4 | 1.654s     | 0.294s        | 5.6x    |

  (int8/int4 prefill is dp4a GEMM on the wide tile; their token loops are
  so fast after dp4a that the batch speedup *ratio* shrinks while the
  absolute TTFT keeps falling.) The prefill path is checked against the
  token loop in `verify`: final logits may differ at float-rounding scale,
  but the greedy argmax must match in every weight/KV mode.
- **The GEMM dispatches on M, because a square tile wastes compute on skinny
  batches.** A 64x64 tile burns 64 rows of FMAs whether M is 512 or 8, and
  that wasted compute — not bandwidth — was the floor of the speculative
  verify pass: verifying 8 draft tokens cost ~6 decode steps, making spec
  decode a net loss. Four tiers fix it: for real prefill (M > 64) a wide
  tile — fp32/fp16 get 128x128x8 with an 8x8 register micro-tile,
  float4-staged transposed A and double-buffered smem (the 01-gemm
  ladder's endgame grafted into the engine, worth 1.7–1.9x: GPT-2 fp32
  0.462 → 0.240s, Qwen fp16 1.821 → 1.026s), and the dp4a paths get the
  int edition — int8 as 128x128 over 32 k-values (8x8 micro-tile of
  dp4a+scale-FMA pairs), int4 as 64x128 (4x8 micro-tile plus a per-tile
  partial so the 32-row fp16 weight scale folds once per tile), worth
  another 36–43% on int prefill (GPT-2 int8 0.392 → 0.252s, TinyLlama
  int8 3.87 → 2.23s); 64-row tiles for mid-size batches;
  16-row tiles below that; and for M <= 8 a multi-row GEMV (`gemm_rows`)
  where each thread owns output columns gemv-style, B streams through once
  with zero wasted FMAs and the 8-row accumulator lives in registers. An
  8-token verify dropped from 49ms to 15ms (GPT-2 int8) — under 2 decode
  steps.
- **Prompt-lookup speculative decoding** (`--spec`, optional `--spec-k N`) uses
  repeated n-grams from the prompt/generated history as draft tokens, verifies
  them with one batched forward, and accepts only tokens that match the full
  model's greedy argmax. Logits never leave the GPU: the verify pass argmaxes
  every row on device and ships back token ids, not `n x n_vocab` floats.
  It is lossless by construction — `verify` compares the speculative output
  token-for-token with ordinary greedy decode (host and device argmax break
  ties the same way, first index, so the paths cannot diverge on equal
  logits). Measured with `bench --spec` (128 tokens, greedy output loops so
  prompt lookup hits constantly):

  | model | mode | greedy | spec | gain |
  |------|------|--------|------|------|
  | GPT-2 | int8 | 266.5 tok/s | 315.2 tok/s | 1.18x |
  | GPT-2 | int4 | 371.1 tok/s | 389.9 tok/s | 1.05x |
  | Qwen2.5-0.5B | int4 | 104.1 tok/s | 129.4 tok/s | 1.24x |
  | TinyLlama-1.1B | int4 | 61.8 tok/s | 74.0 tok/s | 1.20x |

  The spec margins shrank as dp4a made plain decode faster — speculation
  pays in proportion to how expensive a forward pass is; on text with no
  repeats it falls back to one token per forward and costs nothing.
- **int8 weights were instruction-bound until the math went integer.** The
  story has three chapters. The first int8 GEMV issued one byte load +
  convert + FMA per weight — the same instruction count as fp32 for a
  quarter of the data, so it ran at fp32's pace (on Qwen it *lost* to
  fp16, 28 vs 30 tok/s). Chapter two: `char4` loads with 4 outputs per
  thread on wide matrices (`n_out >= 4096`) — Qwen 28 → 52.6 tok/s, GPT-2
  122 → 130; narrow matrices keep one output per thread because cutting
  the thread count 4x starves 3 SMs of latency-hiding warps. Chapter
  three: stop converting at all — dp4a multiplies the packed bytes
  directly (W8A8), and the convert+FMA instruction stream collapses into
  one MAC per 4 weights: GPT-2 130 → 266.5, Qwen 52.6 → 74.3, TinyLlama
  28.9 → 38.9 tok/s. TinyLlama int8 is now bus-bound (~42.8 of 43.8 GB/s
  measured roof); the smaller models still carry per-token fixed costs,
  which is what graphs (+3.6%) and the decode-tail pass (warp-shuffle
  reductions, residual add fused into the GEMV epilogue, float4 K/V
  attention loads: int8 +7%, int4 +10% on GPT-2) keep trimming.
- **int4 weights pack two per byte** (Q4_0-style: one fp16 scale per 32
  weights of an output column, nibbles store q+8). Memory-wise it is the
  only way TinyLlama-1.1B fits comfortably; speed-wise it now beats int8
  on every model by close to the byte ratio (GPT-2 266 → 371, Qwen 74 →
  104, TinyLlama 39 → 62 tok/s) instead of the few percent it managed
  pre-dp4a, because the nibble-unpack instruction wall is gone: both
  nibble planes go straight into dp4a as masked words, the +8 bias is
  subtracted analytically per 32-row group, and no per-weight unpack or
  convert survives in the GEMV. The int4 GEMM unpacks each tile once into
  signed bytes (`__vsubss4`) and reuses the int8 micro-kernel shape, so
  int4 prefill went from 35% behind int8 to nearly level (GPT-2 0.294s vs
  0.252s, TinyLlama 2.71s vs 2.23s @512) — the remaining gap is the
  per-tile weight-scale fold and the half-height wide tile it forces.
- **int8 KV cache** (`--kv8`): K/V rows are quantized on write with one
  absmax scale per (position, head) and dequantized inside the attention
  kernel. The cache shrinks 75.5 → 19.6 MB and its traffic — the only part
  of decode that grows with context — drops 4x. KV traffic only matters at
  long context (at position 900 it is 66 MB/token fp32, more than half the
  int8 weights), so that is where the gain shows (table measured pre-dp4a;
  the ratios are the point, and post-dp4a the lighter weights only make KV
  bytes a *larger* fraction of the step):

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
  load count 4x and flipped the result. The dp4a treatment finished the
  job: q is quantized on the fly inside the kernel (one absmax scale per
  head) and the score dot runs entirely in integers — at 900 ctx with kv8
  that is another +5–9% on the int modes (GPT-2 int4 290 → 316, TinyLlama
  int4 52 → 56 tok/s) and +2–3% even for fp32/fp16 weights, since kv8
  attention is the same kernel regardless of weight dtype. Quality gates
  hold: q carries one scale per 64 values, but unlike the GEMV activations
  (which need 4-wide groups) attention scores tolerate it — verify argmax,
  spec and graph paths all stay green.

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
| GPT-2 | int8 | fp32 | 2047                  | 25.657     |
| GPT-2 | int4 | fp32 | 2047                  | **264.2**  |
| GPT-2 | fp32 | int8 | 2047                  | 25.363     |
| GPT-2 | fp16 | int8 | 2047                  | 25.377     |
| GPT-2 | int8 | int8 | 2047                  | 25.651     |
| Qwen  | fp16 | fp32 | 2047                  | 12.463     |
| Qwen  | int8 | fp32 | 2047                  | 12.468     |
| Qwen  | int4 | fp32 | 2047                  | 14.269     |
| Qwen  | fp16 | int8 | 2047                  | 12.951     |
| Qwen  | int8 | int8 | 2047                  | 12.938     |
| TinyLlama | int8 | fp32 | 2047              | 7.356      |
| TinyLlama | int4 | fp32 | 2047              | 7.694      |
| TinyLlama | int8 | int8 | 2047              | 7.359      |
| TinyLlama | int4 | int8 | 2047              | 7.703     |

(int8/int4 here are the dp4a W8A8/W4A8 paths — activations quantized on the
fly in 4-value absmax groups. The group size matters: 32-wide groups cost
GPT-2 +0.7 ppl because of its activation outliers; at 4 the damage is zero
and the speed identical.)

Several quality stories in one table. Int8 *weights* stay almost free on
every model (Qwen 12.468 vs 12.463 fp16). The int8 *KV cache* depends on GQA
width: free on GPT-2 (12 KV heads, errors average out) and on TinyLlama
(4 KV heads), but costs Qwen +0.48 — with only 2 KV heads each quantized
K/V row is reused by 7 query heads and its error has nowhere to hide.

Int4 *weights* are a clean function of model scale. TinyLlama-1.1B barely
notices (+0.34), Qwen-0.5B pays a real but workable +1.8, and GPT-2 124M
collapses outright (25.7 → 264; greedy output degenerates into "the only,
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
- `kernels/llm.cu` — embed, layernorm/rmsnorm (warp-shuffle reductions on
  the decode side), RoPE, GEMV (fp32 / fp16 storage / int8 and int4 as
  dp4a paths: weights repacked into int32 words, activations quantized
  in-kernel to absmax 4-value groups, int4's +8 nibble bias folded
  analytically; residual add fused into the epilogue via an accum flag),
  fused causal KV-cache attention with GQA (one block per query head,
  online scores in shared memory, float4/char4 K/V loads; fp32 and
  int8-cache variants), batched prefill GEMM in four M-tiers (wide tiles
  for prefill: 128x128 double-buffered 8x8 micro-tile for fp32/fp16,
  128x128 and 64x128 dp4a tiles for int8/int4; 64/16-row tiles below,
  multi-row GEMV for M <= 8; int GEMMs run integer micro-kernels with
  per-tile scale epilogues), flash-style
  attention, quantize-on-write KV kernels, GELU, SwiGLU combine, GPU
  argmax for graph replay and per-row argmax for draft verification.
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
