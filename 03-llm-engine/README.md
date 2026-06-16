# Stage 3 — LLM inference engine in plain CUDA

GPT-2 124M, Qwen2.5-0.5B and TinyLlama-1.1B running on hand-written CUDA
kernels with a Rust host: custom weight format, two tokenizers written from
scratch (byte-level BPE with both pre-tokenization regexes hand-rolled, and
SentencePiece BPE with byte fallback), KV cache (fp32 or int8), fp16 storage,
int8/int4/int3/int2 weight quantization on dp4a integer math. The same
kernel set serves all
three architectures — LayerNorm/RMSNorm, learned positions/RoPE, GELU/SwiGLU,
full attention/GQA, tied/untied lm_head are per-arch dispatches. One command
pipeline (`--model gpt2|qwen|tinyllama`, default gpt2):

```
cargo run -rp llm-engine -- export [--model qwen]     # download + convert weights
cargo run -rp llm-engine -- verify [--model qwen]     # GPU logits vs CPU reference
cargo run -rp llm-engine -- generate "Alan Turing was" -n 40 [--fp16|--int8|--int4|--int4k|--int3|--int2] [--kv8] [--spec] [--temp 0.8 --top-k 40 --top-p 0.95 --seed 1]
cargo run -rp llm-engine -- bench -n 128 [--graphs] [--kv8] [--spec]
cargo run -rp llm-engine -- prefill-bench -n 512 [--kv8] [--prefill-dp4a]  # --prefill-dp4a: opt-in dp4a prefill scores (~6-8%)
cargo run -rp llm-engine -- kbench [--model qwen] [--int8|--int4] [--emit-llama]  # isolated matmul kernels vs llama.cpp MMVQ/MMQ
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
| **ours, int3 / int2**   | 62 / 57 MB | **429.4 / 424.0** (k-quants rungs — see ppl) |
| PyTorch 2.7.1+cu126, GPU fp16 (sm_61) | 249 MB | 42.5 |
| PyTorch 2.7.1+cu126, CPU fp32         | 497 MB | 22.5 |

Qwen2.5-0.5B (24 layers, GQA 14q/2kv, SwiGLU, RoPE, 152k vocab):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, fp16 storage**  | 988 MB  | **30.5**   |
| **ours, int8 (dp4a)**   | 494 MB  | **74.3**   |
| **ours, int4 (dp4a)**   | 278 MB  | **104.1**  |
| **ours, int4 + graph**  | 278 MB  | **108.6**  |
| **ours, int4 + spec**   | 278 MB  | **129.4**  |
| **ours, int3 / int2**   | 244 / 222 MB | **125.2 / 125.4** (see ppl) |
| PyTorch 2.7.1+cu126, GPU fp16 (sm_61) | 988 MB | 20.4 |

TinyLlama-1.1B (22 layers, GQA 32q/4kv, SwiGLU, RoPE, untied lm_head,
n_ctx 2048 — the full trained window):

| engine                  | weights | tokens/sec |
|-------------------------|---------|------------|
| **ours, int8 (dp4a)**   | 1100 MB | **38.9**   |
| **ours, int4 (dp4a)**   | 619 MB  | **61.8**   |
| **ours, int4 + spec**   | 619 MB  | **74.0**   |
| **ours, int3 (dp4a)**   | 528 MB  | **71.7** (+0.65 ppl — usable) |
| **ours, int2 (dp4a)**   | 462 MB  | **79.2** (ppl 40 — bottom rung) |
| PyTorch 2.7.1+cu126, GPU fp16 (sm_61) | 2.2 GB | **OOM (2 GB)** |

Qwen2.5-0.5B in fp32 is ~1.9 GB of weights — it does not fit in 2 GB VRAM,
so fp16/int8 storage is not an optimization here but the only way the model
runs at all. TinyLlama-1.1B pushes that one step further: even fp16 is
2.2 GB, so the model exists on this card only as int8 (1.1 GB, barely) or
int4 (619 MB, comfortably) or int3 (528 MB, +0.65 ppl). PyTorch on this GPU
(cu126 wheels) only has fp16, which OOMs here — so a 1.1B-parameter model
generating coherent text at 62–72 tok/s over its full 2048-token window
on a 2019 laptop card is the engine's closing argument.

A word on the PyTorch baseline, because it is easy to get wrong. Pascal
(sm_61) is *not* unsupported by PyTorch in general — it was dropped only from
the CUDA 12.8/12.9/13.0 wheels (torch 2.8+); the **cu126** wheel line still
ships sm_61 kernels. (Installing CUDA 12.9 here, then running a default
`pip install torch`, lands you on a Pascal-less line and the
`cudaErrorNoKernelImageForDevice` that makes it *look* impossible — it isn't.)
With `torch==2.7.1+cu126` a real GPU baseline runs on this card
(`scripts/hf_baseline.py`): GPT-2 fp16 **42.5 tok/s**, Qwen2.5-0.5B fp16
**20.4** — both *slower* than this engine (GPT-2 fp16 117 / int8 266; Qwen
int8 74), because Pascal's fp16 throughput is poor (this engine stores fp16
but does the math in fp32 for exactly that reason) and HF `generate` carries
per-token host overhead. TinyLlama-1.1B fp16 (~2.2 GB) simply **OOMs** on the
2 GB card under torch, which has no sub-fp16 path here — so the model that runs
at 62 tok/s int4 on this engine does not run under PyTorch at all. The honest
claim is therefore not "PyTorch can't use this GPU" (it can) but "even when it
can, it needs ≥2× the memory and loses on speed."

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

The CPU baselines below use a CPU-only build (`-DGGML_CUDA=OFF`) of the same
sources and `-ngl 0` (`build-cpu`, commit `7dad2f1`).

| model | engine | model storage | prefill / prompt processing | greedy decode |
|-------|--------|---------------|-----------------------------|---------------|
| GPT-2 | llama.cpp CUDA | Q8_0 GGUF, 167.75 MiB | **2756.7 tok/s** (`pp512`) | 144.0 tok/s (`tg128`) |
| GPT-2 | ours | int8 weights, ~124 MiB | 2535 tok/s (`512 / 0.202s`) | **266.5 tok/s** |
| Qwen2.5-0.5B | llama.cpp CUDA | Q8_0 GGUF, 500.79 MiB | **866.0 tok/s** (`pp512`) | 45.4 tok/s (`tg128`) |
| Qwen2.5-0.5B | ours | int8 weights, ~494 MiB | **977 tok/s** (`512 / 0.524s`) | **74.3 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q8_0 GGUF, 1.09 GiB | **384.3 tok/s** (`pp512`) | 22.0 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int8 weights, ~1.1 GB | **431 tok/s** (`512 / 1.188s`) | **38.9 tok/s** |
| TinyLlama-1.1B | llama.cpp CUDA | Q4_0 GGUF, 606.53 MiB | **430.5 tok/s** (`pp512`) | 30.8 tok/s (`tg128`) |
| TinyLlama-1.1B | ours | int4 weights, ~619 MB | 329 tok/s (`512 / 1.556s`) | **61.8 tok/s** |

CPU baselines put the GPU numbers in context — the same llama.cpp build run
with `-ngl 0` (CPU-only build `7dad2f1`, 4 threads on an i5-10210U), plus
PyTorch 2.7.1+cu126 on CPU:

| model | engine | prefill (`pp512`) | decode (`tg128`) |
|-------|--------|-------------------|------------------|
| GPT-2 | llama.cpp CPU, Q8_0 | 913.2 tok/s | 120.3 tok/s |
| GPT-2 | PyTorch CPU, fp32 | — | 22.5 tok/s |
| Qwen2.5-0.5B | llama.cpp CPU, Q8_0 | 139.9 tok/s | 34.3 tok/s |
| Qwen2.5-0.5B | PyTorch CPU, fp32 | — | 6.0 tok/s |
| TinyLlama-1.1B | llama.cpp CPU, Q8_0 | 55.3 tok/s | 17.2 tok/s |
| TinyLlama-1.1B | llama.cpp CPU, Q4_0 | 62.4 tok/s | 26.1 tok/s |
| TinyLlama-1.1B | PyTorch CPU, fp32 | — | 3.6 tok/s |

GPT-2 is small enough that CPU decode (120 tok/s, llama.cpp) rivals llama.cpp
on the GPU (144) — the GPU edge only opens up on the bigger models (Qwen decode
34 CPU vs 74 ours, TinyLlama Q4_0 26 CPU vs 62 ours). PyTorch eager on CPU is
much slower than llama.cpp's quantized CPU kernels throughout.

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
Prefill went the same way once the dp4a GEMMs got their own wide tiles
(int8 128x128, int4 64x128 over 32 k-values), double-buffered smem with
vector-staged loads, and per-model activation groups (AG=8 on the RoPE
models — GPT-2's outliers need AG=4, the others were paying double the
scale-FMAs for nothing): **Qwen and TinyLlama int8 prefill now beat
llama.cpp too** (977 vs 866, 431 vs 384 tok/s). What's left of the table
is GPT-2 prefill at 1.09x (2535 vs 2756.7) and TinyLlama Q4_0 at 1.31x
(329 vs 430.5) — MMQ's per-quant-format shape tuning and int4's
per-tile weight-scale fold, no longer a design gap.

### Kernel-level comparison (MMVQ / MMQ)

End-to-end `tok/s` conflates the matmul kernels with everything around them —
tokenizer, sampler, host loop, residual/norm/attention kernels, and kernel
fusion. To attribute the win or loss to the quantized matmul *itself*, time
each weight matmul in isolation and put it next to llama.cpp's quantized
`MUL_MAT` on the same shape. `llm-engine kbench` does our side; llama.cpp's
side is `test-backend-ops perf -o MUL_MAT`, which dispatches `n=1` (one token,
decode) to **MMVQ** and `n=512` (a 512-token batch, prefill) to **MMQ** —
the same two kernels our `gemv_int8/int4` and `gemm_int8/int4_wide` stand in
for. int8↔`Q8_0`, int4↔`Q4_0`.

```
# our kernels, per matmul shape (no weights loaded — dp4a timing is
# data-independent, so synthetic quantized weights time identically):
llm-engine kbench --model gpt2 > kbench_gpt2.md          # qwen / tinyllama too
llm-engine kbench --model gpt2 --emit-llama              # the test_mul_mat cases to patch in

# llama.cpp side (shapes patched into make_test_cases_perf), stderr split out
# so the one-time "disabling CUDA graphs" warning can't interleave a timing:
test-backend-ops perf -o MUL_MAT > tbo.txt 2>tbo.err

scripts/compare_llama.py tbo.txt kbench_gpt2.md kbench_qwen.md kbench_tinyllama.md
```

Decode is one GEMV per weight per token, so the fair unit is latency
(`us/run`); prefill is compute-bound, so it's `GFLOP/s`. Shapes are the real
qkv / attn_proj / ffn / lm_head matmuls of GPT-2 124M, Qwen2.5-0.5B and
TinyLlama-1.1B (lm_head is decode-only — real prefill projects logits for one
position, not the whole prompt). 15 decode shapes and 12 prefill shapes per
weight mode; speedup is ours ÷ llama.cpp, so >1 means we win:

| category | wins | geomean speedup |
|----------|------|-----------------|
| int8 decode | 15/15 | 1.51x |
| int4 decode | 15/15 | 1.69x |
| int8 prefill | 8/12 | 1.00x |
| int4 prefill | 0/12 | 0.70x |

**Decode goes to this engine on every shape, and the lead widens with the
model.** GPT-2's small matmuls win 1.2–1.4x (int8) and 1.2–2.1x (int4);
TinyLlama's larger ones reach 2.0–2.1x (int8) and 2.1–2.4x (int4, e.g.
lm_head 867 vs 2059 us). MMVQ is general — one kernel for every quant
format and arch; our GEMV is one arch, one layout, activations quantized
straight into shared memory, so it keeps more of the dp4a ceiling. This is
exactly why our `tg128` beats llama.cpp CUDA on *every* model in the table
above. The decode/MMVQ path is the clean PR target for Pascal sm_61.

**Prefill is the other story.** int8 is a wash overall (1.00x geomean):
GPT-2's four small qkv/proj/FFN shapes lose ~0.75x, but Qwen and TinyLlama
win 1.1–1.25x on all eight of theirs — which is why their end-to-end int8
`pp512` already beats llama.cpp. int4 prefill, though, loses on all 12 shapes (0.70x geomean,
0.5x on GPT-2): MMQ never leaves its per-format integer tile and folds the
`Q4_0` weight scale once per tile, work our 64x128 int4 tile pays for in the
epilogue. That single kernel is what the end-to-end TinyLlama Q4_0 `pp512`
gap (1.31x) traces back to. Prefill/MMQ — int4 especially — is not a PR
target here.

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
  | int8 | 2.178s     | 0.202s        | 10.8x   |
  | int4 | 1.654s     | 0.225s        | 7.4x    |

  (int8/int4 prefill is dp4a GEMM on the wide tile; their token loops are
  so fast after dp4a that the batch speedup *ratio* shrinks while the
  absolute TTFT keeps falling.) The prefill path is checked against the
  token loop in `verify`: final logits may differ at float-rounding scale,
  but the greedy argmax must match in every weight/KV mode.
- **Prefill attention scores can run on dp4a too (`--prefill-dp4a`, opt-in).**
  The kv8 path already scores QKᵀ with `__dp4a` (q and the int8-cache K go
  straight in); the same trick works over an fp32 cache by quantizing each K
  tile row to int8 on the fly (symmetric absmax, so no affine term to fold) —
  and dropping the 16 KB fp32 K tile lifts occupancy. Worth ~6-8% on prefill
  (GPT-2 batch TTFT fp32 0.239→0.222, int8 0.196→0.180, int4 0.223→0.208s).
  It stays opt-in, not default, because the scores go int8-approximate while
  non-kv8 decode stays exact fp32: the greedy argmax still matches on every
  meaningful mode (verify green for fp32/fp16/int8/int4/int4k on all three
  models), but the bottom-rung int3/int2 — logits already ~100x off — can flip
  a token. So the exact fp32-score prefill remains the default and the
  decode==prefill invariant holds untouched unless you ask for the speed;
  `verify --prefill-dp4a` exercises the opt-in path and relaxes only int3/int2.
  Decode is left alone deliberately — it is memory-bound (dp4a buys it nothing)
  and its 8e-5 fp32 fidelity is the reference the rest is checked against.
  One kernel, `attn_prefill_body<int KIND>`: 0 exact fp32, 1 kv8 int8-cache,
  2 this dp4a-over-fp32-cache path.
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
  another 36–43% on int prefill, then double-buffered smem with
  vector-staged loads (−20–28% more) and per-model activation groups
  (AG=8 where no outliers force AG=4: −16–28% more on the RoPE models)
  (net: GPT-2 int8 0.392 → 0.202s, TinyLlama
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
  int4 prefill went from 35% behind int8 to nearly level (GPT-2 0.225s vs
  0.202s, TinyLlama 1.56s vs 1.19s @512) — the remaining gap is the
  per-tile weight-scale fold and the half-height wide tile it forces.
- **`--int4k` is the same nibbles with k-quants two-level scales (Q4_K),
  a separate opt-in mode — quality up, speed down.** It reuses the int4
  nibble packing but swaps the single Q4_0 scale for the same two-level
  `w = d*q - m` fit the int3/int2 paths use (16-row sub-blocks, 128-row
  fp16 super-pair, the `-m` term folded through the activation sums), at
  4.75 bits/weight vs 4.5. The quality gain tracks how fragile the model
  is: GPT-2 perplexity 264 → 81 (the 124M model's outliers were wrecking
  one scale per 32 rows), but only 14.27 → 14.00 on Qwen and 6.04 → 5.98
  on TinyLlama, where Q4_0 is already near the 4-bit floor. The cost is
  real on this card: the richer dequant is ~2x the scale work per row at
  the same dp4a density, so decode drops 20-32% and prefill rises ~25%.
  One occupancy lesson on the way: the first wide-tile cut spent 150
  registers (a per-tile `sx[2][8]` plus a doubled per-sub-block fold) and
  fell to one block/SM (12.5% occupancy), nearly doubling prefill;
  `__launch_bounds__(256, 2)` pinned it to exactly 128 registers with
  zero spills — two blocks/SM, same as the int8 wide tile — and a single
  `tacc[8][4]` folded per sub-block (not the `tacc[2][...]` that spilled
  the int3/int2 wide attempts) kept it there. So `--int4` stays the fast
  default and `--int4k` is there when the bits matter more than the
  tokens/sec.
- **int3 and int2 finish the bit ladder — with k-quants two-level
  scales.** int3 packs two int2-style lo-plane words plus a hi-bit word
  per 32-row group (a dp4a plane assembles as `lo2 | hi << 2`); int2 is
  four bit-pair planes in one word. The first cut used one fp16 absmax
  scale per 32 rows, and below 4 bits that simply dies (ppl 152 on
  TinyLlama, 4e4 on Qwen, 1e12 on GPT-2). The fix is the Q2_K/Q3_K
  playbook from llama.cpp, both halves of it. First, two-level
  asymmetric scales: `w = d*q - m` per 16-row sub-block (a grid-search
  least-squares fit, q stored unsigned in the same bit planes), with
  4-bit sub-scales for d and m packed one byte per sub-block and one
  fp16 (d, m) pair per 128-row super-block — 2.75/3.75 bits per weight
  instead of 2.5/3.5. The `-m` term folds analytically into the same
  activation sums the int4 path already tracks (in the GEMMs one extra
  `dp4a(a, 0x01010101)` per word builds them). Second, mixed precision:
  embeddings, lm_head and ffn_down are exactly the tensors llama.cpp
  refuses to take below 4 bits, and the bottom rungs here do the same
  (int4 for embeddings/lm_head on int3/int2, plus ffn_down on int2).
  Result: int3 becomes a real operating point everywhere it wasn't —
  TinyLlama 6.69 ppl (+0.65 over int4, was +1.1) at 528 MB, Qwen 17.1
  (was 27.7) — and int2 climbs out of total collapse on every model
  (TinyLlama 152 → 40, Qwen 4e4 → 82, GPT-2 1e12 → 6268) while staying
  unusable, which is itself the honest result: at 2 bits the scales are
  no longer the problem, the 4 quantization levels are. One trade
  surfaced: the ffn_down bump hands back enough bytes that int2 loses
  its speed edge over int3 (GPT-2 424 vs 429 tok/s, Qwen 125.4 vs
  125.2) — int3 now strictly dominates it on this card. A negative
  result worth recording: importance-weighting the fit by |w| (which
  llama.cpp does) made TinyLlama int2 24x *worse* here — uniform
  least-squares won.
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
| GPT-2 | int4 | fp32 | 2047                  | **264.2** (Q4_0) |
| GPT-2 | int4k | fp32 | 2047                 | **81.25** (Q4_K — 3.25x better than Q4_0) |
| GPT-2 | int3 | fp32 | 2047                  | **408.5** (was 3.2e5 pre-k-quants) |
| GPT-2 | int2 | fp32 | 2047                  | **6268** (was 1.3e12) |
| GPT-2 | fp32 | int8 | 2047                  | 25.363     |
| GPT-2 | fp16 | int8 | 2047                  | 25.377     |
| GPT-2 | int8 | int8 | 2047                  | 25.651     |
| Qwen  | fp16 | fp32 | 2047                  | 12.463     |
| Qwen  | int8 | fp32 | 2047                  | 12.460     |
| Qwen  | int4 | fp32 | 2047                  | 14.269 (Q4_0) |
| Qwen  | int4k | fp32 | 2047                 | 13.998 (Q4_K) |
| Qwen  | int3 | fp32 | 2047                  | 17.083 (was 27.7) |
| Qwen  | int2 | fp32 | 2047                  | **82.4** (was 4.0e4) |
| Qwen  | fp16 | int8 | 2047                  | 12.939     |
| Qwen  | int8 | int8 | 2047                  | 12.953     |
| TinyLlama | int8 | fp32 | 2047              | 5.782      |
| TinyLlama | int4 | fp32 | 2047              | 6.043 (Q4_0) |
| TinyLlama | int4k | fp32 | 2047             | 5.980 (Q4_K) |
| TinyLlama | int3 | fp32 | 2047              | 6.686 (was 7.152) |
| TinyLlama | int2 | fp32 | 2047              | **40.1** (was 152.3) |
| TinyLlama | int8 | int8 | 2047              | 5.786      |
| TinyLlama | int4 | int8 | 2047              | 6.043 (Q4_0) |
| TinyLlama | int4k | int8 | 2047             | 5.981 (Q4_K) |

(TinyLlama rows are over its full 2048-token window — the n_ctx bump from
1024 alone took int8 from 7.356 to 5.782, the biggest quality jump in the
project for zero weight bytes.)

(int8/int4 here are the dp4a W8A8/W4A8 paths — activations quantized on the
fly in 4-value absmax groups. The group size matters: 32-wide groups cost
GPT-2 +0.7 ppl because of its activation outliers; at 4 the damage is zero
and the speed identical.)

(int4 is Q4_0 (one fp16 scale per 32 rows); int4k is the opt-in Q4_K variant
with k-quants two-level sub-block scales — better quality, ~20-32% slower
decode. The gap is largest exactly where Q4_0's single scale hurts most: the
124M GPT-2 (264 → 81), tiny on the bigger models already near the 4-bit floor.)

(int3/int2 are the k-quants-style two-level rows: 16-row sub-blocks fit as
`w = d*q - m` with 4-bit sub-scales under a 128-row fp16 super-scale pair,
and embeddings/lm_head — plus ffn_down on int2 — stay at int4, the same
tensors llama.cpp's Q2_K/Q3_K presets refuse to shrink.)

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
