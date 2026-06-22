# Stage 2 — Flash Attention from scratch

Flash Attention (Dao et al. 2022) in plain CUDA, **forward and backward**, vs a
naive attention baseline that materializes the full N×N score matrix. Single
head, head dim 64, fp32, optional causal mask. The backward pass is the training
half — gradients dQ/dK/dV from the upstream dO. Run:

```
cargo run -rp flash-attention
```

## Results (median of 5 runs, non-causal)

|      N |  naive (S = NxN) |        flash | speedup | naive S extra |
|--------|------------------|--------------|---------|---------------|
|   1024 |          17.0 ms |       1.4 ms |  11.74x |          4 MB |
|   2048 |          64.7 ms |       4.1 ms |  15.64x |         17 MB |
|   4096 |         275.9 ms |      16.3 ms |  16.88x |         67 MB |
|   8192 |        1152.2 ms |      60.8 ms |  18.95x |        268 MB |
|  16384 |        4762.1 ms |     250.4 ms |  19.02x |       1074 MB |
|  32768 |              OOM |     992.9 ms |       - |       4295 MB |

Flash uses **zero** extra global memory and scales right past the point where
the naive version no longer fits in 2 GB of VRAM. Both implementations are
verified against a CPU reference (causal and non-causal) with an
`allclose`-style criterion (rtol 1e-3, atol 1e-4).

## How

**Naive** (`attention_naive.cu`): three kernels — S = QKᵀ/√d (one thread per
score), row-wise softmax in place (one 256-thread block per row, shared-memory
reductions), O = S·V. S costs N²·4 bytes of VRAM and, worse, several full
N² passes through global memory — on a 40 GB/s bus that dominates everything.

**Flash** (`attention_flash.cu`): one thread owns one query row; its Q row,
output accumulator and the online-softmax state (running max m, running sum l)
live entirely in registers. K/V tiles (32×64) are staged through shared memory
and shared by the 64 rows of the block. Each new score rescales the
accumulator by exp(m_old − m_new) — mathematically identical to softmax, no
N×N matrix ever exists. With the causal flag, tiles past the block's last row
are skipped entirely.

The ~19x speedup is the stage-1 lesson restated: same FLOPs, ~N²/8 fewer bytes
of DRAM traffic, and on a 64-bit memory bus traffic is the only thing that
matters.

## Backward pass

The training half. Given Q, K, V, the forward output O and the upstream
gradient dO, compute dQ, dK, dV — again without ever materializing the N×N
matrix. The forward (`attn_flash_lse`) emits one extra scalar per query row, the
log-sum-exp `L_i = m_i + log(l_i)`, so the backward recomputes every score and
recovers `p_ij = exp(s_ij − L_i)` from that single value.

`attention_bwd.cu`, following the softmax-attention VJP:

```
Delta_i = Σ_x dO[i,x]·O[i,x]            (one scalar per row, attn_bwd_preprocess)
dP_ij   = dO_i · v_j
dS_ij   = p_ij·(dP_ij − Delta_i)·scale
dV_j += p_ij·dO_i      dK_j += dS_ij·q_i      dQ_i += dS_ij·k_j
```

One thread owns one query row `i` and accumulates dQ_i privately in registers;
dK_j/dV_j are touched by every query row that attends key j, so those scatter
with `atomicAdd` (native fp32 on sm_61). Correctness is gated two ways: GPU
dQ/dK/dV match a CPU reference (`allclose`, rtol 1e-3/atol 1e-4, causal and
non-causal) and that CPU reference matches a central finite-difference of the
scalar loss `⟨O, dO⟩` (rel err ~1e-5).

### Results (median of 5 runs, causal)

|      N |        fwd |        bwd |   bwd/fwd |
|--------|------------|------------|-----------|
|   1024 |    0.82 ms |   43.11 ms |    52.83x |
|   2048 |    2.57 ms |  163.81 ms |    63.71x |
|   4096 |    9.06 ms |  636.78 ms |    70.30x |

This first backward is correctness-first, not tiled: it reads K/V from global
memory on every key, serializes dK/dV through atomics, and spills the three
head-dim register arrays (q, dO, dQ accumulator) — so it runs 50–70× slower than
the tiled forward, and the gap widens with N. The fix is the **FlashAttention-2
backward**: parallelize over K/V blocks (compute dK_j, dV_j per block, with a
separate dQ pass) to stage tiles in shared memory and drop the atomics. That is
the next step for this stage.
