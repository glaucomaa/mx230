# Stage 2 — Flash Attention from scratch

Forward-pass Flash Attention (Dao et al. 2022) in plain CUDA vs a naive
attention baseline that materializes the full N×N score matrix. Single head,
head dim 64, fp32, optional causal mask. Run:

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
