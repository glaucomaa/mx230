# Stage 1 — SGEMM optimization ladder

Six hand-written CUDA kernels, each one optimization step apart, benchmarked
against cuBLAS on an MX230 (Pascal sm_61, 256 cores, ~40 GB/s). Run:

```
cargo run -rp gemm
```

## Results (GFLOPS, median of 7 runs, % of cuBLAS)

| kernel         |               256 |               512 |              1024 |              2048 |
|----------------|-------------------|-------------------|-------------------|-------------------|
| v1 naive       |      11.5 (   2%) |      11.3 (   2%) |      11.2 (   2%) |      10.6 (   2%) |
| v2 coalesced   |      40.5 (   7%) |      40.9 (   7%) |      40.2 (   7%) |      36.3 (   5%) |
| v3 smem tiled  |     102.7 (  18%) |     109.1 (  18%) |     109.3 (  18%) |     108.5 (  16%) |
| v4 blocktiled  |     235.8 (  42%) |     396.6 (  64%) |     434.8 (  71%) |     454.3 (  67%) |
| v5 vectorized  |     292.6 (  53%) |     469.0 (  75%) |     481.0 (  78%) |     487.2 (  72%) |
| v6 dbuf        |     297.9 (  54%) |     488.2 (  78%) |     502.4 (  82%) |     508.7 (  75%) |
| cuBLAS         |     555.4 (100%) |     622.7 (100%) |     614.1 (100%) |     674.7 (100%) |

48x from naive to double-buffered. The remaining gap to cuBLAS lives below
PTX — SASS-level instruction scheduling and register-bank allocation that
nvcc/ptxas does not expose (Scott Gray's maxas SGEMM showed hand-written
SASS can even beat cuBLAS by ~5-10% on Maxwell).

## The ladder

1. **naive** — one thread per C element, `threadIdx.x` walks rows: adjacent
   threads in a warp read B down a column and write C with stride N — every
   global access is a separate transaction.
2. **coalesced** — swap thread axes so `threadIdx.x` walks columns: warp reads
   consecutive B elements and writes consecutive C. Same instruction count,
   ~3.5x faster — pure memory-traffic win.
3. **smem tiled** — 32×32 block loads tiles of A and B into shared memory;
   each global element is read 32x fewer times. ~2.7x.
4. **blocktiled** — register blocking: 256 threads per block compute a 128×128
   C tile, 8×8 per thread in registers. Arithmetic intensity per shared-memory
   access grows ~64x; the kernel finally becomes compute-bound. ~4x.
5. **vectorized** — `float4` (128-bit) global loads/stores, A tile stored
   transposed in shared memory for contiguous inner-loop reads. ~1.1x.
6. **dbuf** — double buffering: two shared-memory tile buffers; the next
   tile's global loads are issued before the compute loop and land in the
   spare buffer afterwards, overlapping global-memory latency with FMAs and
   halving the `__syncthreads()` count. ~1.05x.

## Roofline sanity check

Peak fp32 ≈ 0.94 TFLOPS (256 cores × 2 × 1.83 GHz boost), DRAM ≈ 40 GB/s →
ridge point at ~24 FLOP/byte. A 2048³ SGEMM does 17.2 GFLOP over ≥50 MB of
mandatory traffic (~340 FLOP/byte): firmly compute-bound *if* you reuse data.
v1–v2 sit on the memory roof (v2's 36 GFLOPS ≈ 40 GB/s × ~1 FLOP/byte fetched);
v3 lifts reuse into shared memory, v4 into registers, after which the kernel
runs at ~50% of peak ALU throughput — within 25% of cuBLAS.

Notes: kernels v3–v6 require sizes divisible by 32/128 (benchmark sizes are);
Nsight Compute no longer supports Pascal, so analysis is cudaEvent timings +
theory only. Correctness is checked against cuBLAS at 512³ (max rel. err = 0).
