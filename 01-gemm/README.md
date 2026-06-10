# Stage 1 — SGEMM optimization ladder

Five hand-written CUDA kernels, each one optimization step apart, benchmarked
against cuBLAS on an MX230 (Pascal sm_61, 256 cores, ~40 GB/s). Run:

```
cargo run -rp gemm
```

## Results (GFLOPS, median of 7 runs, % of cuBLAS)

| kernel         |               256 |               512 |              1024 |              2048 |
|----------------|-------------------|-------------------|-------------------|-------------------|
| v1 naive       |      11.5 (   2%) |      11.3 (   2%) |      11.1 (   2%) |      10.6 (   2%) |
| v2 coalesced   |      40.2 (   7%) |      40.7 (   7%) |      40.1 (   7%) |      36.2 (   5%) |
| v3 smem tiled  |     105.7 (  18%) |     108.6 (  17%) |     108.4 (  18%) |     108.3 (  16%) |
| v4 blocktiled  |     239.2 (  42%) |     394.2 (  63%) |     434.2 (  71%) |     452.4 (  67%) |
| v5 vectorized  |     287.4 (  50%) |     468.1 (  75%) |     477.9 (  78%) |     484.6 (  71%) |
| cuBLAS         |     574.9 (100%) |     624.2 (100%) |     611.8 (100%) |     677.8 (100%) |

45x from naive to vectorized; the remaining gap to cuBLAS is mostly double
buffering and finer-grained scheduling.

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

## Roofline sanity check

Peak fp32 ≈ 0.94 TFLOPS (256 cores × 2 × 1.83 GHz boost), DRAM ≈ 40 GB/s →
ridge point at ~24 FLOP/byte. A 2048³ SGEMM does 17.2 GFLOP over ≥50 MB of
mandatory traffic (~340 FLOP/byte): firmly compute-bound *if* you reuse data.
v1–v2 sit on the memory roof (v2's 36 GFLOPS ≈ 40 GB/s × ~1 FLOP/byte fetched);
v3 lifts reuse into shared memory, v4 into registers, after which the kernel
runs at ~50% of peak ALU throughput — within 30% of cuBLAS.

Notes: kernels v3–v5 require sizes divisible by 32/128 (benchmark sizes are);
Nsight Compute no longer supports Pascal, so analysis is cudaEvent timings +
theory only. Correctness is checked against cuBLAS at 512³ (max rel. err = 0).
