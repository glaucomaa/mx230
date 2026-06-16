#!/usr/bin/env python3
"""Join `llm-engine kbench` output with llama.cpp `test-backend-ops perf -o
MUL_MAT` output into a kernel-vs-kernel comparison table.

Both sides time the same matmul shapes in isolation (no tokenizer, sampling,
host loop, or fusion). ggml convention: m=n_out, n=tokens (1=decode, 512=
prefill), k=n_in. Decode is memory-bound so the fair unit is us/run (latency);
prefill is compute-bound so it's GFLOP/s. We compare our gemv_int8/int4 and
gemm_int8/int4_wide against llama.cpp MMVQ (n=1) and MMQ (n=512).

Usage:
  compare_llama.py <tbo_mulmat.txt> <kbench_gpt2.md> [kbench_qwen.md ...]
"""
import math
import re
import sys

# llama.cpp line:
#   MUL_MAT(type_a=q8_0,type_b=f32,m=2304,n=1,k=768,bs=...): N runs - X us/run - Y MFLOP/run - Z GFLOPS
LL = re.compile(
    r"MUL_MAT\(type_a=(q8_0|q4_0),type_b=f32,m=(\d+),n=(\d+),k=(\d+),"
    r".*?-\s*([\d.]+)\s*us/run.*?-\s*([\d.]+)\s*([GT])FLOPS"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# kbench markdown row:
#  | qkv | 768→2304 | int8 | 48.1 | 37.0 | 2907.1 | 623 |
KB = re.compile(
    r"\|\s*([\w/]+)\s*\|\s*(\d+)→(\d+)\s*\|\s*(int8|int4)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.—]+)\s*\|\s*([\d.—]+)\s*\|"
)
MODE_TYPE = {"int8": "q8_0", "int4": "q4_0"}
# kbench header: "## Gpt2 — isolated matmul kernels ..." -> model name
HDR = re.compile(r"^##\s+(\S+)\s+—")


def parse_llama(path):
    """(type, m, n, k) -> (us_per_run, gflops), CUDA backend only (first seen)."""
    out = {}
    for line in open(path):
        line = ANSI.sub("", line)
        mm = LL.search(line)
        if not mm:
            continue
        ty, m, n, k, us, perf, unit = mm.groups()
        key = (ty, int(m), int(n), int(k))
        gflops = float(perf) * (1000.0 if unit == "T" else 1.0)
        out.setdefault(key, (float(us), gflops))  # first = CUDA0
    return out


def parse_kbench(path):
    """(model_name, [(label, k_in, n_out, mode, dec_us, dec_gbps, pre_us, pre_gflops)])."""
    model = path
    rows = []
    for line in open(path):
        h = HDR.match(line)
        if h:
            model = h.group(1)
        m = KB.search(line)
        if not m:
            continue
        label, k_in, n_out, mode, dus, dgb, pus, pgf = m.groups()
        rows.append((label, int(k_in), int(n_out), mode, float(dus), float(dgb),
                     None if pus == "—" else float(pus),
                     None if pgf == "—" else float(pgf)))
    return model, rows


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ll = parse_llama(sys.argv[1])
    print("| model | matmul | n_in→n_out | mode | decode us (ours/llama) | dec speedup "
          "| prefill GFLOP/s (ours/llama) | pre speedup |")
    print("|-------|--------|-----------|------|------------------------|-------------"
          "|------------------------------|-------------|")
    # speedups (ours / llama.cpp; >1 = we win), keyed by (mode, phase)
    agg = {(m, p): [] for m in ("int8", "int4") for p in ("decode", "prefill")}
    for kb_path in sys.argv[2:]:
        model, rows = parse_kbench(kb_path)
        for (label, k_in, n_out, mode, dus, _dgb, pus, pgf) in rows:
            ty = MODE_TYPE[mode]
            l_dec = ll.get((ty, n_out, 1, k_in))      # decode: n=1
            l_pre = ll.get((ty, n_out, 512, k_in))    # prefill: n=512
            if l_dec:
                dec = f"{dus:.1f} / {l_dec[0]:.1f}"
                dspd = f"{l_dec[0]/dus:.2f}x"
                agg[(mode, "decode")].append(l_dec[0] / dus)
            else:
                dec, dspd = f"{dus:.1f} / ?", "?"
            if pgf and l_pre:
                pre = f"{pgf:.0f} / {l_pre[1]:.0f}"
                pspd = f"{pgf/l_pre[1]:.2f}x"
                agg[(mode, "prefill")].append(pgf / l_pre[1])
            elif pgf:
                pre, pspd = f"{pgf:.0f} / ?", "?"
            else:
                pre, pspd = "—", "—"
            print(f"| {model} | {label} | {k_in}→{n_out} | {mode} | {dec} | {dspd} "
                  f"| {pre} | {pspd} |")

    # Aggregate the per-row speedups into the win-count + geomean the
    # conclusion is built on (geomean, not mean — speedups are ratios).
    print("\n**Summary** — speedup = ours ÷ llama.cpp, so >1.00x means we win.\n")
    print("| category | wins | geomean speedup |")
    print("|----------|------|-----------------|")
    for mode in ("int8", "int4"):
        for phase in ("decode", "prefill"):
            data = agg[(mode, phase)]
            if not data:
                continue
            wins = sum(1 for s in data if s > 1.0)
            print(f"| {mode} {phase} | {wins}/{len(data)} | {geomean(data):.2f}x |")


if __name__ == "__main__":
    main()
