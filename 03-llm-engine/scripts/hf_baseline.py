# HF transformers baseline for the tokens/sec table.
#
# Pascal (sm_61) is NOT unsupported by PyTorch in general — it was dropped only
# from the CUDA 12.8 / 12.9 / 13.0 wheels (torch 2.8+). The cu126 wheel line
# still ships sm_61 kernels, so a real GPU baseline IS possible on this card;
# it just needs the cu126 build instead of whatever `pip install torch` gives
# you today (which defaults to a Pascal-less CUDA line).
#
# Run (GPU baseline, cu126 wheels):
#   uv run --with transformers --with "torch==2.7.1" \
#     --extra-index-url https://download.pytorch.org/whl/cu126 \
#     scripts/hf_baseline.py --model gpt2
#   (or, with a recent uv: add `--torch-backend=cu126` instead of the index url)
#
# --model gpt2|qwen|tinyllama, default gpt2.
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_IDS = {
    "gpt2": "openai-community/gpt2",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=HF_IDS.keys(), default="gpt2")
ap.add_argument("-n", type=int, default=128, help="new tokens to generate")
args = ap.parse_args()
hf_id = HF_IDS[args.model]

print(f"torch={torch.__version__} cuda={torch.version.cuda} "
      f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)} "
          f"cc={torch.cuda.get_device_capability(0)}")

tok = AutoTokenizer.from_pretrained(hf_id)
ids_cpu = tok("The history of computing began", return_tensors="pt").input_ids


def bench(device, dtype):
    """Returns tok/s, or raises on OOM / no-kernel-image."""
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    ids = ids_cpu.to(device)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)  # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        model.generate(ids, max_new_tokens=args.n, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        return args.n / (time.time() - t0)


# GPU first (fp16 — the dtype that actually fits a 2 GB card); on OOM or a
# missing sm_61 kernel image, that failure is itself the result for the table.
if torch.cuda.is_available():
    try:
        tps = bench("cuda", torch.float16)
        print(f"{args.model} GPU fp16: {args.n} tokens = {tps:.1f} tok/s")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        kind = "OOM (2 GB)" if isinstance(e, torch.cuda.OutOfMemoryError) else "FAILED"
        print(f"{args.model} GPU fp16: {kind} — {str(e).splitlines()[0]}")
        torch.cuda.empty_cache()
else:
    print("cuda not available to torch — install the cu126 wheel for a GPU row")

# CPU baseline (fp32) for reference, as before.
tps = bench("cpu", torch.float32)
print(f"{args.model} CPU fp32: {args.n} tokens = {tps:.1f} tok/s")
