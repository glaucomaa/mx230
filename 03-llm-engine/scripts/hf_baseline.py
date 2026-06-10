# HF transformers baseline for the tokens/sec table.
# Run: uv run --with torch --with transformers scripts/hf_baseline.py
import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# torch.cuda.is_available() is true here, but current torch wheels ship no
# sm_61 kernels (cudaErrorNoKernelImageForDevice) — Pascal support is gone,
# so the only PyTorch baseline this GPU can have is the CPU one.
device = "cpu"
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

ids = tok("The history of computing began", return_tensors="pt").input_ids.to(device)
n_new = 128

with torch.no_grad():
    # warmup
    model.generate(ids, max_new_tokens=4, do_sample=False)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
    dt = time.time() - t0

print(f"device={device} torch={torch.__version__}")
print(f"{n_new} tokens in {dt:.2f}s = {n_new / dt:.1f} tok/s")
