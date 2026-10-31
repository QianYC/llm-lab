import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, record_function

# 用法：python hf_baseline.py <roofline.py 测到的带宽 GB/s>
bw = float(sys.argv[1]) * 1e9 if len(sys.argv) > 1 else None
mid = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid, dtype=torch.bfloat16).to("cuda").eval()

nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"weights: {nbytes / 1e9:.2f} GB")

prompt = "Explain how a KV cache speeds up decoding. " * 20
enc = tok(prompt, return_tensors="pt").to("cuda")
n_in = enc.input_ids.shape[1]
print("prompt tokens:", n_in)


@torch.no_grad()
def run(n_new, label):
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function(label):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=n_new,
                         min_new_tokens=n_new, do_sample=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return end - t0, out


run(8, "warm up")                    # 预热
t1, _ = run(1, "prefill")            # ≈ prefill + 第 1 个 token
t2, out = run(129, "decode")
tpot = (t2 - t1) / 128
print(f"TTFT ≈ {t1 * 1e3:.1f} ms")
print(f"TPOT ≈ {tpot * 1e3:.2f} ms  ({1 / tpot:.0f} tok/s)")
if bw:
    bound = nbytes / bw
    print(f"带宽上限 ≈ {bound * 1e3:.2f} ms/token"
          f"  ({1 / bound:.0f} tok/s)")
print(tok.decode(out[0, n_in:n_in + 40]))
