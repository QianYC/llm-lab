import time
import torch
from huggingface_hub import snapshot_download
from qwen3 import load_weights, forward, new_cache
from torch.profiler import profile, ProfilerActivity, record_function

W = load_weights(snapshot_download("Qwen/Qwen3-0.6B"))

def ms(fn, iters=10):
    for _ in range(2):
        fn()                                     # 预热
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print("  长度  无 cache(ms)  有 cache(ms)")
for n in [128, 256, 512, 1024, 2048]:
    ids = torch.randint(0, 151936, (n,), device="cuda")
    t_full = ms(lambda: forward(ids, W, last_only=True))
    cache = new_cache(n + 1, torch.bfloat16)
    forward(ids, W, cache, 0, last_only=True)    # 先把 cache 填好
    last = ids[-1:]

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with record_function("cached forward"):
            t_step = ms(lambda: forward(last, W, cache, n, last_only=True))
    print(f"{n:6d} {t_full:13.2f} {t_step:13.2f}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

before = torch.cuda.memory_allocated()
kv = new_cache(4096, torch.bfloat16)
kib = (torch.cuda.memory_allocated() - before) / 4096 / 1024
print(f"KV cache: {kib:.0f} KiB/token")

del W
before = torch.cuda.memory_allocated()
w = load_weights(snapshot_download("Qwen/Qwen3-0.6B"))
kib = (torch.cuda.memory_allocated() - before) / 1024 / 1024
print(f"Qwen3-0.6B weight: {kib:.0f} MiB")