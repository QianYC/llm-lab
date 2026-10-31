import time
import torch

dev, dt = "cuda", torch.bfloat16


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# 1) 显存带宽：拷贝 1 GiB（读 1 GiB + 写 1 GiB）
a = torch.empty(2**29, device=dev, dtype=dt)
b = torch.empty_like(a)
t = bench(lambda: b.copy_(a))
bw = 2 * a.numel() * a.element_size() / t
print(f"copy bandwidth : {bw / 1e9:7.1f} GB/s")

# 2) 算力峰值：大矩阵乘
m = 8192
x = torch.randn(m, m, device=dev, dtype=dt)
w = torch.randn(m, m, device=dev, dtype=dt)
t = bench(lambda: x @ w, iters=20)
peak = 2 * m**3 / t
print(f"bf16 GEMM peak : {peak / 1e12:7.1f} TFLOPS")
print(f"ridge point    : {peak / bw:7.1f} FLOP/byte")

# 3) 模拟 decode：Y = X(B×d) @ W(d×d)，扫描 batch 大小 B
#    d=8192 时 W 有 128 MiB，比 64 MB 的 Infinity Cache 大
d = 8192
w = torch.randn(d, d, device=dev, dtype=dt)
print("    B   time(us)  TFLOPS  FLOP/byte")
for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    x = torch.randn(B, d, device=dev, dtype=dt)
    t = bench(lambda: x @ w)
    flops = 2 * B * d * d
    nbytes = 2 * (d * d + 2 * B * d)
    print(f"{B:5d} {t * 1e6:10.1f} {flops / t / 1e12:7.2f}"
          f" {flops / nbytes:10.1f}")
