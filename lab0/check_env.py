import torch

print("torch:", torch.__version__, "| HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "| arch:", p.gcnArchName)
print(f"VRAM: {p.total_memory / 2**30:.1f} GiB")
# 矩阵乘后端：ROCm 上的 Cublaslt 指的就是 hipBLASLt
print("BLAS:", torch.backends.cuda.preferred_blas_library())
try:
    import triton
    print("triton:", triton.__version__)
except ImportError:
    print("triton: 没装（实验 3 之前再处理）")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float()
print("bf16 matmul ok:", bool(torch.isfinite(y).all()))
