### check_env.py output

torch: 2.14.0+rocm10.1.0 | HIP: 7.16.26385
GPU available: True
device: AMD Radeon RX 7800 XT | arch: gfx1101
VRAM: 16.0 GiB
BLAS: _BlasBackend.Cublas
triton: 3.8.0
bf16 matmul ok: True

### roofline.py output

copy bandwidth :   505.2 GB/s
bf16 GEMM peak :    58.1 TFLOPS
ridge point    :   115.0 FLOP/byte
    B   time(us)  TFLOPS  FLOP/byte
    1      237.6    0.56        1.0
    2      241.3    1.11        2.0
    4      245.1    2.19        4.0
    8      245.4    4.38        8.0
   16      246.0    8.73       15.9
   32      247.2   17.38       31.8
   64      253.0   33.95       63.0
  128      449.9   38.18      124.1
  256      670.6   51.24      240.9
  512     1259.3   54.57      455.1
 1024     2562.4   53.64      819.2
 2048     5065.3   54.27     1365.3

### hf_baseline.py output

/home/qyc/venvs/llm/lib/python3.14/site-packages/transformers/integrations/sdpa_attention.py:174: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. (Triggered internally at /__w/rockrel/rockrel/external-builds/pytorch/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:498.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
TTFT ≈ 25.8 ms
TPOT ≈ 16.35 ms  (61 tok/s)
带宽上限 ≈ 2.36 ms/token  (424 tok/s)
 Explain how a KV cache speeds up decoding. Explain how a KV cache speeds up decoding. Explain how a KV cache speeds up decoding. Explain how a KV cache speeds up decoding. Explain how a KV

### 思考题

1. 实测roofline是115，跟120差5，因为算力和带宽都没到达标称吧。
2. batch增大到64前耗时几乎不变，这时FLOP/字节基本等于batch大小，符合“白赚”。
3. hf基线的TPOT是带宽上限的1/7。以下是我加的profile代码，但是我无法回答你的问题。
```
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
```
对应的输出：
```
USDT:2026-10-30 18:20:26 64035:64035 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-30 18:20:30 64035:64035 SyncActivityProfilerHandler.cpp:46] profiler_stop
[W1030 18:20:30.493976594 collection.cpp:1211] Warning: Failed to recover relationship between all profiler and kineto events: 566439 vs. 0  reassociated. (function reassociate)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void rocblas_gemvt_kernel<false, 256, rocblas_bfloat...         0.00%       0.000us         0.00%       0.000us       0.000us     698.227ms        39.16%     698.227ms      27.689us           0 B           0 B           0 B           0 B         25217  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us     124.687ms         6.99%     124.687ms       5.738us           0 B           0 B           0 B           0 B         21729  
Cijk_Ailk_Bljk_SB_MT64x128x8_SN_1LDSB0_AMAS0_BL1_BS1...         0.00%       0.000us         0.00%       0.000us       0.000us      92.582ms         5.19%      92.582ms      25.632us           0 B           0 B           0 B           0 B          3612  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us      82.516ms         4.63%      82.516ms       5.661us           0 B           0 B           0 B           0 B         14577  
Cijk_Alik_Bljk_SB_MT64x128x8_SN_1LDSB0_AMAS0_BL1_BS1...         0.00%       0.000us         0.00%       0.000us       0.000us      77.484ms         4.35%      77.484ms      21.452us           0 B           0 B           0 B           0 B          3612  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us      75.946ms         4.26%      75.946ms       5.164us           0 B           0 B           0 B           0 B         14706  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      66.173ms         3.71%      66.173ms       2.591us           0 B           0 B           0 B           0 B         25542  
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us      61.680ms         3.46%      61.680ms       8.605us           0 B           0 B           0 B           0 B          7168  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us      55.150ms         3.09%      55.150ms       7.634us           0 B           0 B           0 B           0 B          7224  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      54.455ms         3.05%      54.455ms       2.469us           0 B           0 B           0 B           0 B         22059  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 983.151ms
Self CUDA time total: 1.783s
```

### 疑问
1. 当我把hf基线脚本里的预热代码去掉后，测出来的TTFT增加到了560ms，但是TPOT却上涨到12ms，为什么？