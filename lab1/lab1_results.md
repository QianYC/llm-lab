### check.py output
rms_norm  0.0
q_norm  0.0
mlp       0.0
rope      0.0 0.0
layer0 attention  0.0
layer 0  0.0e+00
layer 1  0.0e+00
layer 2  0.0e+00
layer 3  0.0e+00
layer 4  0.0e+00
layer 5  0.0e+00
layer 6  0.0e+00
layer 7  0.0e+00
layer 8  0.0e+00
layer 9  0.0e+00
layer10  0.0e+00
layer11  0.0e+00
layer12  0.0e+00
layer13  0.0e+00
layer14  0.0e+00
layer15  0.0e+00
layer16  0.0e+00
layer17  0.0e+00
layer18  0.0e+00
layer19  0.0e+00
layer20  0.0e+00
layer21  0.0e+00
layer22  0.0e+00
layer23  0.0e+00
layer24  0.0e+00
layer25  0.0e+00
layer26  0.0e+00
layer27  0.0e+00
logits   0.0e+00
top-1 一致率 100.00%
有 cache == 无 cache: True
有 cache == HF      : True
 with the invention of the first electronic device, the vacuum tube, which was used to transmit and receive signals. The development of the vacuum tube led to the creation of the first electronic devices, such as the television and the radio. The development of the vacuum tube also led to the creation of the first electronic devices, such
tensor([0.6699, 0.2399, 0.0902, 0.0000, 0.0000])

### bench.py output
 长度  无 cache(ms)  有 cache(ms)                                                                      |  0.00B /  0.00B            
   128         18.37         13.81
   256         20.80         14.23
   512         38.47         15.16
  1024        102.46         17.12
  2048        336.67         20.03
  4096       1132.09         27.18
KV cache: 112 KiB/token

### 思考题
1. 不带 cache 时，第 n 步大约要算多少？（见bench输出）为什么 TPOT 随长度增长（不带cache时的TPOT实际上包含了TTFT，因为需要prefill，而prefill时间会随着输入增加而增加）？
2. 4K 上下文的一个请求，KV cache 有多大？（437.5MiB）16 GB 的卡除去权重，能同时放下多少个这样的请求？（根据实测，Qwen3-0.6B的参数占1137MiB，因此可以放34个请求）
3. 你的 TPOT 离实验 0 的带宽上限还差多少？（实验0测得的上限是2.36ms）差在哪里？（根据如下profile的输出，差距还是因为太多矩阵操作导致的）
USDT:2026-11-02 22:01:09 17296:17296 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-11-02 22:01:10 17296:17296 SyncActivityProfilerHandler.cpp:46] profiler_stop
  2048        336.20         24.34
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void rocblas_gemvt_kernel<false, 256, rocblas_bfloat...         0.00%       0.000us         0.00%       0.000us       0.000us      95.424ms        57.54%      95.424ms      43.256us          2206  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us      20.037ms        12.08%      20.037ms      36.233us           553  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       7.384ms         4.45%       7.384ms       4.494us          1643  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       5.099ms         3.07%       5.099ms       4.623us          1103  
void rocblas_gemvn_kernel<32, 16, int, rocblas_bfloa...         0.00%       0.000us         0.00%       0.000us       0.000us       4.716ms         2.84%       4.716ms      17.275us           273  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       4.105ms         2.48%       4.105ms       3.662us          1121  
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       3.101ms         1.87%       3.101ms       5.712us           543  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.757ms         1.66%       2.757ms       1.962us          1405  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.573ms         1.55%       2.573ms       1.862us          1382  
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.494ms         1.50%       2.494ms       2.280us          1094  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
