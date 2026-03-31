#!/usr/bin/env python3
"""Quick profiler to measure INT8 LLM bottleneck breakdown."""
import torch
import time
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

torch.set_num_threads(os.cpu_count())

# Measure quantization cost at different M values
def quantize_activation_per_row(x):
    x_float = x.float()
    row_max = x_float.abs().amax(dim=1, keepdim=True)
    scale_x = (row_max / 127.0).clamp(min=1e-10)
    x_int8 = (x_float / scale_x).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale_x.squeeze(1)

def quantize_only_real(x_real, buf_256_K):
    """Optimized: only quantize M_actual rows, write into preallocated buf."""
    M = x_real.shape[0]
    x_float = x_real.float()
    row_max = x_float.abs().amax(dim=1)
    scale_x = (row_max / 127.0).clamp(min=1e-10)
    buf_256_K[:M] = (x_float / scale_x[:, None]).round().clamp(-128, 127).to(torch.int8)
    return buf_256_K, scale_x

print("=" * 70)
print("  INT8 LLM Bottleneck Profiler")
print("=" * 70)

# Test 1: Quantization cost for M=1 (decode) vs M=256 (padded)
K_values = [2048, 4096]
for K in K_values:
    print(f"\n--- Quantization cost for K={K} ---")
    
    # Current: quantize 256-row padded tensor (even when M=1)
    x_padded = torch.zeros(256, K, dtype=torch.bfloat16)
    x_padded[0] = torch.randn(K, dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        quantize_activation_per_row(x_padded)
    
    N_iter = 200
    t0 = time.perf_counter()
    for _ in range(N_iter):
        quantize_activation_per_row(x_padded)
    t_padded = (time.perf_counter() - t0) / N_iter * 1000
    
    # Optimized: quantize only 1 row
    x_real = torch.randn(1, K, dtype=torch.bfloat16)
    buf = torch.zeros(256, K, dtype=torch.int8)
    
    for _ in range(10):
        quantize_only_real(x_real, buf)
    
    t0 = time.perf_counter()
    for _ in range(N_iter):
        quantize_only_real(x_real, buf)
    t_opt = (time.perf_counter() - t0) / N_iter * 1000
    
    print(f"  Padded (256×{K}):   {t_padded:.3f} ms")
    print(f"  Real-only (1×{K}):  {t_opt:.3f} ms")
    print(f"  Speedup:            {t_padded/t_opt:.1f}x")

# Test 2: Dequantization cost
print(f"\n--- Dequantization cost ---")
for N in [2048, 8192]:
    c_i32 = torch.randint(-1000, 1000, (256, N), dtype=torch.int32)
    scale_x = torch.randn(1, dtype=torch.float32).abs()
    scale_w = torch.randn(N, dtype=torch.float32).abs()
    
    # Current: c[:1].float() * (scale_x[:1, None] * scale_w[None, :])
    for _ in range(10):
        _ = c_i32[:1].float() * (scale_x[:1, None] * scale_w[None, :])
    
    t0 = time.perf_counter()
    for _ in range(N_iter):
        _ = c_i32[:1].float() * (scale_x[:1, None] * scale_w[None, :])
    t_deq = (time.perf_counter() - t0) / N_iter * 1000
    print(f"  Dequant [1×{N}]:    {t_deq:.3f} ms")

# Test 3: Tensor allocation cost
print(f"\n--- Tensor allocation cost ---")
for _ in range(100):
    _ = torch.empty(256, 2048, dtype=torch.int32)

t0 = time.perf_counter()
for _ in range(N_iter):
    _ = torch.empty(256, 2048, dtype=torch.int32)
t_alloc = (time.perf_counter() - t0) / N_iter * 1000
print(f"  torch.empty(256,2048,i32): {t_alloc:.4f} ms")

t0 = time.perf_counter()
for _ in range(N_iter):
    _ = torch.zeros(256, 2048, dtype=torch.int8)
t_zeros = (time.perf_counter() - t0) / N_iter * 1000
print(f"  torch.zeros(256,2048,i8):  {t_zeros:.4f} ms")

# Test 4: Padding cost
print(f"\n--- Padding cost ---")
x = torch.randn(1, 2048, dtype=torch.bfloat16)
for _ in range(100):
    p = torch.zeros(256, 2048, dtype=torch.bfloat16)
    p[0] = x[0]

t0 = time.perf_counter()
for _ in range(N_iter):
    p = torch.zeros(256, 2048, dtype=torch.bfloat16)
    p[0] = x[0]
t_pad = (time.perf_counter() - t0) / N_iter * 1000
print(f"  zeros+copy (256×2048,bf16): {t_pad:.3f} ms")

# Test 5: bfloat16 → float conversion
print(f"\n--- Dtype conversion cost ---")
x_bf = torch.randn(1, 2048, dtype=torch.bfloat16)
for _ in range(100):
    _ = x_bf.float()
t0 = time.perf_counter()
for _ in range(N_iter):
    _ = x_bf.float()
t_cvt = (time.perf_counter() - t0) / N_iter * 1000
print(f"  bf16→f32 (1×2048):  {t_cvt:.4f} ms")

x_bf256 = torch.randn(256, 2048, dtype=torch.bfloat16)
for _ in range(100):
    _ = x_bf256.float()
t0 = time.perf_counter()
for _ in range(N_iter):
    _ = x_bf256.float()
t_cvt256 = (time.perf_counter() - t0) / N_iter * 1000
print(f"  bf16→f32 (256×2048): {t_cvt256:.3f} ms")

# Test 6: Summary estimate per token
print(f"\n{'='*70}")
print(f"  Estimated savings per decode token (24 layers)")
print(f"{'='*70}")

# Current: 8 padded quants per layer × 24 layers = 192    (but down_proj K-tiled: 9 quants per layer)
# With shared: 5 quants per layer × 24 = 120  (q/k/v shared, gate/up shared)
# With real-only: each quant is much cheaper

quants_current = 9 * 24  # q,k,v,o,gate,up,down×2,+1 for contiguous slice
quants_shared = 5 * 24   # (qkv shared),o,(gate_up shared),down×2
# Current uses padded 256-row quant, optimized uses 1-row
savings_per_quant_ms = t_padded - t_opt  # ms saved per quant by using M=1

total_quant_current = quants_current * t_padded
total_quant_optimized = quants_shared * t_opt
quant_savings = total_quant_current - total_quant_optimized

print(f"  Current quants/token:    {quants_current} × {t_padded:.2f}ms = {total_quant_current:.0f}ms")
print(f"  Optimized quants/token:  {quants_shared} × {t_opt:.3f}ms = {total_quant_optimized:.1f}ms")
print(f"  Quantization savings:    {quant_savings:.0f}ms ({quant_savings/497*100:.0f}% of 497ms budget)")
print(f"")
print(f"  At 497ms/token → {497-quant_savings:.0f}ms → {1000/(497-quant_savings):.2f} tok/s")
