"""Benchmark: GPU SwiGLU vs NPU SwiGLU (including transfer overhead)."""
import torch, time, os
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
gpu = torch.device('cuda:0')

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Measure GPU<->CPU transfer for SwiGLU-sized tensors (8192 elements, bf16)
for n in [8192, 8192*24]:
    x = torch.randn(n, dtype=torch.bfloat16, device=gpu)
    for _ in range(10): y = x.cpu(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): y = x.cpu(); torch.cuda.synchronize()
    g2c = (time.perf_counter() - t0) / 100 * 1000
    y = torch.randn(n, dtype=torch.bfloat16)
    for _ in range(10): z = y.to(gpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): z = y.to(gpu); torch.cuda.synchronize()
    c2g = (time.perf_counter() - t0) / 100 * 1000
    print(f"N={n:>8d} ({n*2/1024:.0f} KB)  GPU->CPU: {g2c:.3f}ms  CPU->GPU: {c2g:.3f}ms  round-trip: {g2c+c2g:.3f}ms")

# GPU SwiGLU
gate_gpu = torch.randn(8192, dtype=torch.bfloat16, device=gpu)
up_gpu = torch.randn(8192, dtype=torch.bfloat16, device=gpu)
for _ in range(10): _ = torch.nn.functional.silu(gate_gpu) * up_gpu; torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100): _ = torch.nn.functional.silu(gate_gpu) * up_gpu
torch.cuda.synchronize()
gpu_swiglu = (time.perf_counter() - t0) / 100 * 1000
print(f"\nGPU SwiGLU (on-device): {gpu_swiglu:.3f}ms")
print(f"NPU SwiGLU (on-device): ~0.260ms")
print(f"Transfer round-trip:    ~{g2c+c2g:.3f}ms")
npu_total = g2c + c2g + 0.260
print(f"NPU total (with xfer):  ~{npu_total:.3f}ms")

if gpu_swiglu < npu_total:
    print(f"\n=> GPU wins for SwiGLU by {npu_total - gpu_swiglu:.3f}ms")
    print("   Recommendation: run EVERYTHING on GPU, skip NPU for activations")
else:
    print(f"\n=> NPU wins for SwiGLU by {gpu_swiglu - npu_total:.3f}ms")
    print("   Recommendation: hybrid GPU (matmul) + NPU (activations)")

print("\n--- Full model projection: GPU-only inference ---")
# Full 24-layer matmul + SwiGLU on GPU
shapes = [
    ('gate_up_proj', 1, 2048, 16384),
    ('down_proj',    1, 8192, 2048),
    ('qkv_proj',    1, 2048, 6144),
    ('o_proj',       1, 2048, 2048),
]
total_gpu = 0
for name, M, K, N in shapes:
    x_gpu = torch.randn(M, K, dtype=torch.bfloat16, device=gpu)
    w_gpu = torch.randn(K, N, dtype=torch.bfloat16, device=gpu)
    for _ in range(10): _ = x_gpu @ w_gpu; torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(24): _ = x_gpu @ w_gpu
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    total_gpu += dt
    print(f"  {name:15s}: {dt:.1f}ms (24 layers)")

# Add SwiGLU and attention on GPU
swiglu_total = gpu_swiglu * 24
print(f"  {'SwiGLU':15s}: {swiglu_total:.1f}ms (24 layers)")
total_gpu += swiglu_total

print(f"\n  Total matmul+activation: {total_gpu:.1f}ms/token")
print(f"  Projected throughput:    {1000/total_gpu:.0f} tokens/s")
