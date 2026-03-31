"""Benchmark all SmolLM2-1.7B matmul shapes on NPU with FastNPUDispatch."""
import torch
import triton
import triton.language as tl
import sys
import os
import time
import importlib

os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
    os.path.dirname(__file__), "..", "matmul", "transform_aie2p.mlir"
)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import benchmark

benchmark.select_npu_backend()
npu_driver = importlib.import_module("triton.backends.amd_triton_npu.driver")


@triton.jit
def bare_matmul(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_block = tl.dot(a_block, b_block)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)


# SmolLM2-1.7B shapes (decode: M=1 padded to 256)
# hidden=2048, intermediate=8192
shapes = [
    ("q_proj",    256, 2048, 2048),
    ("k_proj",    256, 256,  2048),
    ("v_proj",    256, 256,  2048),
    ("o_proj",    256, 2048, 2048),
    ("gate_proj", 256, 8192, 2048),
    ("up_proj",   256, 8192, 2048),
    ("down_proj", 256, 2048, 4096),  # K-tiled: 2x4096 instead of 1x8192
]

results = {}
for name, M, N, K in shapes:
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
    gX = triton.cdiv(M, 256)
    gY = triton.cdiv(N, 256)
    print(f"Compiling {name}: {M}x{N}x{K} (grid={gX}x{gY})...")
    bare_matmul[(gX, gY)](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
    )
    mod = npu_driver._last_dispatched_module
    if mod is None:
        print("  Not captured!")
        continue

    # Correctness check
    c_ref = torch.matmul(a, b).to(torch.float32)
    err = (c - c_ref).abs().max().item()

    # Benchmark with weight-resident: use SAME weight tensor across iterations
    weight_b = torch.randn(K, N, dtype=torch.bfloat16)  # fixed weight
    times = []
    for i in range(20):
        a = torch.randn(M, K, dtype=torch.bfloat16)
        c = torch.empty(M, N, dtype=torch.float32)
        t0 = time.perf_counter()
        mod.launch(
            gX, gY, 1, None, None, None, None,
            a, weight_b, c, M, N, K,
            a.stride(0), a.stride(1), weight_b.stride(0), weight_b.stride(1),
            c.stride(0), c.stride(1), 256, 256, K,
        )
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
    times.sort()
    med = times[len(times) // 2]
    flops = 2 * M * N * K / (med / 1000) / 1e9
    ok = "PASS" if err < 30 else "FAIL"
    results[name] = med
    print(f"  {med:.3f}ms, {flops:.0f} GFLOPS, err={err:.2f} {ok}")

# Total per layer
print()
print("=" * 50)
print("Per-Layer Totals (SmolLM2-1.7B decode)")
print("=" * 50)
layer_time = sum(results.values())
# down_proj needs 2x (K-tiled from 8192 -> 2 x 4096)
dp = results.get("down_proj", 0)
layer_time += dp  # add the second K-tile
for name, ms in results.items():
    extra = ""
    if name == "down_proj":
        extra = " (x2 for K-tile)"
        ms *= 2
    print(f"  {name:12s}: {ms:.3f}ms{extra}")
print(f"  {'TOTAL':12s}: {layer_time:.3f}ms")
print(f"  24 layers      : {24 * layer_time:.1f}ms")
print(f"  Projected tok/s: {1000 / (24 * layer_time):.1f}")
