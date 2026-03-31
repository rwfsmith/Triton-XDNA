# INT8 vs BF16 benchmark using FastNPUDispatch
# Measures raw NPU compute time without Python/Triton overhead

import torch
import triton
import triton.language as tl
import sys, os, time
import importlib

sys.path.append(os.path.abspath(".."))
import benchmark

@triton.jit
def bare_matmul(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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


def compile_and_get_module(M, N, K, dtype_in, dtype_out, transform_script):
    """Compile kernel and return the cached module for fast dispatch"""
    device = "cpu"
    if dtype_in == torch.int8:
        a = torch.randint(-128, 127, (M, K), device=device, dtype=dtype_in)
        b = torch.randint(-128, 127, (K, N), device=device, dtype=dtype_in)
    else:
        a = torch.randn((M, K), device=device, dtype=dtype_in)
        b = torch.randn((K, N), device=device, dtype=dtype_in)
    c = torch.zeros((M, N), device=device, dtype=dtype_out)
    
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script
    
    BLOCK_M = min(M, 256)
    BLOCK_N = min(N, 256)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
    
    bare_matmul[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=K,
    )
    
    # Get the compiled module
    driver_mod = importlib.import_module('triton.backends.amd_triton_npu.driver')
    module = driver_mod._last_dispatched_module
    return module, a, b, c


def fast_dispatch(module, a, b, c, warmup=5, iters=20):
    """Benchmark using direct module.launch() (bypassing Triton JIT overhead)"""
    M, K = a.shape
    _, N = b.shape
    BLOCK_M = min(M, 256)
    BLOCK_N = min(N, 256)
    gX = (M + BLOCK_M - 1) // BLOCK_M
    gY = (N + BLOCK_N - 1) // BLOCK_N
    
    for _ in range(warmup):
        module.launch(
            gX, gY, 1, None, None, None, None,
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, K,
        )
    
    start = time.perf_counter()
    for _ in range(iters):
        module.launch(
            gX, gY, 1, None, None, None, None,
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, K,
        )
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


if __name__ == "__main__":
    benchmark.select_npu_backend()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bf16_transform = os.path.join(os.path.dirname(script_dir), "matmul", "transform_aie2p.mlir")
    i8_transform = os.path.join(script_dir, "transform_aie2p.mlir")
    
    print("=" * 70)
    print("INT8 vs BF16 Matmul - FastNPUDispatch Benchmark")
    print("=" * 70)
    
    shapes = [
        (256, 256, 256),
        (256, 256, 512),
        (256, 256, 1024),
        (256, 256, 2048),
        (256, 256, 4096),
    ]
    
    print(f"\n{'Shape':>20s}  {'BF16 (ms)':>10s}  {'INT8 (ms)':>10s}  {'Speedup':>8s}  {'BF16 GFLOPS':>12s}  {'INT8 GOPS':>10s}")
    print("-" * 76)
    
    for M, N, K in shapes:
        flops = 2 * M * N * K
        
        # BF16 benchmark
        module_bf16, a_bf16, b_bf16, c_bf16 = compile_and_get_module(
            M, N, K, torch.bfloat16, torch.float32, bf16_transform)
        t_bf16 = fast_dispatch(module_bf16, a_bf16, b_bf16, c_bf16)
        gflops_bf16 = flops / t_bf16 / 1e9
        
        # INT8 benchmark
        module_i8, a_i8, b_i8, c_i8 = compile_and_get_module(
            M, N, K, torch.int8, torch.int32, i8_transform)
        t_i8 = fast_dispatch(module_i8, a_i8, b_i8, c_i8)
        gops_i8 = flops / t_i8 / 1e9
        
        speedup = t_bf16 / t_i8
        print(f"  {M}x{N}x{K:>4d}       {t_bf16*1000:>8.2f}    {t_i8*1000:>8.2f}    {speedup:>6.2f}x    {gflops_bf16:>10.0f}      {gops_i8:>8.0f}")
    
    print("\nNote: INT8 GOPS = INT8 multiply-accumulate operations per second")
    print("      BF16 GFLOPS = BF16 floating-point operations per second")
    print("      INT8 elements are half the size of BF16, so bandwidth advantage = 2x")
