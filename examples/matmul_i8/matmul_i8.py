# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# INT8 x INT8 -> INT32 matmul on NPU
# Uses native AIE2P INT8 MAC operations (8x8x8 MMUL shape)
# Expected ~2x memory bandwidth advantage over BF16

import torch
import triton
import triton.language as tl
import sys, os
import time

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def bare_matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
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


def test_matmul_i8(M, N, K, verbose=True):
    """Test INT8 matmul: A[M,K] x B[K,N] -> C[M,N] in INT32"""
    device = "cpu"
    
    # Create random INT8 tensors (range -128 to 127)
    a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
    c = torch.zeros((M, N), device=device, dtype=torch.int32)
    
    # Reference: compute in INT32
    c_ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    
    # Triton kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    # Set the INT8 transform script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(script_dir, "transform_aie2p.mlir")
    
    BLOCK_M = min(M, 256)
    BLOCK_N = min(N, 256)
    compiled_kernel = bare_matmul[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=K,
    )
    
    # Save the IR for debugging
    with open("tt_i8.shared.mlir", "w") as f:
        f.write(str(compiled_kernel.asm["ttsharedir"]))
    
    # Check correctness
    match = torch.equal(c, c_ref)
    max_diff = (c - c_ref).abs().max().item()
    
    if verbose:
        status = "PASS" if match else "FAIL"
        print(f"  INT8 matmul {M}x{N}x{K}: {status} (max_diff={max_diff})")
        if not match:
            print(f"    c[0,:4]     = {c[0,:4].tolist()}")
            print(f"    c_ref[0,:4] = {c_ref[0,:4].tolist()}")
    
    return match


def bench_matmul_i8(M, N, K, warmup=3, iters=10):
    """Benchmark INT8 matmul with warmup"""
    device = "cpu"
    a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
    c = torch.zeros((M, N), device=device, dtype=torch.int32)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(script_dir, "transform_aie2p.mlir")
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    BLOCK_M = min(M, 256)
    BLOCK_N = min(N, 256)
    
    # Warmup
    for _ in range(warmup):
        bare_matmul[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=K,
        )
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        bare_matmul[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=K,
        )
    elapsed = (time.perf_counter() - start) / iters
    
    gflops = 2 * M * N * K / elapsed / 1e9
    print(f"  INT8 {M}x{N}x{K}: {elapsed*1000:.2f} ms, {gflops:.0f} GFLOPS(INT8)")
    return elapsed


if __name__ == "__main__":
    benchmark.select_npu_backend()
    
    print("=" * 60)
    print("INT8 x INT8 -> INT32 Matmul on NPU (AIE2P)")
    print("=" * 60)
    
    # Test correctness first
    print("\n--- Correctness Tests ---")
    sizes = [
        (256, 256, 256),
        (256, 256, 512),
        (256, 256, 1024),
        (256, 256, 2048),
        (256, 256, 4096),
    ]
    
    all_pass = True
    for M, N, K in sizes:
        if not test_matmul_i8(M, N, K):
            all_pass = False
    
    if all_pass:
        print("\nAll correctness tests PASSED!")
        
        # Benchmark
        print("\n--- Benchmarks ---")
        for M, N, K in sizes:
            bench_matmul_i8(M, N, K)
    else:
        print("\nSome tests FAILED - skipping benchmarks")
