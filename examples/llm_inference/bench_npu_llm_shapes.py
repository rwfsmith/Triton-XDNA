"""Benchmark NPU matmul with FastNPUDispatch at LLM-relevant sizes."""
import torch
import triton
import triton.language as tl
import sys
import os
import time

os.environ["TRITON_BACKENDS_IN_TREE"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import benchmark

os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
    os.path.dirname(__file__), "..", "matmul", "transform_aie2p.mlir"
)


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


def compile_and_capture(M, N, K, block_m=256, block_n=256):
    """Compile kernel for shape, return captured module."""
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    bare_matmul[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=K,
    )
    from triton.backends.amd_triton_npu.driver import _last_dispatched_module
    return _last_dispatched_module


def fast_matmul(mod, a, b, c, M, N, K, gridX, gridY, block_m=256, block_n=256):
    """Direct fast dispatch."""
    mod.launch(
        gridX, gridY, 1,
        None, None, None, None,
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        block_m, block_n, K,
    )


if __name__ == "__main__":
    benchmark.select_npu_backend()

    print("=" * 70)
    print("  NPU FastDispatch Matmul: LLM-Relevant Shapes")
    print("=" * 70)

    # The matmul transform IR tiles at BLOCK_SIZE 256x256.
    # For shapes > 256 in M or N, we need multiple tiles (grid > 1).
    # K must fit in BLOCK_SIZE_K (which equals K in the current kernel).
    #
    # SmolLM2-1.7B shapes (decode, M=1, padded to M=256):
    llm_shapes = [
        # (M_pad, N, K, gridX, gridY, description)
        (256, 256, 256, 1, 1, "small (baseline)"),
        (256, 1024, 256, 1, 4, "MLP-like 256x256->1024"),
        (256, 1024, 1024, 1, 4, "MLP-like 256x1024->1024"),
    ]

    # Compile each shape
    print("\n--- Compiling kernels ---")
    modules = {}
    for M, N, K, gx, gy, desc in llm_shapes:
        print(f"  Compiling {M}x{N}x{K} (grid={gx}x{gy})...", end=" ", flush=True)
        t0 = time.perf_counter()
        mod = compile_and_capture(M, N, K)
        dt = (time.perf_counter() - t0) * 1000
        modules[(M, N, K)] = mod
        print(f"{dt:.0f}ms")

    # Benchmark each shape
    print(f"\n--- Benchmark (FastNPUDispatch) ---")
    print(f"{'Shape':>25s}  {'Time':>8s}  {'GFLOPS':>8s}  {'Correct':>8s}")
    print("-" * 60)

    for M, N, K, gx, gy, desc in llm_shapes:
        mod = modules[(M, N, K)]
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)
        c = torch.empty(M, N, dtype=torch.float32)
        c_ref = torch.matmul(a, b).to(torch.float32)

        # Warmup
        for _ in range(3):
            fast_matmul(mod, a, b, c, M, N, K, gx, gy)

        # Benchmark
        times = []
        for _ in range(20):
            a_new = torch.randn(M, K, dtype=torch.bfloat16)
            c_new = torch.empty(M, N, dtype=torch.float32)
            t0 = time.perf_counter()
            fast_matmul(mod, a_new, b, c_new, M, N, K, gx, gy)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        flops = 2 * M * N * K
        gflops = flops / (avg / 1000) / 1e9

        # Correctness
        fast_matmul(mod, a, b, c, M, N, K, gx, gy)
        err = (c - c_ref).abs().max().item()
        status = "PASS" if err < 10 else "FAIL"

        print(f"  {M}x{N}x{K} ({desc:20s})  {avg:6.2f}ms  {gflops:6.1f}  {status} (err={err:.2f})")

    # Projection for full LLM
    print("\n--- NPU-Only LLM Projection ---")
    # All matmuls padded to M=256, tiled in N
    # gate_up: 256x2048 -> 256x16384 (needs K=2048!!)
    # down: 256x8192 -> 256x2048
    # qkv: 256x2048 -> 256x6144
    # o: 256x2048 -> 256x2048
    print("  NOTE: Current kernel requires K <= BLOCK_SIZE_K (single reduction block)")
    print("  For K=2048, we'd need BLOCK_SIZE_K=2048 (or tiled K reduction)")
    print("  This requires a different transform IR or K-tiling in the Triton kernel")
    print()
    print("  With K=256 (current max), we'd need to tile K externally:")
    print("    gate_up: 256x256->16384, 8 K-tiles = 8 dispatches")
    print("    down:    256x256->2048,  32 K-tiles = 32 dispatches")
    print("  Total: too many dispatches, need K-tiled kernel")
