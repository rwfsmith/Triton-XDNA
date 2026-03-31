"""Test NPU matmul and benchmark for LLM-relevant shapes."""
import torch
import triton
import triton.language as tl
import sys
import os
import time

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


def test_matmul(M, N, K, block_m=256, block_n=256):
    """Run a single matmul test and return (pass, time_ms)."""
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
    c_ref = torch.matmul(a, b).to(torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    t0 = time.perf_counter()
    bare_matmul[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=K,
    )
    dt = (time.perf_counter() - t0) * 1000

    max_err = (c - c_ref).abs().max().item()
    passed = max_err < 10
    return passed, dt, max_err


if __name__ == "__main__":
    benchmark.select_npu_backend()

    print("=" * 60)
    print("  NPU Matmul Test (AIE2P, 8x4 core array)")
    print("=" * 60)

    # Phase 1: Basic correctness test
    print("\n--- Phase 1: Correctness (256x256x256) ---")
    passed, dt, err = test_matmul(256, 256, 256)
    print(f"  256x256x256: {'PASS' if passed else 'FAIL'} ({dt:.0f}ms, max_err={err:.4f})")

    if not passed:
        print("  Basic matmul failed, stopping.")
        sys.exit(1)

    # Phase 2: Re-run to see cached time
    print("\n--- Phase 2: Cached dispatch ---")
    passed, dt, err = test_matmul(256, 256, 256)
    print(f"  256x256x256 (cached): {'PASS' if passed else 'FAIL'} ({dt:.1f}ms)")

    # Phase 3: Larger sizes
    print("\n--- Phase 3: Larger sizes ---")
    for M, N, K in [(256, 256, 256), (1024, 1024, 256), (1024, 1024, 1024)]:
        passed, dt, err = test_matmul(M, N, K)
        status = "PASS" if passed else "FAIL"
        print(f"  {M}x{N}x{K}: {status} ({dt:.0f}ms, max_err={err:.4f})")
