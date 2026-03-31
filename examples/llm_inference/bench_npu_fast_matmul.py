"""Test NPU matmul with FastNPUDispatch to measure true NPU compute time
(bypassing Python overhead), and test padded M=1 GEMV."""
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


if __name__ == "__main__":
    benchmark.select_npu_backend()

    print("=" * 70)
    print("  NPU Matmul: FastNPUDispatch + Padded GEMV Test")
    print("=" * 70)

    # Phase 1: First dispatch (slow path — JIT compile)
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    print("\n--- Slow path (JIT compile) ---")
    t0 = time.perf_counter()
    bare_matmul[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
    )
    print(f"  256x256x256 slow path: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Phase 2: Capture the module for FastNPUDispatch
    from triton.backends.amd_triton_npu.driver import _last_dispatched_module
    print(f"\n--- Captured module: {_last_dispatched_module is not None} ---")

    if _last_dispatched_module is not None:
        mod = _last_dispatched_module
        # The matmul kernel has different args than elementwise:
        # gridX=1, gridY=1, gridZ=1, then tensor args, then constexpr vals
        # For 256x256x256 with BLOCK_SIZE=256: grid=(1,1,1)

        # Fast dispatch: time just the module.launch() call
        print("\n--- Fast dispatch (bypass Triton JIT) ---")
        times = []
        for i in range(20):
            a = torch.randn(M, K, dtype=torch.bfloat16)
            b = torch.randn(K, N, dtype=torch.bfloat16)
            c = torch.empty(M, N, dtype=torch.float32)
            t0 = time.perf_counter()
            mod.launch(
                1, 1, 1,       # gridX, gridY, gridZ
                None, None,    # kernel_metadata, launch_metadata
                None, None,    # launch_enter/exit hooks
                a, b, c,      # tensor args
                M, N, K,      # constexpr: M, N, K
                a.stride(0), a.stride(1),  # strides
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                256, 256, K,  # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if i < 3:
                # Verify correctness
                c_ref = torch.matmul(a, b).to(torch.float32)
                err = (c - c_ref).abs().max().item()
                print(f"  Run {i}: {dt:.2f}ms, max_err={err:.4f}")

        avg = sum(times[2:]) / len(times[2:])  # skip first 2
        print(f"\n  Average fast dispatch: {avg:.2f}ms (18 runs)")
        flops = 2 * M * N * K
        gflops = flops / (avg / 1000) / 1e9
        print(f"  {gflops:.1f} GFLOPS at {avg:.2f}ms")

    # Phase 3: Test padded GEMV (M=1 padded to M=256)
    print("\n--- Padded GEMV: M=1 -> padded to M=256, K=256, N=256 ---")
    actual_M = 1
    padded_M = 256
    a_real = torch.randn(actual_M, K, dtype=torch.bfloat16)
    a_padded = torch.zeros(padded_M, K, dtype=torch.bfloat16)
    a_padded[:actual_M] = a_real
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(padded_M, N, dtype=torch.float32)

    # Reference
    c_ref = torch.matmul(a_real, b).to(torch.float32)

    if _last_dispatched_module is not None:
        mod.launch(
            1, 1, 1,
            None, None, None, None,
            a_padded, b, c,
            padded_M, N, K,
            a_padded.stride(0), a_padded.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            256, 256, K,
        )
        # Extract first row only (the actual result)
        c_actual = c[:actual_M]
        err = (c_actual - c_ref).abs().max().item()
        print(f"  Padded GEMV correctness: max_err={err:.4f} ({'PASS' if err < 10 else 'FAIL'})")
