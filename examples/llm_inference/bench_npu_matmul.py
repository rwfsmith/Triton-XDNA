"""Benchmark NPU matmul for LLM-relevant shapes.
Compare cached dispatch time against CPU and GPU."""
import torch
import triton
import triton.language as tl
import sys
import os
import time

os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

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


def npu_matmul(M, N, K, block_m=256, block_n=256):
    """Single NPU matmul dispatch. Returns (result, time_ms)."""
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
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
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=K,
    )
    dt = (time.perf_counter() - t0) * 1000
    return c, dt


if __name__ == "__main__":
    benchmark.select_npu_backend()
    torch.set_num_threads(os.cpu_count())

    print("=" * 70)
    print("  NPU vs CPU vs GPU Matmul Benchmark")
    print("=" * 70)

    # The existing matmul kernel uses BLOCK_SIZE_K = K (full reduction in one shot)
    # and BLOCK_SIZE_M/N = 256. This means:
    # - Max practical M, N = 256 per tile (grid handles multiple tiles)
    # - K must fit in a single block (limit depends on NPU L1 memory)
    #
    # For LLM matmuls (e.g., 1x2048 x 2048x8192), we need:
    # - M=1 (decode), K=2048, N=8192
    # - But BLOCK_SIZE must be powers of 2 and >= actual dimension
    # So the kernel needs to handle M=1 with BLOCK_SIZE_M >= 1

    # Test with shapes the current kernel can handle
    test_shapes = [
        # (M, N, K, description)
        (256, 256, 256, "small square"),
        (1024, 1024, 256, "medium (multi-tile M,N)"),
        (1024, 1024, 1024, "large square"),
    ]

    print("\n--- Phase 1: Warm up (JIT compile each shape) ---")
    for M, N, K, desc in test_shapes:
        _, dt = npu_matmul(M, N, K)
        print(f"  {M}x{N}x{K} ({desc}): {dt:.0f}ms (includes compile)")

    print("\n--- Phase 2: Cached dispatch timing ---")
    print(f"{'Shape':>20s}  {'NPU':>10s}  {'CPU':>10s}  {'GPU':>10s}  {'NPU/CPU':>8s}  {'NPU/GPU':>8s}")
    print("-" * 70)

    has_gpu = torch.cuda.is_available()
    gpu = torch.device('cuda:0') if has_gpu else None

    for M, N, K, desc in test_shapes:
        # NPU (avg of 5 runs)
        times = []
        for _ in range(5):
            _, dt = npu_matmul(M, N, K)
            times.append(dt)
        npu_ms = sum(times) / len(times)

        # CPU
        a_cpu = torch.randn(M, K, dtype=torch.bfloat16)
        b_cpu = torch.randn(K, N, dtype=torch.bfloat16)
        for _ in range(3):
            _ = a_cpu @ b_cpu
        t0 = time.perf_counter()
        for _ in range(10):
            _ = a_cpu @ b_cpu
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000

        # GPU
        if has_gpu:
            a_gpu = a_cpu.to(gpu)
            b_gpu = b_cpu.to(gpu)
            for _ in range(10):
                _ = a_gpu @ b_gpu
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = a_gpu @ b_gpu
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) / 10 * 1000
        else:
            gpu_ms = float('nan')

        print(f"  {M}x{N}x{K}  {npu_ms:8.1f}ms  {cpu_ms:8.2f}ms  {gpu_ms:8.2f}ms  "
              f"{npu_ms/cpu_ms:6.1f}x  {npu_ms/gpu_ms:6.1f}x")

    # Calculate GFLOPS
    print("\n--- Phase 3: GFLOPS ---")
    for M, N, K, desc in test_shapes:
        flops = 2 * M * N * K
        times = []
        for _ in range(5):
            _, dt = npu_matmul(M, N, K)
            times.append(dt)
        npu_ms = sum(times) / len(times)
        gflops = flops / (npu_ms / 1000) / 1e9
        print(f"  {M}x{N}x{K}: {gflops:.1f} GFLOPS (NPU, {npu_ms:.1f}ms)")
