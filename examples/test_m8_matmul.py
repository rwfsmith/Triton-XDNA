#!/usr/bin/env python3
"""Test if M=8 INT8 matmul compiles and runs correctly on NPU.
If so, we can use M=8 kernels for decode (1 token) instead of M=256,
reducing NPU compute by 32x."""
import torch, os, sys, time

os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import benchmark
benchmark.select_npu_backend()

import triton, triton.language as tl
import importlib

torch.set_num_threads(os.cpu_count())

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

os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
    os.path.dirname(__file__), "matmul_i8", "transform_aie2p.mlir"
)

npu_driver = importlib.import_module("triton.backends.amd_triton_npu.driver")

# Test M=8 for each decode shape
test_shapes = [
    (128, 2048, 2048),   # q/k/v/o_proj
    (128, 8192, 2048),   # gate/up_proj
    (128, 2048, 4096),   # down_proj tiles
]

for M, N, K in test_shapes:
    gX = 1  # BLOCK_SIZE_M=M, so one tile row
    gY = N // 256
    print(f"Testing M={M}, N={N}, K={K} (grid={gX}x{gY})...", end=" ", flush=True)

    try:
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8)
        c = torch.zeros(M, N, dtype=torch.int32)

        bare_matmul[(gX, gY)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=M, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
        )

        c_ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        diff = (c - c_ref).abs().max().item()
        status = "PASS" if diff == 0 else "FAIL"
        print(f"{status} (diff={diff})")

        if diff == 0:
            # Benchmark M=8 vs M=256
            mod = npu_driver._last_dispatched_module

            # M=8 benchmark
            N_ITER = 50
            for _ in range(5):  # warmup
                mod.launch(gX, gY, 1, None, None, None, None,
                           a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1), M, 256, K)

            t0 = time.perf_counter()
            for _ in range(N_ITER):
                mod.launch(gX, gY, 1, None, None, None, None,
                           a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1), M, 256, K)
            t_m8 = (time.perf_counter() - t0) / N_ITER * 1000
            print(f"  M=8 dispatch: {t_m8:.3f} ms")

    except Exception as e:
        print(f"FAILED: {e}")
