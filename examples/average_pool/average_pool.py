# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Average pooling kernel for AMD XDNA NPU
# Computes: y[i] = mean(x[i, :]) per row
#
# Uses BLOCK_M=2 (2D tiling) so the Linalg IR has a row dimension that
# can be tiled at [1], avoiding the scalar chain issue where tl.sum
# produces a scalar that can't be fused into a forall.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def avg_pool_kernel(
    X,
    Y,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Load BLOCK_M rows at once (2D block)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(X + offsets)

    # Sum per row in bf16 (AIE2P only supports bf16 vector add)
    row_sum = tl.sum(x, axis=1)  # [BLOCK_M], bf16

    # Divide by N in f32 (divf is f32-only on AIE2P)
    mean = row_sum.to(tl.float32) / N
    y = mean.to(x.dtype)  # [BLOCK_M], bf16

    tl.store(Y + rows, y)


def bench_avg_pool(M, N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    BLOCK_M = 4  # Process 4 rows per invocation (tiled at [2] for DMA alignment)
    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.empty(M, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        y_ref = x.float().mean(dim=-1).to(dtype)
    if provider == "triton" or provider == "test":
        grid = (M // BLOCK_M,)
        compiled_kernel = avg_pool_kernel[grid](
            x,
            y,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=N,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=5e-1, rtol=1e-1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    # N >= 256 required for proper 2D DMA patterns in aircc runtime sequence
    for M in [32, 64]:
        for N in [256]:
            bench_avg_pool(M, N, "test")
