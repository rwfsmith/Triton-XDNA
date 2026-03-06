# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# RMS Normalization kernel for AMD XDNA NPU
# Computes: y = x * rsqrt(mean(x^2) + eps) per row
# Reference: mlir-air/programming_examples/rms_norm/

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

EPS = 1e-5


@triton.jit
def rms_norm_kernel(
    X,
    Y,
    M,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load the row
    x = tl.load(X + row_idx * N + offsets)

    # Compute mean of squares
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N

    # Compute rsqrt(mean_sq + eps)
    # Use rsqrt directly -- AIE has native rsqrt intrinsic
    rstd = tl.math.rsqrt(mean_sq + eps)

    # Normalize
    y = x * rstd
    tl.store(Y + row_idx * N + offsets, y)


def bench_rms_norm(M, N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.empty(M, N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        # Manual RMS norm reference
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + EPS)
        y_ref = (x_f32 * rstd).to(dtype)
    if provider == "triton" or provider == "test":
        grid = (M,)
        compiled_kernel = rms_norm_kernel[grid](
            x,
            y,
            M,
            N,
            EPS,
            BLOCK_SIZE=N,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=5e-1, rtol=1e-1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    # Test with various sizes
    for M in [32, 64]:
        for N in [64, 128]:
            bench_rms_norm(M, N, "test")
