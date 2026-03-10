# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

ALPHA = 0.01  # Standard leaky relu negative slope


@triton.jit
def leaky_relu_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # Leaky ReLU: y = x if x >= 0, else alpha * x
    # tl.where produces arith.cmpf + arith.select in Linalg IR.
    # AIE2 supports vector select (vselect intrinsic).
    y = tl.where(x >= 0, x, 0.01 * x)
    tl.store(Y + offsets[:], y)


def bench_leaky_relu(N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.empty(N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        y_ref = torch.nn.functional.leaky_relu(x, negative_slope=ALPHA)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = leaky_relu_kernel[grid](
            x,
            y,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_leaky_relu(N, "test")
