# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def gelu_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # GELU(x) ≈ x * sigmoid(1.702 * x)
    # Uses sigmoid-based "fast" GELU approximation.
    # sigmoid requires f32 input.
    x_f32 = x.to(tl.float32)
    y = (x_f32 * tl.sigmoid(1.702 * x_f32)).to(x.dtype)
    tl.store(Y + offsets[:], y)


def bench_gelu(N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.empty(N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        # Reference uses sigmoid approximation: x * sigmoid(1.702 * x)
        y_ref = x * torch.sigmoid(1.702 * x.float()).to(dtype)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = gelu_kernel[grid](
            x,
            y,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_gelu(N, "test")
