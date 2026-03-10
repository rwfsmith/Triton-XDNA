# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def silu_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # SiLU(x) = x * sigmoid(x)
    # sigmoid requires f32 input
    x_f32 = x.to(tl.float32)
    sig = tl.sigmoid(x_f32)
    y = (x_f32 * sig).to(x.dtype)
    tl.store(Y + offsets[:], y)


def bench_silu(N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.empty(N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        y_ref = torch.nn.functional.silu(x)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = silu_kernel[grid](
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
        bench_silu(N, "test")
