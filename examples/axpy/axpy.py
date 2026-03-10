# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def axpy_kernel(
    X,
    Y,
    OUT,
    alpha: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    y = tl.load(Y + offsets[:])
    out = alpha * x + y
    tl.store(OUT + offsets[:], out)


def bench_axpy(N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    alpha = 2.0
    x = torch.randn(N, device=device, dtype=dtype)
    y = torch.randn(N, device=device, dtype=dtype)
    out = torch.empty(N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        out_ref = alpha * x + y
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = axpy_kernel[grid](
            x,
            y,
            out,
            alpha,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_axpy(N, "test")
