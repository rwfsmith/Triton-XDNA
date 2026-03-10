# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def swiglu_kernel(
    GATE,
    UP,
    OUT,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    gate = tl.load(GATE + offsets[:])
    up = tl.load(UP + offsets[:])
    # SwiGLU(gate, up) = SiLU(gate) * up = gate * sigmoid(gate) * up
    # sigmoid requires f32 input
    gate_f32 = gate.to(tl.float32)
    sig = tl.sigmoid(gate_f32)
    silu_gate = (gate_f32 * sig).to(gate.dtype)
    out = silu_gate * up
    tl.store(OUT + offsets[:], out)


def bench_swiglu(N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    gate = torch.randn(N, device=device, dtype=dtype)
    up = torch.randn(N, device=device, dtype=dtype)
    out = torch.empty(N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        out_ref = torch.nn.functional.silu(gate) * up
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = swiglu_kernel[grid](
            gate,
            up,
            out,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(out, out_ref, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_swiglu(N, "test")
