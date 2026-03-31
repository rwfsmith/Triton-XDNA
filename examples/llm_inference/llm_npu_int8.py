#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# INT8 NPU LLM Inference with AMD Ryzen AI
# ==========================================
# Loads SmolLM2-1.7B, quantizes weights to INT8 (per-channel symmetric),
# dynamically quantizes activations, and runs INT8×INT8→INT32 matmul on NPU.
#
# Quantization:
#   Weights: Per-output-channel symmetric: scale_w[j] = max(|W[:, j]|) / 127
#   Activations: Per-tensor symmetric: scale_x = max(|x|) / 127
#   Output: C_int32 → float: result = C_int32 * (scale_x * scale_w)
#
# Compared to BF16 version:
#   - Weight memory: 1 byte/elem vs 2 bytes/elem → 2× less bandwidth
#   - NPU INT8 MAC: native 8×8×8 MMUL on AIE2P
#   - Same dispatch overhead, but faster compute for bandwidth-bound shapes

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import os
import sys
import types
import importlib
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
import benchmark

torch.set_num_threads(os.cpu_count())


# =============================================================================
# Triton NPU Matmul Kernel (same kernel, different dtypes)
# =============================================================================

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


# =============================================================================
# Triton NPU SwiGLU Kernel (stays BF16)
# =============================================================================

@triton.jit
def swiglu_kernel(
    GATE, UP, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gate = tl.load(GATE + offsets[:])
    up = tl.load(UP + offsets[:])
    gate_f32 = gate.to(tl.float32)
    silu_gate = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    tl.store(OUT + offsets[:], silu_gate * up)


# =============================================================================
# INT8 Quantization Helpers
# =============================================================================

def quantize_weight_per_channel(weight_fp):
    """Quantize weight to INT8 with per-output-channel symmetric scaling.

    Args:
        weight_fp: [K, N] float/bf16 weight tensor (already transposed)

    Returns:
        weight_int8: [K, N] int8 tensor
        scale_w: [N] float32 scale (one per output channel)
    """
    # Per-column max absolute value
    w_float = weight_fp.float()
    col_max = w_float.abs().amax(dim=0)  # [N]
    scale_w = col_max / 127.0
    scale_w = scale_w.clamp(min=1e-10)  # avoid division by zero

    # Quantize
    weight_int8 = (w_float / scale_w[None, :]).round().clamp(-128, 127).to(torch.int8)
    return weight_int8, scale_w


def quantize_activation_per_row(x):
    """Quantize activation to INT8 with per-row symmetric scaling.

    Per-row scaling preserves much more precision than per-tensor,
    since different tokens/rows have very different magnitudes.

    Args:
        x: [M, K] bf16/float activation tensor

    Returns:
        x_int8: [M, K] int8 tensor
        scale_x: [M] float32 row scales
    """
    x_float = x.float()
    row_max = x_float.abs().amax(dim=1, keepdim=True)  # [M, 1]
    scale_x = (row_max / 127.0).clamp(min=1e-10)
    x_int8 = (x_float / scale_x).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale_x.squeeze(1)  # x_int8: [M,K], scale_x: [M]


# =============================================================================
# NPU INT8 Matmul Dispatcher
# =============================================================================

class NPUInt8Engine:
    """NPU dispatch engine using INT8×INT8→INT32 matmul with dequantization."""

    M_BLOCK = 128  # Block size for matmul M dimension (128 = 2x less NPU compute vs 256)

    def __init__(self):
        self.npu_driver = importlib.import_module(
            "triton.backends.amd_triton_npu.driver"
        )
        self._modules = {}           # shape_key -> (mod, gX, gY, M, N, K)
        self._modules_bf16 = {}      # for SwiGLU (remains BF16)
        self._layers = {}            # layer_name -> layer info
        self._swiglu_mod = None
        self._swiglu_n = None
        self.dispatch_count = 0
        self._compile_count = 0
        self._x_bufs = {}    # K -> pre-allocated [M_BLOCK, K] int8 pad buffer
        self._c_bufs = {}    # N -> pre-allocated [M_BLOCK, N] int32 output buffer
        self._slot_counters = {}     # shape_key -> next available slot ID
        self._c_bufs = {}    # N -> pre-allocated [M_BLOCK, N] int32 output buffer

    def _set_i8_transform(self):
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
            SCRIPT_DIR, "..", "matmul_i8", "transform_aie2p.mlir"
        )

    def _set_swiglu_transform(self):
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
            SCRIPT_DIR, "transform_binary_aie2p.mlir"
        )

    def _compile_i8_shape(self, M, N, K):
        """Compile INT8 matmul kernel and capture module."""
        shape_key = (M, N, K)
        if shape_key in self._modules:
            return

        self._set_i8_transform()
        MB = self.M_BLOCK
        gX = triton.cdiv(M, MB)
        gY = triton.cdiv(N, 256)
        print(f"    Compiling INT8 matmul {M}x{N}x{K} (grid={gX}x{gY})...", end="", flush=True)

        a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8)
        c = torch.zeros(M, N, dtype=torch.int32)
        bare_matmul[(gX, gY)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=MB, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
        )
        mod = self.npu_driver._last_dispatched_module
        if mod is None:
            raise RuntimeError(f"Failed to capture INT8 module for {shape_key}")

        # Verify correctness
        c_ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        match = torch.equal(c, c_ref)
        status = "PASS" if match else "FAIL"
        max_diff = (c - c_ref).abs().max().item()
        print(f" {status} (diff={max_diff})")

        self._modules[shape_key] = (mod, gX, gY, M, N, K)
        self._compile_count += 1

        # Pre-allocate reusable buffers for this shape
        if K not in self._x_bufs:
            self._x_bufs[K] = torch.zeros(MB, K, dtype=torch.int8)
        if N not in self._c_bufs:
            self._c_bufs[N] = torch.empty(MB, N, dtype=torch.int32)

    def compile_swiglu(self, n_elements, block_size=1024):
        """Compile SwiGLU kernel (stays BF16)."""
        if self._swiglu_mod is not None:
            return
        self._set_swiglu_transform()
        print(f"    Compiling SwiGLU N={n_elements}...", end="", flush=True)

        gate = torch.randn(n_elements, dtype=torch.bfloat16)
        up = torch.randn(n_elements, dtype=torch.bfloat16)
        out = torch.empty(n_elements, dtype=torch.bfloat16)
        grid = (triton.cdiv(n_elements, block_size),)
        swiglu_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=block_size)

        mod = self.npu_driver._last_dispatched_module
        if mod is None:
            raise RuntimeError("Failed to capture SwiGLU module")

        self._swiglu_mod = mod
        self._swiglu_n = n_elements
        self._swiglu_grid = grid[0]
        self._swiglu_bs = block_size
        print(" OK")

    def _next_slot(self, shape_key):
        """Get next available weight slot ID for a given shape."""
        slot = self._slot_counters.get(shape_key, 0)
        self._slot_counters[shape_key] = slot + 1
        return slot

    def prepare_layer(self, name, linear, max_K=4096):
        """Quantize weights to INT8 and register for NPU dispatch."""
        K_full = linear.in_features
        N = linear.out_features
        M = self.M_BLOCK

        # Transpose: W[out, in] -> W_t[in, out] = [K, N]
        weight_t = linear.weight.data.T.contiguous().float()

        if K_full > max_K:
            n_tiles = (K_full + max_K - 1) // max_K
            K_tile = K_full // n_tiles
            assert K_tile * n_tiles == K_full

            weight_tiles_i8 = []
            scale_tiles = []
            tile_slots = []
            for i in range(n_tiles):
                w_slice = weight_t[i * K_tile : (i + 1) * K_tile, :]
                w_i8, s_w = quantize_weight_per_channel(w_slice)
                weight_tiles_i8.append(w_i8.contiguous())
                scale_tiles.append(s_w)
                tile_slots.append(self._next_slot((M, N, K_tile)))

            self._compile_i8_shape(M, N, K_tile)
            self._layers[name] = {
                "weight_tiles": weight_tiles_i8,
                "scale_w_tiles": scale_tiles,
                "shape_key": (M, N, K_tile),
                "k_tiles": n_tiles,
                "K_full": K_full,
                "K_tile": K_tile,
                "N": N,
                "slots": tile_slots,
            }
        else:
            w_i8, s_w = quantize_weight_per_channel(weight_t)
            self._compile_i8_shape(M, N, K_full)
            slot = self._next_slot((M, N, K_full))
            self._layers[name] = {
                "weight_tiles": [w_i8.contiguous()],
                "scale_w_tiles": [s_w],
                "shape_key": (M, N, K_full),
                "k_tiles": 1,
                "K_full": K_full,
                "K_tile": K_full,
                "N": N,
                "slots": [slot],
            }

    def matmul(self, name, x):
        """INT8 matmul on NPU with optimized quantization.

        For non-K-tiled shapes, quantizes and dispatches via matmul_preq.
        For K-tiled shapes (down_proj), quantizes each tile slice separately
        using pre-allocated buffers to avoid memory allocation overhead.
        """
        info = self._layers[name]
        M_actual = x.shape[0]

        if info["k_tiles"] == 1:
            x_i8, scale_x = self.quantize_and_pad(x, info["K_full"])
            return self.matmul_preq(name, x_i8, scale_x, M_actual)
        else:
            # K-tiled: quantize each slice using pre-allocated buffers
            K_tile = info["K_tile"]
            mod, gX, gY, M, N, K = self._modules[info["shape_key"]]
            N_actual = info["N"]
            accum_f32 = torch.zeros(M_actual, N_actual, dtype=torch.float32)

            x_float = x.float()  # convert once, slice from this
            buf = self._x_bufs[K_tile]
            c = self._c_bufs[N_actual]

            for t in range(info["k_tiles"]):
                x_slice = x_float[:M_actual, t * K_tile : (t + 1) * K_tile]
                row_max = x_slice.abs().amax(dim=1)
                scale_x = (row_max / 127.0).clamp(min=1e-10)
                buf[:M_actual] = (x_slice / scale_x[:, None]).round().clamp(-128, 127).to(torch.int8)

                weight_i8 = info["weight_tiles"][t]
                scale_w = info["scale_w_tiles"][t]

                mod.set_slot(info["slots"][t])
                mod.set_output_bytes(M_actual * N_actual * 4)
                mod.launch(
                    gX, gY, 1, None, None, None, None,
                    buf, weight_i8, c,
                    M, N_actual, K_tile,
                    buf.stride(0), buf.stride(1),
                    weight_i8.stride(0), weight_i8.stride(1),
                    c.stride(0), c.stride(1),
                    self.M_BLOCK, 256, K_tile,
                )
                self.dispatch_count += 1
                accum_f32 += c[:M_actual].float() * (scale_x[:M_actual, None] * scale_w[None, :])

            return accum_f32.to(torch.bfloat16)

    def quantize_and_pad(self, x, K):
        """Quantize activation per-row, writing only real rows into pre-allocated buffer.

        During decode (M=1), this is 12x faster than quantizing the full 256-row
        padded tensor. The padding rows keep their previous values but this doesn't
        affect correctness since we only read c[:M_actual] from the NPU output.

        Args:
            x: [M_actual, K] bf16 activation (M_actual typically 1 during decode)
            K: K dimension (must match a pre-allocated buffer)

        Returns:
            x_i8: [M_BLOCK, K] int8 buffer (pre-allocated, only [:M_actual] updated)
            scale_x: [M_actual] float32 per-row scales
        """
        M_actual = x.shape[0]
        assert M_actual <= self.M_BLOCK, f"M_actual={M_actual} exceeds M_BLOCK={self.M_BLOCK}"
        buf = self._x_bufs[K]
        x_float = x.float()
        row_max = x_float.abs().amax(dim=1)  # [M_actual]
        scale_x = (row_max / 127.0).clamp(min=1e-10)
        buf[:M_actual] = (x_float / scale_x[:, None]).round().clamp(-128, 127).to(torch.int8)
        return buf, scale_x

    def matmul_preq(self, name, x_i8, scale_x, M_actual):
        """INT8 matmul with pre-quantized activation (non-K-tiled only).

        Used when multiple projections share the same input (e.g., Q/K/V or gate/up)
        to avoid redundant quantization.
        """
        info = self._layers[name]
        mod, gX, gY, M, N, K = self._modules[info["shape_key"]]
        N = info["N"]
        weight_i8 = info["weight_tiles"][0]
        scale_w = info["scale_w_tiles"][0]
        c = self._c_bufs[N]

        mod.set_slot(info["slots"][0])
        mod.set_output_bytes(M_actual * N * 4)
        mod.launch(
            gX, gY, 1, None, None, None, None,
            x_i8, weight_i8, c,
            M, N, K,
            x_i8.stride(0), x_i8.stride(1),
            weight_i8.stride(0), weight_i8.stride(1),
            c.stride(0), c.stride(1),
            self.M_BLOCK, 256, K,
        )
        self.dispatch_count += 1

        result_f32 = c[:M_actual].float() * (scale_x[:M_actual, None] * scale_w[None, :])
        return result_f32.to(torch.bfloat16)

    def swiglu(self, gate_flat, up_flat):
        """SwiGLU on NPU (stays BF16)."""
        out = torch.empty_like(gate_flat)
        self._swiglu_mod.launch(
            self._swiglu_grid, 1, 1,
            None, None, None, None,
            gate_flat, up_flat, out,
            self._swiglu_n, self._swiglu_bs,
        )
        self.dispatch_count += 1
        return out


# =============================================================================
# Model Patching
# =============================================================================

def patch_model_int8(model, engine):
    """Replace all Linear projections with INT8 NPU dispatch."""
    num_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size

    print(f"  Quantizing & registering {num_layers} layers for INT8 NPU dispatch...")

    # Quantize all weights and register
    total_orig_bytes = 0
    total_quant_bytes = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        for proj_name, proj in [
            (f"L{i}.q", attn.q_proj), (f"L{i}.k", attn.k_proj),
            (f"L{i}.v", attn.v_proj), (f"L{i}.o", attn.o_proj),
            (f"L{i}.gate", mlp.gate_proj), (f"L{i}.up", mlp.up_proj),
            (f"L{i}.down", mlp.down_proj),
        ]:
            orig_bytes = proj.weight.numel() * proj.weight.element_size()
            total_orig_bytes += orig_bytes
            engine.prepare_layer(proj_name, proj)
            quant_bytes = proj.weight.numel()  # 1 byte per INT8 element
            total_quant_bytes += quant_bytes

    # Compile SwiGLU kernel (BF16)
    engine.compile_swiglu(intermediate)

    compression = total_orig_bytes / total_quant_bytes
    print(f"  Total weight memory: {total_orig_bytes/1e6:.0f} MB → {total_quant_bytes/1e6:.0f} MB ({compression:.1f}x)")
    print(f"  Compiled {engine._compile_count} unique INT8 NPU kernels")
    total_slots = sum(engine._slot_counters.values())
    print(f"  Weight residency: {total_slots} slots across {len(engine._slot_counters)} shapes")

    # Patch MLP
    def make_mlp_forward(layer_idx):
        def mlp_forward(self, hidden_states):
            x = hidden_states.squeeze(0) if hidden_states.dim() == 3 else hidden_states
            x_bf16 = x.to(torch.bfloat16)
            M_actual = x_bf16.shape[0]

            # Quantize once for gate + up (shared input → saves 1 quant/layer)
            x_i8, scale_x = engine.quantize_and_pad(x_bf16, K=x_bf16.shape[1])
            gate = engine.matmul_preq(f"L{layer_idx}.gate", x_i8, scale_x, M_actual)
            up = engine.matmul_preq(f"L{layer_idx}.up", x_i8, scale_x, M_actual)

            if gate.shape[0] == 1:
                activated = engine.swiglu(gate.view(-1), up.view(-1)).view(gate.shape)
            else:
                activated = torch.empty_like(gate)
                for j in range(gate.shape[0]):
                    activated[j] = engine.swiglu(gate[j].contiguous(), up[j].contiguous())

            down = engine.matmul(f"L{layer_idx}.down", activated)

            if hidden_states.dim() == 3:
                return down.unsqueeze(0).to(hidden_states.dtype)
            return down.to(hidden_states.dtype)
        return mlp_forward

    # Patch attention
    def make_attn_forward(layer_idx):
        def attn_forward(self, hidden_states, position_embeddings=None,
                         attention_mask=None, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)
            M_actual = x.shape[0]

            # Quantize once for Q, K, V (shared input → saves 2 quants/layer)
            x_i8, scale_x = engine.quantize_and_pad(x, K=x.shape[1])
            q = engine.matmul_preq(f"L{layer_idx}.q", x_i8, scale_x, M_actual).to(hidden_states.dtype)
            k = engine.matmul_preq(f"L{layer_idx}.k", x_i8, scale_x, M_actual).to(hidden_states.dtype)
            v = engine.matmul_preq(f"L{layer_idx}.v", x_i8, scale_x, M_actual).to(hidden_states.dtype)

            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(hidden_shape).transpose(1, 2)
            value_states = v.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_idx
                )

            from transformers.models.llama.modeling_llama import (
                ALL_ATTENTION_FUNCTIONS,
                eager_attention_forward,
            )
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states,
                attention_mask, dropout=0.0, scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_out_2d = attn_output.reshape(-1, attn_output.shape[-1]).to(torch.bfloat16)
            # O proj: separate quantization (different input from Q/K/V)
            o_i8, o_scale = engine.quantize_and_pad(attn_out_2d, K=attn_out_2d.shape[1])
            attn_output = engine.matmul_preq(f"L{layer_idx}.o", o_i8, o_scale, M_actual)
            attn_output = attn_output.view(*input_shape, -1).to(hidden_states.dtype)

            return attn_output, attn_weights
        return attn_forward

    for i, layer in enumerate(model.model.layers):
        layer.mlp.forward = types.MethodType(make_mlp_forward(i), layer.mlp)
        layer.self_attn.forward = types.MethodType(make_attn_forward(i), layer.self_attn)

    dispatches_per_token = num_layers * 9
    return dispatches_per_token


# Rotary embedding helper
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Custom Forward Pass — bypasses all HuggingFace overhead
# =============================================================================

def generate_fast(model, engine, tokenizer, prompt, max_tokens):
    """Custom transformer forward that bypasses HuggingFace framework.

    Instead of model(input_ids) → Module.__call__ → hooks → forward → ...,
    this directly implements the transformer layers with raw tensor ops.
    Eliminates ~79% CPU overhead from framework dispatch.
    """
    # ---- Extract model components ----
    cfg = model.config
    num_layers = cfg.num_hidden_layers      # 24
    hidden_size = cfg.hidden_size           # 2048
    num_heads = cfg.num_attention_heads     # 32
    head_dim = cfg.head_dim                 # 64
    intermediate = cfg.intermediate_size    # 8192
    eps = cfg.rms_norm_eps                  # 1e-5
    scale = 1.0 / (head_dim ** 0.5)        # 0.125

    embed_w = model.model.embed_tokens.weight.data          # [vocab, hidden]
    final_norm_w = model.model.norm.weight.data              # [hidden]
    lm_head_w = model.lm_head.weight.data                    # [vocab, hidden]

    # Per-layer norm weights (avoid Module.__call__)
    input_ln_w = [l.input_layernorm.weight.data for l in model.model.layers]
    post_ln_w = [l.post_attention_layernorm.weight.data for l in model.model.layers]

    # ---- RMSNorm (inline, no module overhead) ----
    def rms_norm(x, weight):
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight).to(torch.bfloat16)

    # ---- Rotary embedding (inline) ----
    def apply_rotary(x, cos, sin):
        """x: [heads, seq, head_dim], cos/sin: [1, seq, head_dim]"""
        x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    # ---- Tokenize ----
    messages = [{"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(chat_input, return_tensors="pt")["input_ids"][0]
    prompt_len = input_ids.shape[0]
    max_seq = prompt_len + max_tokens + 1

    # ---- Pre-compute rotary embeddings for all positions ----
    with torch.no_grad():
        dummy = torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16)
        positions = torch.arange(max_seq).unsqueeze(0)
        cos_all, sin_all = model.model.rotary_emb(dummy, positions)
    # cos_all: [1, max_seq, head_dim] → [max_seq, head_dim]
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)

    # ---- Pre-allocate KV cache (avoids torch.cat allocation per layer) ----
    k_caches = [torch.zeros(num_heads, max_seq, head_dim, dtype=torch.bfloat16)
                for _ in range(num_layers)]
    v_caches = [torch.zeros(num_heads, max_seq, head_dim, dtype=torch.bfloat16)
                for _ in range(num_layers)]

    # ---- Core forward pass ----
    def forward_pass(token_ids, start_pos, cache_len):
        """Run transformer on token_ids with position offset and KV cache.

        Args:
            token_ids: [S] int64 tensor
            start_pos: starting position for RoPE
            cache_len: current filled length of KV cache
        Returns:
            logits: [S, vocab] float32
        """
        S = token_ids.shape[0]

        # Embedding (simple indexing, no Module.__call__)
        hidden = embed_w[token_ids].to(torch.bfloat16)  # [S, hidden]

        # RoPE cos/sin for this batch of positions
        cos = cos_all[start_pos:start_pos + S].unsqueeze(0)  # [1, S, head_dim]
        sin = sin_all[start_pos:start_pos + S].unsqueeze(0)

        for i in range(num_layers):
            residual = hidden

            # ---- Input LayerNorm ----
            normed = rms_norm(hidden, input_ln_w[i])

            # ---- Q/K/V with shared quantization (1 quant for 3 projections) ----
            x_i8, sx = engine.quantize_and_pad(normed, K=hidden_size)
            q = engine.matmul_preq(f"L{i}.q", x_i8, sx, S)
            k = engine.matmul_preq(f"L{i}.k", x_i8, sx, S)
            v = engine.matmul_preq(f"L{i}.v", x_i8, sx, S)

            # Reshape to multi-head: [S, hidden] → [heads, S, head_dim]
            q = q.view(S, num_heads, head_dim).permute(1, 0, 2)
            k = k.view(S, num_heads, head_dim).permute(1, 0, 2)
            v = v.view(S, num_heads, head_dim).permute(1, 0, 2)

            # Rotary embeddings
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # KV cache update (in-place, no allocation)
            k_caches[i][:, cache_len:cache_len + S, :] = k
            v_caches[i][:, cache_len:cache_len + S, :] = v
            total_len = cache_len + S

            # Attention (fused SDPA — avoids 6+ intermediate tensor allocs)
            k_full = k_caches[i][:, :total_len, :]      # view, no copy
            v_full = v_caches[i][:, :total_len, :]
            context = F.scaled_dot_product_attention(
                q, k_full, v_full,
                is_causal=(S > 1 and cache_len == 0),
                scale=scale,
            )  # [heads, S, head_dim]

            # Reshape back: [heads, S, head_dim] → [S, hidden]
            context = context.permute(1, 0, 2).reshape(S, hidden_size).to(torch.bfloat16)

            # ---- O projection (separate quantization) ----
            o_i8, o_sx = engine.quantize_and_pad(context, K=hidden_size)
            attn_out = engine.matmul_preq(f"L{i}.o", o_i8, o_sx, S)

            # Residual
            hidden = residual + attn_out.to(residual.dtype)

            # ---- Post-attention LayerNorm ----
            residual = hidden
            normed = rms_norm(hidden, post_ln_w[i])

            # ---- MLP: gate + up with shared quantization ----
            m_i8, m_sx = engine.quantize_and_pad(normed, K=hidden_size)
            gate = engine.matmul_preq(f"L{i}.gate", m_i8, m_sx, S)
            up = engine.matmul_preq(f"L{i}.up", m_i8, m_sx, S)

            # SwiGLU on NPU
            if S == 1:
                activated = engine.swiglu(gate.view(-1), up.view(-1)).view(1, intermediate)
            else:
                activated = torch.empty_like(gate)
                for j in range(S):
                    activated[j] = engine.swiglu(gate[j].contiguous(), up[j].contiguous())

            # Down projection (K-tiled path)
            down = engine.matmul(f"L{i}.down", activated)

            # Residual
            hidden = residual + down.to(residual.dtype)

        # ---- Final norm + LM head ----
        hidden = rms_norm(hidden, final_norm_w)
        logits = torch.matmul(hidden.float(), lm_head_w.float().T)
        return logits

    # ---- Generation loop ----
    print(f"  Prompt: \"{prompt}\"")
    print()

    engine.dispatch_count = 0
    token_times = []
    generated_tokens = []
    cache_len = 0

    with torch.inference_mode():
        # Prefill
        t0 = time.perf_counter()
        logits = forward_pass(input_ids, start_pos=0, cache_len=0)
        cache_len = prompt_len
        next_id = logits[-1].argmax().item()
        generated_tokens.append(next_id)
        dt = time.perf_counter() - t0
        token_times.append(dt)

        tok_str = tokenizer.decode([next_id], skip_special_tokens=True)
        print(f"  Token  1: \"{tok_str}\"  ({dt * 1000:.0f}ms) [prefill]", flush=True)

        # Decode
        for i in range(1, max_tokens):
            t0 = time.perf_counter()
            token_tensor = torch.tensor([next_id], dtype=torch.long)
            logits = forward_pass(token_tensor, start_pos=cache_len, cache_len=cache_len)
            cache_len += 1
            next_id = logits[0].argmax().item()
            generated_tokens.append(next_id)
            dt = time.perf_counter() - t0
            token_times.append(dt)

            tok_str = tokenizer.decode([next_id], skip_special_tokens=True)
            print(f"  Token {i + 1:2d}: \"{tok_str}\"  ({dt * 1000:.0f}ms)", flush=True)

            if next_id == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return full_text, token_times, engine.dispatch_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT8 NPU LLM inference")
    parser.add_argument("--prompt", "-p", type=str,
                        default="Explain what an NPU is in one sentence:")
    parser.add_argument("--max-tokens", "-n", type=int, default=20)
    parser.add_argument("--model", "-m", type=str,
                        default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    args = parser.parse_args()

    print("=" * 70)
    print("  INT8 NPU LLM Inference — AMD Ryzen AI (Strix Halo)")
    print("  All Linear projections INT8×INT8→INT32 on NPU")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Initialize NPU backend
    # ------------------------------------------------------------------
    print("\n[1/5] Activating NPU backend...")
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
    benchmark.select_npu_backend()

    engine = NPUInt8Engine()

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[2/5] Loading {args.model}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.1f}s ({n_params:.2f}B params)")

    # ------------------------------------------------------------------
    # 3. Quantize weights & compile INT8 NPU kernels
    # ------------------------------------------------------------------
    print(f"\n[3/5] Quantizing to INT8 & compiling NPU kernels...")
    t0 = time.perf_counter()
    dispatches_per_token = patch_model_int8(model, engine)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    print(f"  NPU dispatches per decode token: {dispatches_per_token}")

    # ------------------------------------------------------------------
    # 4. Warmup
    # ------------------------------------------------------------------
    print(f"\n[4/5] Warmup...")
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": args.prompt}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt")
    with torch.inference_mode():
        _ = model.generate(inputs["input_ids"], max_new_tokens=1, do_sample=False)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. Generate text
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating text (INT8 NPU matmuls)...")
    print(f"  Prompt: \"{args.prompt}\"")
    print()

    input_len = inputs["input_ids"].shape[1]
    engine.dispatch_count = 0
    token_times = []
    generated_ids = inputs["input_ids"].clone()
    past_key_values = None

    with torch.inference_mode():
        for i in range(args.max_tokens):
            t_tok = time.perf_counter()

            if past_key_values is None:
                outputs = model(input_ids=generated_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            dt = time.perf_counter() - t_tok
            token_times.append(dt)

            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"  Token {i+1:2d}: \"{token_text}\"  ({dt*1000:.0f}ms)", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    full_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    n_tokens = len(token_times)
    total_time = sum(token_times)

    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0

    print(f"\n{'=' * 70}")
    print(f"  Output: \"{full_text}\"")
    print(f"{'=' * 70}")
    print(f"  Tokens:          {n_tokens}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Prefill:         {prefill_time*1000:.0f}ms")
    if decode_times:
        print(f"  Avg decode:      {avg_decode*1000:.0f}ms/token")
        print(f"  Decode tok/s:    {1/avg_decode:.2f}")
    print(f"  NPU dispatches:  {engine.dispatch_count}")
    print(f"  Weight format:   INT8 (per-channel symmetric)")
    print(f"  Activation:      INT8 (per-row dynamic)")
    print(f"  NPU compute:     INT8×INT8→INT32")
