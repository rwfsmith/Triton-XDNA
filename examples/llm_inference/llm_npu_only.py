#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NPU-Only LLM Inference with AMD Ryzen AI
# ==========================================
# Runs ALL matmul compute on the NPU via Triton-XDNA compiled kernels.
# Only embedding lookup, attention score/softmax, and token sampling remain on CPU.
#
# Architecture:
#   NPU: All Linear projections (q/k/v/o/gate/up/down) + SwiGLU activation
#   CPU: Embedding, attention (tiny in batch-1), RMSNorm, LM head, sampling
#
# Optimizations:
#   - Weight-resident BOs: weights loaded to NPU once, only activations transfer per call
#   - FastNPUDispatch: bypasses Triton JIT, calls compiled C extension directly
#   - Fused gate+up: single NPU dispatch instead of two (BUT NPU needs separate for proper tiling)
#   - K-tiling: down_proj (K=8192) split into 2×K=4096 dispatches

import torch
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

# Use all CPU cores for the few remaining CPU ops
torch.set_num_threads(os.cpu_count())


# =============================================================================
# Triton NPU Matmul Kernel
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
# Triton NPU SwiGLU Kernel
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
# NPU Matmul Dispatcher
# =============================================================================

class NPUMatmulEngine:
    """Manages NPU matmul dispatch with weight-resident buffer objects.

    Pre-compiles xclbins for each unique (M_pad, N, K) shape, then dispatches
    via FastNPUDispatch. Weight tensors are kept at fixed memory addresses so
    the C++ dispatch layer skips their memcpy+DMA (weight-resident optimization).
    """

    def __init__(self):
        self.npu_driver = importlib.import_module(
            "triton.backends.amd_triton_npu.driver"
        )
        # Maps shape_key -> (mod, gX, gY, M, N, K)
        self._modules = {}
        # Maps layer_name -> (weight_bf16_t, shape_key, k_tiles)
        self._layers = {}
        # SwiGLU module
        self._swiglu_mod = None
        self._swiglu_n = None
        self.dispatch_count = 0
        self._compile_count = 0

    def _set_matmul_transform(self):
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
            SCRIPT_DIR, "..", "matmul", "transform_aie2p.mlir"
        )

    def _set_swiglu_transform(self):
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
            SCRIPT_DIR, "transform_binary_aie2p.mlir"
        )

    def _compile_matmul_shape(self, M, N, K):
        """Compile and capture NPU module for a specific (M, N, K) shape."""
        shape_key = (M, N, K)
        if shape_key in self._modules:
            return

        self._set_matmul_transform()
        gX = triton.cdiv(M, 256)
        gY = triton.cdiv(N, 256)
        print(f"    Compiling matmul {M}x{N}x{K} (grid={gX}x{gY})...", end="", flush=True)

        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)
        c = torch.empty(M, N, dtype=torch.float32)
        bare_matmul[(gX, gY)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
        )
        mod = self.npu_driver._last_dispatched_module
        if mod is None:
            raise RuntimeError(f"Failed to capture module for shape {shape_key}")

        # Verify correctness
        c_ref = torch.matmul(a, b).to(torch.float32)
        err = (c - c_ref).abs().max().item()
        status = "PASS" if err < 30 else "FAIL"
        print(f" {status} (err={err:.2f})")

        self._modules[shape_key] = (mod, gX, gY, M, N, K)
        self._compile_count += 1

    def compile_swiglu(self, n_elements, block_size=1024):
        """Compile SwiGLU kernel for the given element count."""
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

    def prepare_layer(self, name, linear, max_K=4096):
        """Register a Linear layer for NPU matmul dispatch.

        Args:
            name: Unique layer identifier (e.g., "layer.0.q_proj")
            linear: nn.Linear instance
            max_K: Maximum K per NPU dispatch (K-tiling splits larger K)
        """
        K_full = linear.in_features
        N = linear.out_features
        M = 256  # Padded batch dim

        # Pre-transpose weight for row-major: W[out, in] -> W_t[in, out] = B[K, N]
        weight_t = linear.weight.data.T.contiguous().to(torch.bfloat16)

        # Determine K-tiling
        if K_full > max_K:
            n_tiles = (K_full + max_K - 1) // max_K
            K_tile = K_full // n_tiles  # even split
            assert K_tile * n_tiles == K_full, \
                f"K={K_full} not evenly divisible into {n_tiles} tiles of {K_tile}"
            # Pre-split weights along K dimension
            weight_tiles = [
                weight_t[i * K_tile : (i + 1) * K_tile, :].contiguous()
                for i in range(n_tiles)
            ]
            self._compile_matmul_shape(M, N, K_tile)
            self._layers[name] = {
                "weight_tiles": weight_tiles,
                "shape_key": (M, N, K_tile),
                "k_tiles": n_tiles,
                "K_full": K_full,
                "K_tile": K_tile,
                "N": N,
            }
        else:
            self._compile_matmul_shape(M, N, K_full)
            self._layers[name] = {
                "weight_tiles": [weight_t],
                "shape_key": (M, N, K_full),
                "k_tiles": 1,
                "K_full": K_full,
                "K_tile": K_full,
                "N": N,
            }

    def matmul(self, name, x):
        """Run matmul on NPU: output = x @ weight.T

        Args:
            name: Layer name (registered via prepare_layer)
            x: Input tensor [batch, in_features] in bf16

        Returns:
            output tensor [batch, out_features] in bf16
        """
        info = self._layers[name]
        mod, gX, gY, M, N, K = self._modules[info["shape_key"]]

        # Pad input to M=256
        M_actual = x.shape[0]
        if M_actual < 256:
            x_pad = torch.zeros(256, info["K_full"], dtype=torch.bfloat16)
            x_pad[:M_actual] = x
        else:
            x_pad = x

        if info["k_tiles"] == 1:
            # Single dispatch
            weight = info["weight_tiles"][0]
            c = torch.empty(256, N, dtype=torch.float32)
            mod.launch(
                gX, gY, 1, None, None, None, None,
                x_pad, weight, c,
                M, N, K,
                x_pad.stride(0), x_pad.stride(1),
                weight.stride(0), weight.stride(1),
                c.stride(0), c.stride(1),
                256, 256, K,
            )
            self.dispatch_count += 1
            result = c[:M_actual].to(torch.bfloat16)
        else:
            # K-tiled: multiple dispatches + accumulate
            K_tile = info["K_tile"]
            accum = torch.zeros(256, N, dtype=torch.float32)
            for t, weight_tile in enumerate(info["weight_tiles"]):
                x_slice = x_pad[:, t * K_tile : (t + 1) * K_tile].contiguous()
                c = torch.empty(256, N, dtype=torch.float32)
                mod.launch(
                    gX, gY, 1, None, None, None, None,
                    x_slice, weight_tile, c,
                    M, N, K_tile,
                    x_slice.stride(0), x_slice.stride(1),
                    weight_tile.stride(0), weight_tile.stride(1),
                    c.stride(0), c.stride(1),
                    256, 256, K_tile,
                )
                accum += c
                self.dispatch_count += 1
            result = accum[:M_actual].to(torch.bfloat16)

        return result

    def swiglu(self, gate_flat, up_flat):
        """Run SwiGLU on NPU."""
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

def patch_model_npu(model, engine):
    """Replace all Linear projections in the model with NPU dispatch.

    Returns the number of NPU dispatches expected per decode token.
    """
    num_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size

    print(f"  Preparing {num_layers} layers for NPU dispatch...")

    # Register all Linear layers
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        # Attention projections (K=2048, N=2048)
        engine.prepare_layer(f"L{i}.q", attn.q_proj)
        engine.prepare_layer(f"L{i}.k", attn.k_proj)
        engine.prepare_layer(f"L{i}.v", attn.v_proj)
        engine.prepare_layer(f"L{i}.o", attn.o_proj)

        # MLP projections
        engine.prepare_layer(f"L{i}.gate", mlp.gate_proj)
        engine.prepare_layer(f"L{i}.up", mlp.up_proj)
        engine.prepare_layer(f"L{i}.down", mlp.down_proj)  # K=8192, auto K-tiled

    # Compile SwiGLU kernel
    engine.compile_swiglu(intermediate)

    print(f"  Compiled {engine._compile_count} unique NPU kernels")
    print(f"  Registered {len(engine._layers)} layer dispatches")

    # Patch MLP forward
    def make_mlp_forward(layer_idx):
        def mlp_forward(self, hidden_states):
            x = hidden_states.squeeze(0) if hidden_states.dim() == 3 else hidden_states
            x_bf16 = x.to(torch.bfloat16)

            # NPU: gate and up projections
            gate = engine.matmul(f"L{layer_idx}.gate", x_bf16)
            up = engine.matmul(f"L{layer_idx}.up", x_bf16)

            # NPU: SwiGLU activation (per-row: kernel compiled for intermediate_size)
            if gate.shape[0] == 1:
                activated = engine.swiglu(gate.view(-1), up.view(-1)).view(gate.shape)
            else:
                activated = torch.empty_like(gate)
                for j in range(gate.shape[0]):
                    activated[j] = engine.swiglu(gate[j].contiguous(), up[j].contiguous())

            # NPU: down projection
            down = engine.matmul(f"L{layer_idx}.down", activated)

            if hidden_states.dim() == 3:
                return down.unsqueeze(0).to(hidden_states.dtype)
            return down.to(hidden_states.dtype)
        return mlp_forward

    # Patch attention forward to use NPU matmul for projections
    def make_attn_forward(layer_idx):
        def attn_forward(self, hidden_states, position_embeddings=None,
                         attention_mask=None, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # Flatten for NPU matmul: [bsz, seq, hidden] -> [bsz*seq, hidden]
            x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)

            # NPU: Q/K/V projections
            q = engine.matmul(f"L{layer_idx}.q", x).to(hidden_states.dtype)
            k = engine.matmul(f"L{layer_idx}.k", x).to(hidden_states.dtype)
            v = engine.matmul(f"L{layer_idx}.v", x).to(hidden_states.dtype)

            # Reshape for multi-head attention
            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(hidden_shape).transpose(1, 2)
            value_states = v.view(hidden_shape).transpose(1, 2)

            # Apply rotary embeddings (CPU — tiny)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # KV cache (CPU — in-place update)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_idx
                )

            # Attention (CPU — tiny for batch-1 decode)
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

            # NPU: output projection
            attn_out_2d = attn_output.reshape(-1, attn_output.shape[-1]).to(torch.bfloat16)
            attn_output = engine.matmul(f"L{layer_idx}.o", attn_out_2d)
            attn_output = attn_output.view(*input_shape, -1).to(hidden_states.dtype)

            return attn_output, attn_weights
        return attn_forward

    for i, layer in enumerate(model.model.layers):
        layer.mlp.forward = types.MethodType(make_mlp_forward(i), layer.mlp)
        layer.self_attn.forward = types.MethodType(make_attn_forward(i), layer.self_attn)

    # Count dispatches per decode token
    # Per layer: q + k + v + o + gate + up + swiglu + down (2 K-tiles) = 9 dispatches
    dispatches_per_token = num_layers * 9  # 24 * 9 = 216
    return dispatches_per_token


# Rotary embedding helper (from transformers)
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
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPU-only LLM inference")
    parser.add_argument("--prompt", "-p", type=str,
                        default="Explain what an NPU is in one sentence:")
    parser.add_argument("--max-tokens", "-n", type=int, default=20)
    parser.add_argument("--model", "-m", type=str,
                        default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    args = parser.parse_args()

    print("=" * 70)
    print("  NPU-Only LLM Inference — AMD Ryzen AI (Strix Halo)")
    print("  All Linear projections + SwiGLU on NPU")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Initialize NPU backend
    # ------------------------------------------------------------------
    print("\n[1/5] Activating NPU backend...")
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
    benchmark.select_npu_backend()

    engine = NPUMatmulEngine()

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
    print(f"  Config: hidden={model.config.hidden_size}, "
          f"intermediate={model.config.intermediate_size}, "
          f"layers={model.config.num_hidden_layers}")

    # ------------------------------------------------------------------
    # 3. Compile NPU kernels & register weights
    # ------------------------------------------------------------------
    print(f"\n[3/5] Compiling NPU kernels & loading weights...")
    t0 = time.perf_counter()
    dispatches_per_token = patch_model_npu(model, engine)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    print(f"  NPU dispatches per decode token: {dispatches_per_token}")

    # ------------------------------------------------------------------
    # 4. Warmup (1 token to prime caches)
    # ------------------------------------------------------------------
    print(f"\n[4/5] Warmup...")
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": args.prompt}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(inputs["input_ids"], max_new_tokens=1, do_sample=False)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. Generate text
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating text (NPU-only matmuls)...")
    print(f"  Prompt: \"{args.prompt}\"")
    print()

    input_len = inputs["input_ids"].shape[1]
    engine.dispatch_count = 0
    token_times = []
    generated_ids = inputs["input_ids"].clone()
    past_key_values = None

    with torch.no_grad():
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
            dispatches_so_far = engine.dispatch_count
            print(f"  Token {i+1:2d}: \"{token_text}\"  ({dt*1000:.0f}ms, "
                  f"{dispatches_so_far} NPU dispatches)", flush=True)

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

    # Breakdown
    if decode_times:
        print(f"\n  Performance Breakdown (per decode token):")
        # From benchmarks: matmul shapes timing
        matmul_time_ms = 14.534  # from bench_llm_shapes.py with weight-resident
        swiglu_time_ms = 0.3
        # Per layer = q + k + v + o + gate + up + swiglu + down(x2)
        npu_compute_ms = (matmul_time_ms + swiglu_time_ms) * 1  # per layer
        npu_total_24 = npu_compute_ms * 24
        total_ms = avg_decode * 1000
        cpu_ms = total_ms - npu_total_24
        print(f"    NPU matmul:   ~{npu_total_24:.0f}ms ({24} layers x {matmul_time_ms:.1f}ms)")
        print(f"    CPU overhead: ~{cpu_ms:.0f}ms (attention, RMSNorm, embedding, lm_head)")
        print(f"    NPU share:    ~{npu_total_24/total_ms*100:.0f}%")
        print(f"\n  Note: NPU processes all {dispatches_per_token} Linear projections per token.")
        print(f"  CPU handles only: embedding, attention score/softmax, RMSNorm, lm_head.")
