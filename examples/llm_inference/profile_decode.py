#!/usr/bin/env python3
"""Profile a single decode step to see exactly where CPU time goes."""

import os, sys, time
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("AMD_TRITON_NPU_OUTPUT_FORMAT", "xclbin")
if "XRT_DEV_DIR" not in os.environ:
    for _p in [r"C:\projects\xrt-dev", r"C:\xrt-dev"]:
        if os.path.isdir(_p):
            os.environ["XRT_DEV_DIR"] = _p
            break

import torch
import triton
import triton.language as tl
import types, importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
torch.set_num_threads(os.cpu_count())

import benchmark
benchmark.select_npu_backend()

import llm_npu_int8 as mod

# --- Load and patch ---
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings; warnings.filterwarnings("ignore")

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
model.eval()

engine = mod.NPUInt8Engine()
mod.patch_model_int8(model, engine, verbose=True)

# --- Warmup ---
msgs = [{"role": "user", "content": "Hi"}]
inp = tokenizer(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt")
with torch.inference_mode():
    _ = model(input_ids=inp["input_ids"], use_cache=True)

# --- Profile decode steps ---
print("\n" + "=" * 70)
print("PROFILING DECODE STEPS")
print("=" * 70)

msgs = [{"role": "user", "content": "Count from 1 to 20"}]
inp = tokenizer(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt")

with torch.inference_mode():
    # Prefill
    outputs = model(input_ids=inp["input_ids"], use_cache=True)
    past = outputs.past_key_values
    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ids = torch.cat([inp["input_ids"], next_tok], dim=-1)
    
    # Time 10 decode steps
    times = []
    for i in range(10):
        engine.dispatch_count = 0
        t0 = time.perf_counter()
        outputs = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=True)
        t1 = time.perf_counter()
        past = outputs.past_key_values
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
        
        tok_text = tokenizer.decode([next_tok.item()], skip_special_tokens=True)
        times.append(t1 - t0)
        print(f"  decode[{i}]: {(t1-t0)*1000:.1f}ms  dispatches={engine.dispatch_count}  token='{tok_text}'")
    
    avg = sum(times[1:]) / len(times[1:])  # skip first (may have JIT overhead)
    print(f"\n  Average decode: {avg*1000:.1f}ms ({1/avg:.1f} tok/s)")

# --- Now profile INSIDE a single decode step ---
print("\n" + "=" * 70)
print("BREAKDOWN OF ONE DECODE STEP")
print("=" * 70)

cfg = model.config
num_layers = cfg.num_hidden_layers
hidden_size = cfg.hidden_size
num_heads = cfg.num_attention_heads
head_dim = cfg.head_dim
intermediate = cfg.intermediate_size
eps = cfg.rms_norm_eps
scale = 1.0 / (head_dim ** 0.5)

embed_w = model.model.embed_tokens.weight.data
final_norm_w = model.model.norm.weight.data
lm_head_w = model.lm_head.weight.data
input_ln_w = [l.input_layernorm.weight.data for l in model.model.layers]
post_ln_w = [l.post_attention_layernorm.weight.data for l in model.model.layers]

# Get current state
last_tok = ids[:, -1:]
past_kv = past

# Profile each component
timings = {
    "embed": 0, "rms_norm": 0, "quantize": 0, "qkv_npu": 0,
    "rope": 0, "kv_cache": 0, "attention": 0, "o_proj_npu": 0,
    "residual": 0, "gate_up_npu": 0, "swiglu_npu": 0, "down_npu": 0,
    "final_norm": 0, "lm_head": 0, "argmax": 0,
}

with torch.inference_mode():
    # Embedding
    t = time.perf_counter()
    h = embed_w[last_tok.squeeze()]  # [1, hidden]
    timings["embed"] = time.perf_counter() - t

    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # Input LayerNorm (RMS)
        t = time.perf_counter()
        residual = h
        h_f = h.float()
        rms = torch.sqrt(h_f.pow(2).mean(-1, keepdim=True) + eps)
        h_normed = (h_f / rms * input_ln_w[i]).to(torch.bfloat16)
        timings["rms_norm"] += time.perf_counter() - t

        # Quantize for Q/K/V
        t = time.perf_counter()
        x_2d = h_normed.view(-1, hidden_size)
        x_i8, scale_x = engine.quantize_and_pad(x_2d, K=hidden_size)
        timings["quantize"] += time.perf_counter() - t

        # Q/K/V projections on NPU
        t = time.perf_counter()
        q = engine.matmul_preq(f"L{i}.q", x_i8, scale_x, 1).to(h.dtype)
        k = engine.matmul_preq(f"L{i}.k", x_i8, scale_x, 1).to(h.dtype)
        v = engine.matmul_preq(f"L{i}.v", x_i8, scale_x, 1).to(h.dtype)
        timings["qkv_npu"] += time.perf_counter() - t

        # RoPE
        t = time.perf_counter()
        q_s = q.view(1, 1, num_heads, head_dim).transpose(1, 2)
        k_s = k.view(1, 1, num_heads, head_dim).transpose(1, 2)
        v_s = v.view(1, 1, num_heads, head_dim).transpose(1, 2)
        pos_ids = torch.tensor([[ids.shape[1] - 1]])
        cos, sin = model.model.rotary_emb(v_s, pos_ids)
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        q1 = q_s[..., :head_dim//2]; q2 = q_s[..., head_dim//2:]
        q_s = (q_s * cos_u) + (torch.cat((-q2, q1), -1) * sin_u)
        k1 = k_s[..., :head_dim//2]; k2 = k_s[..., head_dim//2:]
        k_s = (k_s * cos_u) + (torch.cat((-k2, k1), -1) * sin_u)
        timings["rope"] += time.perf_counter() - t

        # KV cache
        t = time.perf_counter()
        k_s, v_s = past_kv.update(k_s, v_s, i)
        timings["kv_cache"] += time.perf_counter() - t

        # Attention (SDPA or manual)
        t = time.perf_counter()
        attn_w = torch.matmul(q_s, k_s.transpose(2, 3)) * scale
        attn_w = torch.nn.functional.softmax(attn_w, dim=-1)
        attn_out = torch.matmul(attn_w, v_s)
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, -1).contiguous()
        timings["attention"] += time.perf_counter() - t

        # O proj on NPU
        t = time.perf_counter()
        o_2d = attn_out.view(-1, hidden_size).to(torch.bfloat16)
        o_i8, o_scale = engine.quantize_and_pad(o_2d, K=hidden_size)
        h = engine.matmul_preq(f"L{i}.o", o_i8, o_scale, 1).to(residual.dtype)
        timings["o_proj_npu"] += time.perf_counter() - t

        # Residual add
        t = time.perf_counter()
        h = residual + h.view(residual.shape)
        timings["residual"] += time.perf_counter() - t

        # Post-attn norm
        t = time.perf_counter()
        residual2 = h
        h_f = h.float()
        rms = torch.sqrt(h_f.pow(2).mean(-1, keepdim=True) + eps)
        h_normed2 = (h_f / rms * post_ln_w[i]).to(torch.bfloat16)
        timings["rms_norm"] += time.perf_counter() - t

        # Gate + Up on NPU (shared quant)
        t = time.perf_counter()
        x_2d = h_normed2.view(-1, hidden_size)
        x_i8, scale_x = engine.quantize_and_pad(x_2d, K=hidden_size)
        timings["quantize"] += time.perf_counter() - t

        t = time.perf_counter()
        gate = engine.matmul_preq(f"L{i}.gate", x_i8, scale_x, 1)
        up = engine.matmul_preq(f"L{i}.up", x_i8, scale_x, 1)
        timings["gate_up_npu"] += time.perf_counter() - t

        # SwiGLU on NPU
        t = time.perf_counter()
        activated = engine.swiglu(gate.view(-1), up.view(-1)).view(gate.shape)
        timings["swiglu_npu"] += time.perf_counter() - t

        # Down proj on NPU
        t = time.perf_counter()
        down = engine.matmul(f"L{i}.down", activated)
        timings["down_npu"] += time.perf_counter() - t

        # Residual add
        t = time.perf_counter()
        h = residual2 + down.view(residual2.shape).to(residual2.dtype)
        timings["residual"] += time.perf_counter() - t

    # Final norm
    t = time.perf_counter()
    h_f = h.float()
    rms = torch.sqrt(h_f.pow(2).mean(-1, keepdim=True) + eps)
    h = (h_f / rms * final_norm_w).to(torch.bfloat16)
    timings["final_norm"] = time.perf_counter() - t

    # LM head (vocab projection)
    t = time.perf_counter()
    logits = torch.nn.functional.linear(h.view(1, -1), lm_head_w)
    timings["lm_head"] = time.perf_counter() - t

    # Argmax
    t = time.perf_counter()
    _ = logits.argmax(dim=-1)
    timings["argmax"] = time.perf_counter() - t

# Print results
total = sum(timings.values())
print(f"\n{'Operation':<20} {'Time (ms)':>10} {'% Total':>10}  {'Where':>8}")
print("-" * 55)
for k, v in sorted(timings.items(), key=lambda x: -x[1]):
    where = "NPU" if "npu" in k else "CPU"
    print(f"  {k:<18} {v*1000:>10.2f} {v/total*100:>9.1f}%  {where:>8}")
print(f"  {'TOTAL':<18} {total*1000:>10.2f}")

npu_time = sum(v for k, v in timings.items() if "npu" in k)
cpu_time = total - npu_time
print(f"\n  NPU total: {npu_time*1000:.1f}ms ({npu_time/total*100:.0f}%)")
print(f"  CPU total: {cpu_time*1000:.1f}ms ({cpu_time/total*100:.0f}%)")
print(f"  Theoretical max if CPU=0: {1/npu_time:.1f} tok/s")
