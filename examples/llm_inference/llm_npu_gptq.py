#!/usr/bin/env python3
"""NPU-Only LLM Inference with GPTQ INT8 quantized model.

Loads GPTQ INT8 weights, dequantizes to bf16, then runs the same NPU pipeline.
Compare with the bf16 model to see if there's any speed difference.
"""
import torch
import time
import os
import sys
import argparse
import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, ".."))


def dequantize_gptq_weight(qweight, qzeros, scales, g_idx, bits=8):
    """Dequantize GPTQ packed INT8 weights to bf16.

    GPTQ packs multiple INT values per int32:
      - 8-bit: 4 values per int32
      - 4-bit: 8 values per int32

    Args:
        qweight: [in_features/pack_factor, out_features] int32
        qzeros: [n_groups, out_features/pack_factor] int32
        scales: [n_groups, out_features] float16
        g_idx: [in_features] int32 — group assignment per input row
        bits: quantization bits (8)

    Returns:
        weight: [out_features, in_features] bf16 (standard nn.Linear layout)
    """
    pack_factor = 32 // bits  # 4 for INT8
    mask = (1 << bits) - 1    # 0xFF for INT8

    in_features = qweight.shape[0] * pack_factor
    out_features = qweight.shape[1]

    # Unpack qweight: [in_features/4, out_features] int32 -> [in_features, out_features]
    weight_unpacked = torch.zeros(in_features, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        weight_unpacked[j::pack_factor] = (qweight >> (bits * j)) & mask

    # Unpack qzeros: [n_groups, out_features/4] int32 -> [n_groups, out_features]
    n_groups = qzeros.shape[0]
    zeros_unpacked = torch.zeros(n_groups, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        zeros_unpacked[:, j::pack_factor] = (qzeros >> (bits * j)) & mask

    # Dequantize: weight_float = (weight_int - zero_point) * scale
    # g_idx maps each input row to its group
    scales_expanded = scales[g_idx]        # [in_features, out_features]
    zeros_expanded = zeros_unpacked[g_idx]  # [in_features, out_features]

    weight_float = (weight_unpacked.float() - zeros_expanded.float()) * scales_expanded.float()

    # Return in nn.Linear layout: [out_features, in_features]
    return weight_float.T.to(torch.bfloat16)


def load_gptq_as_bf16(repo_id):
    """Load a GPTQ model and dequantize all weights to a standard bf16 LlamaForCausalLM."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"  Downloading GPTQ model from {repo_id}...")
    t0 = time.perf_counter()

    # Download config without quantization_config to build a standard model
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Remove quantization config so we get a normal model
    quant_config = config_dict.pop("quantization_config", None)
    bits = quant_config.get("bits", 8) if quant_config else 8
    group_size = quant_config.get("group_size", 128) if quant_config else 128
    print(f"  Quantization: {bits}-bit, group_size={group_size}")

    # Build a standard (unquantized) model config
    config = AutoConfig.for_model(config_dict["model_type"], **config_dict)

    # Create empty bf16 model
    print(f"  Creating empty bf16 model...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.eval()

    # Load quantized weights
    safetensors_path = hf_hub_download(repo_id, "model.safetensors")
    print(f"  Loading quantized weights...")
    state = load_file(safetensors_path)

    # Dequantize and assign weights
    print(f"  Dequantizing INT{bits} → bf16...")
    assigned = set()

    for name, param in model.named_parameters():
        full_key = name

        # Check if this is a quantized linear layer
        base_key = full_key.replace(".weight", "")
        qweight_key = f"{base_key}.qweight"

        if qweight_key in state:
            # Dequantize this layer
            qweight = state[f"{base_key}.qweight"]
            qzeros = state[f"{base_key}.qzeros"]
            scales = state[f"{base_key}.scales"]
            g_idx = state[f"{base_key}.g_idx"]

            dequantized = dequantize_gptq_weight(qweight, qzeros, scales, g_idx, bits)
            param.data.copy_(dequantized)
            assigned.add(full_key)
        elif full_key in state:
            # Non-quantized parameter (layernorm, embedding, etc.)
            param.data.copy_(state[full_key].to(param.dtype))
            assigned.add(full_key)

    # Check coverage
    model_params = set(n for n, _ in model.named_parameters())
    missing = model_params - assigned
    if missing:
        print(f"  WARNING: {len(missing)} parameters not loaded: {list(missing)[:5]}...")

    dt = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded and dequantized in {dt:.1f}s ({n_params:.2f}B params)")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPU-only LLM inference with GPTQ INT8 model")
    parser.add_argument("--prompt", "-p", type=str,
                        default="Explain what an NPU is in one sentence:")
    parser.add_argument("--max-tokens", "-n", type=int, default=30)
    parser.add_argument("--model", "-m", type=str,
                        default="fbaldassarri/HuggingFaceTB_SmolLM2-1.7B-Instruct-auto_gptq-int8-gs128-asym")
    args = parser.parse_args()

    print("=" * 70)
    print("  NPU-Only LLM Inference — GPTQ INT8 → bf16 dequantized")
    print("  AMD Ryzen AI (Strix Halo)")
    print("=" * 70)

    # Setup NPU backend
    print("\n[1/5] Activating NPU backend...")
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
    import benchmark
    benchmark.select_npu_backend()

    # Import NPU engine from our main script
    from llm_npu_only import NPUMatmulEngine, patch_model_npu

    engine = NPUMatmulEngine()

    # Load & dequantize model
    print(f"\n[2/5] Loading GPTQ INT8 model: {args.model}")
    model, tokenizer = load_gptq_as_bf16(args.model)

    # Quick sanity check: compare with bf16 model output
    print(f"\n  Sanity check: first token prediction...")
    messages = [{"role": "user", "content": args.prompt}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_ids"]).logits[0, -1, :]
        tok_id = logits.argmax().item()
        print(f"  First token: {tok_id} = '{tokenizer.decode([tok_id])}'")
        print(f"  Top-5: {logits.topk(5).indices.tolist()}")

    # Compile NPU kernels & register weights
    print(f"\n[3/5] Compiling NPU kernels & loading weights...")
    t0 = time.perf_counter()
    dispatches_per_token = patch_model_npu(model, engine)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.1f}s")
    print(f"  NPU dispatches per decode token: {dispatches_per_token}")

    # Warmup
    print(f"\n[4/5] Warmup...")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(inputs["input_ids"], max_new_tokens=1, do_sample=False)
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.1f}s")

    # Generate
    print(f"\n[5/5] Generating text (GPTQ INT8 dequantized → NPU)...")
    print(f'  Prompt: "{args.prompt}"')
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
            print(f"  Token {i+1:2d}: \"{token_text}\"  ({dt*1000:.0f}ms)", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Results
    full_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    n_tokens = len(token_times)
    total_time = sum(token_times)
    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0

    print(f"\n{'=' * 70}")
    print(f'  Output: "{full_text}"')
    print(f"{'=' * 70}")
    print(f"  Model:           GPTQ INT8 → bf16 dequantized")
    print(f"  Tokens:          {n_tokens}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Prefill:         {prefill_time*1000:.0f}ms")
    if decode_times:
        print(f"  Avg decode:      {avg_decode*1000:.0f}ms/token")
        print(f"  Decode tok/s:    {1/avg_decode:.2f}")
    print(f"  NPU dispatches:  {engine.dispatch_count}")
