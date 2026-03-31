#!/usr/bin/env python3
"""
NPU Chat — Multi-turn conversation on AMD Ryzen AI NPU
=======================================================
All Linear projections run as INT8×INT8→INT32 on the NPU.

Usage:
    python chat.py
    python chat.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct
"""

import torch
import triton
import triton.language as tl
import time
import os
import sys
import types
import importlib
import argparse
import io
import contextlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

torch.set_num_threads(os.cpu_count())


def suppress_stderr():
    """Context manager that suppresses stderr (XRT debug messages, etc.)."""
    return contextlib.redirect_stderr(io.StringIO())


def load_engine():
    """Import and initialize the INT8 NPU engine from llm_npu_int8."""
    # Import everything we need from the main module
    import llm_npu_int8
    return llm_npu_int8


def main():
    parser = argparse.ArgumentParser(description="NPU Chat")
    parser.add_argument("--model", "-m", type=str,
                        default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--system", "-s", type=str,
                        default="You are a helpful assistant. Keep responses concise.")
    args = parser.parse_args()

    # ── Load engine module ──
    mod = load_engine()

    print("╔══════════════════════════════════════════════════════╗")
    print("║          NPU Chat — AMD Ryzen AI (INT8)             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Loading model...", end="", flush=True)

    # ── Init NPU backend (suppress noise) ──
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
    with suppress_stderr():
        import benchmark
        benchmark.select_npu_backend()

    engine = mod.NPUInt8Engine()

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import warnings
    warnings.filterwarnings("ignore")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f" done ({load_time:.1f}s)")

    # ── Quantize & compile ──
    print("  Compiling NPU kernels...", end="", flush=True)
    t0 = time.perf_counter()
    with suppress_stderr(), contextlib.redirect_stdout(io.StringIO()):
        mod.patch_model_int8(model, engine)
    compile_time = time.perf_counter() - t0
    print(f" done ({compile_time:.1f}s)")

    # ── Warmup ──
    print("  Warmup...", end="", flush=True)
    t0 = time.perf_counter()
    warmup_msgs = [{"role": "user", "content": "Hi"}]
    warmup_input = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(warmup_input, return_tensors="pt")
    with torch.inference_mode(), suppress_stderr():
        _ = model.generate(inputs["input_ids"], max_new_tokens=1, do_sample=False)
    warmup_time = time.perf_counter() - t0
    print(f" done ({warmup_time:.1f}s)")

    print()
    print("  Ready! Type your message and press Enter.")
    print("  Commands: /clear (reset conversation), /quit (exit)")
    print()

    # ── Chat loop ──
    conversation = []
    if args.system:
        conversation.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() in ("/clear", "/reset"):
            conversation = []
            if args.system:
                conversation.append({"role": "system", "content": args.system})
            print("  (conversation cleared)\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Build chat input
        chat_input = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_input, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        # Generate
        print("NPU: ", end="", flush=True)
        generated_ids = inputs["input_ids"].clone()
        past_key_values = None
        token_times = []
        response_tokens = []

        with torch.inference_mode(), suppress_stderr():
            while True:
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
                response_tokens.append(next_token.item())

                # Stream the token
                tok_text = tokenizer.decode(
                    [next_token.item()], skip_special_tokens=True
                )
                print(tok_text, end="", flush=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Safety limit
                if len(token_times) >= 512:
                    break

        # Stats
        n_tokens = len(token_times)
        total_time = sum(token_times)
        decode_times = token_times[1:] if n_tokens > 1 else token_times
        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
        tok_s = 1.0 / avg_decode if avg_decode > 0 else 0

        print(f"\n  [{n_tokens} tokens, {tok_s:.1f} tok/s, {total_time:.1f}s]\n")

        # Add response to conversation history
        full_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
