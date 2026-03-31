#!/usr/bin/env python3
"""
NPU Chat — Multi-turn conversation on AMD Ryzen AI NPU
=======================================================
All Linear projections run as INT8×INT8→INT32 on the NPU.

Usage:
    python chat.py
    python chat.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct
"""

import os
import sys
import time
import types
import importlib
import argparse
import io
import contextlib

# Must be set before importing triton so it discovers the NPU backend
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("AMD_TRITON_NPU_OUTPUT_FORMAT", "xclbin")
# XRT headers / libs for compilation
if "XRT_DEV_DIR" not in os.environ:
    _xrt_candidates = [r"C:\projects\xrt-dev", r"C:\xrt-dev"]
    for _p in _xrt_candidates:
        if os.path.isdir(_p):
            os.environ["XRT_DEV_DIR"] = _p
            break


def _ensure_msvc_env():
    """Auto-detect MSVC and Windows SDK paths if cl.exe is not on PATH."""
    import shutil
    if shutil.which("cl"):
        return  # already available
    # Find VS 2022 MSVC
    vs_base = r"C:\Program Files\Microsoft Visual Studio\2022"
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        tools_dir = os.path.join(vs_base, edition, "VC", "Tools", "MSVC")
        if os.path.isdir(tools_dir):
            versions = sorted(os.listdir(tools_dir), reverse=True)
            if versions:
                msvc = os.path.join(tools_dir, versions[0])
                cl_dir = os.path.join(msvc, "bin", "Hostx64", "x64")
                if os.path.isfile(os.path.join(cl_dir, "cl.exe")):
                    os.environ["PATH"] = cl_dir + ";" + os.environ.get("PATH", "")
                    os.environ.setdefault("INCLUDE",
                        os.path.join(msvc, "include"))
                    os.environ.setdefault("LIB",
                        os.path.join(msvc, "lib", "x64"))
                    break
    # Find Windows SDK
    sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
    sdk_inc = os.path.join(sdk_base, "Include")
    sdk_lib = os.path.join(sdk_base, "Lib")
    if os.path.isdir(sdk_inc):
        versions = sorted(os.listdir(sdk_inc), reverse=True)
        if versions:
            v = versions[0]
            sdk_incs = ";".join(os.path.join(sdk_inc, v, d) for d in ("ucrt", "um", "shared"))
            sdk_libs = ";".join(os.path.join(sdk_lib, v, d, "x64") for d in ("ucrt", "um"))
            os.environ["INCLUDE"] = os.environ.get("INCLUDE", "") + ";" + sdk_incs
            os.environ["LIB"] = os.environ.get("LIB", "") + ";" + sdk_libs
    # DIA SDK (needed by some tools)
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        dia = os.path.join(vs_base, edition, "DIA SDK")
        if os.path.isdir(dia):
            try:
                import subprocess
                subprocess.run(["subst", "Z:", dia], capture_output=True)
            except Exception:
                pass
            break


_ensure_msvc_env()

import torch
import triton
import triton.language as tl

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
    with suppress_stderr():
        import benchmark
        benchmark.select_npu_backend()

    engine = mod.NPUInt8Engine(verbose=False)

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TQDM_DISABLE"] = "1"  # suppress loading progress bar
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    del os.environ["TQDM_DISABLE"]
    del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
    print(f" done ({load_time:.1f}s)")

    # ── Quantize & compile ──
    print("  Compiling NPU kernels...", end="", flush=True)
    t0 = time.perf_counter()
    os.environ["TRITON_NPU_QUIET"] = "1"
    with suppress_stderr():
        mod.patch_model_int8(model, engine, verbose=False)
    del os.environ["TRITON_NPU_QUIET"]
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
