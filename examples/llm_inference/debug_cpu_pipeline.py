"""Test: run llm_npu_only pipeline but with CPU matmul instead of NPU.
If output is still garbage, the bug is in the pipeline code (attention/MLP forward).
If output is correct, the bug is in NPU matmul accumulation."""
import torch, os, sys, types, importlib
os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
sys.path.append("examples")
import benchmark; benchmark.select_npu_backend()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal engine that uses CPU matmul but mirrors NPUMatmulEngine's interface
class CPUMatmulEngine:
    def __init__(self):
        self._layers = {}
        self.dispatch_count = 0
        self._compile_count = 0

    def prepare_layer(self, name, linear, max_K=4096):
        weight_t = linear.weight.data.T.contiguous().to(torch.bfloat16)
        K_full = linear.in_features
        N = linear.out_features
        if K_full > max_K:
            n_tiles = (K_full + max_K - 1) // max_K
            K_tile = K_full // n_tiles
            weight_tiles = [weight_t[i*K_tile:(i+1)*K_tile, :].contiguous() for i in range(n_tiles)]
            self._layers[name] = {"weight_tiles": weight_tiles, "k_tiles": n_tiles,
                                  "K_full": K_full, "K_tile": K_tile, "N": N}
        else:
            self._layers[name] = {"weight_tiles": [weight_t], "k_tiles": 1,
                                  "K_full": K_full, "K_tile": K_full, "N": N}

    def matmul(self, name, x):
        """CPU stand-in: mimic NPU pipeline (pad, compute, truncate, bf16)"""
        info = self._layers[name]

        # Pad to M=256 just like NPU path
        M_actual = x.shape[0]
        if M_actual < 256:
            x_pad = torch.zeros(256, info["K_full"], dtype=torch.bfloat16)
            x_pad[:M_actual] = x
        else:
            x_pad = x

        if info["k_tiles"] == 1:
            weight = info["weight_tiles"][0]
            c = torch.matmul(x_pad.float(), weight.float())  # F32 accumulation
            self.dispatch_count += 1
            return c[:M_actual].to(torch.bfloat16)
        else:
            K_tile = info["K_tile"]
            accum = torch.zeros(256, info["N"], dtype=torch.float32)
            for t, wt in enumerate(info["weight_tiles"]):
                x_slice = x_pad[:, t*K_tile:(t+1)*K_tile].contiguous()
                accum += torch.matmul(x_slice.float(), wt.float())
                self.dispatch_count += 1
            return accum[:M_actual].to(torch.bfloat16)

    def swiglu(self, gate_flat, up_flat):
        """CPU SwiGLU for testing."""
        g = gate_flat.float()
        activated = g * torch.sigmoid(g)
        self.dispatch_count += 1
        return (activated.to(gate_flat.dtype) * up_flat)

    def compile_swiglu(self, *args, **kwargs):
        pass

# Import the pipeline code
sys.path.insert(0, os.path.join("examples", "llm_inference"))
from llm_npu_only import (
    patch_model_npu, rotate_half, apply_rotary_pos_emb
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Use CPU engine to test the pipeline logic
engine = CPUMatmulEngine()

# Patch model with CPU engine
print("Patching model with CPU matmul engine (testing pipeline code)...")
dispatches = patch_model_npu(model, engine)
print(f"Dispatches per token: {dispatches}")

# Generate
prompt = "Explain what an NPU is in one sentence:"
messages = [{"role": "user", "content": prompt}]
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_input, return_tensors="pt")

print(f"\nGenerating with CPU matmul through NPU pipeline code...")
with torch.no_grad():
    generated = inputs["input_ids"].clone()
    past_kv = None
    tokens = []
    for i in range(15):
        if past_kv is None:
            out = model(input_ids=generated, use_cache=True)
        else:
            out = model(input_ids=generated[:, -1:], past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)
        tok = tokenizer.decode(next_id[0], skip_special_tokens=True)
        tokens.append(tok)
        print(f"  Token {i+1}: '{tok}'", flush=True)
        if next_id.item() == tokenizer.eos_token_id:
            break

text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nOutput: \"{text}\"")
print(f"Dispatches: {engine.dispatch_count}")
