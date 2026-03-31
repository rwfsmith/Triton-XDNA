"""Debug: compare NPU vs CPU layer-by-layer."""
import torch, os, sys, importlib
os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join("examples", "matmul", "transform_aie2p.mlir")
sys.path.append("examples")
import benchmark; benchmark.select_npu_backend()

from transformers import AutoModelForCausalLM, AutoTokenizer
import triton, triton.language as tl

npu_driver = importlib.import_module("triton.backends.amd_triton_npu.driver")

@triton.jit
def bare_matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_block = tl.dot(a_block, b_block)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)

def npu_matmul(x, weight_t, M_pad, N, K):
    """Run matmul on NPU: result = x @ weight_t"""
    M_actual = x.shape[0]
    x_pad = torch.zeros(M_pad, K, dtype=torch.bfloat16)
    x_pad[:M_actual] = x
    c = torch.empty(M_pad, N, dtype=torch.float32)
    gX = M_pad // 256
    gY = N // 256
    mod = npu_driver._last_dispatched_module
    mod.launch(gX, gY, 1, None, None, None, None,
        x_pad, weight_t, c, M_pad, N, K,
        x_pad.stride(0), x_pad.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        256, 256, K)
    return c[:M_actual].to(torch.bfloat16)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Compile shapes we need (just for the module capture)
print("Compiling NPU kernels...")
for shape in [(256, 2048, 2048)]:
    M, N, K = shape
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
    bare_matmul[(M//256, N//256)](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K)
    mod_2048 = npu_driver._last_dispatched_module
    print(f"  {shape}: captured={mod_2048 is not None}")

# Test with decode (single token)
prompt = "Explain what an NPU is in one sentence:"
messages = [{"role": "user", "content": prompt}]
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_input, return_tensors="pt")

with torch.no_grad():
    # CPU reference for first token
    cpu_out = model(inputs["input_ids"])
    cpu_logits = cpu_out.logits[0, -1, :]
    cpu_token = cpu_logits.argmax().item()
    cpu_text = tokenizer.decode([cpu_token])
    print(f"\nCPU first token: {cpu_token} = '{cpu_text}'")

    # Now compare q_proj for decode (single token after prefill)
    # First do the full prefill
    cpu_prefill = model(inputs["input_ids"], use_cache=True)
    past_kv = cpu_prefill.past_key_values
    first_token_id = cpu_logits.argmax().unsqueeze(0).unsqueeze(0)

    # Get the hidden state for the first decode step
    # We need to trace through the model manually
    embeddings = model.model.embed_tokens(first_token_id)
    print(f"\nDecode embedding: shape={embeddings.shape}, mean={embeddings.float().mean():.6f}")

    # Layer 0 input layernorm
    norm_out = model.model.layers[0].input_layernorm(embeddings)
    x_2d = norm_out[0].to(torch.bfloat16)  # [1, 2048]

    # CPU q_proj
    q_cpu = model.model.layers[0].self_attn.q_proj(norm_out[0])
    print(f"CPU q_proj: shape={q_cpu.shape}, mean={q_cpu.float().mean():.6f}")

    # NPU q_proj
    weight_t = model.model.layers[0].self_attn.q_proj.weight.data.T.contiguous()
    q_npu = npu_matmul(x_2d, weight_t, 256, 2048, 2048)
    print(f"NPU q_proj: shape={q_npu.shape}, mean={q_npu.float().mean():.6f}")

    err = (q_npu.float() - q_cpu.float()).abs()
    print(f"Error: max={err.max():.6f}, mean={err.mean():.6f}")

    # Check cosine similarity (overall direction)
    cos_sim = torch.nn.functional.cosine_similarity(
        q_npu.float().view(1, -1), q_cpu.float().view(1, -1)
    )
    print(f"Cosine similarity: {cos_sim.item():.6f}")
