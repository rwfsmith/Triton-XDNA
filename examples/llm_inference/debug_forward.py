"""Debug: compare full forward pass CPU vs NPU layer by layer."""
import torch, os, sys, importlib, types
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

def compile_and_capture(M, N, K):
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    c = torch.empty(M, N, dtype=torch.float32)
    gX = M // 256; gY = N // 256
    bare_matmul[(gX, gY)](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K)
    return npu_driver._last_dispatched_module, gX, gY

def npu_matmul_2d(mod, gX, gY, x, weight_t, M_pad, N, K):
    """Matmul with padding: result = x @ weight_t"""
    M_actual = x.shape[0]
    x_pad = torch.zeros(M_pad, K, dtype=torch.bfloat16)
    x_pad[:M_actual] = x
    c = torch.empty(M_pad, N, dtype=torch.float32)
    mod.launch(gX, gY, 1, None, None, None, None,
        x_pad, weight_t, c, M_pad, N, K,
        x_pad.stride(0), x_pad.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1), 256, 256, K)
    return c[:M_actual].to(torch.bfloat16)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Check model config
cfg = model.config
print(f"Config: hidden={cfg.hidden_size}, intermediate={cfg.intermediate_size}")
print(f"  num_attention_heads={cfg.num_attention_heads}, num_key_value_heads={cfg.num_key_value_heads}")
print(f"  head_dim={cfg.hidden_size // cfg.num_attention_heads}")

# Check actual layer dimensions
l0 = model.model.layers[0]
print(f"\nLayer 0 linear dimensions:")
print(f"  q_proj: in={l0.self_attn.q_proj.in_features}, out={l0.self_attn.q_proj.out_features}")
print(f"  k_proj: in={l0.self_attn.k_proj.in_features}, out={l0.self_attn.k_proj.out_features}")
print(f"  v_proj: in={l0.self_attn.v_proj.in_features}, out={l0.self_attn.v_proj.out_features}")
print(f"  o_proj: in={l0.self_attn.o_proj.in_features}, out={l0.self_attn.o_proj.out_features}")
print(f"  gate_proj: in={l0.mlp.gate_proj.in_features}, out={l0.mlp.gate_proj.out_features}")
print(f"  up_proj: in={l0.mlp.up_proj.in_features}, out={l0.mlp.up_proj.out_features}")
print(f"  down_proj: in={l0.mlp.down_proj.in_features}, out={l0.mlp.down_proj.out_features}")

# Compile the module for (256, 2048, 2048) - the q/k/v/o shape
print("\nCompiling NPU kernels...")
mod_2048, gX_2048, gY_2048 = compile_and_capture(256, 2048, 2048)
print(f"  (256, 2048, 2048): captured={mod_2048 is not None}")

# Get a test input (single decode token)
prompt = "Explain what an NPU is in one sentence:"
messages = [{"role": "user", "content": prompt}]
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_input, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
print(f"\nPrompt seq_len = {seq_len}")

# Run CPU forward and capture intermediate states
print("\n=== CPU Forward Pass (prefill) ===")
with torch.no_grad():
    # Step through the model manually
    input_ids = inputs["input_ids"]
    
    # Embedding
    embeds = model.model.embed_tokens(input_ids)
    print(f"Embeddings: shape={embeds.shape}, mean={embeds.float().mean():.6f}")
    
    # Layer 0
    hidden = embeds
    layer0 = model.model.layers[0]
    
    # Input layernorm
    normed = layer0.input_layernorm(hidden)
    print(f"After input_layernorm: shape={normed.shape}, mean={normed.float().mean():.6f}")
    
    # CPU Q projection
    q_cpu = layer0.self_attn.q_proj(normed)
    print(f"CPU q_proj: shape={q_cpu.shape}, mean={q_cpu.float().mean():.6f}")
    
    # NPU Q projection
    x_2d = normed.reshape(-1, normed.shape[-1]).to(torch.bfloat16)
    weight_q_t = layer0.self_attn.q_proj.weight.data.T.contiguous()
    q_npu = npu_matmul_2d(mod_2048, gX_2048, gY_2048, x_2d, weight_q_t, 256, 2048, 2048)
    q_npu_3d = q_npu.view(q_cpu.shape)
    
    err = (q_npu_3d.float() - q_cpu.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        q_npu_3d.float().reshape(1, -1), q_cpu.float().reshape(1, -1))
    print(f"NPU q_proj: shape={q_npu.shape}, mean={q_npu.float().mean():.6f}")
    print(f"  Error: max={err.max():.6f}, mean={err.mean():.6f}, cosine={cos.item():.6f}")

    # Now let's test FULL layer 0 end-to-end
    print(f"\n=== Full Layer 0 CPU vs NPU comparison ===")
    
    # CPU full layer forward
    # We need position_embeddings and attention_mask
    # The model's forward method computes these. Let's use hooks to capture them.
    
    # Simply run the full model forward and capture intermediate states
    cpu_out = model(input_ids, use_cache=False, output_hidden_states=True)
    cpu_logits = cpu_out.logits
    cpu_hidden_states = cpu_out.hidden_states  # tuple of (n_layers+1) tensors
    
    print(f"CPU hidden states: {len(cpu_hidden_states)} tensors")
    for i, hs in enumerate(cpu_hidden_states):
        print(f"  Layer {i}: shape={hs.shape}, mean={hs.float().mean():.6f}, "
              f"std={hs.float().std():.6f}, max={hs.float().abs().max():.4f}")
    
    cpu_token = cpu_logits[0, -1, :].argmax().item()
    print(f"\nCPU prediction: token={cpu_token} = '{tokenizer.decode([cpu_token])}'")
    print(f"CPU top-5: {cpu_logits[0, -1, :].topk(5).indices.tolist()}")
    print(f"CPU top-5 values: {cpu_logits[0, -1, :].topk(5).values.float().tolist()}")

# Now test single matmul accuracy for MULTI-TOKEN input (prefill)
print(f"\n=== Multi-token matmul accuracy (seq_len={seq_len}) ===")
with torch.no_grad():
    # Get hidden states after layer 0 input_layernorm
    normed_0 = model.model.layers[0].input_layernorm(cpu_hidden_states[0])
    x_2d = normed_0.reshape(-1, normed_0.shape[-1]).to(torch.bfloat16)
    
    # CPU q_proj
    q_cpu_full = model.model.layers[0].self_attn.q_proj(normed_0)
    
    # NPU q_proj  
    q_npu_full = npu_matmul_2d(mod_2048, gX_2048, gY_2048, x_2d, weight_q_t, 256, 2048, 2048)
    q_npu_full_3d = q_npu_full.view(q_cpu_full.shape)
    
    err = (q_npu_full_3d.float() - q_cpu_full.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        q_npu_full_3d.float().reshape(1, -1), q_cpu_full.float().reshape(1, -1))
    print(f"q_proj full prefill ({seq_len} tokens):")
    print(f"  Error: max={err.max():.6f}, mean={err.mean():.6f}, cosine={cos.item():.6f}")
    
    # Check per-row errors
    for row in range(min(5, seq_len)):
        row_err = (q_npu_full[row].float() - q_cpu_full[0, row].float()).abs()
        print(f"  Row {row}: max_err={row_err.max():.6f}, mean_err={row_err.mean():.6f}")
    if seq_len > 5:
        row = seq_len - 1
        row_err = (q_npu_full[row].float() - q_cpu_full[0, row].float()).abs()
        print(f"  Row {row}: max_err={row_err.max():.6f}, mean_err={row_err.mean():.6f}")

print("\nDone.")
