from huggingface_hub import snapshot_download
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

repo_id = "QWen/QWen3-0.6B"
local_dir = f"{Path(__file__).resolve().parent.parent}/.models"
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir
)

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.bfloat16).to("cuda").eval()

tensors = {}
print("key:\tdata type:\tshape:\tdevice")
with safe_open(f"{local_dir}/model.safetensors", framework="pt", device="cuda") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        tensors[k] = t
        print(f"{k}:\t{t.dtype}:\t{t.shape}:\t{t.device}")

def get_layer(layer):
    in_norm = tensors[f"model.layers.{layer}.input_layernorm.weight"]
    q_proj = tensors[f"model.layers.{layer}.self_attn.q_proj.weight"]
    q_norm = tensors[f"model.layers.{layer}.self_attn.q_norm.weight"]
    k_proj = tensors[f"model.layers.{layer}.self_attn.k_proj.weight"]
    k_norm = tensors[f"model.layers.{layer}.self_attn.k_norm.weight"]
    v_proj = tensors[f"model.layers.{layer}.self_attn.v_proj.weight"]
    o_proj = tensors[f"model.layers.{layer}.self_attn.o_proj.weight"]
    post_norm = tensors[f"model.layers.{layer}.post_attention_layernorm.weight"]
    mlp_up = tensors[f"model.layers.{layer}.mlp.up_proj.weight"]
    mlp_gate = tensors[f"model.layers.{layer}.mlp.gate_proj.weight"]
    mlp_down = tensors[f"model.layers.{layer}.mlp.down_proj.weight"]
    return in_norm, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, post_norm, mlp_up, mlp_gate, mlp_down

def rmsnorm(x, weight, eps=1e-6):
    input_type = x.dtype
    if input_type in [torch.float16, torch.bfloat16]:
        x = x.float()

    mean_square = x.square().mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(mean_square + eps)
    return weight * x.to(input_type)

# my forward implementation
def forward(prompt):
    embeddings = tokenizer(prompt, return_tensors="pt").to("cuda")
    embedding_weight = tensors["model.embed_tokens.weight"]
    input_embeddings = embedding_weight[embeddings.input_ids]

    x = input_embeddings[0] # batch 1
    T, H = x.shape
    for i in range(28):
        in_norm, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, post_norm, mlp_up, mlp_gate, mlp_down = get_layer(i)
        h = rmsnorm(x, in_norm)
        print(q_proj.shape, q_norm.shape)
        print(k_proj.shape, k_norm.shape)

        # calculate the QKV, and reshape to GQA(2 queries head share the same key value head)
        q = (h @ q_proj.T).reshape(T, 16, 128)
        k = (h @ k_proj.T).reshape(T, 8, 128)
        v = (h @ v_proj.T).reshape(T, 8, 128)

        q = rmsnorm(q, q_norm)
        k = rmsnorm(k, k_norm)


# forward using transformers lib
def tf_forward(prompt):
    embeddings = tokenizer(prompt, return_tensors="pt").to("cuda")
    config = GenerationConfig(max_new_tokens=64, min_new_tokens=64, output_hidden_states=True)
    output = model.generate(**embeddings, generation_config=config)
    print(tokenizer.decode(output))

prompt = "This is the Lab 1 implementation: manual forwarding using safe tensors. If you read this, please just output 'Hello World!'"
tf_forward(prompt)
forward(prompt)