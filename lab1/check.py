import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen3 import *
from pathlib import Path

dt = torch.bfloat16
path = snapshot_download("Qwen/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained(path)
ref = AutoModelForCausalLM.from_pretrained(
    path, dtype=dt, attn_implementation="eager").to("cuda").eval()
W = load_weights(path, dt)

# name -> [input_dim, output_dim]
acts = {}
def save(name):
    def hook(mod, args, kwargs, out):
        x = args[0] if args else kwargs["hidden_states"]
        y = out[0] if isinstance(out, tuple) else out
        acts[name] = (x[0], y[0])
    return hook

m = ref.model
for i, lyr in enumerate(m.layers):
    lyr.register_forward_hook(save(f"layer{i}"), with_kwargs=True)

l0 = m.layers[0]
for name, mod in [("ln1", l0.input_layernorm), ("mlp", l0.mlp),
                   ("qnorm", l0.self_attn.q_norm), ("attn", l0.self_attn)]:
    mod.register_forward_hook(save(name), with_kwargs=True)

text = open(os.path.join(Path(__file__).resolve().parent, "sample.txt")).read()
ids = tok(text, return_tensors="pt").input_ids[0, :512].cuda()
print(ids.shape)
# ids[None]: add a dimension to ids -> [1, original_dim]
ref_logits = ref(ids[None]).logits[0].float()
print(ref_logits.shape)

def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max()).item()

L0 = W["layers"][0]
pos = torch.arange(len(ids), device="cuda")
print(pos)

cos, sin = rope_cos_sin(pos, dt)
print(f"cos={cos.shape}, sin={sin.shape}")
x, y = acts["ln1"]
print("rms_norm ", rel(rms_norm(x, L0["ln1"]), y))
x, y = acts["qnorm"]
print("q_norm ", rel(rms_norm(x, L0["qn"]), y))
x, y = acts["mlp"];   print("mlp      ", rel(mlp(x, L0), y))
c, s = m.rotary_emb(torch.zeros(1, dtype=dt, device="cuda"), pos[None])
print("rope     ", rel(cos, c[0]), rel(sin, s[0]))

x, y = acts["attn"]
print("layer0 attention ", rel(attention(x, L0, cos, sin), y))

x = W["embed"][ids]
for i, L in enumerate(W["layers"]):
    x = layer(x, L, cos, sin)
    print(f"layer{i:2d}  {rel(x, acts[f'layer{i}'][1]):.1e}")
logits = forward(ids, W).float()
print(f"logits   {rel(logits, ref_logits):.1e}")
same = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()
print(f"top-1 一致率 {same.item():.2%}")

p = tok("The history of the transistor begins",
        return_tensors="pt").input_ids[0].cuda()
mine = generate(p, W, 64)
nocache = generate_nocache(p, W, 64)
hf = ref.generate(p[None], attention_mask=torch.ones_like(p[None]),
                  max_new_tokens=64, min_new_tokens=64,
                  do_sample=False)[0, len(p):]
print("有 cache == 无 cache:", torch.equal(mine, nocache))
print("有 cache == HF      :", torch.equal(mine, hf))
print(tok.decode(mine))

logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
cnt = torch.zeros(5)
for _ in range(20000):
    cnt[sample(logits, 1.0, 5, 0.9)] += 1
print(cnt / cnt.sum())