import glob, json, os
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

NAMES = dict(
    ln1="input_layernorm",
    ln2="post_attention_layernorm",
    wq="self_attn.q_proj",
    wk="self_attn.k_proj",
    wv="self_attn.v_proj",
    wo="self_attn.o_proj",
    qn="self_attn.q_norm",
    kn="self_attn.k_norm",
    gate="mlp.gate_proj",
    up="mlp.up_proj",
    down="mlp.down_proj"
)

def load_weights(path, dtype=torch.bfloat16, device="cuda"):
    raw = {}
    for fn in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                if k != "lm_head.weight":
                    raw[k] = f.get_tensor(k).to(device, dtype)
    layers = [{k: raw[f"model.layers.{i}.{v}.weight"] for k, v in NAMES.items()} for i in range(28)]
    return dict(embed=raw["model.embed_tokens.weight"],
                norm=raw["model.norm.weight"],
                layers=layers)

def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)

def rope_cos_sin(pos, dtype, head_dim=128, theta=1e6):
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    f = pos.float()[:, None] * inv.to(pos.device)[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]

def mlp(x, L):
    g = F.silu(F.linear(x, L["gate"]))
    return F.linear(g * F.linear(x, L["up"]), L["down"])

def attention(x, L, cos, sin, kc=None, vc=None, start=0):
    """
    x: [T, 1024]
    L: weight of this layer
    kc: key cache
    vc: value cache,
    start: the absolute pos of the first token
    """
    T = x.shape[0]
    # F.linear(L["wq"], x)
    q, k, v = (x @ L["wq"].T).reshape([T, 16, 128]), (x @ L["wk"].T).reshape([T, 8, 128]), (x @ L["wv"].T).reshape([T, 8, 128])
    q, k = rms_norm(q, L["qn"]), rms_norm(k, L["kn"])
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    if kc is not None:
        kc[start:start+T] = k
        vc[start:start+T] = v
        k, v = kc[:start+T], vc[:start+T]

    # k, v: [S, 16, 128]
    S = k.shape[0]
    k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    # scores: [16, T, S]
    scores = q @ k.transpose(1, 2) * (128 ** -0.5)

    qpos = torch.arange(T, device=x.device) + start
    kpos = torch.arange(S, device=x.device)
    scores = scores.masked_fill(kpos[None, :] > qpos[:, None], float("-Inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = (probs @ v).transpose(0, 1).reshape([T, 2048])
    return out @ L["wo"].T

def layer(x, L, cos, sin, kc=None, vc=None, start=0):
    h = x + attention(rms_norm(x, L["ln1"]), L, cos, sin, kc, vc, start)
    return h + mlp(rms_norm(h, L["ln2"]), L)

def forward(ids, W, cache=None, start=0, last_only=False):
    """
    ids: token ids
    """
    x = W["embed"][ids]
    pos = torch.arange(start, start + len(ids), device=ids.device)
    cos, sin = rope_cos_sin(pos, x.dtype)
    for i, l in enumerate(W["layers"]):
        kc = vc = None
        if cache is not None:
            kc = cache[0][i]
            vc = cache[1][i]
        x = layer(x, l, cos, sin, kc, vc, start)
    if last_only:
        x = x[-1:]
    return F.linear(rms_norm(x, W["norm"]), W["embed"])

def generate_nocache(ids, W, n_new):
    for _ in range(n_new):
        logits = forward(ids, W, None, 0, True)
        ids = torch.cat([ids, logits[-1].argmax()[None]])
    return ids[-n_new:]

def new_cache(max_len, dtype, device="cuda"):
    shape = (28, max_len, 8, 128)
    return (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))

def generate(ids, W, n_new, max_len=4096, pick=None):
    pick = pick or (lambda x: x.argmax())
    cache = new_cache(max_len, W["embed"].dtype)
    # prefill
    logits = forward(ids, W, cache, 0, last_only=True)
    out = []
    for step in range(n_new):
        nxt = pick(logits[-1])
        out.append(nxt)
        if step < n_new - 1:
            p = len(ids) + step
            logits = forward(nxt[None], W, cache, p, last_only=True)
    return torch.stack(out)

def sample(logits, temperature=0.6, top_k=20, top_p=0.95):
    logits = logits.float() / temperature
    vals, idx = logits.topk(top_k)
    probs = F.softmax(vals, dim=-1)
    cum = probs.cumsum(dim=-1)
    probs[cum - probs > top_p] = 0
    probs = probs / probs.sum()
    return idx[torch.multinomial(probs, 1)[0]]