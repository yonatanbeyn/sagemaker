"""
gpt2_model.py — shared GPT-2 architecture for the v5 pipeline

Imported by BOTH train_v5.py (training job) and inference_v5.py (endpoint), so
the two can never drift apart. The v3 pipeline duplicated its model classes
across train.py and inference.py with a "must match exactly" comment; this
module removes that footgun.

Why the architecture is plain GPT-2 and not the modern v4 stack (RoPE, RMSNorm,
no-bias Linears): pretrained weights are only meaningful inside the exact
architecture they were trained in. Loading the real gpt2-medium checkpoint means
matching what GPT-2 actually used —

    learned absolute position embeddings (wpe), max 1024
    LayerNorm with bias
    biases on every Linear except the tied lm_head
    GELU with the tanh approximation ("gelu_new")
    fused QKV in a single c_attn projection

Module names deliberately mirror the HuggingFace state_dict layout
(`transformer.h.0.attn.c_attn.weight`) so the checkpoint drops in key-for-key.

Dependencies: torch only — not even `transformers`. load_hf_state_dict() takes
a plain dict, so the caller decides how the tensors were read (train_v5.py
reads model.safetensors directly). That keeps this module usable in the
inference container, which installs nothing but tiktoken.
"""

import torch
import torch.nn as nn

# ── Chat template ────────────────────────────────────────────────────────────
# GPT-2's BPE has no special chat tokens, so these are encoded as ordinary
# tokens. The model simply learns them as delimiters during fine-tuning.
# Training and serving MUST use identical markers or the endpoint will prompt
# the model in a format it never saw.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(user_message):
    """The exact prefix the model is fine-tuned to continue from."""
    return f"{IM_START}user\n{user_message}{IM_END}\n{IM_START}assistant\n"


# ── Architecture ─────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """
    GPT-2 multi-head causal self-attention.

    Q, K and V live in ONE fused projection (`c_attn`: n_embd → 3*n_embd), which
    is how GPT-2 was trained and why the checkpoint has no separate q/k/v
    tensors. Supports a padding mask (batched variable-length SFT examples) and
    a KV cache (fast autoregressive decoding).
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd {n_embd} not divisible by n_head {n_head}"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # fused Q, K, V
        self.c_proj = nn.Linear(n_embd, n_embd)       # output projection

    def forward(self, x, pad_mask=None, past_kv=None):
        B, T, C = x.shape

        # One matmul produces Q, K and V; split along the feature axis.
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache: prepend keys/values already computed for earlier positions.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        T_q, T_k = q.size(2), k.size(2)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask: query i may attend to keys <= i. The `diagonal` offset
        # keeps this correct when T_q < T_k (i.e. when a KV cache is in play).
        causal = torch.ones(T_q, T_k, device=x.device, dtype=torch.bool).tril(
            diagonal=T_k - T_q
        )
        scores = scores.masked_fill(~causal, float("-inf"))

        # Padding mask: (B, T_k), True at padded key positions.
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
            # A query row with every key masked would softmax to NaN and poison
            # the backward pass. Right-padding makes that impossible (position 0
            # is always real), but guard anyway — a NaN here is very hard to
            # trace back from a diverged loss curve.
            fully_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(fully_masked, 0.0)

        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.c_proj(out), (k, v)


class MLP(nn.Module):
    """
    GPT-2 feedforward block: n_embd → 4*n_embd → GELU → n_embd.

    GPT-2 used the tanh approximation of GELU ("gelu_new"). Exact GELU here
    would silently mismatch the pretrained weights — the model would still run,
    just measurably worse.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    """Pre-LayerNorm block: x + attn(ln_1(x)), then x + mlp(ln_2(x))."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, pad_mask=None, past_kv=None):
        attn_out, new_kv = self.attn(self.ln_1(x), pad_mask=pad_mask, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT2(nn.Module):
    """
    Decoder-only GPT-2.

    Submodule names match the HuggingFace state_dict layout exactly, so loading
    pretrained weights is a direct key-for-key copy (see load_hf_state_dict).
    """

    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),       # token embeddings
            wpe=nn.Embedding(n_positions, n_embd),      # learned positions
            h=nn.ModuleList([GPT2Block(n_embd, n_head) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # WEIGHT TYING: the output head reuses the token embedding table.
        # GPT-2 ships tied, so this is not an optimisation we are adding — it is
        # part of matching the checkpoint.
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, pad_mask=None, past_kvs=None, pos_offset=0):
        B, T = idx.shape
        if pos_offset + T > self.n_positions:
            raise ValueError(
                f"position {pos_offset + T} exceeds GPT-2's "
                f"{self.n_positions}-token context window"
            )

        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        h = self.transformer.wte(idx) + self.transformer.wpe(pos)[None, :, :]

        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past = past_kvs[i] if past_kvs is not None else None
            h, kv = block(h, pad_mask=pad_mask, past_kv=past)
            new_kvs.append(kv)

        h = self.transformer.ln_f(h)
        return self.lm_head(h), new_kvs


# ── Loading pretrained GPT-2 weights ─────────────────────────────────────────

# HuggingFace stores these four as Conv1D weights, shaped (in_features,
# out_features) — the TRANSPOSE of what nn.Linear expects. This is the single
# most common source of silent garbage when porting GPT-2 by hand: the square
# ones (c_proj) load without any shape error and just produce nonsense.
NEEDS_TRANSPOSE = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

# Non-parameter buffers holding the old attention mask cache. We rebuild the
# causal mask on the fly, so these are skipped.
SKIP_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def normalize_gpt2_key(key):
    """
    Map a checkpoint key onto our module layout.

    Two layouts exist in the wild and we accept both:
      - GPT2LMHeadModel.state_dict()  ->  "transformer.h.0.attn.c_attn.weight"
      - the Hub's model.safetensors   ->  "h.0.attn.c_attn.weight"
    The Hub files for gpt2/gpt2-medium/... were saved from the BASE GPT2Model,
    so they carry no "transformer." prefix (verified against the published
    gpt2-medium safetensors header: 316 tensors, none prefixed).
    """
    if key.startswith("transformer.") or key == "lm_head.weight":
        return key
    return "transformer." + key


def load_hf_state_dict(model, hf_sd):
    """
    Copy a HuggingFace GPT-2 state_dict into a GPT2 instance.

    Takes a plain dict, so this module never imports `transformers` or
    `safetensors` — the caller decides how the tensors were read.
    Returns (copied, transposed, skipped) counts for logging.
    """
    our_sd = model.state_dict()
    copied = transposed = skipped = 0
    seen = set()

    with torch.no_grad():
        for raw_key, tensor in hf_sd.items():
            if raw_key.endswith(SKIP_SUFFIXES):
                skipped += 1
                continue
            if raw_key == "lm_head.weight":
                skipped += 1          # tied to wte.weight; copied along with it
                continue

            key = normalize_gpt2_key(raw_key)
            seen.add(key)
            if key not in our_sd:
                raise KeyError(
                    f"checkpoint key '{raw_key}' (-> '{key}') has no "
                    f"counterpart in GPT2"
                )

            if key.endswith(NEEDS_TRANSPOSE):
                tensor = tensor.t()
                transposed += 1

            if our_sd[key].shape != tensor.shape:
                raise ValueError(
                    f"shape mismatch for '{key}': ours {tuple(our_sd[key].shape)} "
                    f"vs checkpoint {tuple(tensor.shape)}"
                )
            our_sd[key].copy_(tensor)
            copied += 1

    # Every parameter must have received a pretrained value. Without this a
    # silently renamed key would leave part of the model randomly initialised
    # and the failure would only show up as mediocre output.
    missing = [k for k in our_sd if k not in seen and k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"these parameters got no pretrained values: {missing}")

    # Tying must survive the copy, or the head silently becomes a second,
    # untied 50257 x n_embd matrix.
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()

    return copied, transposed, skipped


# ── Shared sampling / generation ─────────────────────────────────────────────


def filter_logits(logits, temperature=0.8, top_k=50, top_p=0.95,
                  repetition_penalty=1.0, generated_ids=None):
    """
    Apply repetition penalty, temperature, top-k and nucleus (top-p) filtering
    to a 1-D logits vector. Returns the filtered logits.
    """
    logits = logits.float().clone()

    # Repetition penalty (CTRL-style): divide positive logits, multiply
    # negative ones, so the penalty always pushes probability DOWN.
    if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
        unique_ids = torch.tensor(sorted(set(generated_ids)), device=logits.device)
        scores = logits[unique_ids]
        logits[unique_ids] = torch.where(
            scores > 0, scores / repetition_penalty, scores * repetition_penalty
        )

    logits = logits / max(temperature, 1e-6)

    if top_k:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Cumulative probability BEFORE each token, so the top-1 always survives.
        cum_before = probs.cumsum(dim=-1) - probs
        sorted_logits = sorted_logits.masked_fill(cum_before > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    return logits


@torch.no_grad()
def generate(model, enc, prompt, eos_id, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.95, repetition_penalty=1.05, device=None,
             raw=False):
    """
    Sample a reply with a KV cache: the prompt is encoded once, then each new
    token attends against cached keys/values instead of re-running the prefix.

    raw=False wraps `prompt` in the fine-tuning chat template and stops at
    <|im_end|>. raw=True continues the given text verbatim (useful for probing
    the base model before any fine-tuning has happened).

    Returns (text, num_tokens_generated, stop_reason).
    """
    model.eval()
    device = device or next(model.parameters()).device

    text = prompt if raw else build_prompt(prompt)
    prompt_ids = enc.encode(text, allowed_special={"<|endoftext|>"})

    # Leave room to actually generate something.
    max_prompt = model.n_positions - 1
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, kvs = model(x)                    # prime the cache with the prompt
    pos_offset = x.size(1)

    out_ids = []
    stop_reason = "max_tokens"

    for _ in range(max_new_tokens):
        filtered = filter_logits(
            logits[0, -1, :],
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=prompt_ids + out_ids,
        )
        probs = torch.softmax(filtered, dim=-1)

        # Degenerate distributions (all -inf / NaN) would crash multinomial.
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            probs = torch.ones_like(probs) / probs.numel()

        next_id = int(torch.multinomial(probs, 1).item())
        if next_id == eos_id:
            stop_reason = "eos"
            break

        out_ids.append(next_id)

        decoded = enc.decode(out_ids)
        if not raw and IM_END in decoded:
            return decoded.split(IM_END)[0], len(out_ids), "im_end"

        if pos_offset >= model.n_positions:
            stop_reason = "context_full"
            break

        step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, kvs = model(step_input, past_kvs=kvs, pos_offset=pos_offset)
        pos_offset += 1

    return enc.decode(out_ids), len(out_ids), stop_reason