"""
inference.py — SageMaker Endpoint serving script

SageMaker calls four functions when hosting a model:
  model_fn(model_dir)             → load model from /opt/ml/model/
  input_fn(request_body, content_type) → parse incoming HTTP request
  predict_fn(input_data, model)   → run forward pass, return result
  output_fn(prediction, accept)   → serialise prediction to HTTP response

The endpoint receives POST requests like:
  curl -X POST https://<endpoint>/invocations \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "The attention mechanism", "max_tokens": 80, "temperature": 0.8}'

This file uses the same model architecture as train.py — weights are loaded
from the .pt checkpoint saved at the end of training.
"""

import os
import json
import torch
import torch.nn as nn
import tiktoken


# ── Tokenizer (loaded once at startup) ───────────────────────────────────────
enc    = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token   # 50256


# ── Model classes — must match train.py exactly ──────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x, pad_mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / self.scale
        T      = x.size(1)
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~causal, float('-inf'))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
        return torch.softmax(scores, dim=-1) @ V


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn  = SelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_layers):
        super().__init__()
        self.seq_len     = seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(seq_len,    embed_dim)
        self.blocks      = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, pad_mask=None):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.token_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)
        return self.head(x)


# ── SageMaker hook: load model ────────────────────────────────────────────────
def model_fn(model_dir):
    """
    Called once when the endpoint container starts.
    Loads metadata.json to get architecture params, then loads model.pt weights.

    Returns a dict so predict_fn can access both model and metadata.
    """
    print(f"[model_fn] Loading model from {model_dir}")

    # Load metadata — tells us seq_len, embed_dim, num_layers
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    hp         = meta["hyperparameters"]
    vocab_size = meta["vocab_size"]
    seq_len    = hp["seq_len"]
    embed_dim  = hp["embed_dim"]
    num_layers = hp["num_layers"]

    print(f"[model_fn] Architecture: vocab={vocab_size}, seq_len={seq_len}, "
          f"embed_dim={embed_dim}, num_layers={num_layers}")

    # Build model with correct architecture
    model = TinyTransformer(vocab_size, seq_len, embed_dim, num_layers)

    # Load trained weights
    weights_path = os.path.join(model_dir, "model.pt")
    state_dict   = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[model_fn] Weights loaded. model.eval() set.")

    return {"model": model, "meta": meta}


# ── SageMaker hook: parse request ────────────────────────────────────────────
def input_fn(request_body, content_type="application/json"):
    """
    Parse the incoming HTTP request body.

    Expected JSON:
      {
        "prompt":      "The attention mechanism",  (required)
        "max_tokens":  80,                          (optional, default 80)
        "temperature": 0.8                          (optional, default 0.8)
      }
    """
    if content_type == "application/json":
        data = json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    return {
        "prompt":      data.get("prompt", ""),
        "max_tokens":  data.get("max_tokens",  80),
        "temperature": data.get("temperature", 0.8),
    }


# ── SageMaker hook: run inference ────────────────────────────────────────────
def predict_fn(input_data, model_dict):
    """
    Generate text from a prompt.

    input_data  : dict with prompt, max_tokens, temperature (from input_fn)
    model_dict  : dict with model and meta (from model_fn)
    """
    model       = model_dict["model"]
    meta        = model_dict["meta"]
    SEQ_LEN     = meta["hyperparameters"]["seq_len"]
    vocab_size  = meta["vocab_size"]

    prompt      = input_data["prompt"]
    max_tokens  = input_data["max_tokens"]
    temperature = input_data["temperature"]

    print(f"[predict_fn] prompt='{prompt}' max_tokens={max_tokens} temperature={temperature}")

    # Tokenize prompt
    idx        = enc.encode(prompt)
    result_ids = list(idx)

    with torch.no_grad():
        for step in range(max_tokens):

            context    = result_ids[-SEQ_LEN:]
            actual_len = len(context)
            pad_count  = SEQ_LEN - actual_len

            if pad_count > 0:
                context = [EOS_ID] * pad_count + context

            x = torch.tensor([context])

            logits   = model(x, pad_mask=None)
            logits_t = logits[0, -1] / temperature
            logits_t = torch.clamp(logits_t, min=-50, max=50)
            probs    = torch.softmax(logits_t, dim=0)

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones(vocab_size) / vocab_size

            next_id = torch.multinomial(probs, 1).item()

            if next_id == EOS_ID:
                print(f"[predict_fn] EOS at step {step + 1}")
                break

            result_ids.append(next_id)

    generated_text = enc.decode(result_ids)
    tokens_generated = len(result_ids) - len(idx)

    print(f"[predict_fn] Generated {tokens_generated} tokens")

    return {
        "prompt":           prompt,
        "generated_text":   generated_text,
        "tokens_generated": tokens_generated,
        "temperature":      temperature,
    }


# ── SageMaker hook: serialise response ───────────────────────────────────────
def output_fn(prediction, accept="application/json"):
    """
    Serialise the prediction dict to JSON for the HTTP response.
    """
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")


# ── Local test ────────────────────────────────────────────────────────────────
# Run this file directly to test without SageMaker:
#   python inference.py
if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./model_output"

    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        print(f"No model found at {model_dir}. Run train.py first.")
        sys.exit(1)

    model_dict = model_fn(model_dir)

    for prompt, temp in [
        ("The attention mechanism", 0.8),
        ("machine learning is",    0.5),
        ("A transformer is",       0.8),
    ]:
        inp    = input_fn(json.dumps({"prompt": prompt, "temperature": temp}))
        result = predict_fn(inp, model_dict)
        body, _ = output_fn(result)
        print(json.loads(body)["generated_text"])
        print()