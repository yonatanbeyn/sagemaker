"""
inference_v5.py — SageMaker Endpoint serving script for the v5 GPT-2 model

SageMaker calls four functions when hosting a model:
  model_fn(model_dir)                   → load model from /opt/ml/model/
  input_fn(request_body, content_type)  → parse the incoming HTTP request
  predict_fn(input_data, model)         → run generation, return a result dict
  output_fn(prediction, accept)         → serialise to the HTTP response

The endpoint receives POST requests like:
  curl -X POST https://<endpoint>/invocations \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "Explain multi-head attention", "max_tokens": 150}'

Unlike v3's inference.py, this file does NOT redefine the architecture. It
imports gpt2_model.py, which train_v5.py bundles into model.tar.gz alongside
this file — so serving and training provably share one definition, including
the chat template the model was fine-tuned on.

Latency note: on a CPU endpoint (ml.m5.xlarge) expect roughly 10-20 tokens/sec
through 24 layers, so a 150-token reply takes ~10s. SageMaker's default
invocation timeout is 60s, which is why max_tokens is capped below.
"""

import json
import os
import sys

import torch

# The SageMaker inference container adds /opt/ml/model/code to sys.path, but be
# explicit — it makes this file runnable locally against a model dir too.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# tiktoken normally downloads its BPE files on first use. Endpoints in a
# no-egress VPC would hang or crash on that, so train_v5.py bundles the cache
# and we point tiktoken at it BEFORE the encoder is constructed.
_TIKTOKEN_CACHE = os.path.join(_HERE, "tiktoken_cache")
if os.path.isdir(_TIKTOKEN_CACHE) and os.listdir(_TIKTOKEN_CACHE):
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", _TIKTOKEN_CACHE)

import tiktoken  # noqa: E402  (must follow the cache-dir setup above)

from gpt2_model import GPT2, generate  # noqa: E402

# ── Tokenizer (loaded once at container startup) ─────────────────────────────
enc = tiktoken.get_encoding("gpt2")

# Guardrails. A CPU endpoint generating unbounded tokens will blow through the
# 60s invocation timeout and return a confusing 424 to the caller.
MAX_TOKENS_CAP = 512
DEFAULT_MAX_TOKENS = 150


# ── SageMaker hook: load model ───────────────────────────────────────────────
def model_fn(model_dir):
    """
    Called once when the endpoint container starts.

    Reads metadata.json for the architecture, rebuilds GPT2, then loads the
    fine-tuned model.pt weights. Returns a dict so predict_fn can reach the
    model, its metadata and the device.
    """
    print(f"[model_fn] Loading model from {model_dir}")

    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    arch = meta["architecture"]
    vocab_size = meta["vocab_size"]
    print(f"[model_fn] Architecture: vocab={vocab_size} "
          f"n_layer={arch['n_layer']} n_embd={arch['n_embd']} "
          f"n_head={arch['n_head']} n_positions={arch['n_positions']} "
          f"({arch['total_params']:,} params)")
    print(f"[model_fn] Base checkpoint: {meta.get('base_checkpoint')} — "
          f"{meta.get('training_mode')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_fn] Using device: {device}")

    model = GPT2(
        vocab_size=vocab_size,
        n_positions=arch["n_positions"],
        n_embd=arch["n_embd"],
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
    )

    weights_path = os.path.join(model_dir, "model.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[model_fn] Weights loaded, model.eval() set.")

    return {"model": model, "meta": meta, "device": device}


# ── SageMaker hook: parse request ────────────────────────────────────────────
def input_fn(request_body, content_type="application/json"):
    """
    Parse the incoming HTTP request body.

    Expected JSON:
      {
        "prompt":             "Explain multi-head attention",  (required)
        "max_tokens":         150,    (optional)
        "temperature":        0.8,    (optional)
        "top_k":              50,     (optional, 0 disables)
        "top_p":              0.95,   (optional, 1.0 disables)
        "repetition_penalty": 1.05,   (optional, 1.0 disables)
        "raw":                false   (optional — true skips the chat template
                                       and continues the prompt verbatim)
      }
    """
    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {content_type}. Use application/json."
        )

    data = json.loads(request_body)

    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' is required and must be a non-empty string")

    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    if max_tokens > MAX_TOKENS_CAP:
        print(f"[input_fn] max_tokens {max_tokens} capped to {MAX_TOKENS_CAP}")
        max_tokens = MAX_TOKENS_CAP

    return {
        "prompt": prompt,
        "max_tokens": max(1, max_tokens),
        "temperature": float(data.get("temperature", 0.8)),
        "top_k": int(data.get("top_k", 50)),
        "top_p": float(data.get("top_p", 0.95)),
        "repetition_penalty": float(data.get("repetition_penalty", 1.05)),
        "raw": bool(data.get("raw", False)),
    }


# ── SageMaker hook: run inference ────────────────────────────────────────────
def predict_fn(input_data, model_dict):
    """Generate a reply. Shares gpt2_model.generate with the training smoke test."""
    model = model_dict["model"]
    device = model_dict["device"]
    eos_id = model_dict["meta"]["eos_token_id"]

    print(f"[predict_fn] prompt={input_data['prompt'][:80]!r} "
          f"max_tokens={input_data['max_tokens']} "
          f"temperature={input_data['temperature']} raw={input_data['raw']}")

    text, n_tokens, stop_reason = generate(
        model, enc, input_data["prompt"],
        eos_id=eos_id,
        max_new_tokens=input_data["max_tokens"],
        temperature=input_data["temperature"],
        top_k=input_data["top_k"],
        top_p=input_data["top_p"],
        repetition_penalty=input_data["repetition_penalty"],
        device=device,
        raw=input_data["raw"],
    )

    print(f"[predict_fn] Generated {n_tokens} tokens (stop_reason={stop_reason})")

    return {
        "prompt": input_data["prompt"],
        "generated_text": text,
        "tokens_generated": n_tokens,
        "stop_reason": stop_reason,
        "temperature": input_data["temperature"],
    }


# ── SageMaker hook: serialise response ───────────────────────────────────────
def output_fn(prediction, accept="application/json"):
    if accept in ("application/json", "*/*", None, ""):
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")


# ── Local test ───────────────────────────────────────────────────────────────
# Run against a local model dir without SageMaker:
#   python inference_v5.py ./model_output
if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./model_output"

    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        print(f"No model found at {model_dir}. Run train_v5.py first.")
        sys.exit(1)

    model_dict = model_fn(model_dir)

    for prompt in [
        "Explain how multi-head attention distributes work across heads.",
        "Design a resilient rate limiter for a distributed API gateway.",
    ]:
        inp = input_fn(json.dumps({"prompt": prompt, "max_tokens": 120}))
        body, _ = output_fn(predict_fn(inp, model_dict))
        print("\n" + "─" * 70)
        print(f"USER      : {prompt}")
        print(f"ASSISTANT : {json.loads(body)['generated_text']}")