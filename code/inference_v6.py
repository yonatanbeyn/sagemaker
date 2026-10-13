"""
inference_v6.py — SageMaker inference handler for the Qwen2.5-7B LoRA adapter.

At cold start: loads the 4-bit base model, then merges the LoRA adapter stored
in model_dir/lora_adapter/.  Subsequent requests use the merged model in memory.

Input  (JSON): {"inputs": "your question here", "parameters": {...}}
Output (JSON): {"generated_text": "...", "input_tokens": N, "output_tokens": N}
"""

import json
import os
import torch

_model = None
_tokenizer = None

MAX_NEW_TOKENS_CAP = 1024
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1   # low temperature → deterministic reasoning


def model_fn(model_dir):
    global _model, _tokenizer

    from unsloth import FastLanguageModel

    adapter_path = os.path.join(model_dir, "lora_adapter")
    meta_path = os.path.join(model_dir, "metadata.json")

    seq_len = 2048
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        seq_len = meta.get("hyperparameters", {}).get("seq_len", seq_len)
        base_model = meta.get("base_model", "Qwen/Qwen2.5-7B")
    else:
        base_model = "Qwen/Qwen2.5-7B"

    # Detect precision from hardware, same logic as train_v6.py.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = None

    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=seq_len,
        dtype=dtype,
        load_in_4bit=True,
    )

    # Load the LoRA adapter and merge into the base weights for faster inference.
    from peft import PeftModel
    _model = PeftModel.from_pretrained(_model, adapter_path)
    _model = _model.merge_and_unload()

    FastLanguageModel.for_inference(_model)

    # Load the tokenizer with the ChatML tokens injected at train time.
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return _model


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        return json.loads(request_body)
    return {"inputs": request_body.decode("utf-8")}


def predict_fn(data, model):
    prompt_text = data.get("inputs") or data.get("prompt") or ""
    params = data.get("parameters", {})

    max_new_tokens = min(
        int(params.get("max_new_tokens", DEFAULT_MAX_TOKENS)),
        MAX_NEW_TOKENS_CAP,
    )
    temperature = float(params.get("temperature", DEFAULT_TEMPERATURE))
    do_sample = temperature > 0.01

    messages = [
        {"role": "system", "content": "You are an expert math and coding assistant. Reason step-by-step."},
        {"role": "user", "content": prompt_text},
    ]
    inputs_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(inputs_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = _tokenizer.decode(generated_ids, skip_special_tokens=False)

    return {
        "generated_text": generated_text,
        "input_tokens": input_len,
        "output_tokens": len(generated_ids),
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
