"""
train_v6.py — SageMaker Training Job entry point (Qwen2.5-7B Base + reasoning SFT)

How this differs from train_v5.py:
  v5: 355M GPT-2 medium SFT on ansulev/claude-mythos-distilled-25k
      Four compounding failures — duplicate answers, boilerplate, memorisation,
      truncated stop tokens — made the model recite a constant and ignore prompts.
  v6: Qwen/Qwen2.5-7B base SFT on Open-Thoughts/Open-Thoughts-Dataset
      (real chain-of-thought traces: math, code, logic puzzles).
      Uses Unsloth + QLoRA (4-bit) so 7B fits in 24GB A10G VRAM.
      Loss is masked to assistant tokens only via DataCollatorForCompletionOnlyLM.

Key architectural improvements over v5:
  - Model     : Qwen2.5-7B base (~7B params, RoPE, GQA, SwiGLU, 128k native ctx)
                vs GPT-2 medium (355M, learned absolute PE, no GQA, GELU, 1024 ctx)
  - Tokenizer : Qwen's BPE has <|im_start|>/<|im_end|> as SINGLE native tokens.
                GPT-2 splits them into 5-7 subwords, so the model never learned
                when to stop — fixed for free here.
  - Dataset   : Open-Thoughts has real reasoning chains (problem→<thought>→answer).
                Mythos was template-filled synthetic data with cross-contaminated
                answers (V8 JIT answer for supply chain question, etc.).
  - PEFT      : QLoRA (r=16, 4-bit NF4) keeps the base weights frozen; only the
                adapter is updated (~50M params). This prevents catastrophic
                forgetting of the pretrained reasoning knowledge and allows the
                run to fit on a single A10G GPU.
  - Collator  : DataCollatorForCompletionOnlyLM masks the user/system turns from
                the loss, so gradients flow only through assistant completions.
                This is what v5 implemented via manual -100 labels; here TRL
                handles it correctly against Qwen's native ChatML template.

SageMaker environment variables:
  SM_CHANNEL_TRAINING  → /opt/ml/input/data/training/   (optional JSONL override)
  SM_MODEL_DIR         → /opt/ml/model/                 (LoRA adapter + metadata)
  SM_OUTPUT_DATA_DIR   → /opt/ml/output/data/           (metrics, checkpoints)

Instance recommendation: ml.g5.2xlarge (24GB A10G, bfloat16 native).
For very long CoT chains (>2048 tokens) use ml.g5.4xlarge and raise --seq-len.
"""

import subprocess
import sys

# ── Torch ecosystem bootstrap ────────────────────────────────────────────────
# Must run before ANY torch import. The DLC pre-installs torchvision/torchaudio
# compiled for its bundled torch version. Unsloth upgrades torch, which breaks
# those native binaries (.so undefined symbol). Force-reinstalling all three
# from the PyTorch cu121 wheel index ensures they match.
def _bootstrap_torch():
    print("[BOOTSTRAP] Reinstalling torch ecosystem for version consistency ...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet",
         "torch", "torchvision", "torchaudio",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[BOOTSTRAP] pip stderr: {r.stderr[-1000:]}")
    else:
        print("[BOOTSTRAP] torch ecosystem reinstalled OK.")

_bootstrap_torch()

import argparse
import contextlib
import glob
import json
import math
import os
import random
import shutil
import time

import torch

# ── Pre-flight import check ──────────────────────────────────────────────────
# Validates every third-party dependency before any real work starts.
# Fails immediately with a clear message instead of crashing mid-training.
def _check_imports():
    required = {
        "numpy":           "numpy",
        "torch":           "torch",
        "torchvision":     "torchvision",
        "transformers":    "transformers",
        "datasets":        "datasets",
        "peft":            "peft",
        "accelerate":      "accelerate",
        "bitsandbytes":    "bitsandbytes",
        "trl":             "trl",
        "unsloth":         "unsloth",
        "torchaudio":      "torchaudio",
    }
    failed = []
    for label, module in required.items():
        try:
            __import__(module)
        except ImportError as e:
            failed.append(f"  ✗ {label}: {e}")
    if failed:
        print("\n[PRE-FLIGHT] Import check FAILED — missing packages:\n" + "\n".join(failed))
        print("\nInstalled packages:")
        os.system(f"{sys.executable} -m pip list --format=columns 2>/dev/null | grep -E 'torch|trl|unsloth|numpy|peft|bitsandbytes|transformers|datasets|torchvision|accelerate'")
        sys.exit(1)

    # Print resolved versions for the training log
    import importlib.metadata as meta
    print("[PRE-FLIGHT] All imports OK. Key package versions:")
    for pkg in ["torch", "torchvision", "torchaudio", "numpy", "transformers", "trl", "unsloth", "peft", "bitsandbytes", "datasets", "accelerate"]:
        try:
            print(f"  {pkg}=={meta.version(pkg)}")
        except meta.PackageNotFoundError:
            print(f"  {pkg}==<not found via metadata>")

_check_imports()

# ── SageMaker environment ────────────────────────────────────────────────────
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "./data")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "./model_output")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")

os.makedirs(SM_MODEL_DIR, exist_ok=True)
os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B",
                    help="HuggingFace model id. Must be a BASE (not Instruct) model.")
parser.add_argument("--dataset", type=str, default="openai/gsm8k",
                    help="HF dataset id with real CoT traces.")
parser.add_argument("--dataset-split", type=str, default="train",
                    help="Dataset slice. GSM8K train split has ~7.5k examples.")
# 2048 captures ~85% of NuminaMath-CoT traces on a 24GB A10G.
# Raise to 4096 on ml.g5.4xlarge (or with gradient_accumulation to compensate).
# Do NOT set above 4096 with 4-bit QLoRA on a single A10G — OOM during backprop.
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=4,
                    help="Per-device micro-batch. Effective batch = batch-size × grad-accum.")
parser.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps. Effective batch size = 16 by default.")
# 200 steps default: more than the PDF's 100 because the base model starts with
# zero instruct awareness and the cold-start loss spike takes ~20 steps to decay.
# Watch the val_loss curve — if it is still descending at step 200, raise to 300.
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--min-lr-frac", type=float, default=0.1)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--grad-clip", type=float, default=1.0)
# LoRA rank. 16 is the sweet spot for structural alignment on a 7B model —
# high enough to learn the ChatML format and CoT style, low enough to stay
# in memory. Raise to 32 if val loss plateaus early.
parser.add_argument("--lora-rank", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=16)
# Precision detection. A10G supports bfloat16 natively (Ampere).
# T4/V100 (Volta/Turing) must fall back to fp16 — detected automatically
# by checking torch.cuda.is_bf16_supported(). Never rely on the instance type
# name; always check the hardware capability at runtime.
parser.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"])
parser.add_argument("--val-fraction", type=float, default=0.05)
parser.add_argument("--eval-interval", type=int, default=50)
parser.add_argument("--eval-batches", type=int, default=10)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

print("=" * 68)
print("  SageMaker Training Job — v6 Qwen2.5-7B Base + CoT SFT (QLoRA)")
print("=" * 68)
print(f"  base_model        : {args.base_model}")
print(f"  dataset           : {args.dataset}  split={args.dataset_split}")
print(f"  seq_len           : {args.seq_len}")
print(f"  batch_size        : {args.batch_size} × {args.grad_accum} accum "
      f"= {args.batch_size * args.grad_accum} examples/step")
print(f"  steps             : {args.steps}")
print(f"  lr                : {args.lr} (warmup {args.warmup_steps} → cosine to "
      f"{args.lr * args.min_lr_frac:g})")
print(f"  lora              : r={args.lora_rank} alpha={args.lora_alpha}")

# ── Device & precision ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.precision == "auto":
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        # Ampere (A10G, A100) and newer: bfloat16 is stable and requires no
        # loss scaling. This is what the ml.g5 family uses.
        precision = "bf16"
        dtype = torch.bfloat16
    elif device.type == "cuda":
        # Volta (V100) or Turing (T4): bfloat16 hardware is absent.
        # fp16 needs a GradScaler to prevent small gradients flushing to zero.
        precision = "fp16"
        dtype = torch.float16
    else:
        precision = "fp32"
        dtype = None
else:
    precision = args.precision
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

print(f"  device            : {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"  precision         : {precision}")

# ── 1. Load Qwen2.5-7B base with Unsloth + 4-bit QLoRA ──────────────────────
print("\n[1/5] Loading Qwen2.5-7B base model with 4-bit quantization ...")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=args.seq_len,
    dtype=dtype,           # None lets unsloth auto-detect; we pass explicitly
    load_in_4bit=True,     # NF4 quantization — mandatory for 7B on 24GB VRAM
)

# Manually inject ChatML special tokens into the base tokenizer.
# The Instruct variant has these configured already; the BASE model does not.
# Without this, apply_chat_template() fails and the model has no concept of
# <|im_start|>/<|im_end|> turn boundaries.
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Set the ChatML template so apply_chat_template() produces the correct format:
# <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
)

total_base_params = sum(p.numel() for p in model.parameters())
print(f"  Base model loaded. Parameters: {total_base_params / 1e9:.2f}B")

# Wrap with LoRA adapters. Only adapter weights (r=16 per layer) are trained;
# the 4-bit base weights stay frozen. This is why the run fits in 24GB.
model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
    bias="none",
    # Unsloth's gradient checkpointing recalculates activations on the backward
    # pass instead of caching them — prevents OOM on long CoT sequences where
    # the activation cache alone would exceed 24GB.
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  LoRA adapter params (trainable): {trainable_params / 1e6:.1f}M "
      f"({100 * trainable_params / total_base_params:.2f}% of base)")

# ── 2. Load the Open-Thoughts dataset ────────────────────────────────────────
print(f"\n[2/5] Loading reasoning dataset '{args.dataset}' ...")

from datasets import load_dataset

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except ImportError:
        import torch
        from transformers import default_data_collator

        class DataCollatorForCompletionOnlyLM:
            """Inline fallback — masks all tokens before the response template."""
            def __init__(self, response_template, tokenizer, **_):
                self.tpl_ids = tokenizer.encode(response_template, add_special_tokens=False)
                self.tokenizer = tokenizer

            def __call__(self, features):
                batch = default_data_collator(features)
                labels = batch["input_ids"].clone()
                tpl = self.tpl_ids
                for i in range(len(labels)):
                    seq = labels[i].tolist()
                    start = None
                    for j in range(len(seq) - len(tpl) + 1):
                        if seq[j: j + len(tpl)] == tpl:
                            start = j
                    labels[i, : (start + len(tpl)) if start is not None else len(seq)] = -100
                labels[batch.get("attention_mask", torch.ones_like(labels)) == 0] = -100
                batch["labels"] = labels
                return batch

raw_rows = []

# Check if training channel has JSONL overrides first
if os.path.isdir(SM_CHANNEL_TRAINING):
    paths = sorted(
        glob.glob(os.path.join(SM_CHANNEL_TRAINING, "**", "*.jsonl"), recursive=True)
        + glob.glob(os.path.join(SM_CHANNEL_TRAINING, "**", "*.json"), recursive=True)
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
            else:
                payload = json.load(f)
                raw_rows.extend(payload if isinstance(payload, list) else [payload])
        print(f"  Read {path} → {len(raw_rows):,} rows so far")

if not raw_rows:
    print(f"  No JSONL in channel — downloading from HF Hub ...")
    hf_dataset = load_dataset(args.dataset, split=args.dataset_split)
    raw_rows = list(hf_dataset)

print(f"  Loaded {len(raw_rows):,} rows")

# ── 3. Format into ChatML + apply loss masking ───────────────────────────────
print(f"\n[3/5] Formatting conversations with ChatML template ...")

SYSTEM_PROMPT = "You are an expert math and coding assistant. Reason step-by-step."


def extract_messages(row):
    """
    Normalise a dataset row to a messages list.

    Supported schemas (tried in order):
      GSM8K          : {"question": str, "answer": str}
      problem/sol    : {"problem": str, "solution": str}
      Open-Thoughts  : {"system": str, "conversations": [{"from": "human"|"gpt", "value": str}]}
      Generic chat   : {"messages": [{"role": str, "content": str}]}
    """
    # GSM8K: {"question": str, "answer": str}
    if "question" in row and "answer" in row:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            return []
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    # NuminaMath-CoT and similar problem/solution datasets
    if "problem" in row and "solution" in row:
        problem = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            return []
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
    # Open-Thoughts / ShareGPT conversation format
    if "conversations" in row:
        system = row.get("system", SYSTEM_PROMPT) or SYSTEM_PROMPT
        msgs = [{"role": "system", "content": system}]
        for turn in row["conversations"]:
            role_raw = turn.get("from", "user")
            role = "user" if role_raw in ("human", "user") else "assistant"
            content = turn.get("value", "")
            if content:
                msgs.append({"role": role, "content": content})
        return msgs
    # Generic messages format
    if "messages" in row:
        return row["messages"]
    return []


def format_example(row):
    msgs = extract_messages(row)
    if not msgs:
        return None
    # apply_chat_template handles the <|im_start|>role\n...<|im_end|>\n structure
    # correctly for Qwen's vocabulary. Manual string construction was the v5 approach
    # and caused the split-token problem with GPT-2's BPE.
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}


formatted, dropped = [], 0
for row in raw_rows:
    result = format_example(row)
    if result is None:
        dropped += 1
        continue
    formatted.append(result)

print(f"  Formatted : {len(formatted):,} examples  (dropped {dropped:,} empty rows)")

from datasets import Dataset as HFDataset

hf_data = HFDataset.from_list(formatted)

# Train / val split
n_val = max(1, int(len(hf_data) * args.val_fraction)) if len(hf_data) > 20 else 0
hf_data = hf_data.shuffle(seed=args.seed)
train_data = hf_data.select(range(n_val, len(hf_data)))
val_data = hf_data.select(range(n_val)) if n_val > 0 else None
print(f"  Split     : {len(train_data):,} train / {n_val:,} val")

effective_batch = args.batch_size * args.grad_accum
planned_epochs = args.steps * effective_batch / max(len(train_data), 1)
print(f"  Coverage  : {args.steps} steps × {effective_batch} = "
      f"{args.steps * effective_batch:,} examples seen = {planned_epochs:.2f} epochs")
if planned_epochs < 0.8:
    print(f"  NOTE: less than one full pass. Raise --steps to at least "
          f"{math.ceil(len(train_data) / effective_batch)} for one epoch.")
elif planned_epochs > 3:
    print(f"  NOTE: {planned_epochs:.1f} epochs — watch for memorisation (loss → 0).")

# DataCollatorForCompletionOnlyLM masks everything BEFORE the assistant's
# response_template from the loss. This is the canonical TRL way to do
# completion-only SFT and is more robust than manual -100 labels.
# Open-Thoughts uses 'gpt' as the assistant label internally, but after
# apply_chat_template it becomes <|im_start|>assistant\n — so we mask on that.
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# ── 4. SFT training ──────────────────────────────────────────────────────────
print(f"\n[4/5] Fine-tuning for {args.steps} steps ...")

from transformers import TrainingArguments
from trl import SFTTrainer

checkpoint_dir = os.path.join(SM_OUTPUT_DATA_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    warmup_steps=args.warmup_steps,
    max_steps=args.steps,
    learning_rate=args.lr,
    # bf16/fp16 is set based on runtime hardware detection above, not the
    # instance type name. A10G → bf16. T4/V100 → fp16. CPU → fp32.
    bf16=(precision == "bf16"),
    fp16=(precision == "fp16"),
    logging_steps=args.log_interval,
    optim="adamw_8bit",         # 8-bit Adam cuts optimizer state memory in half
    weight_decay=args.weight_decay,
    lr_scheduler_type="cosine",
    seed=args.seed,
    output_dir=checkpoint_dir,
    save_strategy="steps",
    save_steps=args.eval_interval,
    save_total_limit=1,         # rolling checkpoint — keep only the latest
    eval_strategy="steps" if val_data is not None else "no",
    eval_steps=args.eval_interval if val_data is not None else None,
    load_best_model_at_end=(val_data is not None),
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",           # no wandb / tensorboard in the SageMaker container
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field="text",
    max_seq_length=args.seq_len,
    data_collator=collator,
    packing=False,              # packing can mix reasoning traces from different
                                # problems; off keeps each example isolated
    args=training_args,
)

start_time = time.time()
train_result = trainer.train()
train_seconds = time.time() - start_time

print(f"  Training complete in {train_seconds / 60:.1f} min")
print(f"  Final train loss : {train_result.training_loss:.4f}")

# ── 5. Save artifacts ─────────────────────────────────────────────────────────
print("\n[5/5] Saving LoRA adapter and metadata ...")

# Save the LoRA adapter weights and the updated tokenizer (with ChatML tokens).
# inference_v6.py loads these and merges them into the base model at cold start.
adapter_path = os.path.join(SM_MODEL_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
adapter_size_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, dn, filenames in os.walk(adapter_path)
    for f in filenames
) / 1e6
print(f"  Saved LoRA adapter → {adapter_path}/ ({adapter_size_mb:.0f} MB)")

# Pull best val loss from trainer state
best_val_loss = None
if val_data is not None and trainer.state.best_metric is not None:
    best_val_loss = round(float(trainer.state.best_metric), 6)

metadata = {
    "version": "v6",
    "model_name": f"Qwen2.5-7B-Base-CoT-SFT-LoRA-r{args.lora_rank}",
    "base_model": args.base_model,
    "training_mode": "QLoRA SFT (completion-only loss, assistant turns only)",
    "tokenizer": "Qwen BPE with ChatML special tokens",
    "dataset": args.dataset,
    "architecture": {
        "base": "Qwen2.5-7B",
        "pos_encoding": "RoPE (128k context native)",
        "attention": "GQA (Grouped-Query Attention)",
        "activation": "SwiGLU",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        "quantization": "4-bit NF4 (QLoRA)",
        "trainable_params": trainable_params,
        "total_base_params": total_base_params,
    },
    "hyperparameters": {
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": effective_batch,
        "steps": args.steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "min_lr_frac": args.min_lr_frac,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "precision": precision,
        "seed": args.seed,
    },
    "data": {
        "dataset_split": args.dataset_split,
        "train_examples": len(train_data),
        "val_examples": n_val,
        "planned_epochs": round(planned_epochs, 3),
        "response_template": response_template,
    },
    "results": {
        "final_train_loss": round(train_result.training_loss, 6),
        "best_val_loss": best_val_loss,
        "train_seconds": round(train_seconds, 1),
    },
}
meta_path = os.path.join(SM_MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata   → {meta_path}")

metrics = {
    "final_train_loss": round(train_result.training_loss, 6),
    "best_val_loss": best_val_loss,
    "trainable_params": trainable_params,
    "steps": args.steps,
    "train_seconds": round(train_seconds, 1),
    "precision": precision,
}
metrics_path = os.path.join(SM_OUTPUT_DATA_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics    → {metrics_path}")

# Bundle inference code alongside the adapter so the endpoint is self-contained.
code_dir = os.path.join(SM_MODEL_DIR, "code")
os.makedirs(code_dir, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
for fname in ["inference_v6.py"]:
    src = os.path.join(here, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(code_dir, fname))
        print(f"  Bundled {fname} → code/{fname}")

with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write(
        "unsloth>=2024.8\n"
        "transformers>=4.40.0\n"
        "peft>=0.10.0\n"
        "accelerate>=0.29.0\n"
        "bitsandbytes>=0.43.0\n"
    )
print("  Wrote serving requirements.txt")
print("\n  Done.")
