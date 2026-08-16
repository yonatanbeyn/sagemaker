"""
train_v5.py — SageMaker Training Job entry point (GPT-2 base + Mythos SFT)

How this differs from train.py (v3):
  v3: 26M-param TinyTransformer trained FROM SCRATCH on a scraped prose corpus,
      predicting every token in one packed stream.
  v5: 355M-param gpt2-medium loaded PRETRAINED, then supervised-fine-tuned on
      Claude Mythos reasoning traces with the loss masked to assistant tokens
      only. The model already speaks English on arrival; this run only teaches
      it the reasoning format.

SageMaker injects these environment variables:
  SM_CHANNEL_TRAINING   → /opt/ml/input/data/training/    (Mythos traces, JSONL)
  SM_CHANNEL_BASEMODEL  → /opt/ml/input/data/basemodel/   (gpt2-medium weights)
  SM_MODEL_DIR          → /opt/ml/model/                  (tarred + uploaded to S3)
  SM_OUTPUT_DATA_DIR    → /opt/ml/output/data/            (metrics / checkpoints)

Both input channels are staged to S3 by the GitHub Actions workflow so the
training job needs no internet egress and the run is reproducible. If a channel
is absent, this script falls back to downloading from the HuggingFace Hub.

On completion SageMaker tars SM_MODEL_DIR and uploads it to S3, which is what
the endpoint later serves.
"""

import argparse
import array
import contextlib
import glob
import json
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import tiktoken

from gpt2_model import GPT2, IM_START, IM_END, load_hf_state_dict, generate

# ── SageMaker environment ────────────────────────────────────────────────────
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "./data")
SM_CHANNEL_BASEMODEL = os.environ.get("SM_CHANNEL_BASEMODEL", "")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "./model_output")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")

os.makedirs(SM_MODEL_DIR, exist_ok=True)
os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)

# ── Hyperparameters (overridable from the workflow) ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="gpt2-medium",
                    help="HF checkpoint id, used only if the basemodel channel is absent")
parser.add_argument("--dataset", type=str, default="ansulev/claude-mythos-distilled-25k",
                    help="HF dataset id, used only if the training channel is absent")
parser.add_argument("--dataset-split", type=str, default="train",
                    help="full 25k traces by default; slice with e.g. train[:2000]")
parser.add_argument("--seq-len", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
# 3200 steps x 16 examples = 51,200 examples ~= 2 epochs over the full 25k set.
# 2-3 epochs is the usual SFT range; more starts memorising the traces.
parser.add_argument("--steps", type=int, default=3200)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--min-lr-frac", type=float, default=0.1)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--precision", type=str, default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"])
parser.add_argument("--val-fraction", type=float, default=0.05)
parser.add_argument("--eval-interval", type=int, default=200)
parser.add_argument("--eval-batches", type=int, default=20)
parser.add_argument("--log-interval", type=int, default=20)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

SEQ_LEN = args.seq_len
STEPS = args.steps
LR = args.lr

print("=" * 68)
print("  SageMaker Training Job — v5 GPT-2 + Mythos SFT")
print("=" * 68)
print(f"  base_model        : {args.base_model}")
print(f"  dataset           : {args.dataset}")
print(f"  seq_len           : {SEQ_LEN}")
print(f"  batch_size        : {args.batch_size} x {args.grad_accum} accum "
      f"= {args.batch_size * args.grad_accum} examples/step")
print(f"  steps             : {STEPS}")
print(f"  lr                : {LR} (warmup {args.warmup_steps} → cosine to "
      f"{LR * args.min_lr_frac:g})")
print(f"  SM_CHANNEL_TRAINING  : {SM_CHANNEL_TRAINING}")
print(f"  SM_CHANNEL_BASEMODEL : {SM_CHANNEL_BASEMODEL or '(none — will use HF Hub)'}")
print(f"  SM_MODEL_DIR         : {SM_MODEL_DIR}")

random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Device & precision ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.precision == "auto":
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        precision = "bf16"
    elif device.type == "cuda":
        precision = "fp16"
    else:
        precision = "fp32"
else:
    precision = args.precision

# bf16 needs no loss scaling; fp16 does, or small gradients flush to zero.
amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
use_amp = amp_dtype is not None and device.type == "cuda"

# torch.amp.GradScaler is the modern spelling (2.3+); the SageMaker PyTorch 2.1
# DLC only has torch.cuda.amp.GradScaler, which newer torch warns on. Support
# both so the same file runs on the DLC and locally.
_scaler_enabled = (precision == "fp16" and use_amp)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)


def amp_context():
    """autocast when AMP is on, otherwise a no-op — passing dtype=None or
    float32 to autocast behaves inconsistently across torch versions."""
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return contextlib.nullcontext()

print(f"  device            : {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"  precision         : {precision}"
      + ("  [AMP autocast]" if use_amp else ""))

# ── Tokenizer ────────────────────────────────────────────────────────────────
# Pin tiktoken's download cache to a known directory so it can be bundled into
# model.tar.gz below. Without it the endpoint would try to fetch the BPE files
# from openaipublic.blob.core.windows.net on every cold start — which fails
# outright in a no-egress VPC.
TIKTOKEN_CACHE = os.path.join(SM_OUTPUT_DATA_DIR, "tiktoken_cache")
os.makedirs(TIKTOKEN_CACHE, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token      # 50256, doubles as pad (padding is masked out of the loss)
vocab_size = enc.n_vocab    # 50257

# ── 1. Load the pretrained GPT-2 base ────────────────────────────────────────
print("\n[1/5] Loading pretrained GPT-2 base ...")

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    raise SystemExit("transformers is required at training time. "
                     "Check requirements.txt in the source dir.")

if SM_CHANNEL_BASEMODEL and os.path.isdir(SM_CHANNEL_BASEMODEL) and os.listdir(SM_CHANNEL_BASEMODEL):
    base_source = SM_CHANNEL_BASEMODEL
    print(f"  Loading from S3-staged channel: {base_source}")
    print(f"  Channel contents: {sorted(os.listdir(base_source))}")
else:
    base_source = args.base_model
    print(f"  No basemodel channel — downloading '{base_source}' from the HF Hub")

hf_model = GPT2LMHeadModel.from_pretrained(base_source)
cfg = hf_model.config
print(f"  Config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
      f"n_positions={cfg.n_positions} vocab={cfg.vocab_size}")

if cfg.vocab_size != vocab_size:
    raise ValueError(f"tokenizer vocab {vocab_size} != checkpoint vocab {cfg.vocab_size}")
if SEQ_LEN > cfg.n_positions:
    raise ValueError(f"seq_len {SEQ_LEN} exceeds the {cfg.n_positions}-token context window")

# Build our model from the checkpoint's own config so the two cannot drift.
model = GPT2(
    vocab_size=cfg.vocab_size,
    n_positions=cfg.n_positions,
    n_embd=cfg.n_embd,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
)

copied, transposed, skipped = load_hf_state_dict(model, hf_model.state_dict())
del hf_model
print(f"  Copied {copied} tensors ({transposed} transposed from HF's Conv1D "
      f"layout), skipped {skipped} (tied/buffers)")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,} (~{total_params / 1e6:.0f}M)")

# Sanity check the port before spending an hour of GPU time on it. A healthy
# pretrained GPT-2 scores plain English around 3-4 (perplexity ~20-60). A bad
# transpose or a GELU mismatch shows up here as a loss above ~10 — and the
# square c_proj matrices transpose without any shape error, so this numeric
# check is the only thing that catches them.
model.eval()
with torch.no_grad():
    probe = torch.tensor(
        [enc.encode("The capital of France is Paris, and the capital of Italy is Rome.")]
    )
    probe_logits, _ = model(probe)
    probe_loss = F.cross_entropy(probe_logits[0, :-1].float(), probe[0, 1:]).item()

print(f"  Sanity check — loss on plain English: {probe_loss:.3f} "
      f"(perplexity {math.exp(probe_loss):.1f})")
if probe_loss > 6.0:
    raise RuntimeError(
        f"base model scores {probe_loss:.2f} on plain English — the weight port "
        f"is wrong. Refusing to burn GPU time fine-tuning a broken model."
    )

model.to(device)

# ── 2. Load and format the Mythos traces ─────────────────────────────────────
print(f"\n[2/5] Loading Mythos reasoning traces ...")


def read_jsonl_channel(channel_dir):
    """Read every .jsonl/.json file staged into the training channel."""
    paths = sorted(
        glob.glob(os.path.join(channel_dir, "**", "*.jsonl"), recursive=True)
        + glob.glob(os.path.join(channel_dir, "**", "*.json"), recursive=True)
    )
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            else:
                payload = json.load(f)
                rows.extend(payload if isinstance(payload, list) else [payload])
        print(f"  Read {path} → {len(rows):,} rows so far")
    return rows


raw_rows = []
if os.path.isdir(SM_CHANNEL_TRAINING):
    raw_rows = read_jsonl_channel(SM_CHANNEL_TRAINING)

if raw_rows:
    print(f"  Loaded {len(raw_rows):,} conversations from the S3 channel")
else:
    print(f"  No JSONL in the training channel — falling back to the HF Hub")
    from datasets import load_dataset
    raw_rows = list(load_dataset(args.dataset, split=args.dataset_split))
    print(f"  Loaded {len(raw_rows):,} conversations from '{args.dataset}'")


def encode_conversation(messages):
    """
    Turn one conversation into (input_ids, labels).

    labels[i] is aligned with input_ids[i]. A label of -100 means "never ask the
    model to predict this token" — used for every user/system token and for the
    assistant's role header. Assistant CONTENT and its closing <|im_end|> stay
    unmasked, so the model learns both what to say and when to stop.
    """
    ids, labels = [], []
    for msg in messages:
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        if not content:
            continue

        header = enc.encode(f"{IM_START}{role}\n")
        body = enc.encode(content)
        footer = enc.encode(f"{IM_END}\n")

        ids += header + body + footer
        if role == "assistant":
            labels += [-100] * len(header) + body + footer
        else:
            labels += [-100] * (len(header) + len(body) + len(footer))

    ids.append(EOS_ID)
    labels.append(EOS_ID if any(l != -100 for l in labels) else -100)
    return ids, labels


print("  Encoding conversations and masking prompt tokens ...")
encode_start = time.time()
examples, dropped, truncated = [], 0, 0
for row in raw_rows:
    messages = row.get("messages") or row.get("conversations") or []
    if not messages:
        dropped += 1
        continue

    ids, labels = encode_conversation(messages)

    # +1 because each example is split into x=ids[:-1], y=labels[1:].
    if len(ids) > SEQ_LEN + 1:
        ids, labels = ids[:SEQ_LEN + 1], labels[:SEQ_LEN + 1]
        truncated += 1

    # Truncation can leave an example with nothing to learn from — e.g. a very
    # long question that pushed the whole answer past the cut.
    if len(ids) < 2 or all(l == -100 for l in labels[1:]):
        dropped += 1
        continue

    # Stored as int32 arrays rather than Python lists. Across the full 25k
    # traces that is ~0.11GB instead of ~0.93GB — Python boxes every token id
    # above 255 as a separate ~28-byte object, which adds up fast at 12.8M
    # tokens. Slices stay array.array and torch.tensor consumes them directly.
    examples.append((array.array("i", ids), array.array("i", labels)))

if not examples:
    raise ValueError("no usable training examples — check the dataset schema")

total_tokens = sum(len(i) for i, _ in examples)
supervised_tokens = sum(sum(1 for l in lb[1:] if l != -100) for _, lb in examples)
print(f"  Usable examples   : {len(examples):,}")
print(f"  Dropped           : {dropped:,}")
print(f"  Truncated to {SEQ_LEN:<4} : {truncated:,}")
print(f"  Total tokens      : {total_tokens:,}")
print(f"  Supervised tokens : {supervised_tokens:,} "
      f"({100 * supervised_tokens / max(total_tokens, 1):.1f}% — rest is masked prompt)")
print(f"  Encoding took     : {time.time() - encode_start:.1f}s")

random.shuffle(examples)
n_val = max(1, int(len(examples) * args.val_fraction)) if len(examples) > 20 else 0
val_set = examples[:n_val]
train_set = examples[n_val:]
print(f"  Split             : {len(train_set):,} train / {len(val_set):,} val")

# Make coverage explicit. It is the number that decides whether a run
# under-trains or memorises, and it moves whenever steps, batch size or the
# dataset slice changes — easy to get wrong silently.
effective_batch = args.batch_size * args.grad_accum
planned_epochs = STEPS * effective_batch / max(len(train_set), 1)
print(f"  Coverage          : {STEPS:,} steps x {effective_batch} = "
      f"{STEPS * effective_batch:,} examples seen = {planned_epochs:.2f} epochs")
if planned_epochs < 0.9:
    print(f"  NOTE: under one full pass over the data — raise --steps to at "
          f"least {math.ceil(len(train_set) / effective_batch):,} for one epoch.")
elif planned_epochs > 4:
    print(f"  NOTE: {planned_epochs:.1f} epochs is well past the usual 2-3 for "
          f"SFT; expect the model to start memorising the traces.")


def make_batch(batch_examples):
    """Collate into right-padded tensors: x, y (-100 where masked), pad_mask."""
    xs = [ids[:-1] for ids, _ in batch_examples]
    ys = [lb[1:] for _, lb in batch_examples]
    max_len = max(len(x) for x in xs)

    x_out = torch.full((len(xs), max_len), EOS_ID, dtype=torch.long)
    y_out = torch.full((len(xs), max_len), -100, dtype=torch.long)
    pad_mask = torch.ones((len(xs), max_len), dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        y_out[i, :len(y)] = torch.tensor(y, dtype=torch.long)
        pad_mask[i, :len(x)] = False

    return (x_out.to(device, non_blocking=True),
            y_out.to(device, non_blocking=True),
            pad_mask.to(device, non_blocking=True))


class EpochSampler:
    """Yields shuffled batches, reshuffling at each epoch boundary."""

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.order = []
        self.served = 0     # examples handed out, for exact epoch accounting

    def next_batch(self):
        if len(self.order) < self.batch_size:
            self.order = list(range(len(self.data)))
            random.shuffle(self.order)
        picks, self.order = self.order[:self.batch_size], self.order[self.batch_size:]
        self.served += len(picks)
        return make_batch([self.data[i] for i in picks])

    @property
    def epochs(self):
        """Fractional passes completed. Counting reshuffles instead would
        over-report, since the pass in progress is counted the moment it
        starts — that read as 3 epochs on a run that had done 2.02."""
        return self.served / max(len(self.data), 1)


train_sampler = EpochSampler(train_set, args.batch_size)
val_sampler = EpochSampler(val_set, args.batch_size) if val_set else None

# ── 3. Optimizer ─────────────────────────────────────────────────────────────
print("\n[3/5] Configuring optimizer ...")

# Weight decay belongs on matmul weights, not on LayerNorm gains or biases —
# shrinking a normalisation gain toward zero just fights the layer.
decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
optimizer = optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, 0.95), eps=1e-8,
)
print(f"  AdamW: {sum(p.numel() for p in decay):,} decayed params, "
      f"{sum(p.numel() for p in no_decay):,} undecayed")


def lr_at(step):
    """Linear warmup, then cosine decay to LR * min_lr_frac."""
    if step <= args.warmup_steps:
        return LR * step / max(args.warmup_steps, 1)
    progress = min((step - args.warmup_steps) / max(STEPS - args.warmup_steps, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)


def batch_loss(x, y, pad_mask):
    """Mean cross-entropy over unmasked (assistant) positions only."""
    with amp_context():
        logits, _ = model(x, pad_mask=pad_mask)
    # Compute the loss in fp32 regardless of autocast — softmax over a 50k
    # vocab in fp16 loses real precision.
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100
    )


@torch.no_grad()
def evaluate():
    if val_sampler is None:
        return None
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y, pm = val_sampler.next_batch()
        losses.append(batch_loss(x, y, pm).item())
    model.train()
    return sum(losses) / len(losses)


# ── 4. Fine-tuning loop ──────────────────────────────────────────────────────
print(f"\n[4/5] Fine-tuning for {STEPS:,} steps ...")

checkpoint_dir = os.path.join(SM_OUTPUT_DATA_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

training_log, eval_log = [], []
best_val = float("inf")
best_state = None
start_time = time.time()

model.train()
optimizer.zero_grad(set_to_none=True)

for step in range(1, STEPS + 1):
    lr_now = lr_at(step)
    for group in optimizer.param_groups:
        group["lr"] = lr_now

    # Gradient accumulation: several forward/backward passes per update. Each
    # micro-batch loss is divided so the accumulated gradient is a mean.
    step_loss = 0.0
    for _ in range(args.grad_accum):
        x, y, pm = train_sampler.next_batch()
        loss = batch_loss(x, y, pm) / args.grad_accum
        scaler.scale(loss).backward()
        step_loss += loss.item()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step == 1 or step % args.log_interval == 0 or step == STEPS:
        elapsed = time.time() - start_time
        # "train_loss=" / "val_loss=" prefixes are picked up by the
        # MetricDefinitions regexes in the GitHub Actions workflow, which is
        # what makes these show up as CloudWatch metrics.
        print(f"  step {step:5d}/{STEPS}  train_loss={step_loss:.4f}  "
              f"ppl={math.exp(min(step_loss, 20)):.1f}  lr={lr_now:.2e}  "
              f"grad_norm={float(grad_norm):.2f}  "
              f"{elapsed / step:.2f}s/step", flush=True)
        training_log.append({
            "step": step,
            "loss": round(step_loss, 6),
            "lr": lr_now,
            "grad_norm": round(float(grad_norm), 4),
        })

    if step % args.eval_interval == 0 or step == STEPS:
        val_loss = evaluate()
        if val_loss is not None:
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                # Keep the best weights on CPU so they survive later steps
                # without competing for GPU memory.
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                marker = "  <- best"
            print(f"  step {step:5d}/{STEPS}  val_loss={val_loss:.4f}  "
                  f"ppl={math.exp(min(val_loss, 20)):.1f}{marker}", flush=True)
            eval_log.append({"step": step, "val_loss": round(val_loss, 6)})

        # ROLLING checkpoint — keep only the most recent. Each one is ~1.4GB
        # for gpt2-medium, so writing one per eval over a full 25k run would
        # put >20GB of redundant weights into output.tar.gz. The best weights
        # are already held in memory (best_state) and restored after the loop;
        # this file exists only so a crashed run leaves something behind.
        ckpt = os.path.join(checkpoint_dir, f"checkpoint-{step}.pt")
        torch.save(model.state_dict(), ckpt)
        for old in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*.pt")):
            if old != ckpt:
                os.remove(old)

train_seconds = time.time() - start_time
print(f"  Training complete in {train_seconds / 60:.1f} min. "
      f"Final train loss: {training_log[-1]['loss']:.4f}")

# Serve the best-validation weights, not simply the last ones — with only a few
# thousand traces the tail of the run can overfit.
if best_state is not None:
    print(f"  Restoring best checkpoint (val_loss={best_val:.4f})")
    model.load_state_dict(best_state)

# ── 5. Smoke test + save artifacts ───────────────────────────────────────────
print("\n[5/5] Smoke test and artifact export ...")

smoke_prompt = "Explain how multi-head attention distributes work across heads."
smoke_text, smoke_n, smoke_stop = generate(
    model, enc, smoke_prompt, eos_id=EOS_ID,
    max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.95, device=device,
)
print(f"  prompt : {smoke_prompt}")
print(f"  output : {smoke_text[:400]!r}")
print(f"  ({smoke_n} tokens, stop_reason={smoke_stop})")

# 1. Weights. fp32 state dict, ~1.4GB for gpt2-medium.
#    Deliberately NOT dumping learned_params.json the way v3 does — the JSON
#    form of 355M floats is roughly 7GB and would make model.tar.gz unusable.
model_path = os.path.join(SM_MODEL_DIR, "model.pt")
torch.save(model.state_dict(), model_path)
print(f"  Saved weights  → {model_path} "
      f"({os.path.getsize(model_path) / 1e9:.2f} GB)")

# 2. Metadata — inference_v5.py reads this to rebuild the architecture.
metadata = {
    "version": "v5",
    "model_name": f"TinyTransformer-v5-{args.base_model}-MythosSFT",
    "base_checkpoint": args.base_model,
    "training_mode": "supervised fine-tuning (assistant-only loss)",
    "tokenizer": "tiktoken gpt2 BPE",
    "dataset": args.dataset,
    "vocab_size": vocab_size,
    "eos_token_id": EOS_ID,
    "chat_template": {"im_start": IM_START, "im_end": IM_END},
    "architecture": {
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_positions": cfg.n_positions,
        "pos_encoding": "learned absolute (wpe)",
        "norm": "LayerNorm",
        "activation": "GELU (tanh approx)",
        "weight_tying": True,
        "total_params": total_params,
    },
    "hyperparameters": {
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "steps": STEPS,
        "lr": LR,
        "warmup_steps": args.warmup_steps,
        "min_lr_frac": args.min_lr_frac,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "precision": precision,
        "seed": args.seed,
    },
    "data": {
        "examples": len(examples),
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "planned_epochs": round(planned_epochs, 3),
        "epochs_seen": round(train_sampler.epochs, 3),
    },
    "base_model_english_loss": round(probe_loss, 4),
    "smoke_test": {
        "prompt": smoke_prompt,
        "output": smoke_text,
        "tokens": smoke_n,
        "stop_reason": smoke_stop,
    },
}
meta_path = os.path.join(SM_MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata → {meta_path}")

# 3. Metrics for SageMaker Experiments / offline analysis.
metrics = {
    "final_train_loss": training_log[-1]["loss"],
    "best_val_loss": None if best_val == float("inf") else round(best_val, 6),
    "final_val_loss": eval_log[-1]["val_loss"] if eval_log else None,
    "base_model_english_loss": round(probe_loss, 4),
    "total_params": total_params,
    "steps": STEPS,
    "train_seconds": round(train_seconds, 1),
    "precision": precision,
    "training_log": training_log,
    "eval_log": eval_log,
}
metrics_path = os.path.join(SM_OUTPUT_DATA_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics  → {metrics_path}")

# 4. Bundle serving code into model.tar.gz.
#    The SageMaker inference container adds model_dir/code/ to sys.path and
#    looks there for the handler named by SAGEMAKER_PROGRAM. gpt2_model.py must
#    ride along or inference_v5.py cannot import the architecture.
code_dir = os.path.join(SM_MODEL_DIR, "code")
os.makedirs(code_dir, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
for fname in ["inference_v5.py", "gpt2_model.py"]:
    src = os.path.join(here, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{fname} missing from the source dir — the "
                                f"endpoint would fail to start")
    shutil.copy(src, os.path.join(code_dir, fname))
    print(f"  Bundled {fname} → code/{fname}")

# Serving needs only tiktoken on top of the container's torch. Deliberately not
# reusing the training requirements — transformers and datasets are ~1GB of
# dependencies the endpoint would install on every cold start and never use.
with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write("tiktoken>=0.5.0\n")
print(f"  Wrote serving requirements.txt (tiktoken only)")

# 5. Bundle the tiktoken BPE cache so the endpoint never needs to reach
#    openaipublic.blob.core.windows.net at cold start. inference_v5.py points
#    TIKTOKEN_CACHE_DIR at this directory before building the encoder.
cache_dest = os.path.join(code_dir, "tiktoken_cache")
if os.path.isdir(TIKTOKEN_CACHE) and os.listdir(TIKTOKEN_CACHE):
    shutil.copytree(TIKTOKEN_CACHE, cache_dest, dirs_exist_ok=True)
    n_files = len(os.listdir(cache_dest))
    print(f"  Bundled tiktoken cache → code/tiktoken_cache/ ({n_files} files)")
else:
    print("  WARNING: tiktoken cache is empty — the endpoint will need internet "
          "egress to download GPT-2 BPE files at startup")

print("\n  Done.")