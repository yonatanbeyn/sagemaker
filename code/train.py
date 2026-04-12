"""
train.py — SageMaker Training Job entry point

Adapted from genai_transformer_v3.py for AWS SageMaker.

SageMaker injects environment variables:
  SM_CHANNEL_TRAINING  → /opt/ml/input/data/training/   (S3 training data)
  SM_MODEL_DIR         → /opt/ml/model/                 (saved model destination)
  SM_OUTPUT_DATA_DIR   → /opt/ml/output/data/           (metrics / artifacts)
  SM_HP_*              → hyperparameters passed from pipeline

On completion, SageMaker tars SM_MODEL_DIR and uploads it to S3 automatically.

Architecture (v3, unchanged):
  token_embed : (50257, 256)
  pos_embed   : (100,   256)
  2 × TransformerBlock (SelfAttention + FeedForward 256→1024→256)
  LayerNorm + head Linear(256 → 50257)
  Total params ≈ 26M
"""
import os
import json
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

# ── SageMaker environment variables ─────────────────────────────────────────
# These are set automatically inside a SageMaker Training Job container.
# When running locally, fall back to current directory.
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", ".")
SM_MODEL_DIR        = os.environ.get("SM_MODEL_DIR",        "./model_output")
SM_OUTPUT_DATA_DIR  = os.environ.get("SM_OUTPUT_DATA_DIR",  "./output")

os.makedirs(SM_MODEL_DIR,       exist_ok=True)
os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)

# ── Hyperparameters (overridable from SageMaker pipeline) ────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seq-len",    type=int,   default=100)
parser.add_argument("--embed-dim",  type=int,   default=256)
parser.add_argument("--num-layers", type=int,   default=2)
parser.add_argument("--steps",      type=int,   default=50000)
parser.add_argument("--lr",         type=float, default=0.001)
args = parser.parse_args()

SEQ_LEN    = args.seq_len
EMBED_DIM  = args.embed_dim
NUM_LAYERS = args.num_layers
STEPS      = args.steps
LR         = args.lr

print("=" * 60)
print("  SageMaker Training Job — TinyTransformer v3")
print("=" * 60)
print(f"  SM_CHANNEL_TRAINING : {SM_CHANNEL_TRAINING}")
print(f"  SM_MODEL_DIR        : {SM_MODEL_DIR}")
print(f"  seq_len             : {SEQ_LEN}")
print(f"  embed_dim           : {EMBED_DIM}")
print(f"  num_layers          : {NUM_LAYERS}")
print(f"  steps               : {STEPS}")
print(f"  lr                  : {LR}")

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device              : {device}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
enc        = tiktoken.get_encoding("gpt2")
EOS_ID     = enc.eot_token   # 50256
vocab_size = enc.n_vocab      # 50257

# ── Load training text from S3-backed channel ────────────────────────────────
# SageMaker downloads the S3 object to SM_CHANNEL_TRAINING before calling this script.
# The file name must match what you upload to S3.
train_file = os.path.join(SM_CHANNEL_TRAINING, "train.txt")

if os.path.exists(train_file):
    with open(train_file, "r") as f:
        text = f.read()
    print(f"\n  Loaded training text from {train_file} ({len(text)} chars)")
else:
    # Fallback: built-in training text (same as v3)
    print(f"\n  {train_file} not found — using built-in training text")
    text = (
        "The Quick Brown Fox Jumps Over The Lazy Dog. "
        "the quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
        "Sphinx of black quartz judge my vow. "
        "A Big Cat Dances Every Friday Going Home In January. "
        "a big cat dances every friday going home in january. "
        "Kings Learn Many New Outstanding Principles Quietly. "
        "kings learn many new outstanding principles quietly. "
        "Really Smart Turtles Use Very Warm eXtra Yellow Zones. "
        "really smart turtles use very warm extra yellow zones. "
        "Hello World. My name is Transformer. I learn from text. "
        "hello world. my name is transformer. i learn from text. "
        "The cat sat on the mat. The dog sat on the log. "
        "the cat sat on the mat. the dog sat on the log. "
        "I think therefore I am. We learn therefore we grow. "
        "i think therefore i am. we learn therefore we grow. "
        "Machine learning is a method of data analysis that automates analytical model building. "
        "A transformer is a deep learning architecture that relies on the attention mechanism. "
        "The attention mechanism allows the model to focus on different parts of the input sequence. "
        "Natural language processing is a subfield of linguistics and artificial intelligence. "
        "The embedding layer converts token indices into dense vector representations. "
        "Backpropagation is the algorithm used to train neural networks by computing gradients. "
        "The context window determines how many tokens the model can see at one time. "
        "Larger context windows allow the model to capture longer range dependencies in text. "
    )

# ── Encode ───────────────────────────────────────────────────────────────────
token_ids = enc.encode(text)
encoded   = []
for tok_id in token_ids:
    encoded.append(tok_id)
    if "." in enc.decode([tok_id]):
        encoded.append(EOS_ID)

data = torch.tensor(encoded).to(device)

print(f"  BPE tokens  : {len(token_ids)}")
print(f"  With EOS    : {len(encoded)}")
print(f"  Context win : {SEQ_LEN}")

if len(encoded) < SEQ_LEN + 10:
    raise ValueError(
        f"Training text too short ({len(encoded)} tokens) for seq_len={SEQ_LEN}. "
        f"Need at least {SEQ_LEN + 10}."
    )


# ── Model definition ─────────────────────────────────────────────────────────

def get_batch():
    i = torch.randint(len(data) - SEQ_LEN, (1,))
    x = data[i:i + SEQ_LEN]
    y = data[i + 1:i + SEQ_LEN + 1]
    return x, y


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
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_embed   = nn.Embedding(SEQ_LEN,    EMBED_DIM)
        self.blocks      = nn.ModuleList([TransformerBlock(EMBED_DIM) for _ in range(NUM_LAYERS)])
        self.norm        = nn.LayerNorm(EMBED_DIM)
        self.head        = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x, pad_mask=None):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.token_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)
        return self.head(x)


# ── Build model ───────────────────────────────────────────────────────────────
model       = TinyTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Total parameters: {total_params:,}")

eos_weights         = torch.ones(vocab_size, device=device)
eos_weights[EOS_ID] = 0.1
loss_fn   = nn.CrossEntropyLoss(weight=eos_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n  Training for {STEPS:,} steps ...")
training_log = []

for step in range(STEPS):
    x, y = get_batch()
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    logits = model(x, pad_mask=None)
    loss   = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_log.append({"step": step, "loss": round(loss.item(), 6)})

    if step % 5000 == 0:
        print(f"  step {step:6d}  loss: {loss.item():.4f}")

print(f"  Training complete. Final loss: {training_log[-1]['loss']:.4f}")

# ── Save model artifacts to SM_MODEL_DIR ─────────────────────────────────────
# SageMaker will tar this directory and upload it to S3 after training.

# 1. PyTorch checkpoint (.pt) — binary, fast to load
checkpoint_path = os.path.join(SM_MODEL_DIR, "model.pt")
torch.save(model.state_dict(), checkpoint_path)
print(f"\n  Saved model checkpoint → {checkpoint_path}")

# 2. JSON weights — language-agnostic, useful for debugging / cross-language inference
learned_params = {}
for name, param in model.named_parameters():
    learned_params[name] = {
        "shape":  list(param.shape),
        "values": param.detach().tolist()
    }
json_path = os.path.join(SM_MODEL_DIR, "learned_params.json")
with open(json_path, "w") as f:
    json.dump(learned_params, f)
print(f"  Saved JSON weights      → {json_path}")

# 3. Model metadata — needed by inference.py to rebuild architecture
metadata = {
    "version":      "v3",
    "tokenizer":    "tiktoken gpt2 BPE",
    "vocab_size":   vocab_size,
    "eos_token_id": EOS_ID,
    "text":         text,
    "token_count":  len(token_ids),
    "hyperparameters": {
        "seq_len":    SEQ_LEN,
        "embed_dim":  EMBED_DIM,
        "num_layers": NUM_LAYERS,
        "lr":         LR,
        "steps":      STEPS
    },
    "training_log": training_log
}
meta_path = os.path.join(SM_MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata          → {meta_path}")

# 4. Metrics for SageMaker Model Monitor / Experiments
metrics = {
    "final_loss":  training_log[-1]["loss"],
    "total_params": total_params,
    "steps":        STEPS,
}
metrics_path = os.path.join(SM_OUTPUT_DATA_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics           → {metrics_path}")

# 5. Bundle inference code into model.tar.gz
# SageMaker inference container looks for custom handlers in model_dir/code/.
# Placing inference.py + requirements.txt there activates our model_fn/predict_fn
# instead of the default TorchScript handler.
code_dir = os.path.join(SM_MODEL_DIR, "code")
os.makedirs(code_dir, exist_ok=True)
for fname in ["inference.py", "requirements.txt"]:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(code_dir, fname))
        print(f"  Bundled {fname}         → {code_dir}/{fname}")

print("\n  Done.")
