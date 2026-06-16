"""
train.py — SageMaker Training Job entry point (v4 — Text + Image)

Two modes:
  --mode text      Original text-only training (default, backward compatible)
  --mode text2img  Text-to-image generation

Text mode (v3, unchanged):
  token_embed : (50257, 256)
  pos_embed   : (100,   256)
  2 × TransformerBlock (SelfAttention + FeedForward 256→1024→256)
  LayerNorm + head Linear(256 → 50257)
  Total params ≈ 26M

Text2Img mode (v4):
  Phase 1 — Train VQ-VAE to tokenize 32×32 images into 16 image tokens
  Phase 2 — Train transformer on [text_prompt, SEP, img_tok0, ..., img_tok15, EOS]
            Same next-token prediction, same loss, same backprop
            Vocabulary expanded: 50257 text + 256 image + 1 SEP = 50514

  Synthetic training data: colored shapes (red circle, blue square, etc.)
  with text descriptions. No external images needed.
"""

import os
import json
import math
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken

# ── SageMaker environment variables ─────────────────────────────────────────
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", ".")
SM_MODEL_DIR        = os.environ.get("SM_MODEL_DIR",        "./model_output")
SM_OUTPUT_DATA_DIR  = os.environ.get("SM_OUTPUT_DATA_DIR",  "./output")

os.makedirs(SM_MODEL_DIR,       exist_ok=True)
os.makedirs(SM_OUTPUT_DATA_DIR, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",           type=str,   default="text", choices=["text", "text2img"])
parser.add_argument("--seq-len",        type=int,   default=100)
parser.add_argument("--embed-dim",      type=int,   default=256)
parser.add_argument("--num-layers",     type=int,   default=2)
parser.add_argument("--steps",          type=int,   default=50000)
parser.add_argument("--lr",             type=float, default=0.001)
parser.add_argument("--image-size",     type=int,   default=32)
parser.add_argument("--codebook-size",  type=int,   default=256)
parser.add_argument("--vqvae-steps",    type=int,   default=5000)
parser.add_argument("--num-images",     type=int,   default=200)
args = parser.parse_args()

MODE           = args.mode
SEQ_LEN        = args.seq_len
EMBED_DIM      = args.embed_dim
NUM_LAYERS     = args.num_layers
STEPS          = args.steps
LR             = args.lr
IMAGE_SIZE     = args.image_size
CODEBOOK_SIZE  = args.codebook_size
VQVAE_STEPS    = args.vqvae_steps
NUM_IMAGES     = args.num_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tokenizer ────────────────────────────────────────────────────────────────
enc        = tiktoken.get_encoding("gpt2")
EOS_ID     = enc.eot_token   # 50256
TEXT_VOCAB  = enc.n_vocab      # 50257

if MODE == "text":
    vocab_size = TEXT_VOCAB
    SEP_ID     = None
    IMG_OFFSET = None
else:
    IMG_OFFSET = TEXT_VOCAB       # image token IDs start at 50257
    SEP_ID     = TEXT_VOCAB + CODEBOOK_SIZE  # 50257 + 256 = 50513
    vocab_size = TEXT_VOCAB + CODEBOOK_SIZE + 1  # 50514
    IMG_TOKENS = (IMAGE_SIZE // 8) ** 2  # 32/8 = 4, 4×4 = 16 image tokens
    SEQ_LEN    = max(SEQ_LEN, 50 + IMG_TOKENS + 2)  # prompt + SEP + 16 img + EOS

print("=" * 60)
print(f"  SageMaker Training Job — TinyTransformer v4 ({MODE})")
print("=" * 60)
print(f"  device              : {device}")
print(f"  mode                : {MODE}")
print(f"  seq_len             : {SEQ_LEN}")
print(f"  embed_dim           : {EMBED_DIM}")
print(f"  num_layers          : {NUM_LAYERS}")
print(f"  vocab_size          : {vocab_size}")
if MODE == "text2img":
    print(f"  image_size          : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  codebook_size       : {CODEBOOK_SIZE}")
    print(f"  image_tokens        : {IMG_TOKENS} per image ({IMAGE_SIZE//8}×{IMAGE_SIZE//8} grid)")
    print(f"  vqvae_steps         : {VQVAE_STEPS}")
    print(f"  num_images          : {NUM_IMAGES}")


# ══════════════════════════════════════════════════════════════════════════════
# VQ-VAE — Image Tokenizer (text2img mode only)
# Converts a 32×32 RGB image into 16 integer token IDs (4×4 grid)
# Each token is an index into a learned codebook of 256 "visual words"
# ══════════════════════════════════════════════════════════════════════════════

class VectorQuantizer(nn.Module):
    """Quantizes continuous vectors to nearest codebook entry."""
    def __init__(self, codebook_size, embed_dim):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        # z: (B, D, H, W) → reshape to (B*H*W, D)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

        # find nearest codebook vector for each spatial position
        distances = torch.cdist(z_flat, self.codebook.weight)  # (B*H*W, codebook_size)
        indices   = distances.argmin(dim=-1)  # (B*H*W,)

        # look up quantized vectors
        z_q_flat = self.codebook(indices)  # (B*H*W, D)
        z_q      = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # straight-through estimator: gradient flows through z_q as if it were z
        z_q_st = z + (z_q - z).detach()

        # losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss   = F.mse_loss(z.detach(), z_q)

        return z_q_st, indices.reshape(B, H, W), commitment_loss + codebook_loss


class VQVAE(nn.Module):
    """
    Encoder: 32×32×3 image → 4×4×128 latent grid
    Quantizer: 4×4×128 → 16 codebook indices (the "image tokens")
    Decoder: 16 codebook vectors → 32×32×3 reconstructed image
    """
    def __init__(self, codebook_size=256, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),     # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),   # 16→8
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, stride=2, padding=1),  # 8→4
            nn.ReLU(),
        )

        self.quantizer = VectorQuantizer(codebook_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),         # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),           # 16→32
            nn.Sigmoid(),
        )

    def encode(self, x):
        """32×32 image → 16 token IDs"""
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices  # (B, 4, 4)

    def decode_from_indices(self, indices):
        """16 token IDs → 32×32 image"""
        B = indices.shape[0]
        z_q = self.quantizer.codebook(indices)  # (B, 4, 4, latent_dim)
        z_q = z_q.permute(0, 3, 1, 2)          # (B, latent_dim, 4, 4)
        return self.decoder(z_q)

    def forward(self, x):
        z         = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_recon   = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, indices, recon_loss + 0.25 * vq_loss


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic Image Data Generator
# Creates simple colored shapes with text descriptions
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "red":    [1.0, 0.0, 0.0],
    "green":  [0.0, 1.0, 0.0],
    "blue":   [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "orange": [1.0, 0.5, 0.0],
    "white":  [1.0, 1.0, 1.0],
    "cyan":   [0.0, 1.0, 1.0],
}

SHAPES = ["circle", "square", "triangle", "diamond"]

BG_COLORS = {
    "black":     [0.0, 0.0, 0.0],
    "dark gray": [0.2, 0.2, 0.2],
    "gray":      [0.5, 0.5, 0.5],
}


def draw_shape(shape, color_rgb, bg_rgb, size=32):
    """Generate a simple 32×32 image tensor (3, 32, 32) with a colored shape."""
    img = torch.zeros(3, size, size)
    for c in range(3):
        img[c, :, :] = bg_rgb[c]

    cx, cy = size // 2, size // 2
    r = size // 3

    for y_px in range(size):
        for x_px in range(size):
            draw = False

            if shape == "circle":
                if (x_px - cx) ** 2 + (y_px - cy) ** 2 <= r ** 2:
                    draw = True

            elif shape == "square":
                if abs(x_px - cx) <= r and abs(y_px - cy) <= r:
                    draw = True

            elif shape == "triangle":
                if y_px >= cy - r and y_px <= cy + r:
                    progress = (y_px - (cy - r)) / (2 * r) if r > 0 else 0
                    half_w = r * progress
                    if abs(x_px - cx) <= half_w:
                        draw = True

            elif shape == "diamond":
                if abs(x_px - cx) + abs(y_px - cy) <= r:
                    draw = True

            if draw:
                for c in range(3):
                    img[c, y_px, x_px] = color_rgb[c]

    return img


def generate_training_pairs(num_images=200):
    """Generate (text_description, image_tensor) pairs."""
    pairs = []
    color_names = list(COLORS.keys())
    bg_names    = list(BG_COLORS.keys())

    for i in range(num_images):
        shape    = SHAPES[i % len(SHAPES)]
        color_name = color_names[i % len(color_names)]
        bg_name    = bg_names[i % len(bg_names)]
        color_rgb  = COLORS[color_name]
        bg_rgb     = BG_COLORS[bg_name]

        img = draw_shape(shape, color_rgb, bg_rgb, IMAGE_SIZE)

        descriptions = [
            f"a {color_name} {shape}",
            f"{color_name} {shape} on {bg_name}",
            f"draw a {color_name} {shape}",
            f"a {shape} that is {color_name}",
        ]
        desc = descriptions[i % len(descriptions)]
        pairs.append((desc, img))

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Transformer — shared between text and text2img modes
# ══════════════════════════════════════════════════════════════════════════════

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
    def __init__(self, vocab, seq_len, embed_dim, num_layers):
        super().__init__()
        self.token_embed = nn.Embedding(vocab, embed_dim)
        self.pos_embed   = nn.Embedding(seq_len, embed_dim)
        self.blocks      = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, vocab)

    def forward(self, x, pad_mask=None):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.token_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: text — original v3 text-only training
# ══════════════════════════════════════════════════════════════════════════════

if MODE == "text":
    train_file = os.path.join(SM_CHANNEL_TRAINING, "train.txt")

    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            text = f.read()
        print(f"\n  Loaded training text from {train_file} ({len(text)} chars)")
    else:
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

    def get_batch():
        i = torch.randint(len(data) - SEQ_LEN, (1,))
        x = data[i:i + SEQ_LEN]
        y = data[i + 1:i + SEQ_LEN + 1]
        return x, y

    model       = TinyTransformer(vocab_size, SEQ_LEN, EMBED_DIM, NUM_LAYERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    eos_weights         = torch.ones(vocab_size, device=device)
    eos_weights[EOS_ID] = 0.1
    loss_fn   = nn.CrossEntropyLoss(weight=eos_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

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


# ══════════════════════════════════════════════════════════════════════════════
# MODE: text2img — text-to-image generation
# ══════════════════════════════════════════════════════════════════════════════

elif MODE == "text2img":

    # ── Phase 1: Train VQ-VAE ────────────────────────────────────────────────
    # The VQ-VAE learns to compress 32×32 images into 16 discrete tokens
    # and reconstruct them back. This is like building a "visual tokenizer".
    print(f"\n{'─'*60}")
    print(f"  Phase 1: Train VQ-VAE (image tokenizer)")
    print(f"{'─'*60}")

    print(f"\n  Generating {NUM_IMAGES} synthetic training images ...")
    pairs = generate_training_pairs(NUM_IMAGES)
    all_images = torch.stack([img for _, img in pairs]).to(device)  # (N, 3, 32, 32)
    print(f"  Generated {len(pairs)} text-image pairs")
    print(f"  Shapes: {list(set(SHAPES))}")
    print(f"  Colors: {list(COLORS.keys())}")
    print(f"  Example: '{pairs[0][0]}', '{pairs[1][0]}', '{pairs[2][0]}'")

    vqvae = VQVAE(CODEBOOK_SIZE, latent_dim=128).to(device)
    vqvae_params = sum(p.numel() for p in vqvae.parameters())
    print(f"  VQ-VAE parameters: {vqvae_params:,}")

    vqvae_optimizer = optim.Adam(vqvae.parameters(), lr=0.001)

    print(f"\n  Training VQ-VAE for {VQVAE_STEPS} steps ...")
    for step in range(VQVAE_STEPS):
        idx = torch.randint(len(all_images), (min(16, len(all_images)),))
        batch = all_images[idx]

        x_recon, indices, loss = vqvae(batch)

        vqvae_optimizer.zero_grad()
        loss.backward()
        vqvae_optimizer.step()

        if step % 1000 == 0:
            print(f"  vqvae step {step:5d}  loss: {loss.item():.4f}")

    print(f"  VQ-VAE training complete. Final loss: {loss.item():.4f}")

    # verify: encode → decode roundtrip
    vqvae.eval()
    with torch.no_grad():
        test_img    = all_images[:1]
        test_tokens = vqvae.encode(test_img)
        test_recon  = vqvae.decode_from_indices(test_tokens)
        recon_error = F.mse_loss(test_recon, test_img).item()
        print(f"  Roundtrip reconstruction error: {recon_error:.4f}")
        print(f"  Image tokens for '{pairs[0][0]}': {test_tokens[0].flatten().tolist()}")

    # ── Phase 2: Build text+image training sequences ─────────────────────────
    # Each sequence: [text_tok0, ..., text_tokN, SEP, img_tok0, ..., img_tok15, EOS]
    # Same format as text-only training, just with image tokens appended
    print(f"\n{'─'*60}")
    print(f"  Phase 2: Build text+image training sequences")
    print(f"{'─'*60}")

    sequences = []
    with torch.no_grad():
        for desc, img in pairs:
            text_tokens = enc.encode(desc)

            img_indices = vqvae.encode(img.unsqueeze(0).to(device))  # (1, 4, 4)
            img_tokens  = img_indices[0].flatten().tolist()          # 16 ints (0-255)
            img_tokens  = [t + IMG_OFFSET for t in img_tokens]       # shift to 50257+

            seq = text_tokens + [SEP_ID] + img_tokens + [EOS_ID]
            sequences.append(seq)

    # pad all sequences to same length
    max_len = max(len(s) for s in sequences)
    SEQ_LEN = max(SEQ_LEN, max_len)

    print(f"  Total sequences    : {len(sequences)}")
    print(f"  Max sequence length: {max_len}")
    print(f"  SEQ_LEN (adjusted) : {SEQ_LEN}")
    print(f"  Vocab layout:")
    print(f"    text tokens   : 0 — {TEXT_VOCAB - 1}")
    print(f"    image tokens  : {IMG_OFFSET} — {IMG_OFFSET + CODEBOOK_SIZE - 1}")
    print(f"    SEP token     : {SEP_ID}")
    print(f"    EOS token     : {EOS_ID}")
    print(f"  Example sequence for '{pairs[0][0]}':")
    s = sequences[0]
    labels = []
    for t in s:
        if t == SEP_ID:
            labels.append("SEP")
        elif t == EOS_ID:
            labels.append("EOS")
        elif t >= IMG_OFFSET:
            labels.append(f"img{t - IMG_OFFSET}")
        else:
            labels.append(repr(enc.decode([t])))
    print(f"    {labels}")

    padded_sequences = []
    for seq in sequences:
        pad_len = SEQ_LEN + 1 - len(seq)  # +1 because we need x and y (shifted by 1)
        padded  = [EOS_ID] * pad_len + seq
        padded_sequences.append(padded)

    all_seq = torch.tensor(padded_sequences, device=device)  # (N, SEQ_LEN+1)
    print(f"  Padded tensor shape: {list(all_seq.shape)}")

    # ── Phase 3: Train transformer on text+image sequences ───────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 3: Train transformer on text→image sequences")
    print(f"{'─'*60}")

    model = TinyTransformer(vocab_size, SEQ_LEN, EMBED_DIM, NUM_LAYERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Transformer parameters: {total_params:,}")
    print(f"  token_embed: ({vocab_size}, {EMBED_DIM})")
    print(f"  pos_embed:   ({SEQ_LEN}, {EMBED_DIM})")
    print(f"  head:        ({EMBED_DIM}, {vocab_size})")

    loss_fn   = nn.CrossEntropyLoss(ignore_index=EOS_ID)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n  Training for {STEPS:,} steps ...")
    training_log = []

    for step in range(STEPS):
        idx = torch.randint(len(all_seq), (1,)).item()
        seq = all_seq[idx]

        x = seq[:-1].unsqueeze(0)  # (1, SEQ_LEN)
        y = seq[1:].unsqueeze(0)   # (1, SEQ_LEN) shifted by 1

        # pad mask: ignore EOS padding at the start
        pad_mask = (x[0] == EOS_ID)
        first_non_pad = (~pad_mask).nonzero()
        if len(first_non_pad) > 0:
            first_real = first_non_pad[0].item()
            pad_mask[:first_real] = True
            pad_mask[first_real:] = False
        else:
            pad_mask[:] = False
        pad_mask = pad_mask.unsqueeze(0)  # (1, SEQ_LEN)

        logits = model(x, pad_mask)
        loss   = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_log.append({"step": step, "loss": round(loss.item(), 6)})

        if step % 5000 == 0:
            print(f"  step {step:6d}  loss: {loss.item():.4f}")

    print(f"  Training complete. Final loss: {training_log[-1]['loss']:.4f}")

    # ── Generate sample images ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Generating sample images from text prompts")
    print(f"{'─'*60}")

    model.eval()
    vqvae.eval()

    def generate_image(prompt, temperature=0.8):
        """Generate image tokens from a text prompt, decode to pixels."""
        text_tokens = enc.encode(prompt) + [SEP_ID]
        result = list(text_tokens)

        with torch.no_grad():
            for _ in range(IMG_TOKENS):
                # pad to SEQ_LEN
                n_pad   = max(0, SEQ_LEN - len(result))
                context = [EOS_ID] * n_pad + result[-SEQ_LEN:]
                x_in    = torch.tensor([context], device=device)

                pad_mask = torch.zeros(1, len(context), dtype=torch.bool, device=device)
                if n_pad > 0:
                    pad_mask[0, :n_pad] = True

                logits = model(x_in, pad_mask)
                logits_last = logits[0, -1] / temperature

                # only allow image tokens during image generation
                mask = torch.full((vocab_size,), float('-inf'), device=device)
                mask[IMG_OFFSET:IMG_OFFSET + CODEBOOK_SIZE] = 0
                logits_last = logits_last + mask

                probs   = torch.softmax(logits_last, dim=0)
                next_id = torch.multinomial(probs, 1).item()
                result.append(next_id)

        # extract image tokens and decode
        img_token_ids = [t - IMG_OFFSET for t in result[-IMG_TOKENS:]]
        grid_size     = int(math.sqrt(IMG_TOKENS))
        indices       = torch.tensor(img_token_ids, device=device).reshape(1, grid_size, grid_size)
        image         = vqvae.decode_from_indices(indices)
        return image[0].cpu(), img_token_ids  # (3, 32, 32), list of ints

    test_prompts = [
        "a red circle",
        "blue square on black",
        "draw a green triangle",
        "a yellow diamond",
    ]

    generated_images = {}
    for prompt in test_prompts:
        img_tensor, img_tokens = generate_image(prompt)
        generated_images[prompt] = {
            "image_tokens": img_tokens,
            "pixel_values": img_tensor.tolist(),
        }
        print(f"  '{prompt}' → image tokens: {img_tokens[:8]}...")

    # save generated images as raw tensors
    gen_path = os.path.join(SM_OUTPUT_DATA_DIR, "generated_images.json")
    with open(gen_path, "w") as f:
        json.dump(generated_images, f)
    print(f"\n  Saved generated images → {gen_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Save model artifacts
# ══════════════════════════════════════════════════════════════════════════════

checkpoint_path = os.path.join(SM_MODEL_DIR, "model.pt")
torch.save(model.state_dict(), checkpoint_path)
print(f"\n  Saved model checkpoint → {checkpoint_path}")

if MODE == "text2img":
    vqvae_path = os.path.join(SM_MODEL_DIR, "vqvae.pt")
    torch.save(vqvae.state_dict(), vqvae_path)
    print(f"  Saved VQ-VAE checkpoint → {vqvae_path}")

learned_params = {}
for name, param in model.named_parameters():
    learned_params[name] = {
        "shape":  list(param.shape),
        "values": param.detach().cpu().tolist()
    }
json_path = os.path.join(SM_MODEL_DIR, "learned_params.json")
with open(json_path, "w") as f:
    json.dump(learned_params, f)
print(f"  Saved JSON weights      → {json_path}")

metadata = {
    "version":      "v4",
    "mode":         MODE,
    "tokenizer":    "tiktoken gpt2 BPE",
    "vocab_size":   vocab_size,
    "eos_token_id": EOS_ID,
    "hyperparameters": {
        "seq_len":    SEQ_LEN,
        "embed_dim":  EMBED_DIM,
        "num_layers": NUM_LAYERS,
        "lr":         LR,
        "steps":      STEPS,
    },
    "training_log": training_log,
}
if MODE == "text2img":
    metadata["image"] = {
        "image_size":     IMAGE_SIZE,
        "codebook_size":  CODEBOOK_SIZE,
        "img_tokens":     IMG_TOKENS,
        "img_offset":     IMG_OFFSET,
        "sep_id":         SEP_ID,
        "vqvae_steps":    VQVAE_STEPS,
        "num_images":     NUM_IMAGES,
    }

meta_path = os.path.join(SM_MODEL_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata          → {meta_path}")

metrics = {
    "final_loss":   training_log[-1]["loss"],
    "total_params": total_params,
    "steps":        STEPS,
    "mode":         MODE,
}
metrics_path = os.path.join(SM_OUTPUT_DATA_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics           → {metrics_path}")

code_dir = os.path.join(SM_MODEL_DIR, "code")
os.makedirs(code_dir, exist_ok=True)
for fname in ["inference.py", "requirements.txt"]:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(code_dir, fname))
        print(f"  Bundled {fname}         → {code_dir}/{fname}")

print("\n  Done.")
