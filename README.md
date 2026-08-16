# TinyTransformer — AWS SageMaker Deployment

Two independent pipelines share one AWS stack:

| | **v3** | **v5** |
|---|---|---|
| Model | TinyTransformer, 26M params | gpt2-medium, 355M params |
| Weights at start | random init | **pretrained GPT-2** |
| Training | from scratch on scraped prose | **supervised fine-tuning** on Claude Mythos reasoning traces |
| Loss | every token | **assistant tokens only** (prompt masked to -100) |
| Entry points | `code/train.py`, `code/inference.py` | `code/train_v5.py`, `code/inference_v5.py` |
| Workflow | `.github/workflows/train-deploy.yml` | `.github/workflows/train-deploy-v5.yml` |
| Endpoint | `tiny-transformer-endpoint` | `tiny-transformer-v5-endpoint` |

They deploy to separate endpoints and separate Model Registry groups, so a bad
v5 run can never take down v3.

---

## Folder structure

```
sagemaker/
├── code/
│   ├── train.py             v3 training entry point   (SageMaker container)
│   ├── inference.py         v3 serving entry point    (SageMaker endpoint)
│   ├── requirements.txt     v3 deps — torch, tiktoken, datasets
│   ├── gpt2_model.py        v5 GPT-2 architecture — SHARED by train + serve
│   ├── train_v5.py          v5 training entry point
│   ├── inference_v5.py      v5 serving entry point
│   └── requirements-v5.txt  v5 training deps (+ transformers)
├── cloudformation/
│   └── stack.yaml           one-click AWS infrastructure provisioning
├── data/
│   └── train.txt            (optional) custom v3 training text — uploaded to S3
└── .github/
    └── workflows/
        ├── train-deploy.yml      v3 CI/CD: push → train → register → deploy
        └── train-deploy-v5.yml   v5 CI/CD: stage → train → register → deploy
```

`gpt2_model.py` exists because v3 duplicates its model classes across `train.py`
and `inference.py` with a "must match exactly" comment. v5 imports one shared
definition instead, and `train_v5.py` bundles it into `model.tar.gz` so the
endpoint provably runs the same architecture — and the same chat template — that
training used.

---

## Architecture recap (v3)

```
Input tokens  →  token_embed (50257 × 256)
                 +
              →  pos_embed   (100 × 256)
                 ↓
              →  TransformerBlock #1
                   LayerNorm → SelfAttention (Q·Kᵀ/√256, causal mask) → residual
                   LayerNorm → FeedForward (256 → 1024 → 256)         → residual
                 ↓
              →  TransformerBlock #2   (same structure)
                 ↓
              →  LayerNorm
                 ↓
              →  Linear head (256 → 50257)
                 ↓
              →  softmax → next token probabilities
```

Key numbers:
- Context window : 100 tokens
- Embedding dim  : 256
- Attention ops  : 100×100 = 10,000 per layer per step
- Parameters     : ~26 million
- Tokenizer      : GPT-2 BPE via tiktoken (vocab = 50,257)

---

## How training and inference are separated

```
Training (SageMaker Training Job)          Inference (SageMaker Endpoint)
─────────────────────────────────          ──────────────────────────────
train.py runs in container                 inference.py runs in container
reads training text from S3                reads model.pt from S3
runs 50,000 optimisation steps             model_fn()   → loads weights once
saves model.pt + metadata.json → S3        input_fn()   → parses HTTP JSON
                                           predict_fn() → generates text
                                           output_fn()  → serialises response
```

Both containers use the **identical model class** (`TinyTransformer`).
Training sets `model.train()`, inference sets `model.eval()` + `torch.no_grad()`.

---

## AWS resources provisioned

```
CloudFormation stack: tiny-transformer-stack
│
├── S3 bucket:  tiny-transformer-<account>-<region>
│   ├── data/train.txt              ← training text (optional upload)
│   ├── code/                       ← train.py, inference.py uploaded by CI/CD
│   └── models/<job-name>/output/   ← model.pt + metadata.json saved here
│
├── IAM role: tiny-transformer-sagemaker-role
│   ├── AmazonSageMakerFullAccess
│   ├── s3:GetObject / PutObject on the bucket
│   └── CloudWatch logs
│
├── IAM role: tiny-transformer-github-actions-role
│   ├── Assumed via GitHub OIDC (no long-lived keys)
│   ├── sagemaker:CreateTrainingJob / CreateEndpoint / UpdateEndpoint
│   └── s3:PutObject on the bucket
│
├── GitHub OIDC Provider
│   └── token.actions.githubusercontent.com
│
└── SageMaker Model Package Groups
    ├── tiny-transformer-models      (v3 versions)
    └── tiny-transformer-v5-models   (v5 versions)
        └── Each trained version registered here for approval before deploy
```

---

## One-time setup

### Step 1 — Deploy CloudFormation stack

```bash
aws cloudformation deploy \
  --stack-name tiny-transformer-stack \
  --template-file cloudformation/stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1 \
  --parameter-overrides \
    ProjectName=tiny-transformer \
    GitHubOrg=yonatanbeyn \
    GitHubRepo=sagemaker \
    GitHubBranch=main
```

This creates all AWS resources. Takes ~2 minutes.

> **Re-run this exact command before the first v5 pipeline run,** even if the
> stack already exists. v5 needs two things the original stack did not grant:
> `s3:DeleteObject` on the GitHub Actions role (the workflow clears the dataset
> prefix before uploading a new split) and the `tiny-transformer-v5-models`
> registry group. `cloudformation deploy` updates in place and is a no-op for
> everything else. Without it, the v5 run fails at the dataset-staging step.

### Step 2 — Add GitHub secret

Get the GitHubActionsRole ARN from the stack output:

```bash
aws cloudformation describe-stacks \
  --stack-name tiny-transformer-stack \
  --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='GitHubActionsRoleArn'].OutputValue" \
  --output text
```

Add it to your GitHub repository:
```
Settings → Secrets and variables → Actions → New repository secret
Name:  AWS_ROLE_ARN
Value: arn:aws:iam::<account>:role/tiny-transformer-github-actions-role
```

### Step 3 — (Optional) Add manual approval gate for production

```
Settings → Environments → New environment → Name: production
Enable "Required reviewers" → add yourself
```

The deploy job will pause and wait for your approval before updating the endpoint.

---

## CI/CD pipeline flow

```
git push main
    │
    ▼
GitHub Actions: train-deploy.yml
    │
    ├─ [train job]
    │   ├─ Assume GitHubActionsRole via OIDC
    │   ├─ Upload code/ to S3
    │   ├─ Upload data/train.txt to S3  (if present)
    │   ├─ aws sagemaker create-training-job
    │   │     ├─ Instance: ml.m5.large
    │   │     ├─ Container: pytorch-training:2.1.0
    │   │     ├─ Hyperparams: steps=50000 seq-len=100 embed-dim=256
    │   │     └─ Runs train.py inside container
    │   └─ Waits for Completed status
    │
    ├─ [register job]
    │   └─ Registers model.pt in SageMaker Model Registry
    │      Status: PendingManualApproval
    │
    └─ [deploy job]  ← pauses here if production environment gate is set
        ├─ aws sagemaker create-model
        ├─ aws sagemaker create-endpoint-config
        ├─ aws sagemaker create-endpoint  (or update-endpoint if exists)
        ├─ Waits for InService
        └─ Smoke test: POST /invocations → checks response
```

---

## Invoking the deployed endpoint

```bash
# Replace with your actual endpoint name and region
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name tiny-transformer-endpoint \
  --region us-east-1 \
  --content-type application/json \
  --body '{"prompt": "The attention mechanism", "max_tokens": 80, "temperature": 0.8}' \
  response.json

cat response.json
```

Expected response:
```json
{
  "prompt": "The attention mechanism",
  "generated_text": "The attention mechanism allows the model to focus on ...",
  "tokens_generated": 18,
  "temperature": 0.8
}
```

Python SDK equivalent:
```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="tiny-transformer-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "prompt":      "A transformer is",
        "max_tokens":  60,
        "temperature": 0.8,
    })
)

result = json.loads(response["Body"].read())
print(result["generated_text"])
```

---

## Running locally (no AWS needed)

**v3:**
```bash
cd sagemaker/
pip install torch tiktoken

# Train (saves to ./model_output/)
python code/train.py --steps 5000 --seq-len 100 --embed-dim 256

# Infer from saved model
python code/inference.py ./model_output
```

**v5** — same entry points the training job and endpoint use. Without the
SageMaker channel env vars it downloads the base model and dataset from the
HuggingFace Hub directly:

```bash
cd sagemaker/
pip install -r code/requirements-v5.txt

# Fine-tune. Start with the 124M base — gpt2-medium on CPU is impractically slow.
SM_MODEL_DIR=./model_output_v5 SM_OUTPUT_DATA_DIR=./output_v5 \
python code/train_v5.py \
  --base-model gpt2 \
  --steps 200 \
  --batch-size 2 --grad-accum 2 \
  --seq-len 256

# Serve locally through the same four SageMaker hooks
python code/inference_v5.py ./model_output_v5
```

To rehearse the exact S3-channel path the training job takes, point the channel
env vars at local directories:

```bash
SM_CHANNEL_BASEMODEL=/path/to/gpt2-medium \   # config.json + model.safetensors
SM_CHANNEL_TRAINING=/path/to/jsonl_dir \      # *.jsonl of {"messages":[...]}
SM_MODEL_DIR=./model_output_v5 \
python code/train_v5.py --steps 200
```

---

## v5 — GPT-2 base + Mythos reasoning-trace fine-tuning

### What the pipeline does

```
GitHub Actions (ubuntu runner)
  1. snapshot_download gpt2-medium  → s3://BUCKET/v5/basemodel/gpt2-medium/
     (skipped when already staged — this is the slow 1.4GB step)
  2. load_dataset mythos → JSONL    → s3://BUCKET/v5/data/mythos.jsonl
  3. tar train_v5 + inference_v5 + gpt2_model + requirements
                                    → s3://BUCKET/v5/code/sourcedir.tar.gz
       ↓
SageMaker Training Job (ml.g5.xlarge, bf16 AMP)
  4. port pretrained weights into gpt2_model.GPT2
  5. sanity-check the port, then fine-tune with assistant-only loss
  6. save model.pt + metadata.json + bundled serving code → model.tar.gz
       ↓
SageMaker Model Registry (tiny-transformer-v5-models, PendingManualApproval)
       ↓
SageMaker Endpoint (ml.m5.xlarge) + smoke test
```

Both inputs are staged to S3 rather than downloaded inside the job, so training
needs no internet egress and a rerun uses byte-identical inputs.

### Why the architecture reverts to GPT-2 primitives

v4 used RoPE, RMSNorm and bias-free Linears. Pretrained weights are only
meaningful inside the architecture they were trained in, so `gpt2_model.py`
matches what GPT-2 actually used: learned position embeddings, LayerNorm with
bias, biases on every Linear except the tied `lm_head`, GELU with the tanh
approximation, and fused QKV in a single `c_attn`.

The one thing that silently breaks a hand-written GPT-2 port: HuggingFace stores
`c_attn`, `c_proj` and `c_fc` as **Conv1D** weights shaped `(in, out)` — the
transpose of what `nn.Linear` expects. The square ones load with no shape error
and simply produce nonsense. `train_v5.py` therefore scores the ported model on
a plain English sentence before training and **aborts** if the loss exceeds 6.0,
rather than spending an hour of GPU time fine-tuning a broken model.

### Assistant-only loss masking

Each conversation is kept as its own example (v3 packs everything into one
stream) and formatted with a ChatML-style template:

```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n
```

Every user token, and the assistant's role header, get label `-100`. Only the
assistant's content and its closing `<|im_end|>` are scored — so the model
learns to *answer* and to *stop*, not to imitate the user's questions. Roughly
40-45% of tokens end up supervised; the training log prints the exact share.

The template markers are plain BPE tokens (GPT-2 has no special chat tokens) and
live in `gpt2_model.py`, shared by training and serving. A mismatch here would
silently prompt the model in a format it never saw.

### Running it

Manual trigger, Actions → *Train and Deploy v5* → Run workflow:

| Input | Default | Notes |
|---|---|---|
| `steps` | `3200` | optimizer steps, not forward passes |
| `base_model` | `gpt2-medium` | `gpt2` (124M) trains ~3x faster |
| `dataset_split` | `train` | **all 25k traces**; slice with e.g. `train[:2000]` |
| `batch_size` / `grad_accum` | `4` / `4` | effective batch = 16 examples |
| `lr` | `3e-5` | ~10x lower than v3 — fine-tuning nudges weights |
| `training_instance` | `ml.g5.xlarge` | A10G 24GB, bf16 |
| `skip_deploy` | `false` | train + register only, leave the endpoint alone |

### Epoch coverage — the number that actually matters

Steps alone say nothing about how much data the model sees. The relationship is:

```
examples seen = steps x batch_size x grad_accum
epochs        = examples seen / training examples
```

At the defaults: `3200 x 4 x 4 = 51,200` examples over ~23,750 training rows
(25k minus the 5% validation split) = **~2.1 epochs**, which is the normal range
for SFT. `train_v5.py` prints this at startup and warns if a run is under one
full pass or past four epochs — the two ways this silently goes wrong when you
change `steps`, the batch size, or the split independently.

| steps | examples seen | epochs over 25k |
|---|---|---|
| 1500 | 24,000 | ~1.0 |
| 3200 | 51,200 | ~2.1 (default) |
| 4700 | 75,200 | ~3.2 |

Going much past 3 epochs on 25k traces mostly teaches the model to reproduce
them verbatim.

### Invoking the v5 endpoint

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name tiny-transformer-v5-endpoint \
  --region us-east-1 \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"prompt":"Explain how multi-head attention distributes work across heads.","max_tokens":150,"temperature":0.8}' \
  /dev/stdout
```

| Field | Default | Notes |
|---|---|---|
| `prompt` | required | wrapped in the chat template automatically |
| `max_tokens` | `150` | capped at 512 — CPU generation is ~10-20 tok/s |
| `temperature` | `0.8` | |
| `top_k` / `top_p` | `50` / `0.95` | `0` / `1.0` disables |
| `repetition_penalty` | `1.05` | `1.0` disables |
| `raw` | `false` | `true` skips the template and continues the text verbatim |

### Sizing notes (learned the hard way)

- **`ml.t2.medium` cannot serve v5.** 355M fp32 params is 1.4GB of weights
  before activations; the v5 workflow uses `ml.m5.xlarge` (16GB).
- **The default 600s startup health check is not enough.** The container pulls a
  ~1.4GB `model.tar.gz`, pip-installs tiktoken, then loads 355M params on CPU.
  The endpoint config sets both `ModelDataDownloadTimeoutInSeconds` and
  `ContainerStartupHealthCheckTimeoutInSeconds` to 1800.
- **`MaxRuntimeInSeconds` is 14400 (4h), not v3's 3600.** fp32 on a T4 is ~4s per
  step, so 3200 steps would blow a one-hour cap several times over. bf16 AMP on
  g5 brings it to ~1-2s/step, so a full 25k run is ~1-2h.
- **Checkpoints roll, they do not accumulate.** One eval checkpoint per 200
  steps at 1.4GB each would be ~23GB of redundant weights in `output.tar.gz`
  over a 3200-step run. Only the most recent is kept on disk; the *best*
  weights are held in memory and restored before the final save.
- **Examples are stored as `array.array('i')`, not Python lists.** Across the
  full 25k traces that is ~0.11GB instead of ~0.93GB — Python boxes every token
  id above 255 as a separate object, which adds up over 12.8M tokens.
- **No `learned_params.json` for v5.** v3 dumps its weights as JSON; the same
  dump for 355M floats is roughly 7GB and would make `model.tar.gz` unusable.
- **tiktoken's BPE cache is bundled** into `model.tar.gz/code/tiktoken_cache/`,
  so the endpoint never reaches out to `openaipublic.blob.core.windows.net` at
  cold start — which would fail outright in a no-egress VPC.

---

## Cost estimate

| Resource | Instance | Est. cost |
|---|---|---|
| v3 training job (50k steps) | ml.m5.large | ~$0.05 per run |
| v3 endpoint (24h) | ml.t2.medium | ~$0.06/hr = $1.44/day |
| **v5 training job (3200 steps, full 25k)** | ml.g5.xlarge | ~$1.41/hr × ~1.5h = **~$2.10 per run** |
| **v5 endpoint (24h)** | ml.m5.xlarge | ~$0.23/hr = **~$5.52/day** |
| S3 storage | ~100MB (v3) / ~3GB (v5 artifacts + base) | ~$0.07/month |

The v5 endpoint is the dominant cost. Delete it when idle — the model artifact
stays in S3, and rerunning only the `deploy` job brings it back.

```bash
# v3
aws sagemaker delete-endpoint --endpoint-name tiny-transformer-endpoint --region us-east-1
# v5
aws sagemaker delete-endpoint --endpoint-name tiny-transformer-v5-endpoint --region us-east-1
```

---

## Differences from local v3

| Aspect            | Local (genai_transformer_v3.py)     | SageMaker (train.py)                  |
|-------------------|-------------------------------------|---------------------------------------|
| Training data     | Hardcoded string in script          | Loaded from S3 (falls back to builtin)|
| Model saving      | JSON in current directory           | .pt checkpoint + JSON to SM_MODEL_DIR |
| Hyperparameters   | Constants at top of file            | argparse args from SageMaker job config|
| Inference         | generate_verbose() in same file     | Separate inference.py with 4 SM hooks |
| Weights loading   | Not needed (trained in same process)| model.pt via torch.load()             |
| Serving           | Direct Python function call         | HTTP POST to /invocations              |
