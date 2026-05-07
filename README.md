# TinyTransformer v3 — AWS SageMaker Deployment

Productionised version of `simplegenai/genai_transformer_v3.py`.
Same model weights, same architecture — moved to AWS for scalable training and serving.

---

## Folder structure

```
sagemaker/
├── code/
│   ├── train.py          training entry point  (runs inside SageMaker container)
│   ├── inference.py      serving entry point   (runs inside SageMaker endpoint)
│   └── requirements.txt  torch, tiktoken
├── cloudformation/
│   └── stack.yaml        one-click AWS infrastructure provisioning
├── data/
│   └── train.txt         (optional) custom training text — uploaded to S3
└── .github/
    └── workflows/
        └── train-deploy.yml   CI/CD: push → train → register → deploy
```

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
└── SageMaker Model Package Group: tiny-transformer-models
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

```bash
cd sagemaker/

# Install dependencies
pip install torch tiktoken

# Train (saves to ./model_output/)
python code/train.py \
  --steps 5000 \
  --seq-len 100 \
  --embed-dim 256

# Infer from saved model
python code/inference.py ./model_output
```

---

## Cost estimate

| Resource               | Instance     | Est. cost          |
|------------------------|--------------|--------------------|
| Training job (50k steps) | ml.m5.large  | ~$0.05 per run     |
| Endpoint (24h)         | ml.t2.medium | ~$0.06/hr = $1.44/day |
| S3 storage             | ~100MB       | ~$0.002/month      |

Stop the endpoint when not in use:
```bash
aws sagemaker delete-endpoint --endpoint-name tiny-transformer-endpoint --region us-east-1
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
