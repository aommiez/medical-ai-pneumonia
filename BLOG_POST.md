# Building a Pneumonia Detector in One Afternoon — DenseNet121 + Modal Labs + Cloudflare Tunnel

*From `git init` to live demo in 4 hours. Total cost: $0.50.*

---

## TL;DR

I trained a DenseNet121 chest X-ray classifier to **0.91 AUROC** on the Kaggle pneumonia dataset, then deployed it as a live web demo at https://medical-ai.x2.dev — all in about 4 hours, costing less than a coffee.

**Stack:** PyTorch + MONAI for training · Modal Labs T4 GPU for compute · Gradio for UI · Cloudflare Tunnel for hosting · WandB for tracking.

The interesting part is **not the model** — it's the workflow. Here's the breakdown.

---

## The problem

Medical imaging AI projects are notorious for being expensive and slow:
- "You need an A100 cluster"
- "You need a HIPAA-compliant data pipeline"
- "Hospital partnerships take months"

That's all true for *production* medical AI. But for **a portfolio piece showing you understand medical CV**, you can move much faster.

The goals:
- ✅ Train a model that beats CheXNet's published baseline (AUROC 0.768)
- ✅ Use **only open data** (no PHI, no IRB needed)
- ✅ Deploy a live demo with **Grad-CAM** (so reviewers can see model attention)
- ✅ Write reproducible code + documentation
- ✅ Total cost: target < $5

---

## Step 1 — Dataset (5 minutes)

I used the Kaggle [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney.

| Split | Normal | Pneumonia |
|-------|--------|-----------|
| Train | 1,341 | 3,875 |
| Test | 234 | 390 |

Notable: pediatric population (Guangzhou Women and Children's Medical Center), frontal views only, ~5k training images. Small but well-curated — perfect for a 1-day project.

```python
# data/download.py
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "paultimothymooney/chest-xray-pneumonia",
    "--unzip",
], check=True)
```

---

## Step 2 — Model (10 minutes)

DenseNet121 with ImageNet pretraining. MONAI makes this a one-liner:

```python
from monai.networks.nets import DenseNet121

model = DenseNet121(
    spatial_dims=2,
    in_channels=3,      # grayscale replicated to 3 channels
    out_channels=1,     # binary
    pretrained=True,
)
```

Why DenseNet121?
- **Proven** for medical imaging (CheXNet uses it)
- **Compact** — 6.95M params, fits any GPU
- **Pretrained on ImageNet** — texture features transfer well

---

## Step 3 — Training on Modal Labs (15 minutes setup, 5 minutes train)

Here's where things get interesting. Instead of fighting with Colab disconnects or buying a GPU machine, I used **Modal Labs**.

Modal is essentially "serverless GPU." You write Python with decorators:

```python
import modal

app = modal.App("pneumonia-classifier")

@app.function(gpu="T4", timeout=2*3600, secrets=[KAGGLE, WANDB])
def train_pneumonia(epochs: int = 10) -> bytes:
    # ... training code runs on Modal's T4 GPU ...
    return checkpoint_bytes

@app.local_entrypoint()
def main(epochs: int = 10):
    ckpt = train_pneumonia.remote(epochs=epochs)
    Path("checkpoints/best.pt").write_bytes(ckpt)
```

Then:
```bash
modal run modal_train.py --epochs 10
```

That's it. Modal spins up a T4 GPU, runs my function, returns the bytes. **No SSH, no `screen`, no babysitting.**

Training took **5 minutes**. Cost: **$0.30**. Modal's free tier gives $30/month, so this was effectively free.

Compared to:
- Colab free: needs me to keep tab open + click + sometimes disconnects
- Renting a dedicated GPU instance: persistent cost even when idle
- Local GPU: $1500+ upfront

For one-shot training jobs, Modal is the right tool.

---

## Step 4 — Results

After 10 epochs:

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.9138** |
| F1 | 0.886 |
| Sensitivity (recall on pneumonia) | **99.5%** |
| Specificity (recall on normal) | 58.1% |

That **0.91 AUROC beats CheXNet's 0.768** on the multi-label NIH dataset. Of course this is a different (easier) dataset, but the bar is set.

The model is **biased toward catching pneumonia** (false-positive on normals). This is intentional — `pos_weight` in the loss function. For a screening tool, missing a true positive is worse than a false alarm.

---

## Step 5 — Calibration analysis (the part most tutorials skip)

The default 0.5 threshold is rarely optimal. After running `calibrate.py`:

| Use case | Threshold | Sensitivity | Specificity |
|----------|-----------|-------------|-------------|
| Default | 0.50 | 99.5% | 58.1% |
| **Youden's J (balanced)** | **0.96** | **93.1%** | **80.8%** |
| F1-max | 0.93 | 95.6% | 77.8% |
| Screening (≥99% sens) | 0.01 | 100% | 19.2% |

The model is **overconfident** — pushes probabilities toward extremes. Moving the threshold from 0.5 to 0.96 *dramatically* improves specificity while keeping high sensitivity.

This is a **classic finding for class-weighted training**, and a great talking point in interviews.

---

## Step 6 — Grad-CAM (visualizing model attention)

For medical AI, visualization is non-negotiable. You need to show the model isn't picking up spurious correlations (e.g., text labels in the corner of the image).

```python
from pytorch_grad_cam import GradCAM

target_layer = model.features.denseblock4
cam = GradCAM(model=model, target_layers=[target_layer])
heatmap = cam(input_tensor=x, targets=None)[0]
```

In the demo, every prediction comes with a Grad-CAM overlay. Reviewers can verify the model attends to lung fields, not background artifacts.

---

## Step 7 — Deploy via Cloudflare Tunnel (15 minutes)

I already had a VPS with Cloudflare Tunnel running (for [hermes-agent](https://hermes-agent.nousresearch.com) infrastructure). Adding the medical demo:

1. **Gradio app** (`demo/app.py`) — drag-drop X-ray, output prediction + Grad-CAM
2. **systemd service** — auto-restart on crash
3. **Tunnel ingress** — `medical-ai.x2.dev → 127.0.0.1:7860`
4. **DNS CNAME** auto-created via `cloudflared tunnel route dns`

```yaml
# ~/.cloudflared/config.yml
ingress:
  - hostname: medical-ai.x2.dev
    service: http://127.0.0.1:7860
```

No nginx config, no SSL certbot, no port forwarding. Cloudflare handles everything.

---

## Step 8 — Auto-deploy pipeline

Here's the actual end-to-end pipeline:

```
$ modal run modal_train.py --epochs 10
  ↓
Modal T4 GPU trains
  ↓
checkpoint downloaded to local
  ↓
systemd watcher detects new checkpoint
  ↓
medical-ai-demo service restarts
  ↓
Cloudflare Tunnel routes traffic
  ↓
🌐 https://medical-ai.x2.dev (live in 5 minutes)
```

`hermes-agent` even sent me a Telegram notification when training finished. Total handoff: zero clicks.

---

## What I learned

1. **Modal is dramatically underrated for one-shot ML jobs.** No infra to manage, sub-second billing, generous free tier.

2. **Calibration analysis is mandatory.** A model trained with `pos_weight` will be miscalibrated by default — must tune threshold post-hoc.

3. **Grad-CAM is sufficient for "model interpretability"** in portfolio context. SHAP, integrated gradients are nicer but Grad-CAM gets 80% of the value.

4. **Cloudflare Tunnel makes hosting trivial.** I no longer think about ingress, SSL, port forwarding.

5. **Disclaimer prominently.** This is a research project. Real medical AI requires FDA clearance, hospital validation, IRB approval. I included clear disclaimers everywhere.

---

## What's next

The current model is a **screening tool**, not a diagnostic tool. To make it real-world useful:

- Train on full NIH ChestX-ray14 (112k images, 14 diseases — not just pneumonia)
- External validation on PadChest / MIMIC-CXR
- Calibrate to clinical operating points
- IRB partnership for prospective evaluation
- ONNX export for mobile inference

If you want to try the live demo: **https://medical-ai.x2.dev**

Code: **https://github.com/aommiez/medical-ai-pneumonia**

---

*This article is part of a series on building production-adjacent AI systems quickly. Follow for more.*

**Disclaimer:** Educational project only. The model has not been validated for clinical use. Do not use for actual diagnosis.
