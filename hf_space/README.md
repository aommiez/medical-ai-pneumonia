---
title: Pneumonia Detector — Chest X-Ray
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
tags:
  - medical-imaging
  - radiology
  - pneumonia
  - chest-xray
  - pytorch
  - monai
  - densenet
---

# 🫁 Pneumonia Detection from Chest X-Rays

Binary classifier (DenseNet121) trained on Kaggle's [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

**Test AUROC: 0.9138** · **Sensitivity: 99.5%** · **Specificity: 58.1%** (default threshold)

## ⚠️ Disclaimer

**Educational demo only — not for clinical use.**
- No regulatory clearance (FDA / Thai FDA / CE)
- Performance validated only on a single held-out test set
- Output is decision-support information, not a diagnosis
- A licensed radiologist must interpret all medical images for clinical decisions

## Links

- **GitHub repo:** https://github.com/aommiez/medical-ai-pneumonia
- **VPS demo (alt):** https://medical-ai.x2.dev
- **WandB training:** https://wandb.ai/aommiez-savel-co-ltd/medical-ai-pneumonia
- **Blog post:** [BLOG_POST.md](https://github.com/aommiez/medical-ai-pneumonia/blob/main/BLOG_POST.md)

## How it works

1. Upload a frontal chest X-ray image (JPG/PNG)
2. The model resizes to 224×224, normalizes intensity, runs DenseNet121 inference
3. Output: pneumonia probability + Grad-CAM heatmap (showing attention)

## Architecture

- DenseNet121 (6.95M params) pretrained on ImageNet
- Fine-tuned 10 epochs on Modal Labs T4 GPU
- Loss: BCEWithLogitsLoss with `pos_weight` for class imbalance
- Inference: CPU (HF Space free tier), ~3-5s per image
