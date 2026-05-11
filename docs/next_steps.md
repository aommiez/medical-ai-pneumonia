# Next Steps — How to actually finish this portfolio

The scaffold is built. To complete the project:

## Phase 1 — Get the data (1-2 hours)

```bash
cd medical-ai/
pip install kaggle

# Get kaggle.json from https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download the smaller pneumonia subset first
python data/download.py --subset
```

You should end up with:
```
data/raw/chest_xray/
├── train/  (NORMAL/ + PNEUMONIA/)
├── val/
└── test/
```

## Phase 2 — Train on Colab (3-12 hours of GPU time)

1. Upload `medical-ai/` to Google Drive (or push to GitHub, then `git clone` in Colab)
2. Open `notebooks/02_train_colab.py` → File → New notebook → paste cells
3. Runtime → Change runtime type → **T4 GPU** (or A100 if Pro)
4. Mount Drive → fetch project → run cells

Expected: AUROC 0.90+ on val set (subset is easier than full CXR14).

## Phase 3 — Evaluate + Visualize

In Colab or locally with checkpoint:
- Compute test AUROC, AUPRC, sensitivity/specificity
- Plot ROC curve, confusion matrix
- Run Grad-CAM on a few test images
- Add results to README.md

## Phase 4 — Deploy demo on Hermes VPS

The VPS already has Cloudflare Tunnel. Set up new ingress:

```bash
# On VPS, install demo deps
pip install --user gradio monai torch

# Copy code + checkpoint to VPS
scp -r medical-ai/ hermes@45.154.26.57:~/
scp checkpoints/best.pt hermes@45.154.26.57:~/medical-ai/checkpoints/

# Run as systemd service
# (see ../scripts/medical-ai-demo.service)

# Add to Cloudflare Tunnel config
# notes.x2.dev style:
# hostname: medical-ai.x2.dev → http://127.0.0.1:7860
```

## Phase 5 — Publish to GitHub

```bash
cd medical-ai/
git init
git add .
git commit -m "Initial pneumonia classifier"

# Create on GitHub via gh CLI
gh repo create medical-ai-pneumonia --public --source=. --push
```

Add to LinkedIn / portfolio sites.

## Phase 6 — Stretch goals

- [ ] Compare to CheXNet baseline (Rajpurkar et al. — they got AUROC 0.768)
- [ ] Train on full CXR14 multi-label (14 diseases, not just pneumonia)
- [ ] Ensemble with EfficientNet / ResNet-50
- [ ] External validation on PadChest or MIMIC-CXR
- [ ] Add segmentation (U-Net for lung region) → bounding box of pneumonia region
- [ ] Write up as Medium blog post / arXiv preprint
- [ ] Submit to Kaggle competition for ranking
