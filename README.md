# Pneumonia Detection from Chest X-Rays

Deep learning model that classifies chest radiographs (CXRs) into pneumonia / no-pneumonia, trained on the NIH ChestX-ray14 public dataset.

> **Status**: 🚧 In development · target AUROC ≥ 0.85 on held-out test set
> **Stack**: PyTorch + MONAI · DenseNet121 backbone · Gradio demo
> **Compute**: Trained on Google Colab A100 (free tier with credits)
> **License**: MIT (code) · NIH dataset has its own usage terms

---

## ⚠️ Disclaimer

**This is an educational/research project. Not for clinical use.**

- The model has NOT been validated on Thai patient population
- No regulatory clearance (FDA / Thai FDA / CE)
- Performance metrics are on held-out NIH data only — real-world generalization unverified
- Output is decision-support information, not diagnosis
- A licensed radiologist must interpret all medical images for clinical decisions

---

## Quick Demo

```bash
# Try the live demo (auth required via Cloudflare Access)
https://medical-ai.x2.dev          # (deployed via Hermes Agent infrastructure)
```

Upload any chest X-ray → model returns:
- Pneumonia probability (0-1)
- Heatmap showing what the model "looked at" (Grad-CAM)
- Confidence interval from MC dropout

---

## Project Structure

```
medical-ai/
├── README.md              # this file
├── requirements.txt       # pip install -r requirements.txt
├── docs/
│   ├── dataset_card.md    # NIH ChestX-ray14 details
│   ├── model_card.md      # architecture, training, biases
│   └── methodology.md     # full pipeline description
├── data/
│   ├── download.py        # fetch NIH dataset (~45GB)
│   └── splits/            # train/val/test CSVs (committed)
├── notebooks/
│   ├── 01_explore.ipynb   # EDA: class balance, image stats
│   ├── 02_train.ipynb     # Colab notebook for training
│   └── 03_evaluate.ipynb  # metrics + visualizations
├── src/
│   ├── model.py           # MONAI DenseNet121 setup
│   ├── dataset.py         # PyTorch Dataset + augmentation
│   ├── train.py           # CLI training entry point
│   ├── eval.py            # metric computation
│   └── inference.py       # single-image inference + GradCAM
├── demo/
│   └── app.py             # Gradio web demo
├── checkpoints/           # model weights (gitignored)
└── .gitignore
```

---

## Methodology

### Dataset
- **Source**: NIH ChestX-ray14 (Wang et al. 2017)
- **Size**: 112,120 frontal-view CXRs from 30,805 unique patients
- **Labels**: 14 disease classes (multi-label), mined from radiology reports via NLP
- **Subset used**: Binary task — "Pneumonia" vs "Not Pneumonia"
- **Split**: Official train/val/test splits maintained (patient-level, no leakage)

### Model
- **Backbone**: DenseNet121 pretrained on ImageNet (via MONAI)
- **Head**: Global average pool → linear → sigmoid (binary)
- **Input**: 224×224 grayscale → 3-channel replication
- **Augmentation**: random horizontal flip, slight rotation, intensity jitter (MONAI transforms)

### Training
- **Optimizer**: AdamW, lr=1e-4, weight decay=1e-5
- **Loss**: BCEWithLogitsLoss + class weighting (pneumonia is minority class)
- **Schedule**: cosine annealing, 30 epochs, early stopping on val AUROC
- **Hardware**: Google Colab A100 (free tier credits)
- **Time**: ~3 hours/epoch · ~90 hours total

### Evaluation
- Primary: **AUROC** (area under receiver operating characteristic)
- Secondary: AUPRC, sensitivity at 95% specificity, F1
- External: comparison to CheXNet (Rajpurkar et al. 2017, AUROC 0.768)

---

## Results

> _(filled in after training completes — placeholder values)_

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| AUROC | TBD | TBD | TBD |
| AUPRC | TBD | TBD | TBD |
| Sensitivity @ 95% Specificity | TBD | TBD | TBD |
| F1 (threshold=0.5) | TBD | TBD | TBD |

ROC curve, confusion matrix, calibration plot → see `notebooks/03_evaluate.ipynb`

---

## Failure Analysis

> _(filled in after training)_

Common failure modes:
- Lateral views (model expects frontal only)
- Pediatric images (model trained on adults)
- Atypical pneumonia presentations
- Hardware artifacts (lines, leads)

---

## Reproducibility

```bash
# 1. Setup
git clone https://github.com/aommiez/medical-ai-pneumonia.git
cd medical-ai-pneumonia
pip install -r requirements.txt

# 2. Download dataset (~45GB, takes ~1 hour)
python data/download.py

# 3. Train (use Colab notebook for GPU, or local with --device cpu for inference only)
python src/train.py --config configs/default.yaml

# 4. Evaluate
python src/eval.py --checkpoint checkpoints/best.pt

# 5. Launch demo
python demo/app.py
# → opens http://localhost:7860
```

---

## References

1. Wang X, et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* CVPR 2017.
2. Rajpurkar P, et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv 2017.
3. MONAI Project. *Medical Open Network for AI.* https://monai.io
4. Selvaraju RR, et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.

---

## Author

[@aommiez](https://github.com/aommiez)

Built as part of personal AI portfolio · 2026
