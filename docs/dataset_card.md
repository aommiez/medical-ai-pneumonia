# Dataset Card — NIH ChestX-ray14

## Source

**Name**: NIH ChestX-ray14 (also known as ChestX-ray8 / CXR14)
**Released**: 2017 by NIH Clinical Center
**Paper**: Wang X, et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." CVPR 2017.
**Official download**: https://nihcc.app.box.com/v/ChestXray-NIHCC
**Mirror on Kaggle**: https://www.kaggle.com/datasets/nih-chest-xrays/data

## Statistics

| Property | Value |
|----------|-------|
| Total images | 112,120 |
| Unique patients | 30,805 |
| Image format | PNG, 1024×1024, grayscale |
| Total size | ~45GB |
| View positions | PA (posterior-anterior) and AP (anterior-posterior) |
| Age range | 1-95 years (mean 47) |
| Sex | ~56% male, ~44% female |

## Labels

14 disease classes, multi-label (an image can have 0, 1, or several):
- Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Emphysema
- Fibrosis · Hernia · Infiltration · Mass · Nodule · **Pneumonia** · Pleural Thickening · Pneumothorax

For this project: **binary classification — Pneumonia vs Not Pneumonia**

## Label distribution

- Pneumonia positive: ~1,431 images (1.3%) — **highly imbalanced**
- Pneumonia negative: ~110,689 images (98.7%)

We address imbalance via:
- `pos_weight` in BCEWithLogitsLoss (~77x for pneumonia class)
- Stratified sampling per batch
- AUROC as primary metric (robust to imbalance)

## Splits

Official splits provided by NIH:
- Train: 86,524 images
- Val: 6,335 images
- Test: 25,596 images

**Patient-level splits** — no patient appears in multiple splits (prevents leakage).

## Known Limitations

⚠️ **Labels are noisy.** They were mined automatically from radiology reports using NLP (no manual review). Estimated label accuracy: 60-80%. Some authors recommend treating CXR14 as a "weakly supervised" dataset.

⚠️ **No segmentation.** Only image-level labels, no bounding boxes (except for a small held-out 880-image evaluation set).

⚠️ **Demographic bias.**
- All images from NIH Clinical Center (USA) — not representative globally
- Skewed toward older patients (mean age 47)
- May not generalize to Thai patient population without additional validation

⚠️ **No pediatric population.** Adult images only.

⚠️ **Hospital-acquired data.** Patients are typically inpatient → more severe pathology than general population.

## Ethics & Usage

The dataset was released by NIH for research purposes. No PHI (Protected Health Information) — images are anonymized.

For your research / portfolio:
- ✅ Free to use for research, education, publication
- ✅ Cite original paper
- ❌ Do NOT use for clinical product without revalidation
- ❌ Do NOT distribute the images further (download from official source)

## Comparison to Other CXR Datasets

| Dataset | Images | Labels | Quality | License |
|---------|--------|--------|---------|---------|
| **NIH CXR14** | 112k | 14, NLP-mined | Moderate | Public domain |
| CheXpert (Stanford) | 224k | 14, uncertainty labels | Higher | Research-only |
| MIMIC-CXR (MIT) | 377k | 14, radiologist-reviewed | High | DUA required |
| PadChest (Spain) | 160k | 174 labels, manual+NLP | High | Open |
| OpenI (NLM) | 7,470 | report-based | High | Public |

NIH is chosen here for: **size + simplicity + no DUA + accepted benchmark**

## How to Download

```bash
# Option 1: Kaggle (full mirror, requires kaggle.json)
kaggle datasets download -d nih-chest-xrays/data
unzip data.zip -d data/raw/

# Option 2: Direct from NIH (12 tar files, ~3-4GB each)
wget https://nihcc.box.com/shared/static/<each_tar>
# (see https://nihcc.app.box.com/v/ChestXray-NIHCC for current links)

# Option 3: Pre-split pneumonia subset (Kaggle, ~2GB, easier for first try)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```
