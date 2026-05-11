"""Cross-domain evaluation: Pediatric (Kaggle) vs Adult (NIH) generalization.

Tests:
1. Pediatric model → Kaggle test (in-domain) ← already measured
2. Pediatric model → NIH pneumonia subset (out-of-domain)
3. NIH multi-label model (pneumonia output) → NIH test (in-domain)
4. NIH multi-label model → Kaggle pneumonia test (out-of-domain)

Highlights generalization gap when domain shifts (pediatric ↔ adult).

Usage:
    python src/cross_domain_eval.py \
        --pediatric-ckpt checkpoints/best.pt \
        --nih-ckpt checkpoints/multilabel_sample.pt \
        --kaggle-data data/raw/chest_xray \
        --nih-data /vol/nih_sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model


NIH_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
PNEUMONIA_IDX = NIH_DISEASES.index("Pneumonia")


class ImgLabelDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
        return self.transform(arr), torch.tensor([label], dtype=torch.float32)


def load_kaggle_test(root):
    """Returns list of (path, pneumonia_label)."""
    items = []
    test_dir = Path(root) / "test"
    for cls in ["NORMAL", "PNEUMONIA"]:
        for p in (test_dir / cls).glob("*.jpeg"):
            items.append((p, 1 if cls == "PNEUMONIA" else 0))
    return items


def load_nih_test(root, max_samples=2000):
    """Returns list of (path, pneumonia_label) from NIH dataset.

    Uses Data_Entry_2017.csv or sample_labels.csv to find pneumonia labels.
    """
    root_p = Path(root)
    csvs = list(root_p.glob("**/Data_Entry_2017.csv")) + list(root_p.glob("**/sample_labels.csv"))
    if not csvs:
        raise FileNotFoundError(f"No labels CSV in {root}")
    df = pd.read_csv(csvs[0])
    img_map = {p.name: p for p in root_p.glob("**/*.png")}
    df["path"] = df["Image Index"].map(img_map)
    df = df.dropna(subset=["path"])
    df["pneumonia"] = df["Finding Labels"].apply(lambda s: int("Pneumonia" in str(s).split("|")))
    # balanced sampling: all positives + equal negatives
    pos = df[df.pneumonia == 1]
    neg = df[df.pneumonia == 0].sample(min(len(pos), len(df) - len(pos), max_samples // 2), random_state=42)
    pos = pos.head(min(len(pos), max_samples // 2))
    combined = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    return [(row.path, int(row.pneumonia)) for row in combined.itertuples()]


def run_inference(model, items, device, output_slice=None):
    """Run model on items. If output_slice given (e.g. PNEUMONIA_IDX), extract that channel."""
    tfm = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
    loader = DataLoader(ImgLabelDataset(items, tfm), batch_size=16, shuffle=False, num_workers=2)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if output_slice is not None:
                p = torch.sigmoid(logits[:, output_slice]).cpu().numpy()
            else:
                p = torch.sigmoid(logits).cpu().numpy().squeeze()
            ys.extend(np.atleast_1d(y.numpy().squeeze()).tolist())
            ps.extend(np.atleast_1d(p).tolist())
    return np.array(ys), np.array(ps)


def plot_cross_domain_roc(results, out_path):
    plt.figure(figsize=(8, 8))
    colors = ["#0066cc", "#cc3300", "#009933", "#cc6600"]
    for (label, ys, ps), c in zip(results, colors):
        fpr, tpr, _ = roc_curve(ys, ps)
        auroc = roc_auc_score(ys, ps)
        plt.plot(fpr, tpr, label=f"{label}: AUROC={auroc:.3f}", color=c, linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Domain Generalization — Pediatric ↔ Adult")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pediatric-ckpt", required=True)
    ap.add_argument("--nih-ckpt", required=True)
    ap.add_argument("--kaggle-data", required=True)
    ap.add_argument("--nih-data", required=True)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading pediatric model: {args.pediatric_ckpt}")
    ped_model = build_model(num_classes=1, pretrained=False)
    ped_model.load_state_dict(torch.load(args.pediatric_ckpt, map_location=args.device))
    ped_model.to(args.device).eval()

    print(f"Loading NIH multi-label model: {args.nih_ckpt}")
    nih_model = build_model(num_classes=14, pretrained=False)
    nih_model.load_state_dict(torch.load(args.nih_ckpt, map_location=args.device))
    nih_model.to(args.device).eval()

    # Load data
    print("Loading Kaggle test data...")
    kaggle_items = load_kaggle_test(args.kaggle_data)
    print(f"  {len(kaggle_items)} items")

    print("Loading NIH test data...")
    nih_items = load_nih_test(args.nih_data, max_samples=1000)
    print(f"  {len(nih_items)} items")

    # 4 experiments
    print("\n[1/4] Pediatric model → Kaggle (in-domain)...")
    y1, p1 = run_inference(ped_model, kaggle_items, args.device)
    auroc1 = roc_auc_score(y1, p1)

    print("[2/4] Pediatric model → NIH (out-of-domain adults)...")
    y2, p2 = run_inference(ped_model, nih_items, args.device)
    auroc2 = roc_auc_score(y2, p2)

    print("[3/4] NIH model → NIH (in-domain adults, pneumonia slice)...")
    y3, p3 = run_inference(nih_model, nih_items, args.device, output_slice=PNEUMONIA_IDX)
    auroc3 = roc_auc_score(y3, p3)

    print("[4/4] NIH model → Kaggle (out-of-domain pediatric)...")
    y4, p4 = run_inference(nih_model, kaggle_items, args.device, output_slice=PNEUMONIA_IDX)
    auroc4 = roc_auc_score(y4, p4)

    print("\n=== CROSS-DOMAIN RESULTS ===")
    print(f"  Pediatric model on Kaggle (in-domain):  AUROC = {auroc1:.4f}")
    print(f"  Pediatric model on NIH (out-of-domain): AUROC = {auroc2:.4f}  [drop: {auroc1-auroc2:+.4f}]")
    print(f"  NIH model on NIH (in-domain):          AUROC = {auroc3:.4f}")
    print(f"  NIH model on Kaggle (out-of-domain):   AUROC = {auroc4:.4f}  [drop: {auroc3-auroc4:+.4f}]")

    # Plot
    results = [
        ("Pediatric → Kaggle (in-domain)", y1, p1),
        ("Pediatric → NIH adults (out-of-domain)", y2, p2),
        ("NIH adults → NIH (in-domain)", y3, p3),
        ("NIH adults → Kaggle pediatric (out-of-domain)", y4, p4),
    ]
    plot_cross_domain_roc(results, out_dir / "cross_domain_roc.png")

    # Save metrics
    metrics = {
        "pediatric_on_kaggle_in_domain": float(auroc1),
        "pediatric_on_nih_ood": float(auroc2),
        "nih_on_nih_in_domain": float(auroc3),
        "nih_on_kaggle_ood": float(auroc4),
        "ped_to_adult_drop": float(auroc1 - auroc2),
        "adult_to_ped_drop": float(auroc3 - auroc4),
    }
    (out_dir / "cross_domain_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {out_dir}/cross_domain_metrics.json")


if __name__ == "__main__":
    main()
