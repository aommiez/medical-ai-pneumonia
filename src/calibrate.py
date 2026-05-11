"""Calibration analysis + threshold tuning.

Generates:
- Calibration plot (reliability diagram)
- Threshold vs F1 / Precision / Sensitivity / Specificity curves
- Operating-point analysis (Youden's J, F1-max, screening, confirmation)
- Threshold suggestions for different use cases

Usage:
    python src/calibrate.py --checkpoint checkpoints/best.pt --data-root data/raw/chest_xray
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
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model


class CXRDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
        return self.transform(arr), torch.tensor([label], dtype=torch.float32)


def run_inference(model, data_root, device):
    tfm = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
    items = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for p in (Path(data_root) / "test" / cls).glob("*.jpeg"):
            items.append((p, 1 if cls == "PNEUMONIA" else 0))
    loader = DataLoader(CXRDataset(items, tfm), batch_size=32, shuffle=False, num_workers=2)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy().squeeze()
            ys.extend(np.atleast_1d(y.numpy().squeeze()).tolist())
            ps.extend(np.atleast_1d(p).tolist())
    return np.array(ys), np.array(ps)


def plot_calibration(y_true, y_prob, out_path):
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfectly calibrated")
    plt.plot(mean_pred, fraction_pos, "o-", color="#0066cc", linewidth=2,
             markersize=8, label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Reliability Diagram)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(y_true, y_prob, out_path):
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics = {"f1": [], "precision": [], "sensitivity": [], "specificity": []}
    for t in thresholds:
        pred = (y_prob > t).astype(int)
        metrics["f1"].append(f1_score(y_true, pred, zero_division=0))
        metrics["precision"].append(precision_score(y_true, pred, zero_division=0))
        metrics["sensitivity"].append(recall_score(y_true, pred, zero_division=0))
        # specificity = recall on negative class
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        metrics["specificity"].append(tn / max(tn + fp, 1))

    plt.figure(figsize=(8, 6))
    for k, c in zip(metrics, ["#cc3300", "#cc6600", "#0066cc", "#009933"]):
        plt.plot(thresholds, metrics[k], label=k, color=c, linewidth=2)
    plt.axvline(0.5, linestyle=":", color="gray", label="Default (0.5)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Metrics vs Decision Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return thresholds, metrics


def find_operating_points(y_true, y_prob):
    """Find recommended thresholds for different clinical use cases."""
    points = {}

    # 1. Youden's J (maximizes TPR - FPR)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    points["youden_j"] = {
        "threshold": float(thr[idx]),
        "use_case": "Balanced (maximize TPR-FPR)",
        "sensitivity": float(tpr[idx]),
        "specificity": float(1 - fpr[idx]),
    }

    # 2. F1-max
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    idx = int(np.argmax(f1s))
    pred = (y_prob > thresholds[idx]).astype(int)
    tn = ((pred == 0) & (y_true == 0)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    points["f1_max"] = {
        "threshold": float(thresholds[idx]),
        "use_case": "Maximum F1 (best balance precision/recall)",
        "f1": float(f1s[idx]),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
    }

    # 3. High sensitivity (screening) — sensitivity ≥ 99%
    for t in thresholds:
        pred = (y_prob > t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        if sens >= 0.99:
            tn = ((pred == 0) & (y_true == 0)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            spec = tn / max(tn + fp, 1)
            points["screening"] = {
                "threshold": float(t),
                "use_case": "Screening (sensitivity ≥ 99%)",
                "sensitivity": float(sens),
                "specificity": float(spec),
            }
            break

    # 4. High specificity (confirmation) — specificity ≥ 95%
    for t in reversed(thresholds):
        pred = (y_prob > t).astype(int)
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        spec = tn / max(tn + fp, 1)
        sens = recall_score(y_true, pred, zero_division=0)
        if spec >= 0.95:
            points["confirmation"] = {
                "threshold": float(t),
                "use_case": "Confirmation (specificity ≥ 95%)",
                "sensitivity": float(sens),
                "specificity": float(spec),
            }
            break

    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = build_model(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    print("Running inference...")
    y_true, y_prob = run_inference(model, args.data_root, args.device)

    print("Generating calibration plot...")
    plot_calibration(y_true, y_prob, out_dir / "calibration.png")

    print("Generating threshold sweep...")
    plot_threshold_sweep(y_true, y_prob, out_dir / "threshold_sweep.png")

    print("Finding operating points...")
    points = find_operating_points(y_true, y_prob)
    for name, info in points.items():
        print(f"\n{name}: {info}")

    (out_dir / "operating_points.json").write_text(json.dumps(points, indent=2))
    print(f"\nSaved operating_points.json")


if __name__ == "__main__":
    main()
