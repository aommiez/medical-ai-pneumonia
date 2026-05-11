"""Evaluation script: compute test metrics + generate ROC curve + Grad-CAM examples.

Run on VPS with checkpoint available:
    python src/eval.py --checkpoint checkpoints/best.pt --data-root data/raw/chest_xray
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model


def evaluate(model, data_root, device, batch_size=32):
    """Run inference on test split. Returns (y_true, y_prob, image_paths)."""
    from torch.utils.data import DataLoader, Dataset

    class CXRDataset(Dataset):
        def __init__(self, paths_labels, transform):
            self.items = paths_labels
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
            t = self.transform(arr)
            return t, torch.tensor([label], dtype=torch.float32), str(path)

    tfm = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
    items = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for img in (Path(data_root) / "test" / cls).glob("*.jpeg"):
            items.append((img, 1 if cls == "PNEUMONIA" else 0))

    ds = CXRDataset(items, tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    ys, ps, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy().squeeze()
            ys.extend(np.atleast_1d(y.numpy().squeeze()).tolist())
            ps.extend(np.atleast_1d(prob).tolist())
            paths.extend(p)
    return np.array(ys), np.array(ps), paths


def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}", linewidth=2, color="#0066cc")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve — Pneumonia Classifier (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return roc_auc


def plot_pr(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", linewidth=2, color="#cc3300")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return ap


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    classes = ["Normal", "Pneumonia"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=color, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def gradcam_examples(model, img_paths, out_dir, device):
    """Generate Grad-CAM overlays for 2 sample images (1 normal, 1 pneumonia)."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("grad-cam not installed, skipping")
        return []

    tfm = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
    target_layer = model.features.denseblock4
    cam = GradCAM(model=model, target_layers=[target_layer])

    out_paths = []
    for label, path in img_paths:
        img = Image.open(path).convert("RGB").resize((224, 224))
        rgb = np.array(img).astype(np.float32) / 255.0
        arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
        x = tfm(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        grayscale_cam = cam(input_tensor=x, targets=None)[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"Original — {label}")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM — prediction: {prob:.1%}")
        axes[1].axis("off")
        plt.tight_layout()
        out_path = Path(out_dir) / f"gradcam_{label.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        out_paths.append(out_path)
        print(f"saved {out_path}")
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True, help="path to chest_xray dir (containing test/)")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint} to {args.device}")
    model = build_model(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    print("Running inference on test set...")
    y_true, y_prob, paths = evaluate(model, args.data_root, args.device)

    # metrics
    y_pred = (y_prob > 0.5).astype(int)
    auroc = plot_roc(y_true, y_prob, out_dir / "roc_curve.png")
    ap_score = plot_pr(y_true, y_prob, out_dir / "pr_curve.png")
    cm = plot_confusion(y_true, y_pred, out_dir / "confusion_matrix.png")
    f1 = f1_score(y_true, y_pred)

    # sensitivity at 95% specificity
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr[fpr <= 0.05])
    sens_at_95 = tpr[fpr <= 0.05][idx] if len(tpr[fpr <= 0.05]) else 0.0

    report = classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"], digits=4)
    print(report)

    # Save metrics JSON
    import json
    metrics = {
        "test_auroc": float(auroc),
        "test_auprc": float(ap_score),
        "test_f1": float(f1),
        "sensitivity_at_95_specificity": float(sens_at_95),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_true)),
        "test_positives": int(y_true.sum()),
        "test_negatives": int((y_true == 0).sum()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")

    # Grad-CAM examples
    normal_paths = [p for p, y in zip(paths, y_true) if y == 0]
    pneumonia_paths = [p for p, y in zip(paths, y_true) if y == 1]
    sample = [
        ("Normal", normal_paths[0]),
        ("Pneumonia", pneumonia_paths[0]),
    ]
    gradcam_examples(model, sample, out_dir, args.device)

    print(f"\n✓ Done. All artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
