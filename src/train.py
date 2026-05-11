"""Training entry point for pneumonia classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from monai.transforms import (
    Compose,
    EnsureType,
    RandFlip,
    RandRotate,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NIHPneumoniaDataset
from src.model import build_model


def get_transforms(train: bool):
    base = [Resize((224, 224)), ScaleIntensity()]
    if train:
        base += [RandFlip(prob=0.5, spatial_axis=1),
                 RandRotate(range_x=0.1, prob=0.5)]
    base += [ToTensor(), EnsureType()]
    return Compose(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="checkpoints/best.pt")
    args = ap.parse_args()

    train_ds = NIHPneumoniaDataset(args.data_root, args.train_csv, get_transforms(True))
    val_ds = NIHPneumoniaDataset(args.data_root, args.val_csv, get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(num_classes=1, pretrained=True).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    pos_weight = train_ds.class_weights().to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    best_auroc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                p = torch.sigmoid(model(x)).cpu().numpy().squeeze()
                ys.extend(y.numpy().squeeze().tolist())
                ps.extend(p.tolist())
        auroc = roc_auc_score(ys, ps)
        print(f"epoch {epoch+1}: val AUROC = {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), args.output)
            print(f"  → saved best to {args.output}")

    print(f"\nDone. Best val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
