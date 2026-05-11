"""Modal multi-label training: NIH ChestX-ray14 (14 diseases).

Trains DenseNet121 multi-label classifier on NIH CXR14 dataset.
Uses Kaggle's "sample" subset by default (~6k images, fast) or full (112k images).

Usage:
    modal run modal_train_multilabel.py --epochs 5 --subset sample
    modal run modal_train_multilabel.py --epochs 15 --subset full --gpu A10G
"""
from __future__ import annotations

import modal

app = modal.App("nih-cxr14-multilabel")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "monai==1.4.0",
        "pandas",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "Pillow",
        "wandb",
        "kaggle",
    )
)

volume = modal.Volume.from_name("nih-cxr14-data", create_if_missing=True)
KAGGLE_SECRET = modal.Secret.from_name("kaggle-secret")
WANDB_SECRET = modal.Secret.from_name("wandb-secret")


NIH_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    secrets=[KAGGLE_SECRET, WANDB_SECRET],
    volumes={"/vol": volume},
)
def train_multilabel(epochs: int = 5, batch_size: int = 32, lr: float = 1e-4,
                     subset: str = "sample") -> dict:
    """Train multi-label classifier. Returns dict with checkpoint bytes + metrics."""
    import os
    import subprocess
    from pathlib import Path

    # Kaggle creds
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text(
        f'{{"username":"{os.environ["KAGGLE_USERNAME"]}","key":"{os.environ["KAGGLE_KEY"]}"}}'
    )
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

    # Download NIH dataset (sample or full)
    if subset == "sample":
        slug = "nih-chest-xrays/sample"
        data_root = Path("/vol/nih_sample")
    elif subset == "full":
        slug = "nih-chest-xrays/data"
        data_root = Path("/vol/nih_full")
    else:
        raise ValueError(f"unknown subset: {subset}")

    if not data_root.exists() or not list(data_root.glob("**/*.png"))[:1]:
        print(f"Downloading {slug} ({subset}) ...")
        data_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(data_root), "--unzip"],
            check=True,
        )
        volume.commit()
    else:
        print(f"Dataset cached at {data_root}")

    # Find labels CSV
    csv_paths = list(data_root.glob("**/Data_Entry_2017.csv")) + list(data_root.glob("**/sample_labels.csv"))
    if not csv_paths:
        raise RuntimeError(f"no labels CSV found in {data_root}")
    csv_path = csv_paths[0]
    print(f"Using labels: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Find image directories
    img_dirs = [p.parent for p in data_root.glob("**/*.png")][:3]
    print(f"sample image dirs: {img_dirs[:3]}")
    img_root_map = {}
    for png in data_root.glob("**/*.png"):
        img_root_map[png.name] = png
        if len(img_root_map) >= 200000:  # safety cap
            break
    print(f"indexed {len(img_root_map)} images")

    df["path"] = df["Image Index"].map(img_root_map)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    print(f"matched {len(df)} rows to images")

    # Multi-hot labels
    import numpy as np
    for d in NIH_DISEASES:
        df[d] = df["Finding Labels"].apply(lambda s: int(d in str(s).split("|")))

    # Patient-level split (avoid leakage)
    patients = df["Patient ID"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    n = len(patients)
    train_p = set(patients[: int(n * 0.8)])
    val_p = set(patients[int(n * 0.8): int(n * 0.9)])
    test_p = set(patients[int(n * 0.9):])

    def split_of(pid):
        if pid in train_p: return "train"
        if pid in val_p: return "val"
        return "test"
    df["split"] = df["Patient ID"].apply(split_of)
    print(df.groupby("split").size())

    # Training
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from monai.networks.nets import DenseNet121
    from monai.transforms import Compose, EnsureType, RandFlip, RandRotate, Resize, ScaleIntensity, ToTensor
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    from PIL import Image
    import wandb

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    class MultiLabelDataset(Dataset):
        def __init__(self, df_split, transform):
            self.df = df_split.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")
            arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
            t = self.transform(arr)
            label = torch.tensor([row[d] for d in NIH_DISEASES], dtype=torch.float32)
            return t, label

    train_tfm = Compose([Resize((224, 224)), ScaleIntensity(),
                         RandFlip(prob=0.5, spatial_axis=1),
                         RandRotate(range_x=0.1, prob=0.5),
                         ToTensor(), EnsureType()])
    eval_tfm = Compose([Resize((224, 224)), ScaleIntensity(),
                        ToTensor(), EnsureType()])

    train_ds = MultiLabelDataset(df[df.split == "train"], train_tfm)
    val_ds = MultiLabelDataset(df[df.split == "val"], eval_tfm)
    test_ds = MultiLabelDataset(df[df.split == "test"], eval_tfm)
    print(f"sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=14, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # pos_weight per class for imbalance
    pos_counts = torch.tensor([train_ds.df[d].sum() for d in NIH_DISEASES], dtype=torch.float32)
    neg_counts = len(train_ds) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    wandb.init(project="nih-cxr14-multilabel",
               name=f"densenet121-{subset}-{epochs}ep",
               config={"epochs": epochs, "batch_size": batch_size, "lr": lr,
                       "subset": subset, "diseases": NIH_DISEASES})

    Path("/vol/checkpoints").mkdir(exist_ok=True)
    best_auroc = 0.0
    ckpt_path = Path(f"/vol/checkpoints/best_multilabel_{subset}.pt")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # val eval — compute per-class AUROC
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                p = torch.sigmoid(model(x)).cpu().numpy()
                all_y.append(y.numpy())
                all_p.append(p)
        all_y = np.concatenate(all_y)
        all_p = np.concatenate(all_p)

        per_class = {}
        for i, d in enumerate(NIH_DISEASES):
            if len(set(all_y[:, i])) > 1:
                per_class[d] = roc_auc_score(all_y[:, i], all_p[:, i])
        mean_auroc = np.mean(list(per_class.values())) if per_class else 0.0

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"  epoch {epoch+1}: loss={avg_loss:.4f}  mean_val_auroc={mean_auroc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "mean_val_auroc": mean_auroc,
                   **{f"val_auroc/{d}": v for d, v in per_class.items()}})

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved best")

    # Test set eval
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            all_y.append(y.numpy())
            all_p.append(p)
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)

    test_per_class = {}
    for i, d in enumerate(NIH_DISEASES):
        if len(set(all_y[:, i])) > 1:
            test_per_class[d] = float(roc_auc_score(all_y[:, i], all_p[:, i]))
    test_mean = float(np.mean(list(test_per_class.values())))

    print(f"\n=== TEST mean AUROC: {test_mean:.4f} ===")
    for d, v in sorted(test_per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {d:25s} {v:.4f}")

    wandb.log({"test_mean_auroc": test_mean,
               **{f"test_auroc/{d}": v for d, v in test_per_class.items()}})
    wandb.finish()
    volume.commit()

    return {
        "checkpoint": ckpt_path.read_bytes(),
        "test_mean_auroc": test_mean,
        "test_per_class": test_per_class,
        "subset": subset,
    }


@app.local_entrypoint()
def main(epochs: int = 5, batch_size: int = 32, lr: float = 1e-4, subset: str = "sample"):
    from pathlib import Path
    import json
    print(f"Launching multi-label training: subset={subset}, epochs={epochs}")
    result = train_multilabel.remote(epochs=epochs, batch_size=batch_size, lr=lr, subset=subset)
    Path("checkpoints").mkdir(exist_ok=True)
    out = Path(f"checkpoints/multilabel_{subset}.pt")
    out.write_bytes(result["checkpoint"])
    print(f"Saved {len(result['checkpoint'])/1e6:.1f} MB to {out}")
    print(f"Test mean AUROC: {result['test_mean_auroc']:.4f}")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/multilabel_metrics.json").write_text(json.dumps({
        "test_mean_auroc": result["test_mean_auroc"],
        "test_per_class": result["test_per_class"],
        "subset": result["subset"],
    }, indent=2))
    print("Saved docs/multilabel_metrics.json")
