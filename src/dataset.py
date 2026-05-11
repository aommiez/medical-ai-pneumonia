"""NIH ChestX-ray14 dataset loader for binary pneumonia classification."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class NIHPneumoniaDataset(Dataset):
    """Binary classifier dataset: image → {0: no_pneumonia, 1: pneumonia}.

    Expects NIH ChestX-ray14 layout:
        root/
            images/
                00000001_000.png
                ...
            Data_Entry_2017.csv

    The CSV column "Finding Labels" is split on "|" — image is positive iff
    "Pneumonia" appears in the labels.
    """

    def __init__(
        self,
        root: str | Path,
        split_csv: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.df = pd.read_csv(split_csv)
        self.transform = transform

        # build binary label
        self.df["target"] = self.df["Finding Labels"].apply(
            lambda s: int("Pneumonia" in s.split("|"))
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.root / "images" / row["Image Index"]
        img = Image.open(img_path).convert("RGB")  # replicate to 3 channels
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor([row["target"]], dtype=torch.float32)
        return img, label

    def class_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

        pos_weight = negatives / positives.
        """
        pos = (self.df["target"] == 1).sum()
        neg = (self.df["target"] == 0).sum()
        return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
