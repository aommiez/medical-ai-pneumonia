"""Download NIH ChestX-ray14 dataset.

Two strategies supported:
  --kaggle    Use kaggle CLI to fetch the full mirror (requires kaggle.json)
  --subset    Use the smaller pre-split pneumonia subset (~2GB, recommended first try)
  --nih       Stream from official NIH box.com URLs (slow, no auth)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"


def download_kaggle_subset():
    """Download Paul Mooney's pneumonia subset (~2GB, pre-split)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "paultimothymooney/chest-xray-pneumonia",
        "-p", str(DATA_DIR),
        "--unzip",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. Data in {DATA_DIR}/chest_xray/")


def download_kaggle_full():
    """Download full NIH CXR14 mirror from Kaggle (~45GB)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "nih-chest-xrays/data",
        "-p", str(DATA_DIR),
        "--unzip",
    ]
    print("running:", " ".join(cmd))
    print("WARNING: ~45GB download — this will take 1+ hours")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--subset", action="store_true",
                       help="Pneumonia binary subset, ~2GB (recommended)")
    group.add_argument("--kaggle", action="store_true",
                       help="Full NIH CXR14 via Kaggle mirror, ~45GB")
    group.add_argument("--nih", action="store_true",
                       help="Direct from NIH (TODO — not implemented)")
    args = ap.parse_args()

    if args.subset:
        download_kaggle_subset()
    elif args.kaggle:
        download_kaggle_full()
    elif args.nih:
        print("Not implemented yet — use --subset or --kaggle (with kaggle.json).")
        print("Manual: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        sys.exit(1)


if __name__ == "__main__":
    main()
