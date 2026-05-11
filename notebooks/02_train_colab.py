# %% [markdown]
# # Pneumonia Classifier — Colab Training Notebook
#
# Run this on Google Colab with **GPU enabled** (Runtime → Change runtime type → A100/T4).
#
# Cells use `# %%` separators — paste into Colab cells or import as `.ipynb`.

# %% [markdown]
# ## 1. Setup environment

# %%
# !pip install --quiet monai torch torchvision wandb scikit-learn tqdm grad-cam gradio

# %%
import os
import torch
print("PyTorch:", torch.__version__, "· CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# %% [markdown]
# ## 2. Download dataset (NIH ChestX-ray14 subset)
#
# Full dataset is ~45GB across 12 tar files. For a 2-day Colab training, use the
# subset from Kaggle (smaller, already split). Or download the official splits below.

# %%
# Option A — Kaggle (pre-split pneumonia binary subset, ~2GB)
# Requires kaggle.json in /content/
#
# !mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# !unzip -q chest-xray-pneumonia.zip -d /content/data

# Option B — Direct NIH (full 112k images, 45GB, takes 1-2 hours)
# !wget https://nihcc.box.com/v/ChestXray-NIHCC -O nih.tar.gz

# %% [markdown]
# ## 3. Mount Google Drive for checkpoint persistence

# %%
# from google.colab import drive
# drive.mount('/content/drive')
# CHECKPOINT_DIR = "/content/drive/MyDrive/medical-ai/checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_DIR = "/content/checkpoints"  # local Colab if no drive
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# %% [markdown]
# ## 4. Login to WandB for experiment tracking

# %%
# import wandb
# wandb.login()  # paste your key
# wandb.init(project="pneumonia-classifier", name="densenet121-baseline")

# %% [markdown]
# ## 5. Build model + dataset

# %%
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType, RandFlip, RandRotate, Resize, ScaleIntensity, ToTensor
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=1, pretrained=True).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 6. Training loop
#
# Use the same `src/train.py` interface — paste in or upload via Files sidebar.

# %%
# Example training command (adjust paths to your Kaggle subset):
#
# !python src/train.py \
#     --data-root /content/data/chest_xray \
#     --train-csv /content/data/splits/train.csv \
#     --val-csv /content/data/splits/val.csv \
#     --epochs 30 \
#     --batch-size 32 \
#     --lr 1e-4 \
#     --device cuda \
#     --output /content/checkpoints/best.pt

# %% [markdown]
# ## 7. Evaluation + visualization
#
# After training, run `notebooks/03_evaluate.ipynb` locally with the downloaded checkpoint.

# %% [markdown]
# ## 8. Tips for Colab
#
# - **Free tier**: T4 GPU, ~12 hours/day, may disconnect — use Drive checkpointing
# - **Pro ($10/mo)**: more priority + longer runtime + better GPUs
# - **Pro+ ($50/mo)**: A100 available
# - **Pay-as-you-go**: ~$10 buys ~50 A100 compute units (~50 hours)
#
# Cost estimate for this project: ~$5-10 on Colab Pro or free tier with patience.
