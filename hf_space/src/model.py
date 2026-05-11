"""DenseNet121 model for pneumonia classification using MONAI."""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import DenseNet121


def build_model(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Build a DenseNet121 binary/multiclass classifier."""
    return DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        pretrained=pretrained,
    )
