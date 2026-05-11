"""DenseNet121 model for pneumonia classification using MONAI."""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


def build_model(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Build a DenseNet121 classifier.

    Args:
        num_classes: 1 for binary (sigmoid), >1 for multi-class softmax.
        pretrained: load ImageNet weights via torchvision (recommended).

    Returns:
        nn.Module producing logits of shape (B, num_classes).
    """
    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,         # we replicate grayscale to 3 channels
        out_channels=num_classes,
        pretrained=pretrained,
    )
    return model


if __name__ == "__main__":
    # Smoke test
    m = build_model(num_classes=1, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 1), f"unexpected shape {y.shape}"
    print(f"OK: model parameters = {sum(p.numel() for p in m.parameters()):,}")
