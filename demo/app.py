"""Gradio demo for pneumonia classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model


def load_model(checkpoint: str, device: str = "cpu"):
    model = build_model(num_classes=1, pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def predict(image: Image.Image):
    if MODEL is None:
        return "Model not loaded", None

    img = image.convert("RGB")
    transform = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
    x = transform(np.array(img).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(MODEL(x)).item()

    label = "Pneumonia (probability: {:.2%})".format(prob) if prob > 0.5 else "No Pneumonia (probability: {:.2%})".format(prob)
    return label, {"No Pneumonia": 1 - prob, "Pneumonia": prob}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    global MODEL, DEVICE
    DEVICE = args.device
    MODEL = load_model(args.checkpoint, args.device) if Path(args.checkpoint).exists() else None

    desc = """
# Pneumonia Detection — Chest X-Ray

Upload a frontal-view chest X-ray. The model outputs the probability that the
image contains radiographic features of pneumonia.

⚠️ **Educational demo only. Not for clinical decision-making.**
"""

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Chest X-Ray"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Label(label="Probability"),
        ],
        title="Pneumonia Classifier",
        description=desc,
        allow_flagging="never",
    )
    iface.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
