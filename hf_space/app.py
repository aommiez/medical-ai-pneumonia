"""HuggingFace Space app for pneumonia classifier."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import build_model


CKPT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"
DEVICE = "cpu"
MODEL = None
GRAD_CAM = None
TRANSFORM = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])


def load_model():
    global MODEL, GRAD_CAM
    if MODEL is not None:
        return
    MODEL = build_model(num_classes=1, pretrained=False)
    state = torch.load(CKPT, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval().to(DEVICE)
    try:
        from pytorch_grad_cam import GradCAM
        GRAD_CAM = GradCAM(model=MODEL, target_layers=[MODEL.features.denseblock4])
        print("Grad-CAM enabled")
    except Exception as e:
        print(f"Grad-CAM disabled: {e}")
        GRAD_CAM = None


def predict(image: Image.Image):
    if image is None:
        return "Please upload an image", {}, None

    load_model()
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
    x = TRANSFORM(arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(MODEL(x)).item()

    label = (
        f"⚠️ Pneumonia features detected ({prob:.1%} confidence)"
        if prob > 0.5
        else f"✅ No pneumonia features ({(1-prob):.1%} confidence)"
    )

    cam_img = None
    if GRAD_CAM is not None:
        try:
            from pytorch_grad_cam.utils.image import show_cam_on_image
            rgb = np.array(img).astype(np.float32) / 255.0
            grayscale_cam = GRAD_CAM(input_tensor=x, targets=None)[0]
            cam_img = Image.fromarray(show_cam_on_image(rgb, grayscale_cam, use_rgb=True))
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return label, {"No Pneumonia": 1 - prob, "Pneumonia": prob}, cam_img


CSS = ".disclaimer { background:#fff4e0; border-left:4px solid #ff9800; padding:12px; margin:8px 0; }"


with gr.Blocks(title="Pneumonia Classifier") as app:
    gr.Markdown(
        """
        # 🫁 Pneumonia Detection from Chest X-Rays
        DenseNet121 · Test AUROC 0.9138 ·
        [GitHub](https://github.com/aommiez/medical-ai-pneumonia) ·
        [WandB](https://wandb.ai/aommiez-savel-co-ltd/medical-ai-pneumonia)
        """
    )
    gr.HTML(
        """
        <div class="disclaimer">
        <strong>⚠️ Educational demo only — not for clinical use.</strong>
        Output is decision-support information; only a licensed radiologist may interpret
        medical images for clinical decisions.
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload Chest X-Ray", height=400)
            btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        with gr.Column():
            out_label = gr.Textbox(label="Diagnosis", lines=2)
            out_probs = gr.Label(label="Probability", num_top_classes=2)
            out_cam = gr.Image(label="Grad-CAM", height=400)

    btn.click(predict, inputs=inp, outputs=[out_label, out_probs, out_cam])

    with gr.Accordion("ℹ️ About this model", open=False):
        gr.Markdown(
            """
            **Architecture:** DenseNet121 (6.95M params) pretrained on ImageNet
            **Training data:** Kaggle [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
            (5,216 train / 16 val / 624 test images)
            **Training time:** 5 minutes on Modal Labs T4 GPU
            **Test AUROC:** 0.9138 (vs CheXNet baseline 0.768 on similar tasks)

            ### Known limitations
            - Pediatric population only (no adult validation)
            - Frontal views only (lateral X-rays will give unreliable output)
            - Trained on data from one institution — may not generalize globally
            - Model is biased toward catching pneumonia (high sensitivity, lower specificity)

            ### Calibration tip
            Default threshold (0.5) is over-sensitive. For balanced operation, use threshold ~0.93.
            See [calibration analysis](https://github.com/aommiez/medical-ai-pneumonia/blob/main/docs/operating_points.json) on GitHub.
            """
        )


if __name__ == "__main__":
    app.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
