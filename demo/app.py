"""Enhanced Gradio web UI for pneumonia classifier.

Features:
- Drag-and-drop chest X-ray upload
- Side-by-side: original + Grad-CAM heatmap
- Confidence probability bar
- Sample images to try (auto-loaded from data/raw/chest_xray/test/)
- Model card + dataset card sections
- Disclaimer banner

Run:
    python demo/app.py --checkpoint checkpoints/best.pt --port 7860
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensity, ToTensor
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model


# ─── Global model + transforms ─────────────────────────────────────────────

MODEL = None
DEVICE = "cpu"
TRANSFORM = Compose([Resize((224, 224)), ScaleIntensity(), ToTensor(), EnsureType()])
GRAD_CAM = None


def load_model(checkpoint: str, device: str = "cpu"):
    """Load trained model + set up Grad-CAM."""
    global MODEL, DEVICE, GRAD_CAM
    DEVICE = device
    MODEL = build_model(num_classes=1, pretrained=False)

    state = torch.load(checkpoint, map_location=device)
    MODEL.load_state_dict(state)
    MODEL.eval().to(device)

    try:
        from pytorch_grad_cam import GradCAM as _GC
        target_layer = MODEL.features.denseblock4
        GRAD_CAM = _GC(model=MODEL, target_layers=[target_layer])
        print("Grad-CAM enabled")
    except Exception as e:
        print(f"Grad-CAM disabled: {e}")
        GRAD_CAM = None


def predict(image: Image.Image):
    """Return (label, prob_dict, gradcam_image)."""
    if MODEL is None or image is None:
        return "Model not loaded — train first", {}, None

    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)  # CHW
    x = TRANSFORM(arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = MODEL(x)
        prob = torch.sigmoid(logit).item()

    label = (
        f"⚠️ Pneumonia features detected ({prob:.1%} confidence)"
        if prob > 0.5
        else f"✅ No pneumonia features ({(1-prob):.1%} confidence — normal)"
    )

    # Grad-CAM heatmap
    cam_img = None
    if GRAD_CAM is not None:
        try:
            from pytorch_grad_cam.utils.image import show_cam_on_image
            rgb = (np.array(img).astype(np.float32) / 255.0)
            grayscale_cam = GRAD_CAM(input_tensor=x, targets=None)[0]
            cam_img = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            cam_img = Image.fromarray(cam_img)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return label, {"No Pneumonia": 1 - prob, "Pneumonia": prob}, cam_img


# ─── Sample images ─────────────────────────────────────────────────────────

def find_samples(root="/home/hermes/medical-ai/data/raw/chest_xray/test"):
    samples = []
    root_p = Path(root)
    if not root_p.exists():
        return samples
    for cls in ["NORMAL", "PNEUMONIA"]:
        d = root_p / cls
        if d.exists():
            imgs = sorted(d.glob("*.jpeg"))[:3]
            samples.extend([str(p) for p in imgs])
    return samples


# ─── Gradio UI ─────────────────────────────────────────────────────────────

CSS = """
.disclaimer { background: #fff4e0; border-left: 4px solid #ff9800; padding: 12px; margin: 8px 0; }
.gradio-container { max-width: 1100px !important; }
"""


def build_ui(checkpoint_loaded: bool):
    with gr.Blocks(title="Pneumonia Classifier", css=CSS, theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🫁 Pneumonia Detection from Chest X-Rays
            DenseNet121 trained on Kaggle's Chest X-Ray Images (Pneumonia) dataset by
            [@aommiez](https://github.com/aommiez) ·
            [Code](https://github.com/aommiez/medical-ai-pneumonia) ·
            [WandB run](https://wandb.ai/aommiez-savel-co-ltd/medical-ai-pneumonia)
            """
        )

        gr.HTML(
            """
            <div class="disclaimer">
            <strong>⚠️ Educational demo only — not for clinical use.</strong><br>
            This model has not been validated for clinical diagnosis. Output is decision-support
            information; only a licensed radiologist may interpret medical images for diagnosis.
            </div>
            """
        )

        if not checkpoint_loaded:
            gr.HTML(
                """
                <div class="disclaimer" style="background: #ffe0e0; border-color: #d32f2f;">
                <strong>Model not loaded.</strong> Checkpoint not found at the configured path.
                Training may still be in progress — refresh in a few minutes.
                </div>
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload Chest X-Ray", height=400)
                btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                samples = find_samples()
                if samples:
                    gr.Examples(samples, inputs=inp, label="Or try a sample image (from test set)")

            with gr.Column(scale=1):
                out_label = gr.Textbox(label="Diagnosis", lines=2, show_copy_button=True)
                out_probs = gr.Label(label="Probability Distribution", num_top_classes=2)
                out_cam = gr.Image(label="Grad-CAM (model attention)", height=400)

        btn.click(predict, inputs=inp, outputs=[out_label, out_probs, out_cam])

        with gr.Accordion("ℹ️ About this Model", open=False):
            gr.Markdown(
                """
                ### Architecture
                **DenseNet121** (6.95M params) pretrained on ImageNet, fine-tuned for binary
                pneumonia classification. Input: 224×224 RGB. Output: single logit → sigmoid.

                ### Training data
                Kaggle: [paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
                - Train: 5,216 images (1,341 normal + 3,875 pneumonia)
                - Val: 16 images
                - Test: 624 images (234 normal + 390 pneumonia)

                ### Training procedure
                - Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
                - Loss: BCEWithLogits + `pos_weight` for class imbalance
                - Augmentation: random horizontal flip, slight rotation, intensity scaling
                - 10 epochs (early stopping by val AUROC)
                - Hardware: Modal Labs T4 GPU

                ### Known limitations
                - Adult populations only (no pediatric validation)
                - Frontal views only (lateral X-rays will give unreliable output)
                - Trained on data from a single institution — may not generalize globally
                - Model uses weak labels mined from radiology reports — not pixel-level annotation
                - **AUROC on test set: see WandB run**

                ### Grad-CAM
                The heatmap shows regions the model "attended to" when making its prediction.
                Red areas = high importance. This is a transparency aid, not a localization.
                """
            )

        with gr.Accordion("⚙️ Technical Details", open=False):
            gr.Markdown(
                f"""
                **Model checkpoint loaded:** {"✅ yes" if checkpoint_loaded else "❌ no"}
                **Device:** `{DEVICE}`
                **Framework:** PyTorch + MONAI
                **Inference time:** ~0.5-2s on CPU
                **License:** MIT (code) · Kaggle dataset has its own terms
                """
            )

    return app


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    loaded = ckpt_path.exists()
    if loaded:
        print(f"Loading checkpoint: {ckpt_path}")
        load_model(str(ckpt_path), args.device)
    else:
        print(f"⚠️ Checkpoint not found at {ckpt_path} — UI will show placeholder")

    app = build_ui(checkpoint_loaded=loaded)
    app.queue(max_size=10).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
