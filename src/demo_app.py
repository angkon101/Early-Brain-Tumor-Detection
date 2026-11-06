"""Interactive Real-Time Brain MRI Upload & Diagnostic System.

Supports:
1. Gradio Web Application (Runs in Colab inline or via public share link).
2. Real-time multi-task prediction:
   - 4-Class Diagnosis (Glioma, Meningioma, Pituitary, No Tumor)
   - Probability distribution bar chart
   - Evidential Epistemic Uncertainty (u) & Clinical Reliability (omega)
   - Auxiliary pixel-level tumor segmentation mask overlay
   - CARE-XAI Attention Rollout Heatmap overlay
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.explainability.care_xai import CareXAI
from src.models.x_r_mitnet import XRMiTNet

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_COLORS = {
    "Glioma": (220, 53, 69),      # Red
    "Meningioma": (255, 140, 0),  # Orange
    "Pituitary": (111, 66, 193),  # Purple
    "No Tumor": (40, 167, 69),    # Green
}


class TumorDiagnosticEngine:
    """Inference and explainability pipeline for interactive image uploads."""

    def __init__(self, checkpoint_path: Optional[str] = "checkpoints/best_model.pth", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = XRMiTNet(
            img_size=224,
            in_channels=1,
            num_classes=4,
            dims=[64, 128, 256, 512],
            depths=[2, 2, 4, 2],
            num_heads=[2, 4, 8, 16],
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).is_file():
            chk = torch.load(checkpoint_path, map_location=self.device)
            state_dict = chk.get("model_state_dict", chk)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path}")
        else:
            print("Running diagnostic engine with initialized model.")

        self.model.eval()
        self.xai = CareXAI(self.model, device=str(self.device))

    def preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocesses uploaded image to (1, 1, 224, 224) normalized tensor."""
        img_gray = pil_image.convert("L").resize((224, 224), Image.BILINEAR)
        img_np = np.array(img_gray, dtype=np.float32) / 255.0
        # Normalize
        mean = 0.28
        std = 0.22
        norm = (img_np - mean) / std
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, img_np

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image],
        rollout_discard_ratio: float = 0.85,
        mask_threshold: float = 0.5,
    ) -> Tuple[Dict[str, float], str, str, np.ndarray, np.ndarray, np.ndarray]:
        """Executes full inference and generates interactive visualization overlays.

        Returns:
            confidences: dict of class probabilities for Gradio Label
            summary_markdown: clinical report text
            reliability_text: reliability and uncertainty metrics
            orig_rgb: 224x224 RGB image
            mask_overlay: MRI with tumor segmentation overlay
            xai_overlay: MRI with CARE-XAI attention rollout heatmap overlay
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        else:
            pil_img = image_input

        tensor, img_np = self.preprocess_image(pil_img)

        # 1. Forward Pass
        with torch.no_grad():
            outputs = self.model(tensor, return_attention=True)
            probs = outputs["probabilities"][0].cpu().numpy()
            unc = outputs["uncertainty"][0].item()
            pred_mask = outputs["pred_mask"][0, 0].cpu().numpy()

        # 2. XAI Attention Rollout
        rollout = self.xai.compute_attention_rollout(tensor, discard_ratio=rollout_discard_ratio)

        # 3. Format Metrics
        confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf = probs[top_idx]
        reliability = max(0.0, 1.0 - unc)

        # Lesion area computation
        binary_mask = (pred_mask > mask_threshold).astype(np.uint8)
        tumor_pixel_count = int(np.sum(binary_mask))
        rel_area = tumor_pixel_count / (224 * 224)
        is_small = (rel_area > 0 and rel_area < 0.015)

        # Overlays
        orig_255 = (img_np * 255).astype(np.uint8)
        orig_rgb = cv2.cvtColor(orig_255, cv2.COLOR_GRAY2RGB)

        # Mask Overlay (Red lesion color)
        mask_overlay = orig_rgb.copy()
        if tumor_pixel_count > 0:
            color_mask = np.zeros_like(orig_rgb)
            color_mask[binary_mask == 1] = [220, 53, 69]
            mask_overlay = cv2.addWeighted(mask_overlay, 0.7, color_mask, 0.3, 0)
            # Find contours and draw boundary
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, (255, 0, 0), 1)

        # XAI Heatmap Overlay (Magma colormap)
        heatmap_uint8 = (rollout * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_MAGMA)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        xai_overlay = cv2.addWeighted(orig_rgb, 0.5, heatmap_color, 0.5, 0)

        # Report Markdown
        status_color = "#28a745" if top_class == "No Tumor" else "#dc3545"
        risk_alert = "⚠️ **Early / Small Tumour Detected** (<1.5% brain area)" if is_small else (
            "✅ **No Significant Lesion Mass**" if top_class == "No Tumor" else "🔍 **Standard Lesion Mass Detected**"
        )

        summary_markdown = f"""
### Diagnostic Assessment: <span style="color:{status_color}">{top_class}</span>
- **Confidence:** **{top_conf*100:.2f}%**
- **Clinical Reliability (ω):** **{reliability:.3f}** (Threshold: >0.85)
- **Epistemic Uncertainty (u):** **{unc:.3f}**
- **Lesion Status:** {risk_alert}
- **Estimated Relative Area:** {rel_area*100:.2f}% ({tumor_pixel_count} pixels)
"""

        rel_metrics_markdown = f"""
| Metric | Clinical Score | Status |
|---|---|---|
| **Diagnosis** | **{top_class}** | {'Normal' if top_class == 'No Tumor' else 'Pathological'} |
| **Probability** | **{top_conf*100:.2f}%** | {'High' if top_conf > 0.8 else 'Moderate'} |
| **Epistemic Uncertainty (u)** | **{unc:.4f}** | {'Low (Calibrated)' if unc < 0.2 else 'High (Caution)'} |
| **Clinical Reliability (ω)** | **{reliability:.4f}** | {'Verified' if reliability > 0.8 else 'Review Advised'} |
"""

        return confidences, summary_markdown, rel_metrics_markdown, orig_rgb, mask_overlay, xai_overlay


def build_gradio_app(checkpoint_path: Optional[str] = "checkpoints/best_model.pth"):
    """Creates the Gradio Blocks interactive interface."""
    import gradio as gr

    engine = TumorDiagnosticEngine(checkpoint_path=checkpoint_path)

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="X-R-MiTNet Brain Tumor Diagnosis") as demo:
        gr.Markdown(
            """
            # 🧠 X-R-MiTNet: Explainable Reliability-Aware Brain Tumor Diagnostic Suite
            ### Real-Time MRI Upload, Multi-Class Classification, Auxiliary Segmentation & CARE-XAI Attention Rollout
            *Developed for Scientific Research & Journal Publication Protocol.*
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(type="pil", label="Upload Brain MRI Scan (Axial, Coronal, Sagittal)")

                with gr.Accordion("Advanced Diagnostic Parameters", open=False):
                    discard_slider = gr.Slider(
                        minimum=0.5, maximum=0.98, value=0.85, step=0.01,
                        label="Attention Rollout Noise Discard Ratio (Prunes background noise)"
                    )
                    mask_thresh_slider = gr.Slider(
                        minimum=0.2, maximum=0.8, value=0.50, step=0.05,
                        label="Auxiliary Segmentation Binarization Cutoff"
                    )

                diagnose_btn = gr.Button("🔬 Run Real-Time Diagnostic Analysis", variant="primary", size="lg")

            with gr.Column(scale=5):
                report_out = gr.Markdown(value="*Upload an MRI image and click Diagnose to begin analysis.*")
                conf_out = gr.Label(num_top_classes=4, label="Class Probabilities")
                metrics_table = gr.Markdown()

        gr.Markdown("### 🖼️ Real-Time Visual Interpretability & Segmentation Overlays")
        with gr.Row():
            orig_view = gr.Image(label="1. Standardized 224x224 MRI")
            mask_view = gr.Image(label="2. Auxiliary Segmentation Mask Overlay")
            xai_view = gr.Image(label="3. CARE-XAI Attention Rollout Overlay")

        diagnose_btn.click(
            fn=engine.predict,
            inputs=[input_image, discard_slider, mask_thresh_slider],
            outputs=[conf_out, report_out, metrics_table, orig_view, mask_view, xai_view],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_gradio_app(checkpoint_path=args.checkpoint)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

