"""Script to generate notebooks/colab_master_pipeline.ipynb."""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""# 🧠 X-R-MiTNet & CARE-XAI: Master Google Colab Pipeline
## Explainable Reliability-Aware Multimodal Vision Transformer for Brain Tumor Detection and Auxiliary Segmentation

**Targeted for High-Impact Scopus/SCI Medical AI Journal Publication (IEEE TMI / Nature Med / Elsevier CBM)**

This notebook provides a complete, turnkey end-to-end framework:
1. **Colab Environment Setup & GPU Diagnostics** (Nvidia T4/V100/A100 Acceleration)
2. **Dataset Loading & Cross-Platform Path Handling** (BRISC 2025 Multi-Task Benchmark)
3. **Model Instantiation**: Pure CNN-Free Vision Transformer with Bilateral Symmetry Attention ($S$-MHSA), Evidential Dirichlet Head, and Auxiliary Segmentation Decoder
4. **Multi-Task Training**: Mixed-Precision training with Annealed KL Loss and Cosine Annealing
5. **Journal-Grade Metrics & Visualizations**:
   - 4-Class Classification Metrics (Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, Macro/Weighted F1, MCC, Cohen's Kappa, AUC-ROC)
   - Evidential Calibration & Epistemic Uncertainty Quantification (ECE, Brier Score, Separation $t$-test)
   - Auxiliary Pixel-Wise Segmentation (Dice Similarity Coefficient, Jaccard Index / IoU)
   - Small-Tumour Subgroup Analysis (Defensible Early Detection Claim)
   - 7 High-Resolution (300 DPI) Publication Figures (Confusion Matrix, ROC Curves, PR Curves, Convergence Curves, Reliability Diagram, Small Tumor Comparison, CARE-XAI Panels)
   - Auto-generated IEEE/Nature style LaTeX tables
6. **Real-Time Interactive Photo Upload & Diagnostic System**:
   - In-notebook Native Interactive Upload Widget
   - One-Click Gradio Web App with public shareable link for live real-time clinical testing!
""")

    # Section 1
    add_md("""---
## Section 1: Google Colab Setup & GPU Diagnostics

Connect your Google Colab instance to a GPU runtime (**Runtime > Change runtime type > T4 GPU or A100 GPU**).
This cell mounts Google Drive (optional), clones or navigates to the repository, and installs all dependencies.
""")

    add_code("""# 1.1 Check GPU Diagnostics
import os
import sys
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"CUDA Available: {gpu_available}")

if gpu_available:
    device = torch.device("cuda")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("WARNING: Running on CPU. For fast training, enable GPU under Runtime > Change runtime type!")

# 1.2 Setup Working Directory & Google Drive (if on Google Colab)
try:
    from google.colab import drive
    print("\\n[Colab Detected] You can optionally mount Google Drive:")
    # Uncomment the line below to mount your drive:
    # drive.mount('/content/drive')
except ImportError:
    print("\\n[Local / Non-Colab Environment Detected]")

# 1.3 Install Dependencies
!pip install -q albumentations scikit-learn seaborn matplotlib tqdm gradio ipywidgets
print("\\nDependencies successfully verified!")
""")

    # Section 2
    add_md("""---
## Section 2: Data Loading & Multi-Task Split Verification

We verify the dataset paths. Thanks to the integrated `resolve_file_path` mechanism in `src/data/dataset.py`, paths recorded in CSVs are automatically detected regardless of whether running locally or on Google Colab (`/content/...`).
""")

    add_code("""# 2.1 Import Dataset and verify path resolution
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import BrainTumorDataset, CLASS_NAMES if 'CLASS_NAMES' in locals() else None
from torch.utils.data import DataLoader

# Initialize datasets
print("Initializing datasets...")
train_dataset = BrainTumorDataset(cohort="multitask", split="train", is_training=True, augment=True)
val_dataset = BrainTumorDataset(cohort="multitask", split="val", is_training=False, augment=False)
test_dataset = BrainTumorDataset(cohort="multitask", split="test", is_training=False, augment=False)
small_tumor_test = BrainTumorDataset(cohort="multitask", split="small_tumor_test", is_training=False, augment=False)

print(f"Train samples:      {len(train_dataset)}")
print(f"Val samples:        {len(val_dataset)}")
print(f"Test samples:       {len(test_dataset)}")
print(f"Small-Tumour Test:  {len(small_tumor_test)}")

# 2.2 Visualize Sample Batch with Ground-Truth Masks
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for i in range(8):
    sample = train_dataset[i * 20]
    img = sample["image"].squeeze().numpy()
    mask = sample["mask"].squeeze().numpy()
    lbl = sample["label"].item()
    is_small = sample["is_small_tumor"].item() > 0.5
    
    r, c = i // 4, i % 4
    # Normalize for display
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axs[r, c].imshow(img_disp, cmap="gray")
    if mask.max() > 0:
        axs[r, c].imshow(mask, cmap="Reds", alpha=0.45)
    
    small_tag = " [Small]" if is_small else ""
    axs[r, c].set_title(f"{CLASS_NAMES[lbl]}{small_tag}", fontsize=11, fontweight="bold")
    axs[r, c].axis("off")

plt.suptitle("Sample Training MRI Scans with Radiologist Tumor Masks (Red Overlay)", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()
""")

    # Section 3
    add_md("""---
## Section 3: Model Architecture Instantiation (X-R-MiTNet)

**X-R-MiTNet** is a pure CNN-free Vision Transformer featuring:
1. **Linear Patch Embedding:** Extracts non-overlapping 16x16 patches from 224x224 MRI slices.
2. **Bilateral Symmetry-Contrastive Multi-Head Self-Attention ($S$-MHSA):** Explicitly compares spatial patches across the brain midline ($c \leftrightarrow W - 1 - c$) to highlight asymmetric pathological lesions.
3. **Evidential Deep Learning Head:** Dirichlet distribution modeling over class probabilities, yielding calibrated epistemic uncertainty ($u$) and reliability ($\omega = 1 - u$).
4. **Auxiliary Pure-Transformer Segmentation Decoder:** Upsamples 14x14 patch tokens to 224x224 mask resolution via progressive linear projections and PixelShuffle.
""")

    add_code("""# 3.1 Initialize X-R-MiTNet
from src.models.x_r_mitnet import XRMiTNet

model = XRMiTNet(
    img_size=224,
    patch_size=16,
    in_channels=1,
    num_classes=4,
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    drop_rate=0.1,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"X-R-MiTNet Architecture Summary:")
print(f" - Total Parameters:     {total_params:,}")
print(f" - Trainable Parameters: {trainable_params:,}")
print(f" - Embedding Dimension:  256")
print(f" - Transformer Depth:    6 Blocks with S-MHSA")
print(f" - Attention Heads:      8 Heads")
print(f" - Primary Head:         Evidential Dirichlet (4 Classes)")
print(f" - Auxiliary Head:       Pure Transformer PixelShuffle Decoder (224x224)")

# Test Dummy Forward Pass
dummy_input = torch.randn(2, 1, 224, 224, device=device)
with torch.no_grad():
    out = model(dummy_input, return_attention=True)
print(f"\\nSanity Check Forward Pass:")
print(f" - Probabilities shape: {out['probabilities'].shape}")
print(f" - Uncertainty shape:   {out['uncertainty'].shape}")
print(f" - Mask Logits shape:   {out['mask_logits'].shape}")
print(f" - Attention Blocks:    {len(out['attention_weights'])} layers")
""")

    # Section 4
    add_md("""---
## Section 4: High-Performance Multi-Task Training

We train the model using:
- **Evidential Loss** with annealed KL divergence regularization: $\\mathcal{L}_{\\text{evidential}} = \\mathcal{L}_{\\text{ACE}} + \\lambda_{\\text{KL}}(t) \\mathcal{L}_{\\text{KL}}$
- **Auxiliary Segmentation Loss**: $\\mathcal{L}_{\\text{seg}} = \\text{DiceBCELoss}(\\hat{M}, M)$
- **Joint Objective**: $\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{evidential}} + \\lambda_{\\text{seg}} \\mathcal{L}_{\\text{seg}}$
- **Optimizer**: AdamW with Cosine Annealing learning rate schedule and FP16 Mixed Precision.
""")

    add_code("""# 4.1 Run Training Pipeline
from src.train import train_model

# Adjust epochs as needed (20-30 epochs recommended for convergence)
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

print(f"Starting Multi-Task Training for {EPOCHS} epochs on {device}...")
model, history = train_model(
    checkpoint_dir="checkpoints",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    seg_weight=1.0,
    kl_coeff=0.2,
    device=str(device),
    num_workers=2 if device.type == 'cuda' else 0,
    seed=42
)
print("Training Completed! Best checkpoint saved to checkpoints/best_model.pth")
""")

    # Section 5
    add_md("""---
## Section 5: Journal Paper Output Generation & Visualization Suite

In this section, we generate all required metrics and all **7 publication-grade figures** (at 300 DPI) formatted to IEEE, Nature, and Elsevier standards.
""")

    add_code("""# 5.1 Run Comprehensive Test Evaluation
from src.evaluate import run_full_journal_evaluation

report = run_full_journal_evaluation(
    model_checkpoint="checkpoints/best_model.pth",
    output_dir="reports",
    device=str(device),
    batch_size=BATCH_SIZE
)
""")

    add_md("""### 5.2 Display All 7 Publication-Quality Figures Inline (300 DPI)""")

    add_code("""# Display All 7 Generated Figures
from IPython.display import Image, display

figures = [
    ("Figure 1: Normalized Confusion Matrix", "reports/figures/fig1_confusion_matrix.png"),
    ("Figure 2: Multi-Class One-vs-Rest ROC Curves", "reports/figures/fig2_multiclass_roc.png"),
    ("Figure 3: Multi-Class Precision-Recall Curves", "reports/figures/fig3_precision_recall_curves.png"),
    ("Figure 4: Multi-Task Training Convergence Dynamics", "reports/figures/fig4_training_dynamics.png"),
    ("Figure 5: Evidential Reliability Diagram & Epistemic Uncertainty Separation", "reports/figures/fig5_reliability_and_calibration.png"),
    ("Figure 6: Early/Small-Tumour Comparative Analysis", "reports/figures/fig6_small_tumor_comparison.png"),
    ("Figure 7: CARE-XAI Multi-Panel Clinical Interpretability & Attention Rollout", "reports/figures/fig7_care_xai_interpretability_panel.png"),
]

for title, path in figures:
    if os.path.isfile(path):
        print(f"\\n{'='*70}\\n{title}\\n{'='*70}")
        display(Image(filename=path, width=750))
    else:
        print(f"Note: {path} not found.")
""")

    add_md("""### 5.3 Publication-Ready LaTeX Table (For Direct Journal Paper Insertion)""")

    add_code("""# Print LaTeX table ready to copy-paste into Overleaf or LaTeX editor
latex_file = "reports/metrics/latex_table.tex"
if os.path.isfile(latex_file):
    with open(latex_file, "r") as f:
        latex_content = f.read()
    print("--- LaTeX Table for Paper Submission ---\\n")
    print(latex_content)
else:
    print("Run Section 5.1 first to generate the LaTeX table.")
""")

    # Section 6
    add_md("""---
## Section 6: Real-Time Interactive Photo Upload & Inference System

You can test any brain MRI scan in real time!
We provide two intuitive methods:
1. **Method A: Native In-Notebook Upload Widget** (Upload any MRI image directly below, adjust sliders, and instantly see prediction, uncertainty, segmentation mask, and CARE-XAI attention heatmap).
2. **Method B: Gradio Web Application with Public Link** (Launches a full web interface with a shareable URL so you or medical reviewers can test it on any device).
""")

    add_md("""### Method A: Native In-Notebook Interactive Upload Widget""")

    add_code("""import io
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from src.demo_app import TumorDiagnosticEngine

# Initialize diagnostic engine
engine = TumorDiagnosticEngine(checkpoint_path="checkpoints/best_model.pth", device=str(device))

upload_btn = widgets.FileUpload(
    accept='.jpg,.jpeg,.png,.tif',
    multiple=False,
    description='📁 Upload MRI',
    button_style='primary'
)

discard_slider = widgets.FloatSlider(
    value=0.85, min=0.5, max=0.98, step=0.01,
    description='XAI Discard:',
    continuous_update=False
)

output_area = widgets.Output()

def on_upload_change(change):
    with output_area:
        clear_output()
        if not upload_btn.value:
            return
        
        # Extract uploaded image
        uploaded_file = next(iter(upload_btn.value.values()))
        content = uploaded_file['content']
        pil_img = Image.open(io.BytesIO(content))
        
        print("Analyzing uploaded scan with X-R-MiTNet & CARE-XAI...")
        
        # Run inference
        confidences, summary_md, rel_md, orig_rgb, mask_overlay, xai_overlay = engine.predict(
            pil_img,
            rollout_discard_ratio=discard_slider.value,
            mask_threshold=0.5
        )
        
        # Display side-by-side visualization
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(orig_rgb)
        axs[0].set_title("1. Standardized Input MRI", fontsize=12, fontweight="bold")
        axs[0].axis("off")
        
        axs[1].imshow(mask_overlay)
        axs[1].set_title("2. Predicted Lesion Segmentation", fontsize=12, fontweight="bold")
        axs[1].axis("off")
        
        axs[2].imshow(xai_overlay)
        axs[2].set_title("3. CARE-XAI Attention Rollout Overlay", fontsize=12, fontweight="bold")
        axs[2].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Print diagnostic metrics
        top_class = max(confidences, key=confidences.get)
        top_prob = confidences[top_class]
        print(f"\\n------------------------------------------------------------")
        print(f"  DIAGNOSTIC RESULT:  {top_class.upper()} ({top_prob*100:.2f}%)")
        print(f"------------------------------------------------------------")
        for cls_name, prob in confidences.items():
            bar = '█' * int(prob * 30)
            print(f"  {cls_name:<12}: {bar:<30} {prob*100:.1f}%")
        print(f"------------------------------------------------------------\\n")

upload_btn.observe(on_upload_change, names='value')
discard_slider.observe(on_upload_change, names='value')

print("Click 'Upload MRI' below to select any brain scan image:")
display(widgets.HBox([upload_btn, discard_slider]))
display(output_area)
""")

    add_md("""### Method B: Gradio Web Application (Interactive Dashboard & Public Share Link)

Run the cell below to launch the interactive Gradio web application.
In Google Colab, setting `share=True` generates a temporary public URL (`https://xxxx.gradio.live`) that you can share with co-authors, clinical reviewers, or open on mobile devices.
""")

    add_code("""# Run Gradio Web UI
from src.demo_app import build_gradio_app

gradio_app = build_gradio_app(checkpoint_path="checkpoints/best_model.pth")
# In Colab: share=True creates a public URL to test from any browser!
gradio_app.launch(inline=True, share=True)
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("notebooks/colab_master_pipeline.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Master Google Colab Notebook successfully written to: {out_file.resolve()}")

if __name__ == "__main__":
    create_notebook()

