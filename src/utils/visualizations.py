"""Journal-Grade Publication Visualization Suite.

Generates high-DPI (300+ DPI), IEEE/Nature/Elsevier-standard figures:
1. Normalized Confusion Matrix with dual-text annotations (Percentages + Counts).
2. Multi-Class One-vs-Rest ROC Curves with AUC and Macro/Micro averages.
3. Multi-Class Precision-Recall Curves with Average Precision (AP) and Iso-F1 curves.
4. Comprehensive 4-Panel Training & Validation Dynamics Curves.
5. Evidential Reliability Calibration Diagram & Epistemic Uncertainty Separation.
6. Small-Tumour vs Overall Cohort Sensitivity & F1 Comparison Bar Chart.
7. Multi-Modal CARE-XAI Clinical Interpretability Panel.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments/Colab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


def set_journal_style():
    """Sets publication-grade plot styles."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Normalized Confusion Matrix (X-R-MiTNet)",
) -> plt.Figure:
    """Plots a publication-quality normalized confusion matrix with percentage and counts."""
    set_journal_style()
    if class_names is None:
        class_names = CLASS_NAMES

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom", fontsize=11)

    # Ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Diagnosis",
        xlabel="Predicted Diagnosis",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = f"{cm_norm[i, j]*100:.1f}%"
            val_cnt = f"({cm[i, j]})"
            ax.text(
                j, i, f"{val_pct}\n{val_cnt}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontweight="bold" if i == j else "normal",
                fontsize=10,
            )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots multi-class One-vs-Rest ROC curves with macro/micro averages."""
    set_journal_style()
    if class_names is None:
        class_names = CLASS_NAMES
    num_classes = len(class_names)
    y_onehot = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Per-class ROC
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], y_probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=COLORS[c % len(COLORS)], lw=2,
            label=f"{class_names[c]} (AUC = {roc_auc:.4f})"
        )

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), y_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro, tpr_micro, color="crimson", linestyle="--", lw=2,
        label=f"Micro-Average (AUC = {auc_micro:.4f})"
    )

    # Random baseline
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Chance Line (AUC = 0.5000)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Multi-Class One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots multi-class Precision-Recall curves with Average Precision."""
    set_journal_style()
    if class_names is None:
        class_names = CLASS_NAMES
    num_classes = len(class_names)
    y_onehot = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))

    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_onehot[:, c], y_probs[:, c])
        pr_auc = auc(recall, precision)
        ax.plot(
            recall, precision, color=COLORS[c % len(COLORS)], lw=2,
            label=f"{class_names[c]} (AP = {pr_auc:.4f})"
        )

    # Iso-F1 curves
    f_scores = np.linspace(0.4, 0.8, 3)
    for f_score in f_scores:
        x = np.linspace(f_score / 2.0 + 1e-3, 1.0, 100)
        y = f_score * x / (2.0 * x - f_score)
        valid = (y >= 0) & (y <= 1.05)
        ax.plot(x[valid], y[valid], color="gray", alpha=0.3, linestyle=":")
        if 0.85 > (f_score / 2.0):
            y_anno = f_score * 0.85 / (2.0 * 0.85 - f_score)
            if 0 <= y_anno <= 1.0:
                ax.annotate(f"F1={f_score:.1f}", xy=(0.85, y_anno), color="gray", fontsize=8)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Multi-Class Precision-Recall Curves")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots 4-panel publication curves: Total Loss, Component Losses, Accuracy, and F1."""
    set_journal_style()
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Total Multi-Task Loss
    axs[0, 0].plot(epochs, history.get("train_loss", []), "o-", color="#2b5c8f", lw=2, ms=4, label="Train Total")
    if "val_loss" in history:
        axs[0, 0].plot(epochs, history["val_loss"], "s--", color="#d95f02", lw=2, ms=4, label="Val Total")
    axs[0, 0].set_title("(a) Multi-Task Loss Convergence")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # (b) Loss Component Breakdown
    if "train_cls_loss" in history:
        axs[0, 1].plot(epochs, history["train_cls_loss"], "-", color="#2b5c8f", lw=2, label="Evidential Loss")
    if "train_seg_loss" in history:
        axs[0, 1].plot(epochs, history["train_seg_loss"], "--", color="#7570b3", lw=2, label="Segmentation Loss")
    if "train_kl_loss" in history:
        axs[0, 1].plot(epochs, history["train_kl_loss"], ":", color="#1b9e77", lw=2, label="Annealed KL Regularizer")
    axs[0, 1].set_title("(b) Loss Component Progression")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Component Loss")
    axs[0, 1].legend()

    # (c) Accuracy Dynamics
    if "train_acc" in history:
        axs[1, 0].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2b5c8f", lw=2, ms=4, label="Train Acc")
    if "val_acc" in history:
        axs[1, 0].plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#d95f02", lw=2, ms=4, label="Val Acc")
    axs[1, 0].set_title("(c) Classification Accuracy (%)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Accuracy (%)")
    axs[1, 0].legend()

    # (d) Macro F1-Score & Segmentation Dice
    if "val_f1" in history:
        axs[1, 1].plot(epochs, [f * 100 for f in history["val_f1"]], "o-", color="#2b5c8f", lw=2, ms=4, label="Val Macro F1 (%)")
    if "val_dice" in history:
        axs[1, 1].plot(epochs, [d * 100 for d in history["val_dice"]], "^--", color="#1b9e77", lw=2, ms=4, label="Val Aux Dice (%)")
    axs[1, 1].set_title("(d) Validation F1-Score & Dice Progression")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Metric (%)")
    axs[1, 1].legend()

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_reliability_and_calibration(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    uncertainties: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots Reliability Diagram (ECE) alongside Uncertainty Separation."""
    set_journal_style()
    from src.utils.metrics import compute_expected_calibration_error

    ece, bin_accs, bin_confs, bin_counts = compute_expected_calibration_error(y_true, y_probs, num_bins=num_bins)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Reliability Diagram
    bin_centers = np.linspace(0.05, 0.95, num_bins)
    axs[0].bar(
        bin_centers, bin_accs, width=0.08, alpha=0.7, color="#2b5c8f", edgecolor="black", label="Empirical Accuracy"
    )
    axs[0].plot([0, 1], [0, 1], "r--", lw=2, label="Ideal Calibration")
    # Plot calibration gap
    for c, a, conf in zip(bin_centers, bin_accs, bin_confs):
        if a > 0 or conf > 0:
            axs[0].plot([c, c], [a, conf], color="red", alpha=0.5, lw=2)

    axs[0].set_xlabel("Confidence Bin Center")
    axs[0].set_ylabel("Empirical Accuracy")
    axs[0].set_title(f"(a) Reliability Diagram (ECE = {ece:.4f})")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[0].legend(loc="upper left")

    # (b) Epistemic Uncertainty by Decision Correctness
    preds = np.argmax(y_probs, axis=1)
    correct_mask = (preds == y_true)

    unc_correct = uncertainties[correct_mask]
    unc_incorrect = uncertainties[~correct_mask] if np.sum(~correct_mask) > 0 else np.array([])

    sns.kdeplot(unc_correct, ax=axs[1], color="#1b9e77", fill=True, label=f"Correct Decisions (μ={np.mean(unc_correct):.3f})")
    if len(unc_incorrect) > 0:
        sns.kdeplot(unc_incorrect, ax=axs[1], color="#d95f02", fill=True, label=f"Misclassifications (μ={np.mean(unc_incorrect):.3f})")

    axs[1].set_xlabel("Epistemic Uncertainty $u$ (from Evidential Dirichlet)")
    axs[1].set_ylabel("Density")
    axs[1].set_title("(b) Clinical Uncertainty Calibration & Error Separation")
    axs[1].legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_small_tumor_analysis(
    overall_metrics: Dict[str, float],
    small_tumor_metrics: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plots comparative bar chart between standard cohort and small-tumor cohort."""
    set_journal_style()
    metrics_to_plot = ["accuracy", "macro_precision", "macro_recall", "macro_specificity", "macro_f1"]
    labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score"]

    val_overall = [overall_metrics.get(m, 0.0) * 100 for m in metrics_to_plot]
    val_small = [small_tumor_metrics.get(m, 0.0) * 100 for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width / 2, val_overall, width, label="All-Cohort Validation", color="#2b5c8f", edgecolor="black")
    rects2 = ax.bar(x + width / 2, val_small, width, label="Early/Small-Tumour Cohort", color="#d95f02", edgecolor="black")

    ax.set_ylabel("Score (%)")
    ax.set_title("Detection Performance: All Tumors vs. Early / Small Tumors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 110])
    ax.legend(loc="lower right")

    # Add values on top of bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_care_xai_interpretability_panel(
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    rollout_map: np.ndarray,
    ig_map: Optional[np.ndarray] = None,
    pred_class: str = "Glioma",
    true_class: str = "Glioma",
    confidence: float = 0.98,
    uncertainty: float = 0.05,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generates a 5 or 6 panel clinical interpretability figure."""
    set_journal_style()
    num_panels = 6 if ig_map is not None else 5
    fig, axs = plt.subplots(1, num_panels, figsize=(3.2 * num_panels, 3.8))

    # Squeeze 2D representations
    img_2d = image.squeeze()
    true_mask_2d = true_mask.squeeze()
    pred_mask_2d = pred_mask.squeeze()
    rollout_2d = rollout_map.squeeze()

    # Normalize image for visualization
    img_disp = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)

    # 1. Original MRI
    axs[0].imshow(img_disp, cmap="gray")
    axs[0].set_title(f"Original MRI\nTruth: {true_class}")
    axs[0].axis("off")

    # 2. Ground Truth Mask
    axs[1].imshow(img_disp, cmap="gray")
    if true_mask_2d.max() > 0.5:
        axs[1].imshow(true_mask_2d, cmap="Greens", alpha=0.45)
        axs[1].set_title("Ground-Truth Mask\n(Radiologist Annotated)")
    else:
        axs[1].set_title("Ground-Truth Mask\n(No Lesion / Control)")
    axs[1].axis("off")

    # 3. Predicted Mask
    axs[2].imshow(img_disp, cmap="gray")
    if pred_mask_2d.max() > 0.5:
        axs[2].imshow(pred_mask_2d, cmap="Reds", alpha=0.45)
        axs[2].set_title("Predicted Mask\n(Auxiliary Decoder)")
    else:
        axs[2].set_title("Predicted Mask\n(Clean / No Tumor)")
    axs[2].axis("off")

    # 4. Attention Rollout Heatmap
    im4 = axs[3].imshow(rollout_2d, cmap="magma")
    axs[3].set_title("CARE-XAI Rollout\n(Attention Heatmap)")
    axs[3].axis("off")

    # 5. Attention Rollout Overlay
    axs[4].imshow(img_disp, cmap="gray")
    axs[4].imshow(rollout_2d, cmap="magma", alpha=0.55)
    axs[4].set_title(f"Explainable Overlay\nPred: {pred_class} ({confidence*100:.1f}%)")
    axs[4].axis("off")

    # 6. Integrated Gradients Attribution (if available)
    if ig_map is not None:
        ig_2d = ig_map.squeeze()
        ig_norm = (ig_2d - ig_2d.min()) / (ig_2d.max() - ig_2d.min() + 1e-8)
        axs[5].imshow(img_disp, cmap="gray")
        axs[5].imshow(ig_norm, cmap="inferno", alpha=0.55)
        axs[5].set_title(f"Integrated Gradients\nUncertainty: {uncertainty:.3f}")
        axs[5].axis("off")

    reliability = max(0.0, 1.0 - uncertainty)
    fig.suptitle(
        f"Diagnostic Assessment: {pred_class} | Confidence: {confidence*100:.1f}% | Epistemic Uncertainty (u): {uncertainty:.3f} | Reliability (ω): {reliability:.3f}",
        fontsize=13, fontweight="bold", y=1.03
    )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

