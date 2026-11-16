"""Journal Paper Comprehensive Evaluation & Figure Generation Script.

Executes:
1. Primary Multi-Task Test Cohort Evaluation.
2. Early / Small-Tumour Subgroup Analysis (Defensible Early Detection Claim).
3. Calibration & Epistemic Uncertainty Quantification.
4. Auxiliary Segmentation Evaluation (Dice, IoU).
5. Generation of all 7 Publication-Quality Figures (300 DPI PNG + PDF).
6. Auto-generation of IEEE/Nature style LaTeX tables.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorDataset, IDX_TO_LABEL
from src.explainability.care_xai import CareXAI
from src.models.x_r_mitnet import XRMiTNet
from src.utils.metrics import (
    CLASS_NAMES,
    compute_comprehensive_classification_metrics,
    compute_segmentation_metrics,
    generate_latex_table,
)
from src.utils.visualizations import (
    plot_care_xai_interpretability_panel,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_precision_recall_curves,
    plot_reliability_and_calibration,
    plot_small_tumor_analysis,
    plot_training_curves,
)


@torch.no_grad()
def evaluate_dataset(
    model: XRMiTNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference across all test samples and computes journal metrics."""
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_unc = []
    all_images = []

    pred_masks_list = []
    true_masks_list = []

    for batch in tqdm(loader, desc="Running Evaluation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        outputs = model(images)
        probs = outputs["probabilities"]
        unc = outputs["uncertainty"]
        pred_mask = outputs["pred_mask"]

        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_unc.extend(unc.cpu().numpy())
        all_images.append(images.cpu().numpy())

        # Collect masks
        has_mask_np = has_mask.cpu().numpy().astype(bool)
        if np.any(has_mask_np):
            pred_masks_list.append(pred_mask.cpu().numpy()[has_mask_np])
            true_masks_list.append(masks.cpu().numpy()[has_mask_np])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    uncertainties = np.array(all_unc)
    all_imgs_arr = np.concatenate(all_images, axis=0)

    # Classification Metrics
    metrics = compute_comprehensive_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        uncertainties=uncertainties,
    )

    # Segmentation Metrics
    pred_masks_arr = np.empty((0, 1, 224, 224))
    true_masks_arr = np.empty((0, 1, 224, 224))
    if len(pred_masks_list) > 0:
        pred_masks_arr = np.concatenate(pred_masks_list, axis=0)
        true_masks_arr = np.concatenate(true_masks_list, axis=0)
        seg_metrics = compute_segmentation_metrics(pred_masks_arr, true_masks_arr)
        metrics["segmentation"] = seg_metrics

    return metrics, y_true, y_pred, y_probs, uncertainties, pred_masks_arr, true_masks_arr, all_imgs_arr


def run_full_journal_evaluation(
    model_checkpoint: Union[str, Path] = "checkpoints/best_model.pth",
    data_root: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "reports",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> Dict:
    """Executes full journal evaluation, exports metrics, and saves all 7 figures."""
    out_path = Path(output_dir)
    fig_path = out_path / "figures"
    metrics_path = out_path / "metrics"
    fig_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Running Journal Evaluation on {target_device} ---")

    # 1. Load Model
    model = XRMiTNet(
        img_size=224,
        in_channels=1,
        num_classes=4,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
    ).to(target_device)

    chk_file = Path(model_checkpoint)
    if chk_file.is_file():
        checkpoint = torch.load(chk_file, map_location=target_device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {chk_file}")
    else:
        print(f"[Warning] Checkpoint not found at {chk_file}. Running evaluation with current model weights.")

    model.eval()

    # 2. Evaluate on Primary Test Split
    print("\n[1/4] Evaluating Primary Multi-Task Test Cohort...")
    test_dataset = BrainTumorDataset(
        cohort="multitask", split="test", data_root=data_root, is_training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    (
        test_metrics,
        y_true,
        y_pred,
        y_probs,
        uncertainties,
        pred_masks,
        true_masks,
        test_images,
    ) = evaluate_dataset(model, test_loader, target_device)

    # 3. Evaluate on Early/Small-Tumour Test Split
    print("[2/4] Evaluating Early/Small-Tumour Subgroup...")
    small_dataset = BrainTumorDataset(
        cohort="multitask", split="small_tumor_test", data_root=data_root, is_training=False
    )
    small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=False)
    small_metrics, _, _, _, _, _, _, _ = evaluate_dataset(model, small_loader, target_device)

    # 4. Generate all 7 Publication Figures
    print("\n[3/4] Generating Journal-Grade Figures (300 DPI)...")

    # Figure 1: Confusion Matrix
    print("  -> Generating Fig 1: Confusion Matrix...")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=fig_path / "fig1_confusion_matrix.png",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=fig_path / "fig1_confusion_matrix.pdf",
    )

    # Figure 2: Multi-Class ROC Curves
    print("  -> Generating Fig 2: One-vs-Rest ROC Curves...")
    plot_multiclass_roc(
        y_true, y_probs, class_names=CLASS_NAMES, save_path=fig_path / "fig2_multiclass_roc.png"
    )
    plot_multiclass_roc(
        y_true, y_probs, class_names=CLASS_NAMES, save_path=fig_path / "fig2_multiclass_roc.pdf"
    )

    # Figure 3: Precision-Recall Curves
    print("  -> Generating Fig 3: Precision-Recall Curves...")
    plot_precision_recall_curves(
        y_true, y_probs, class_names=CLASS_NAMES, save_path=fig_path / "fig3_precision_recall_curves.png"
    )
    plot_precision_recall_curves(
        y_true, y_probs, class_names=CLASS_NAMES, save_path=fig_path / "fig3_precision_recall_curves.pdf"
    )

    # Figure 4: Training Dynamics (if history JSON exists)
    history_file = Path(chk_file).parent / "training_history.json"
    if history_file.is_file():
        print("  -> Generating Fig 4: Training & Validation Convergence...")
        with open(history_file, "r") as f:
            history = json.load(f)
        plot_training_curves(history, save_path=fig_path / "fig4_training_dynamics.png")
        plot_training_curves(history, save_path=fig_path / "fig4_training_dynamics.pdf")

    # Figure 5: Reliability Diagram & Uncertainty Calibration
    print("  -> Generating Fig 5: Reliability & Calibration Diagram...")
    plot_reliability_and_calibration(
        y_true, y_probs, uncertainties, save_path=fig_path / "fig5_reliability_and_calibration.png"
    )
    plot_reliability_and_calibration(
        y_true, y_probs, uncertainties, save_path=fig_path / "fig5_reliability_and_calibration.pdf"
    )

    # Figure 6: Small Tumor vs All Cohort Analysis
    print("  -> Generating Fig 6: Early/Small-Tumour Comparative Analysis...")
    plot_small_tumor_analysis(
        test_metrics, small_metrics, save_path=fig_path / "fig6_small_tumor_comparison.png"
    )
    plot_small_tumor_analysis(
        test_metrics, small_metrics, save_path=fig_path / "fig6_small_tumor_comparison.pdf"
    )

    # Figure 7: CARE-XAI Multi-Panel Clinical Interpretability
    print("  -> Generating Fig 7: CARE-XAI Interpretability Panel...")
    care_xai = CareXAI(model, device=str(target_device))
    # Pick a tumor slice with mask for illustration
    sample_idx = 0
    for idx in range(len(test_dataset)):
        s = test_dataset[idx]
        if s["has_mask"].item() > 0.5 and s["label"].item() != 3:
            sample_idx = idx
            break

    sample = test_dataset[sample_idx]
    sample_img = sample["image"].unsqueeze(0)  # (1, 1, 224, 224)
    sample_mask = sample["mask"].numpy()
    sample_label = sample["label"].item()

    # Compute rollout
    rollout_map = care_xai.compute_attention_rollout(sample_img, discard_ratio=0.85)

    with torch.no_grad():
        sample_out = model(sample_img.to(target_device))
        pred_label_idx = torch.argmax(sample_out["probabilities"], dim=1).item()
        pred_conf = sample_out["probabilities"][0, pred_label_idx].item()
        pred_unc = sample_out["uncertainty"][0].item()
        pred_mask_sample = sample_out["pred_mask"].cpu().numpy()

    plot_care_xai_interpretability_panel(
        image=sample_img.squeeze().numpy(),
        true_mask=sample_mask,
        pred_mask=pred_mask_sample,
        rollout_map=rollout_map,
        pred_class=CLASS_NAMES[pred_label_idx],
        true_class=CLASS_NAMES[sample_label],
        confidence=pred_conf,
        uncertainty=pred_unc,
        save_path=fig_path / "fig7_care_xai_interpretability_panel.png",
    )
    plot_care_xai_interpretability_panel(
        image=sample_img.squeeze().numpy(),
        true_mask=sample_mask,
        pred_mask=pred_mask_sample,
        rollout_map=rollout_map,
        pred_class=CLASS_NAMES[pred_label_idx],
        true_class=CLASS_NAMES[sample_label],
        confidence=pred_conf,
        uncertainty=pred_unc,
        save_path=fig_path / "fig7_care_xai_interpretability_panel.pdf",
    )

    # 5. Export LaTeX Table & JSON Report
    print("\n[4/4] Exporting LaTeX Table & JSON Report...")
    latex_code = generate_latex_table(test_metrics)
    with open(metrics_path / "latex_table.tex", "w") as f:
        f.write(latex_code)

    report = {
        "primary_test_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in test_metrics.items()
        },
        "small_tumor_subgroup_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in small_metrics.items()
        },
    }
    with open(metrics_path / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=======================================================")
    print(f"             EVALUATION REPORT SUMMARY                 ")
    print(f"=======================================================")
    print(f"Overall Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:         {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:            {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:         {test_metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Specificity:         {test_metrics['macro_specificity']*100:.2f}%")
    print(f"Cohen's Kappa (κ):         {test_metrics['cohen_kappa']:.4f}")
    print(f"Matthews Corr (MCC):       {test_metrics['matthews_corrcoef']:.4f}")
    print(f"Expected Calib Error (ECE):{test_metrics.get('expected_calibration_error', 0.0):.4f}")
    if "segmentation" in test_metrics:
        print(f"Auxiliary Dice (DSC):      {test_metrics['segmentation']['dice_coefficient']*100:.2f}%")
        print(f"Auxiliary IoU:             {test_metrics['segmentation']['iou_jaccard']*100:.2f}%")
    print(f"--- Small-Tumour Subgroup ---")
    print(f"Small-Tumour Sensitivity:  {small_metrics['macro_recall']*100:.2f}%")
    print(f"Small-Tumour F1-Score:     {small_metrics['macro_f1']*100:.2f}%")
    print(f"=======================================================")
    print(f"All 7 figures saved to: {fig_path.resolve()}")
    print(f"LaTeX table saved to:   {(metrics_path / 'latex_table.tex').resolve()}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate X-R-MiTNet on Test Sets")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_full_journal_evaluation(
        model_checkpoint=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

