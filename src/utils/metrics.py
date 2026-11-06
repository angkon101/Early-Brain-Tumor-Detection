"""Journal-Standard Comprehensive Evaluation Metrics Suite.

Calculates:
1. Multi-Class Classification:
   - Accuracy, Balanced Accuracy
   - Macro/Weighted/Micro Precision, Recall (Sensitivity), Specificity, F1-Score
   - Per-Class Metrics with 95% Confidence Intervals
   - Matthews Correlation Coefficient (MCC), Cohen's Kappa
   - Multi-Class One-vs-Rest AUC-ROC & PR-AUC
2. Evidential Uncertainty & Clinical Reliability:
   - Expected Calibration Error (ECE)
   - Brier Score
   - Epistemic Uncertainty Distribution (Correct vs Error separation)
   - Clinical Reliability Index (omega = 1 - u)
3. Auxiliary Segmentation:
   - Dice Similarity Coefficient (DSC)
   - Intersection over Union (IoU / Jaccard Index)
   - Mask Precision and Recall
4. LaTeX Table Generator for direct journal paper insertion.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


def compute_expected_calibration_error(
    y_true: np.ndarray, y_probs: np.ndarray, num_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Expected Calibration Error (ECE) and binning statistics.

    Args:
        y_true: (N,) ground-truth class labels
        y_probs: (N, K) predicted probabilities
        num_bins: number of confidence bins (standard is 10 or 15)
    Returns:
        ece: scalar ECE value in [0, 1]
        bin_accuracies: (num_bins,) empirical accuracy per bin
        bin_confidences: (num_bins,) mean confidence per bin
        bin_counts: (num_bins,) sample count per bin
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins, dtype=int)

    n_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        bin_counts[i] = count

        if count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf
            ece += (count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """Computes Specificity (True Negative Rate) for each class from a confusion matrix."""
    num_classes = cm.shape[0]
    specificities = np.zeros(num_classes)
    for c in range(num_classes):
        # TN is sum of all elements outside row c and column c
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        if (tn + fp) > 0:
            specificities[c] = tn / (tn + fp)
        else:
            specificities[c] = 0.0
    return specificities


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Calculates non-parametric percentile bootstrap 95% Confidence Intervals.

    Essential for peer-reviewed medical imaging journals (IEEE TMI / Nature Med).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    alpha_lower = (1.0 - confidence_level) / 2.0 * 100.0
    alpha_upper = (1.0 + confidence_level) / 2.0 * 100.0

    boot_acc = []
    boot_bal_acc = []
    boot_macro_f1 = []
    boot_weighted_f1 = []
    boot_macro_prec = []
    boot_macro_rec = []
    boot_mcc = []
    boot_kappa = []
    boot_auc = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, size=n_samples)
        b_true = y_true[indices]
        b_pred = y_pred[indices]

        # Require at least 2 distinct classes to compute metrics safely
        if len(np.unique(b_true)) < 2:
            continue

        boot_acc.append(accuracy_score(b_true, b_pred))
        boot_bal_acc.append(balanced_accuracy_score(b_true, b_pred))
        boot_macro_f1.append(f1_score(b_true, b_pred, average="macro", zero_division=0))
        boot_weighted_f1.append(f1_score(b_true, b_pred, average="weighted", zero_division=0))
        boot_macro_prec.append(precision_score(b_true, b_pred, average="macro", zero_division=0))
        boot_macro_rec.append(recall_score(b_true, b_pred, average="macro", zero_division=0))
        boot_mcc.append(matthews_corrcoef(b_true, b_pred))
        boot_kappa.append(cohen_kappa_score(b_true, b_pred))

        if y_probs is not None:
            b_probs = y_probs[indices]
            try:
                boot_auc.append(roc_auc_score(b_true, b_probs, multi_class="ovr", average="macro"))
            except Exception:
                pass

    def get_ci(vals: List[float]) -> Tuple[float, float]:
        if len(vals) == 0:
            return (0.0, 0.0)
        return (float(np.percentile(vals, alpha_lower)), float(np.percentile(vals, alpha_upper)))

    ci_results = {
        "accuracy": get_ci(boot_acc),
        "balanced_accuracy": get_ci(boot_bal_acc),
        "macro_f1": get_ci(boot_macro_f1),
        "weighted_f1": get_ci(boot_weighted_f1),
        "macro_precision": get_ci(boot_macro_prec),
        "macro_recall": get_ci(boot_macro_rec),
        "matthews_corrcoef": get_ci(boot_mcc),
        "cohen_kappa": get_ci(boot_kappa),
    }
    if len(boot_auc) > 0:
        ci_results["auc_roc_macro"] = get_ci(boot_auc)

    return ci_results


def compute_comprehensive_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_probs: Optional[Union[List[List[float]], np.ndarray]] = None,
    uncertainties: Optional[Union[List[float], np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    compute_bootstrap: bool = False,
    n_bootstraps: int = 1000,
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """Calculates all journal-grade metrics for multi-class tumor classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Overall Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Averages
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Specificities
    class_specificities = compute_specificity_per_class(cm)
    macro_spec = float(np.mean(class_specificities))

    # Per-class breakdowns
    per_class = {}
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for c, name in enumerate(class_names):
        support = int(np.sum(y_true == c))
        per_class[name] = {
            "precision": float(prec_per_class[c]),
            "recall": float(rec_per_class[c]),
            "specificity": float(class_specificities[c]),
            "f1_score": float(f1_per_class[c]),
            "support": support,
        }

    metrics: Dict[str, Union[float, Dict, np.ndarray]] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_specificity": macro_spec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "matthews_corrcoef": mcc,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "per_class": per_class,
    }

    # AUC-ROC, PR-AUC, and Brier Score if probabilities provided
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float)
        try:
            # One-vs-Rest AUC-ROC
            auc_macro = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            auc_weighted = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))
            metrics["auc_roc_macro"] = auc_macro
            metrics["auc_roc_weighted"] = auc_weighted

            # Per-class AUC & PR-AUC
            y_onehot = np.eye(num_classes)[y_true]
            for c, name in enumerate(class_names):
                if len(np.unique(y_onehot[:, c])) > 1:
                    c_auc = float(roc_auc_score(y_onehot[:, c], y_probs[:, c]))
                    c_ap = float(average_precision_score(y_onehot[:, c], y_probs[:, c]))
                    metrics["per_class"][name]["auc_roc"] = c_auc
                    metrics["per_class"][name]["pr_auc"] = c_ap
                else:
                    metrics["per_class"][name]["auc_roc"] = 1.0
                    metrics["per_class"][name]["pr_auc"] = 1.0

            # Multi-Class Brier Score: \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (y_{ik} - p_{ik})^2
            brier = float(np.mean(np.sum((y_onehot - y_probs) ** 2, axis=1)))
            metrics["brier_score"] = brier

        except Exception as e:
            metrics["auc_roc_macro"] = 0.0
            metrics["auc_roc_weighted"] = 0.0
            metrics["brier_score"] = 0.0

        # Calibration (ECE & Brier score)
        ece, bin_acc, bin_conf, bin_counts = compute_expected_calibration_error(y_true, y_probs)
        metrics["expected_calibration_error"] = ece
        metrics["ece_bin_accuracies"] = bin_acc
        metrics["ece_bin_confidences"] = bin_conf
        metrics["ece_bin_counts"] = bin_counts

    # Uncertainty Quantification metrics
    if uncertainties is not None:
        unc = np.asarray(uncertainties, dtype=float)
        correct_mask = (y_pred == y_true)
        unc_correct = unc[correct_mask] if np.sum(correct_mask) > 0 else np.array([0.0])
        unc_incorrect = unc[~correct_mask] if np.sum(~correct_mask) > 0 else np.array([1.0])

        metrics["mean_uncertainty_overall"] = float(np.mean(unc))
        metrics["mean_reliability_overall"] = float(np.mean(1.0 - unc))
        metrics["mean_uncertainty_correct"] = float(np.mean(unc_correct))
        metrics["mean_uncertainty_incorrect"] = float(np.mean(unc_incorrect))

        # Separation T-test
        if len(unc_incorrect) > 1 and len(unc_correct) > 1:
            t_stat, p_val = stats.ttest_ind(unc_incorrect, unc_correct, equal_var=False)
            metrics["uncertainty_separation_t_stat"] = float(t_stat)
            metrics["uncertainty_separation_p_value"] = float(p_val)
        else:
            metrics["uncertainty_separation_t_stat"] = 0.0
            metrics["uncertainty_separation_p_value"] = 1.0

    # Non-parametric Percentile Bootstrap 95% Confidence Intervals (B=1000)
    if compute_bootstrap:
        metrics["bootstrap_ci"] = compute_bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            n_bootstraps=n_bootstraps,
        )

    return metrics


def compute_segmentation_metrics(
    pred_masks: np.ndarray, true_masks: np.ndarray, threshold: float = 0.5, eps: float = 1e-7
) -> Dict[str, float]:
    """Computes pixel-wise segmentation metrics (Dice, IoU, Precision, Recall).

    Args:
        pred_masks: (N, 1, H, W) or (N, H, W) probabilities in [0, 1]
        true_masks: (N, 1, H, W) or (N, H, W) binary masks
        threshold: binarization cutoff
    """
    preds_binary = (pred_masks > threshold).astype(np.float32).reshape(-1)
    targets_binary = (true_masks > 0.5).astype(np.float32).reshape(-1)

    intersection = np.sum(preds_binary * targets_binary)
    pred_sum = np.sum(preds_binary)
    target_sum = np.sum(targets_binary)

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (pred_sum + target_sum - intersection + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return {
        "dice_coefficient": float(dice),
        "iou_jaccard": float(iou),
        "mask_precision": float(precision),
        "mask_recall": float(recall),
    }


def generate_latex_table(metrics: Dict[str, Union[float, Dict]]) -> str:
    """Generates an IEEE/Nature style LaTeX table snippet summarizing evaluation results with 95% CIs."""
    ci = metrics.get("bootstrap_ci", {})

    def fmt_val(key: str, mult: float = 100.0, decimals: int = 2) -> str:
        val = metrics.get(key, 0.0) * mult
        if key in ci:
            low, high = ci[key][0] * mult, ci[key][1] * mult
            return f"{val:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"
        return f"{val:.{decimals}f}"

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comprehensive Evaluation of X-R-MiTNet on Multi-Cohort Brain MRI Dataset (Mean [95\% Bootstrap Confidence Intervals]).}",
        r"\label{tab:evaluation_metrics}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Class / Subgroup} & \textbf{Precision (\%)} & \textbf{Recall / Sens (\%)} & \textbf{Specificity (\%)} & \textbf{F1-Score (\%)} & \textbf{AUC-ROC} \\",
        r"\midrule",
    ]

    per_class = metrics.get("per_class", {})
    for name, c_dict in per_class.items():
        p = c_dict.get("precision", 0.0) * 100
        r = c_dict.get("recall", 0.0) * 100
        s = c_dict.get("specificity", 0.0) * 100
        f1 = c_dict.get("f1_score", 0.0) * 100
        auc = c_dict.get("auc_roc", 0.0)
        latex_lines.append(f"{name} & {p:.2f} & {r:.2f} & {s:.2f} & {f1:.2f} & {auc:.4f} \\\\")

    latex_lines.extend([
        r"\midrule",
        f"\\textbf{{Macro Average}} & {fmt_val('macro_precision')} & {fmt_val('macro_recall')} & {metrics.get('macro_specificity', 0.0)*100:.2f} & {fmt_val('macro_f1')} & {fmt_val('auc_roc_macro', 1.0, 4)} \\\\",
        f"\\textbf{{Weighted Average}} & {metrics.get('weighted_precision', 0.0)*100:.2f} & {metrics.get('weighted_recall', 0.0)*100:.2f} & - & {fmt_val('weighted_f1')} & - \\\\",
        r"\midrule",
        f"\\textbf{{Overall Accuracy}} & \\multicolumn{{5}}{{c}}{{{fmt_val('accuracy')}\\% (Balanced: {fmt_val('balanced_accuracy')}\\%)}} \\\\",
        f"\\textbf{{Cohen's Kappa ($\\kappa$)}} & \\multicolumn{{5}}{{c}}{{{fmt_val('cohen_kappa', 1.0, 4)}}} \\\\",
        f"\\textbf{{Matthews Corr (MCC)}} & \\multicolumn{{5}}{{c}}{{{fmt_val('matthews_corrcoef', 1.0, 4)}}} \\\\",
        f"\\textbf{{Expected Calib Error (ECE)}} & \\multicolumn{{5}}{{c}}{{{metrics.get('expected_calibration_error', 0.0):.4f}}} \\\\",
        f"\\textbf{{Brier Score}} & \\multicolumn{{5}}{{c}}{{{metrics.get('brier_score', 0.0):.4f}}} \\\\",
    ])

    if "segmentation" in metrics:
        seg = metrics["segmentation"]
        latex_lines.extend([
            r"\midrule",
            f"\\textbf{{Auxiliary Lesion Dice (DSC)}} & \\multicolumn{{5}}{{c}}{{{seg.get('dice_coefficient', 0.0)*100:.2f}\\% (IoU: {seg.get('iou_jaccard', 0.0)*100:.2f}\\%)}} \\\\",
            f"\\textbf{{Auxiliary Mask Prec / Recall}} & \\multicolumn{{5}}{{c}}{{Precision: {seg.get('mask_precision', 0.0)*100:.2f}\\% \\quad Recall: {seg.get('mask_recall', 0.0)*100:.2f}\\%}} \\\\",
        ])

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(latex_lines)

