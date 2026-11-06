"""High-Performance Multi-Task Training Engine for X-R-MiTNet.

Features:
1. Multi-Task Supervision:
   - Evidential Deep Learning for Primary Detection / 4-Class Classification
   - Pure-Transformer Auxiliary Pixel-Level Segmentation
   - Annealed Kullback-Leibler (KL) Regularization
2. Optimization:
   - AdamW with weight decay
   - Cosine Annealing Learning Rate Schedule with Warmup
   - Native PyTorch Mixed-Precision (torch.cuda.amp)
3. Checkpoint Management:
   - Best Validation Macro-F1 Model Saving
   - Training history JSON serialization
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorDataset
from src.models.evidential import EvidentialLoss, FocalEvidentialLoss
from src.models.segmentation_head import DiceBCELoss
from src.models.x_r_mitnet import XRMiTNet
from src.utils.metrics import compute_comprehensive_classification_metrics, compute_segmentation_metrics


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    evidential_loss_fn: nn.Module,
    dice_bce_loss_fn: DiceBCELoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    seg_weight: float = 1.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Trains the model for one epoch."""
    model.train()
    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    seg_loss_accum = 0.0
    kl_loss_accum = 0.0

    correct_predictions = 0
    total_samples = 0

    epoch_ratio = min(1.0, (epoch + 1) / max(1, total_epochs // 2))

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{total_epochs:02d} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(amp_device, enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            alpha = outputs["alpha"]
            mask_logits = outputs["mask_logits"]

            # 1. Evidential Classification Loss
            cls_loss, ace_loss, kl_loss = evidential_loss_fn(alpha, labels, epoch_ratio=epoch_ratio)

            # 2. Auxiliary Segmentation Loss (only on slices with masks)
            mask_count = has_mask.sum()
            if mask_count > 0:
                # Mask out negative control slices without annotation
                seg_loss = (dice_bce_loss_fn(mask_logits, masks) * has_mask.view(-1, 1, 1, 1)).sum() / (mask_count + 1e-8)
            else:
                seg_loss = torch.tensor(0.0, device=device)

            # 3. Total Multi-Task Loss
            total_loss = cls_loss + (seg_weight * seg_loss)

        # Backward pass with Scaler
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        bs = images.size(0)
        total_samples += bs
        preds = torch.argmax(outputs["probabilities"], dim=1)
        correct_predictions += (preds == labels).sum().item()

        total_loss_accum += total_loss.item() * bs
        cls_loss_accum += cls_loss.item() * bs
        seg_loss_accum += seg_loss.item() * bs
        kl_loss_accum += kl_loss.item() * bs

        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "Cls": f"{cls_loss.item():.4f}",
            "Seg": f"{seg_loss.item():.4f}",
            "Acc": f"{(correct_predictions/total_samples)*100:.1f}%",
        })

    return {
        "train_loss": total_loss_accum / total_samples,
        "train_cls_loss": cls_loss_accum / total_samples,
        "train_seg_loss": seg_loss_accum / total_samples,
        "train_kl_loss": kl_loss_accum / total_samples,
        "train_acc": correct_predictions / total_samples,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    evidential_loss_fn: EvidentialLoss,
    dice_bce_loss_fn: DiceBCELoss,
    device: torch.device,
    seg_weight: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, Union[float, Dict]]]:
    """Evaluates the model on validation data."""
    model.eval()
    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    seg_loss_accum = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_probs = []
    all_unc = []

    pred_masks_list = []
    true_masks_list = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        outputs = model(images)
        alpha = outputs["alpha"]
        probs = outputs["probabilities"]
        unc = outputs["uncertainty"]
        mask_logits = outputs["mask_logits"]
        pred_mask = outputs["pred_mask"]

        cls_loss, _, _ = evidential_loss_fn(alpha, labels, epoch_ratio=1.0)

        mask_count = has_mask.sum()
        if mask_count > 0:
            seg_loss = (dice_bce_loss_fn(mask_logits, masks) * has_mask.view(-1, 1, 1, 1)).sum() / (mask_count + 1e-8)
        else:
            seg_loss = torch.tensor(0.0, device=device)

        total_loss = cls_loss + (seg_weight * seg_loss)

        bs = images.size(0)
        total_samples += bs
        total_loss_accum += total_loss.item() * bs
        cls_loss_accum += cls_loss.item() * bs
        seg_loss_accum += seg_loss.item() * bs

        preds = torch.argmax(probs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_unc.extend(unc.cpu().numpy())

        # Collect segmentation masks for slices with masks
        has_mask_np = has_mask.cpu().numpy().astype(bool)
        if np.any(has_mask_np):
            pred_masks_list.append(pred_mask.cpu().numpy()[has_mask_np])
            true_masks_list.append(masks.cpu().numpy()[has_mask_np])

    # Classification Metrics
    metrics = compute_comprehensive_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_probs=all_probs,
        uncertainties=all_unc,
    )

    # Segmentation Metrics
    val_dice = 0.0
    if len(pred_masks_list) > 0:
        pred_masks_arr = np.concatenate(pred_masks_list, axis=0)
        true_masks_arr = np.concatenate(true_masks_list, axis=0)
        seg_metrics = compute_segmentation_metrics(pred_masks_arr, true_masks_arr)
        val_dice = seg_metrics["dice_coefficient"]
        metrics["segmentation"] = seg_metrics

    summary = {
        "val_loss": total_loss_accum / total_samples,
        "val_cls_loss": cls_loss_accum / total_samples,
        "val_seg_loss": seg_loss_accum / total_samples,
        "val_acc": float(metrics["accuracy"]),
        "val_f1": float(metrics["macro_f1"]),
        "val_dice": val_dice,
    }

    return summary, metrics


def train_model(
    data_root: Optional[Union[str, Path]] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    seg_weight: float = 1.0,
    kl_coeff: float = 0.2,
    device: Optional[str] = None,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Full training pipeline for Google Colab and local execution."""
    set_seed(seed)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"--- Training X-R-MiTNet on device: {target_device} ---")
    if target_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Prepare Datasets & Dataloaders
    train_dataset = BrainTumorDataset(
        cohort="multitask", split="train", data_root=data_root, is_training=True, augment=True
    )
    val_dataset = BrainTumorDataset(
        cohort="multitask", split="val", data_root=data_root, is_training=False, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(target_device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(target_device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print(f"Dataset Loaded: Train = {len(train_dataset)} scans, Val = {len(val_dataset)} scans.")

    # 2. Instantiate Model
    model = XRMiTNet(
        img_size=224,
        in_channels=1,
        num_classes=4,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        drop_rate=0.1,
    ).to(target_device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"X-R-MiTNet Initialized: {num_params:,} trainable parameters.")

    # 3. Losses, Optimizer, Scheduler, Scaler
    evidential_loss_fn = FocalEvidentialLoss(num_classes=4, kl_coeff=kl_coeff, gamma=2.0)
    dice_bce_loss_fn = DiceBCELoss(smooth=1.0, bce_weight=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    amp_dev = "cuda" if target_device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_dev, enabled=(target_device.type == "cuda"))

    # 4. Training Loop
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_seg_loss": [],
        "train_kl_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_seg_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_dice": [],
    }

    best_val_f1 = -1.0
    best_model_path = checkpoint_path / "best_model.pth"

    for epoch in range(epochs):
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            evidential_loss_fn=evidential_loss_fn,
            dice_bce_loss_fn=dice_bce_loss_fn,
            device=target_device,
            epoch=epoch,
            total_epochs=epochs,
            seg_weight=seg_weight,
            use_amp=(target_device.type == "cuda"),
        )

        val_res, val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            evidential_loss_fn=evidential_loss_fn,
            dice_bce_loss_fn=dice_bce_loss_fn,
            device=target_device,
            seg_weight=seg_weight,
        )

        scheduler.step()

        # Update History
        for k, v in train_res.items():
            history[k].append(v)
        for k, v in val_res.items():
            history[k].append(v)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {train_res['train_loss']:.4f} (Acc: {train_res['train_acc']*100:.1f}%) | "
            f"Val Loss: {val_res['val_loss']:.4f} (Acc: {val_res['val_acc']*100:.1f}%, Macro-F1: {val_res['val_f1']*100:.1f}%, Dice: {val_res['val_dice']*100:.1f}%)"
        )

        # Save Best Checkpoint based on Validation Macro-F1
        if val_res["val_f1"] > best_val_f1:
            best_val_f1 = val_res["val_f1"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_metrics,
            }, best_model_path)
            print(f"  >>> Best model saved with Val Macro-F1: {best_val_f1*100:.2f}% to {best_model_path}")

    # Save latest model & training history
    torch.save(model.state_dict(), checkpoint_path / "latest_model.pth")
    with open(checkpoint_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n--- Training Complete! Best Validation Macro-F1: {best_val_f1*100:.2f}% ---")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train X-R-MiTNet on Brain Tumor Dataset")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seg_weight", type=float, default=1.0, help="Weight for auxiliary segmentation loss")
    parser.add_argument("--kl_coeff", type=float, default=0.2, help="KL divergence regularization coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seg_weight=args.seg_weight,
        kl_coeff=args.kl_coeff,
        seed=args.seed,
    )

