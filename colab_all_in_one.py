# ==============================================================================
# 🧠 X-R-MiTNet & CARE-XAI: ALL-IN-ONE MASTER PIPELINE FOR GOOGLE COLAB
# Targeted for High-Impact Scopus/SCI Medical AI Journal Publication
# ==============================================================================
# INSTRUCTIONS FOR RUNNING ON GOOGLE COLAB:
# 1. Enable GPU: In Google Colab top menu, go to:
#    Runtime -> Change runtime type -> Select 'T4 GPU' (or A100/V100) -> Save.
#
# 2. Provide the Data (Choose ONE option):
#    Option A (Google Drive - Recommended):
#      - Put 'data_colab.zip' (created in your repo) into your Google Drive root.
#      - Uncomment 'drive.mount("/content/drive")' below. It will auto-unzip!
#    Option B (Direct Upload):
#      - Drag & drop 'data_colab.zip' into the left Colab Files panel.
#      - This script will automatically detect and unzip it!
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 0: DEPENDENCY INSTALLATION & SYSTEM IMPORTS
# ------------------------------------------------------------------------------
import os
import sys
import io
import math
import time
import json
import zipfile
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Auto-install dependencies if running on Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("[Environment] Google Colab Detected. Installing dependencies...")
    get_ipython().system("pip install -q albumentations scikit-learn seaborn matplotlib tqdm gradio ipywidgets opencv-python")
except (ImportError, NameError):
    IN_COLAB = False
    print("[Environment] Standard Python Environment Detected.")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, auc
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Set seed for scientific reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[WARNING] Currently running on CPU! In Colab, please switch to GPU: Runtime -> Change runtime type -> T4 GPU")

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


# ------------------------------------------------------------------------------
# AUTOMATIC DATASET UNPACKING & SEARCH
# ------------------------------------------------------------------------------
def auto_prepare_data():
    """Detects and unzips dataset archives from /content or cwd if images are missing."""
    def has_images(directory):
        p = Path(directory)
        if not p.is_dir():
            return False
        for root, dirs, files in os.walk(p):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return True
        return False

    # 1. Check if actual MRI images already exist
    for cand_dir in [
        "data/01_primary_multitask",
        "01_primary_multitask",
        "/content/data/01_primary_multitask",
        "/content/Early-Brain-Tumor-Detection/data/01_primary_multitask",
    ]:
        if has_images(cand_dir):
            print(f"[OK] Verified: Real MRI scans & masks present in '{cand_dir}'.")
            return

    print("[*] MRI images not yet extracted. Scanning for uploaded .zip files...")

    # 2. Search for uploaded zip files across /content and current directory
    search_dirs = [
        Path.cwd(),
        Path("/content"),
        Path("/content/Early-Brain-Tumor-Detection"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/Brain_Tumor")
    ]

    zip_candidates = []
    for s_dir in search_dirs:
        if s_dir.is_dir():
            for zf in s_dir.glob("*.zip"):
                zip_candidates.append(zf)

    target_zip = None
    for zf in zip_candidates:
        fname = zf.name.lower()
        if any(k in fname for k in ["data", "colab", "brain", "tumor", "brisc"]):
            target_zip = zf
            break

    if target_zip is None and len(zip_candidates) > 0:
        target_zip = zip_candidates[0]

    if target_zip is not None and target_zip.is_file():
        print(f"[ZIP] Found uploaded dataset archive at: '{target_zip}'")
        print(f"      Unzipping files into '{Path.cwd().resolve()}' (takes ~15 seconds)...")
        with zipfile.ZipFile(target_zip, "r") as z:
            z.extractall(Path.cwd())
        print("[OK] Extraction complete! All MRI images & masks are now available.")
        return

    print("[!] Could not locate zip archive automatically. In Colab run: !unzip -q /content/*.zip")

auto_prepare_data()


# ==============================================================================
# STEP 1: NEURAL ARCHITECTURE: X-R-MiTNet (CNN-Free Vision Transformer)
# ==============================================================================

class LinearPatchEmbedding(nn.Module):
    """Pure CNN-Free Linear Patch Embedding for 224x224 grayscale MRI scans."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P, G = self.patch_size, self.grid_size
        patches = x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, G * G, -1)
        return self.proj(patches)


class SymmetryContrastiveAttention(nn.Module):
    """Bilateral Symmetry-Contrastive Multi-Head Self-Attention (S-MHSA)."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, grid_size: int = 14, drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.grid_size = grid_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        # Bilateral Asymmetry Gate
        self.asymmetry_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid(),
        )

        mirror_indices = []
        for r in range(grid_size):
            for c in range(grid_size):
                mirror_indices.append(r * grid_size + (grid_size - 1 - c))
        self.register_buffer("mirror_indices", torch.tensor(mirror_indices, dtype=torch.long))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, S, D = x.shape
        cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]

        mirror_tokens = patch_tokens[:, self.mirror_indices, :]
        asym_gate = self.asymmetry_proj(torch.abs(patch_tokens - mirror_tokens))
        modulated_patches = patch_tokens * (1.0 + asym_gate)
        x_mod = torch.cat([cls_token, modulated_patches], dim=1)

        qkv = self.qkv(x_mod).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn_weights)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj_drop(self.proj(out))

        return (out, attn_weights) if return_attention else (out, None)


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and S-MHSA."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, grid_size: int = 14, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SymmetryContrastiveAttention(embed_dim, num_heads, grid_size, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class EvidentialHead(nn.Module):
    """Evidential Deep Learning Head (Dirichlet distribution over probabilities)."""
    def __init__(self, in_features: int, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mlp(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probabilities = alpha / S
        uncertainty = (self.num_classes / S).squeeze(-1)
        reliability = torch.clamp(1.0 - uncertainty, min=0.0, max=1.0)
        return {
            "evidence": evidence, "alpha": alpha, "probabilities": probabilities,
            "uncertainty": uncertainty, "reliability": reliability, "logits": logits
        }


class PureTransformerSegmentationDecoder(nn.Module):
    """Auxiliary CNN-Free PixelShuffle Segmentation Decoder."""
    def __init__(self, embed_dim: int = 256, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        dim1 = embed_dim // 2
        dim2 = dim1 // 2
        dim3 = dim2 // 2

        self.expand1 = nn.Conv2d(embed_dim, dim1 * 4, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim1)
        self.expand2 = nn.Conv2d(dim1, dim2 * 4, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim2)
        self.expand3 = nn.Conv2d(dim2, dim3 * 4, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(dim3)
        self.expand4 = nn.Conv2d(dim3, 1 * 4, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.act = nn.GELU()

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = patch_tokens.shape
        H = W = self.grid_size
        x = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        x = self.act(self.norm1(self.pixel_shuffle(self.expand1(x))))
        x = self.act(self.norm2(self.pixel_shuffle(self.expand2(x))))
        x = self.act(self.norm3(self.pixel_shuffle(self.expand3(x))))
        mask_logits = self.pixel_shuffle(self.expand4(x))
        pred_mask = torch.sigmoid(mask_logits)
        return mask_logits, pred_mask


class XRMiTNet(nn.Module):
    """Explainable Reliability-Aware Multimodal Imaging Transformer Network."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, num_classes: int = 4, embed_dim: int = 256, depth: int = 6, num_heads: int = 8):
        super().__init__()
        self.patch_embed = LinearPatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, self.patch_embed.grid_size, drop=0.1) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.evidential_head = EvidentialHead(embed_dim, num_classes)
        self.seg_decoder = PureTransformerSegmentationDecoder(embed_dim, self.patch_embed.grid_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B = x.shape[0]
        patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = self.pos_drop(torch.cat((cls_tokens, patches), dim=1) + self.pos_embed)

        attention_matrices = []
        for block in self.blocks:
            tokens, attn = block(tokens, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_matrices.append(attn)

        tokens = self.norm(tokens)
        cls_feat, patch_feat = tokens[:, 0], tokens[:, 1:]

        evidential_out = self.evidential_head(cls_feat)
        mask_logits, pred_mask = self.seg_decoder(patch_feat)

        out = {
            **evidential_out,
            "mask_logits": mask_logits,
            "pred_mask": pred_mask,
        }
        if return_attention:
            out["attention_weights"] = attention_matrices
        return out


# ==============================================================================
# STEP 2: LOSS FUNCTIONS & CARE-XAI ATTENTION ROLLOUT
# ==============================================================================

class EvidentialLoss(nn.Module):
    """Dirichlet Adjusted Cross-Entropy with Annealed KL Divergence."""
    def __init__(self, num_classes: int = 4, kl_coeff: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.kl_coeff = kl_coeff

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch_ratio: float = 1.0):
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        ace_loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1).mean()

        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        kl = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1)
            + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1)
        ).mean()

        total_loss = ace_loss + min(1.0, epoch_ratio) * self.kl_coeff * kl
        return total_loss, ace_loss, kl


class DiceBCELoss(nn.Module):
    """Combined Soft Dice + BCE Loss for Auxiliary Tumor Segmentation."""
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.sigmoid(logits).view(-1)
        target_flat = target.view(-1)
        intersection = (probs * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs.sum() + target_flat.sum() + self.smooth)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        return (1.0 - self.bce_weight) * dice_loss + self.bce_weight * bce_loss


def compute_attention_rollout(model: nn.Module, image: torch.Tensor, discard_ratio: float = 0.85, device: torch.device = device) -> np.ndarray:
    """Computes Transformer Attention Rollout for CARE-XAI Heatmap."""
    model.eval()
    with torch.no_grad():
        out = model(image.to(device), return_attention=True)
        attn_list = out["attention_weights"]

    fused_attns = []
    for attn in attn_list:
        fused = attn.mean(dim=1)
        if discard_ratio > 0.0:
            flat = fused.view(fused.size(0), -1)
            num_discard = int(flat.size(-1) * discard_ratio)
            indices = flat.argsort(dim=-1)[:, :num_discard]
            flat.scatter_(1, indices, 0.0)
            fused = flat.view(fused.shape)
        I = torch.eye(fused.size(-1), device=fused.device).unsqueeze(0)
        fused_aug = 0.5 * fused + 0.5 * I
        fused_aug = fused_aug / fused_aug.sum(dim=-1, keepdim=True)
        fused_attns.append(fused_aug)

    rollout = fused_attns[0]
    for l in range(1, len(fused_attns)):
        rollout = torch.bmm(fused_attns[l], rollout)

    cls_attn = rollout[0, 0, 1:]
    grid_size = int(np.sqrt(cls_attn.size(0)))
    heatmap_2d = cls_attn.view(1, 1, grid_size, grid_size)
    heatmap_full = F.interpolate(heatmap_2d, size=(224, 224), mode="bicubic", align_corners=False)
    heatmap_np = heatmap_full.squeeze().cpu().numpy()

    h_min, h_max = heatmap_np.min(), heatmap_np.max()
    return (heatmap_np - h_min) / (h_max - h_min + 1e-8) if h_max - h_min > 1e-8 else np.zeros_like(heatmap_np)


# ==============================================================================
# STEP 3: CROSS-PLATFORM DATASET LOADER (GOOGLE COLAB & WINDOWS)
# ==============================================================================

def resolve_file_path(path_str: Optional[Union[str, Path]], data_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Dynamically resolves dataset file paths across Windows, Linux, and Google Colab."""
    if path_str is None or pd.isna(path_str):
        return None
    s = str(path_str).strip()
    if s.lower() in ["none", "nan", ""]:
        return None

    direct_p = Path(s)
    if direct_p.is_file():
        return direct_p

    norm_s = s.replace("\\", "/")
    fname = direct_p.name

    # Check relative candidates
    candidates = []
    if "data/" in norm_s:
        idx = norm_s.index("data/")
        candidates.extend([norm_s[idx:], norm_s[idx + len("data/"): ]])
    elif "01_primary_multitask" in norm_s:
        idx = norm_s.index("01_primary_multitask")
        candidates.extend(["data/" + norm_s[idx:], norm_s[idx:]])

    search_bases = [
        Path.cwd(),
        Path.cwd() / "data",
        Path("/content"),
        Path("/content/data"),
        Path("/content/Early-Brain-Tumor-Detection"),
        Path("/content/Early-Brain-Tumor-Detection/data"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/Early-Brain-Tumor-Detection")
    ]
    if data_root is not None:
        p_root = Path(data_root)
        search_bases.extend([p_root, p_root / "data"])

    for base in search_bases:
        for cand in candidates:
            target = base / cand
            if target.is_file():
                return target

    # Search by filename in primary folders
    for base in search_bases:
        for sub in [
            "01_primary_multitask/segmentation/Train/images", "01_primary_multitask/segmentation/Test/images",
            "01_primary_multitask/segmentation/Train/masks", "01_primary_multitask/segmentation/Test/masks",
            "data/01_primary_multitask/segmentation/Train/images", "data/01_primary_multitask/segmentation/Test/images",
            "data/01_primary_multitask/segmentation/Train/masks", "data/01_primary_multitask/segmentation/Test/masks",
        ]:
            target = base / sub / fname
            if target.is_file():
                return target

    return None


LABEL_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3, "notumor": 3}

class BrainTumorDataset(Dataset):
    """Unified Multimodal/Multi-Task Brain Tumor Dataset."""
    def __init__(self, split_csv: Optional[Union[str, Path]] = None, split: str = "train", data_root: Optional[Union[str, Path]] = None, is_training: bool = False, augment: bool = False):
        self.is_training = is_training
        self.augment = augment and is_training
        self.data_root = data_root

        # Resolve split CSV
        if split_csv is not None:
            resolved_csv = resolve_file_path(split_csv, data_root) or Path(split_csv)
            self.df = pd.read_csv(resolved_csv)
        else:
            rel_path = f"data/splits/multitask_{split}.csv" if split in ["train", "val", "test"] else "data/splits/small_tumor_test.csv"
            resolved_csv = resolve_file_path(rel_path, data_root) or Path(rel_path)
            if not resolved_csv.is_file():
                # Colab auto-search
                for candidate_root in [Path.cwd(), Path("/content"), Path("/content/Early-Brain-Tumor-Detection")]:
                    if (candidate_root / rel_path).is_file():
                        resolved_csv = candidate_root / rel_path
                        break
            if not resolved_csv.is_file():
                raise FileNotFoundError(
                    f"\n[Error] Could not locate split CSV '{rel_path}'.\n"
                    f"Please make sure you have unzipped 'data_colab.zip' or cloned the repo:\n"
                    f"  !unzip -q data_colab.zip\n"
                )
            self.df = pd.read_csv(resolved_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_p = resolve_file_path(row["image_path"], self.data_root)
        if img_p is None or not img_p.is_file():
            raise FileNotFoundError(
                f"\n[Missing Image Error]: Cannot find MRI image file:\n  {row['image_path']}\n\n"
                f"WHY THIS HAPPENS: Medical images (*.jpg, *.png) are not stored inside git repo due to size limits.\n"
                f"HOW TO FIX ON COLAB (takes 1 minute):\n"
                f"  1. Locate 'data_colab.zip' on your local computer in 'E:\\Brain tumor\\data_colab.zip'.\n"
                f"  2. Upload 'data_colab.zip' to Google Drive or directly to the left files pane in Colab.\n"
                f"  3. Run: !unzip -q data_colab.zip\n"
            )

        img = Image.open(img_p).convert("L").resize((224, 224), Image.BILINEAR)

        has_mask = bool(row.get("has_mask", False))
        mask_p = resolve_file_path(row.get("mask_path", None), self.data_root) if has_mask else None
        if has_mask and mask_p is not None and mask_p.is_file():
            mask = Image.open(mask_p).convert("L").resize((224, 224), Image.NEAREST)
        else:
            mask = Image.new("L", (224, 224), 0)

        # Training Augmentation
        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_norm = (img_np - 0.28) / 0.22
        img_t = torch.from_numpy(img_norm).unsqueeze(0)

        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        raw_label = str(row["tumor_label"]).lower().strip().replace("notumor", "no_tumor")
        label_idx = LABEL_TO_IDX.get(raw_label, 3)

        return {
            "image": img_t, "mask": mask_t,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "has_mask": torch.tensor(1.0 if (has_mask and mask_p is not None) else 0.0, dtype=torch.float32),
            "is_small_tumor": torch.tensor(1.0 if bool(row.get("is_small_tumor", False)) else 0.0, dtype=torch.float32),
            "case_id": str(row.get("case_id", f"idx_{idx}"))
        }


# ==============================================================================
# STEP 4: JOURNAL-GRADE METRICS & FIGURE PLOTTING SUITE
# ==============================================================================

def compute_expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, num_bins: int = 10):
    """Calculates Expected Calibration Error (ECE) for clinical reliability assessment."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = np.zeros(num_bins), np.zeros(num_bins)

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_accs[i], bin_confs[i] = bin_acc, bin_conf
            ece += (count / len(y_true)) * np.abs(bin_acc - bin_conf)

    return float(ece), bin_accs, bin_confs


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, uncertainties: np.ndarray):
    """Calculates all journal classification and calibration metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    specs = []
    for c in range(4):
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        specs.append(tn / (tn + fp + 1e-8))
    macro_spec = float(np.mean(specs))

    try:
        auc_roc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except:
        auc_roc = 0.0

    ece, bin_accs, bin_confs = compute_expected_calibration_error(y_true, y_probs)

    correct_mask = (y_pred == y_true)
    unc_correct = uncertainties[correct_mask] if np.sum(correct_mask) > 0 else np.array([0.0])
    unc_error = uncertainties[~correct_mask] if np.sum(~correct_mask) > 0 else np.array([1.0])

    t_stat, p_val = stats.ttest_ind(unc_error, unc_correct, equal_var=False) if len(unc_error) > 1 and len(unc_correct) > 1 else (0.0, 1.0)

    return {
        "accuracy": acc, "balanced_accuracy": bal_acc, "matthews_corrcoef": mcc, "cohen_kappa": kappa,
        "macro_precision": macro_prec, "macro_recall": macro_rec, "macro_specificity": macro_spec,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1, "auc_roc": auc_roc,
        "expected_calibration_error": ece, "bin_accs": bin_accs, "bin_confs": bin_confs,
        "confusion_matrix": cm, "mean_unc_correct": float(np.mean(unc_correct)),
        "mean_unc_error": float(np.mean(unc_error)), "uncertainty_p_value": float(p_val)
    }


def compute_segmentation_dice(pred_masks: np.ndarray, true_masks: np.ndarray, threshold: float = 0.5):
    """Computes Dice Similarity Coefficient and Jaccard IoU."""
    preds = (pred_masks > threshold).astype(np.float32).reshape(-1)
    targets = (true_masks > 0.5).astype(np.float32).reshape(-1)
    intersection = np.sum(preds * targets)
    dice = (2.0 * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)
    iou = (intersection + 1e-7) / (preds.sum() + targets.sum() - intersection + 1e-7)
    return float(dice), float(iou)


def generate_all_7_figures(metrics: Dict, y_true: np.ndarray, y_probs: np.ndarray, uncertainties: np.ndarray, small_metrics: Dict, sample_img: np.ndarray, sample_true_mask: np.ndarray, sample_pred_mask: np.ndarray, sample_rollout: np.ndarray, history: Optional[Dict] = None, output_dir: str = "reports/figures"):
    """Generates all 7 publication-standard figures at 300 DPI."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

    # Fig 1: Confusion Matrix
    cm = metrics["confusion_matrix"]
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           title="Normalized Confusion Matrix (X-R-MiTNet)", ylabel="Ground-Truth Diagnosis", xlabel="Predicted Diagnosis")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontweight="bold" if i == j else "normal", fontsize=9)
    fig.savefig(f"{output_dir}/fig1_confusion_matrix.png")
    fig.savefig(f"{output_dir}/fig1_confusion_matrix.pdf")
    plt.close()

    # Fig 2: One-vs-Rest ROC Curves
    y_onehot = np.eye(4)[y_true]
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(4):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], y_probs[:, c])
        ax.plot(fpr, tpr, color=COLORS[c], lw=2, label=f"{CLASS_NAMES[c]} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Chance Line")
    ax.set(xlabel="False Positive Rate (1 - Specificity)", ylabel="True Positive Rate (Sensitivity)", title="Multi-Class One-vs-Rest ROC Curves")
    ax.legend(loc="lower right")
    fig.savefig(f"{output_dir}/fig2_multiclass_roc.png")
    fig.savefig(f"{output_dir}/fig2_multiclass_roc.pdf")
    plt.close()

    # Fig 3: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(4):
        prec, rec, _ = precision_recall_curve(y_onehot[:, c], y_probs[:, c])
        ax.plot(rec, prec, color=COLORS[c], lw=2, label=f"{CLASS_NAMES[c]} (AP={auc(rec, prec):.4f})")
    ax.set(xlabel="Recall (Sensitivity)", ylabel="Precision (PPV)", title="Multi-Class Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.savefig(f"{output_dir}/fig3_precision_recall_curves.png")
    fig.savefig(f"{output_dir}/fig3_precision_recall_curves.pdf")
    plt.close()

    # Fig 4: Training Dynamics (if history present)
    if history:
        fig, axs = plt.subplots(2, 2, figsize=(11, 9))
        epochs = range(1, len(history["train_loss"]) + 1)
        axs[0, 0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2b5c8f")
        axs[0, 0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#d95f02")
        axs[0, 0].set(title="(a) Multi-Task Loss Convergence", xlabel="Epoch", ylabel="Loss")
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, history.get("train_cls_loss", []), label="Evidential Loss", color="#2b5c8f")
        axs[0, 1].plot(epochs, history.get("train_seg_loss", []), "--", label="Segmentation Loss", color="#7570b3")
        axs[0, 1].set(title="(b) Component Loss Dynamics", xlabel="Epoch", ylabel="Component Loss")
        axs[0, 1].legend()

        axs[1, 0].plot(epochs, [a * 100 for a in history.get("train_acc", [])], "o-", label="Train Acc (%)", color="#2b5c8f")
        axs[1, 0].plot(epochs, [a * 100 for a in history.get("val_acc", [])], "s--", label="Val Acc (%)", color="#d95f02")
        axs[1, 0].set(title="(c) Classification Accuracy Progression", xlabel="Epoch", ylabel="Accuracy (%)")
        axs[1, 0].legend()

        axs[1, 1].plot(epochs, [f * 100 for f in history.get("val_f1", [])], "o-", label="Val Macro F1 (%)", color="#2b5c8f")
        axs[1, 1].plot(epochs, [d * 100 for d in history.get("val_dice", [])], "^--", label="Val Dice (%)", color="#1b9e77")
        axs[1, 1].set(title="(d) Validation F1 & Auxiliary Dice Progression", xlabel="Epoch", ylabel="Score (%)")
        axs[1, 1].legend()

        fig.tight_layout()
        fig.savefig(f"{output_dir}/fig4_training_dynamics.png")
        fig.savefig(f"{output_dir}/fig4_training_dynamics.pdf")
        plt.close()

    # Fig 5: Reliability Diagram & Uncertainty Separation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    bin_centers = np.linspace(0.05, 0.95, 10)
    axs[0].bar(bin_centers, metrics["bin_accs"], width=0.08, alpha=0.7, color="#2b5c8f", edgecolor="black", label="Empirical Accuracy")
    axs[0].plot([0, 1], [0, 1], "r--", lw=2, label="Perfect Calibration")
    axs[0].set(xlabel="Confidence Bin", ylabel="Accuracy", title=f"(a) Reliability Diagram (ECE = {metrics['expected_calibration_error']:.4f})")
    axs[0].legend()

    preds = np.argmax(y_probs, axis=1)
    correct_mask = (preds == y_true)
    sns.kdeplot(uncertainties[correct_mask], ax=axs[1], color="#1b9e77", fill=True, label=f"Correct (μ={metrics['mean_unc_correct']:.3f})")
    if np.sum(~correct_mask) > 0:
        sns.kdeplot(uncertainties[~correct_mask], ax=axs[1], color="#d95f02", fill=True, label=f"Error (μ={metrics['mean_unc_error']:.3f})")
    axs[1].set(xlabel="Epistemic Uncertainty $u$", ylabel="Density", title=f"(b) Uncertainty Separation (p-val: {metrics['uncertainty_p_value']:.2e})")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig5_reliability_and_calibration.png")
    fig.savefig(f"{output_dir}/fig5_reliability_and_calibration.pdf")
    plt.close()

    # Fig 6: Small-Tumour Comparative Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score"]
    m_all = [metrics["accuracy"]*100, metrics["macro_precision"]*100, metrics["macro_recall"]*100, metrics["macro_specificity"]*100, metrics["macro_f1"]*100]
    m_sm = [small_metrics["accuracy"]*100, small_metrics["macro_precision"]*100, small_metrics["macro_recall"]*100, small_metrics["macro_specificity"]*100, small_metrics["macro_f1"]*100]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, m_all, w, label="All-Cohort Test", color="#2b5c8f")
    ax.bar(x + w/2, m_sm, w, label="Early/Small-Tumour Subgroup", color="#d95f02")
    ax.set(xticks=x, xticklabels=labels, ylabel="Score (%)", title="Detection Performance: Standard Cohort vs. Early/Small Tumors", ylim=[0, 110])
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig6_small_tumor_comparison.png")
    fig.savefig(f"{output_dir}/fig6_small_tumor_comparison.pdf")
    plt.close()

    # Fig 7: CARE-XAI Multi-Panel Clinical Interpretability
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.8))
    img_disp = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
    axs[0].imshow(img_disp, cmap="gray")
    axs[0].set(title="1. Original MRI", xticks=[], yticks=[])

    axs[1].imshow(img_disp, cmap="gray")
    if sample_true_mask.max() > 0:
        axs[1].imshow(sample_true_mask, cmap="Greens", alpha=0.45)
    axs[1].set(title="2. Radiologist Mask", xticks=[], yticks=[])

    axs[2].imshow(img_disp, cmap="gray")
    if sample_pred_mask.max() > 0:
        axs[2].imshow(sample_pred_mask, cmap="Reds", alpha=0.45)
    axs[2].set(title="3. Predicted Mask", xticks=[], yticks=[])

    axs[3].imshow(sample_rollout, cmap="magma")
    axs[3].set(title="4. CARE-XAI Rollout Heatmap", xticks=[], yticks=[])

    axs[4].imshow(img_disp, cmap="gray")
    axs[4].imshow(sample_rollout, cmap="magma", alpha=0.55)
    axs[4].set(title="5. Decision Attribution Overlay", xticks=[], yticks=[])

    fig.suptitle("Clinical Interpretability Panel: Multimodal Attribution and Segmentation Localization", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fig7_care_xai_interpretability_panel.png")
    fig.savefig(f"{output_dir}/fig7_care_xai_interpretability_panel.pdf")
    plt.close()

    print(f"✅ All 7 publication-grade figures saved to '{output_dir}/' (PNG & PDF @ 300 DPI)")


# ==============================================================================
# STEP 5: END-TO-END TRAINING & EVALUATION PIPELINE
# ==============================================================================

def train_and_evaluate_pipeline(epochs=20, batch_size=32, lr=3e-4):
    """Executes model training, evaluation, and auto-generates all journal artifacts."""
    print(f"\n{'='*70}\n[STEP 1/3] INITIALIZING DATASET & MODEL ARCHITECTURE\n{'='*70}")
    train_ds = BrainTumorDataset(split="train", is_training=True, augment=True)
    val_ds = BrainTumorDataset(split="val", is_training=False)
    test_ds = BrainTumorDataset(split="test", is_training=False)
    small_ds = BrainTumorDataset(split="small_tumor_test", is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    small_loader = DataLoader(small_ds, batch_size=batch_size, shuffle=False)

    model = XRMiTNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Modern PyTorch GradScaler
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    evidential_loss_fn = EvidentialLoss(num_classes=4, kl_coeff=0.2)
    dice_bce_loss_fn = DiceBCELoss()

    history = {"train_loss": [], "train_cls_loss": [], "train_seg_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_dice": []}
    best_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n{'='*70}\n[STEP 2/3] STARTING MULTI-TASK TRAINING (EPOCHS: {epochs})\n{'='*70}")
    for epoch in range(epochs):
        model.train()
        t_loss, t_cls, t_seg, correct, total = 0.0, 0.0, 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d}", leave=False)

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(images)
                cls_loss, _, _ = evidential_loss_fn(out["alpha"], labels, epoch_ratio=(epoch + 1) / epochs)
                mask_cnt = has_mask.sum()
                seg_loss = (dice_bce_loss_fn(out["mask_logits"], masks) * has_mask.view(-1, 1, 1, 1)).sum() / (mask_cnt + 1e-8) if mask_cnt > 0 else torch.tensor(0.0, device=device)
                total_loss = cls_loss + seg_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            total += bs
            correct += (torch.argmax(out["probabilities"], dim=1) == labels).sum().item()
            t_loss += total_loss.item() * bs
            t_cls += cls_loss.item() * bs
            t_seg += seg_loss.item() * bs

        scheduler.step()

        # Validation
        model.eval()
        v_loss, v_labels, v_preds, v_pred_masks, v_true_masks = 0.0, [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)
                out = model(images)
                cls_loss, _, _ = evidential_loss_fn(out["alpha"], labels)
                v_loss += cls_loss.item() * images.size(0)
                v_labels.extend(labels.cpu().numpy())
                v_preds.extend(torch.argmax(out["probabilities"], dim=1).cpu().numpy())
                v_pred_masks.append(out["pred_mask"].cpu().numpy())
                v_true_masks.append(masks.cpu().numpy())

        v_acc = accuracy_score(v_labels, v_preds)
        v_f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)
        v_dice, _ = compute_segmentation_dice(np.concatenate(v_pred_masks), np.concatenate(v_true_masks))

        history["train_loss"].append(t_loss / total)
        history["train_cls_loss"].append(t_cls / total)
        history["train_seg_loss"].append(t_seg / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(v_loss / len(val_ds))
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["val_dice"].append(v_dice)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {t_loss/total:.4f} (Acc: {correct/total*100:.1f}%) | Val Acc: {v_acc*100:.1f}% | Val Macro F1: {v_f1*100:.1f}% | Aux Dice: {v_dice*100:.1f}%")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

    print(f"\n{'='*70}\n[STEP 3/3] EXHAUSTIVE TEST SET EVALUATION & PUBLICATION REPORT\n{'='*70}")
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    def run_eval(loader):
        all_l, all_p, all_prob, all_u, p_masks, t_masks, imgs = [], [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)
                out = model(images)
                all_l.extend(labels.cpu().numpy())
                all_p.extend(torch.argmax(out["probabilities"], dim=1).cpu().numpy())
                all_prob.extend(out["probabilities"].cpu().numpy())
                all_u.extend(out["uncertainty"].cpu().numpy())
                p_masks.append(out["pred_mask"].cpu().numpy())
                t_masks.append(masks.cpu().numpy())
                imgs.append(images.cpu().numpy())
        return np.array(all_l), np.array(all_p), np.array(all_prob), np.array(all_u), np.concatenate(p_masks), np.concatenate(t_masks), np.concatenate(imgs)

    t_true, t_pred, t_probs, t_unc, t_pred_m, t_true_m, t_imgs = run_eval(test_loader)
    sm_true, sm_pred, sm_probs, sm_unc, _, _, _ = run_eval(small_loader)

    test_metrics = compute_all_metrics(t_true, t_pred, t_probs, t_unc)
    small_metrics = compute_all_metrics(sm_true, sm_pred, sm_probs, sm_unc)
    dice_score, iou_score = compute_segmentation_dice(t_pred_m, t_true_m)

    sample_img = t_imgs[0, 0]
    sample_true_m = t_true_m[0, 0]
    sample_pred_m = (t_pred_m[0, 0] > 0.5).astype(float)
    sample_rollout = compute_attention_rollout(model, torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0), device=device)

    generate_all_7_figures(test_metrics, t_true, t_probs, t_unc, small_metrics, sample_img, sample_true_m, sample_pred_m, sample_rollout, history)

    print(f"\n=======================================================")
    print(f"      JOURNAL PUBLICATION PERFORMANCE METRICS          ")
    print(f"=======================================================")
    print(f"Overall Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:         {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:            {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:         {test_metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Specificity:         {test_metrics['macro_specificity']*100:.2f}%")
    print(f"Macro Precision:           {test_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall (Sens):       {test_metrics['macro_recall']*100:.2f}%")
    print(f"AUC-ROC (One-vs-Rest):     {test_metrics['auc_roc']:.4f}")
    print(f"Cohen's Kappa (κ):         {test_metrics['cohen_kappa']:.4f}")
    print(f"Matthews Corr (MCC):       {test_metrics['matthews_corrcoef']:.4f}")
    print(f"Calibration Error (ECE):   {test_metrics['expected_calibration_error']:.4f}")
    print(f"Auxiliary Segmentation Dice:{dice_score*100:.2f}% (IoU: {iou_score*100:.2f}%)")
    print(f"-------------------------------------------------------")
    print(f"Small-Tumour Sensitivity:  {small_metrics['macro_recall']*100:.2f}%")
    print(f"Small-Tumour Macro F1:     {small_metrics['macro_f1']*100:.2f}%")
    print(f"=======================================================\n")

    return model


# ==============================================================================
# STEP 6: REAL-TIME PHOTO UPLOAD & INFERENCE SYSTEMS
# ==============================================================================

class InteractiveDiagnosticEngine:
    """Preprocesses user-uploaded MRI scans and generates real-time diagnostics."""
    def __init__(self, model_path="checkpoints/best_model.pth"):
        self.model = XRMiTNet().to(device)
        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def analyze_image(self, pil_image: Image.Image, discard_ratio: float = 0.85):
        img_gray = pil_image.convert("L").resize((224, 224), Image.BILINEAR)
        img_np = np.array(img_gray, dtype=np.float32) / 255.0
        norm = (img_np - 0.28) / 0.22
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = self.model(tensor, return_attention=True)
            probs = out["probabilities"][0].cpu().numpy()
            unc = out["uncertainty"][0].item()
            pred_mask = (out["pred_mask"][0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        rollout = compute_attention_rollout(self.model, tensor, discard_ratio=discard_ratio, device=device)

        orig_rgb = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        mask_overlay = orig_rgb.copy()
        if np.sum(pred_mask) > 0:
            color_mask = np.zeros_like(orig_rgb)
            color_mask[pred_mask == 1] = [220, 53, 69]
            mask_overlay = cv2.addWeighted(mask_overlay, 0.7, color_mask, 0.3, 0)

        heatmap = cv2.applyColorMap((rollout * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        xai_overlay = cv2.addWeighted(orig_rgb, 0.5, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.5, 0)

        top_idx = int(np.argmax(probs))
        confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        return confidences, CLASS_NAMES[top_idx], probs[top_idx], unc, orig_rgb, mask_overlay, xai_overlay


def launch_in_notebook_uploader():
    """Method A: Native in-notebook widget for uploading and testing MRI scans."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    engine = InteractiveDiagnosticEngine()
    upload_btn = widgets.FileUpload(accept='.jpg,.png,.jpeg', multiple=False, description='📁 Upload MRI', button_style='primary')
    out_widget = widgets.Output()

    def on_file_uploaded(change):
        with out_widget:
            clear_output()
            if not upload_btn.value:
                return
            file_info = next(iter(upload_btn.value.values()))
            pil_img = Image.open(io.BytesIO(file_info['content']))

            confidences, top_class, top_conf, unc, orig, mask_view, xai_view = engine.analyze_image(pil_img)

            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
            axs[0].imshow(orig)
            axs[0].set(title="Input MRI Scan", xticks=[], yticks=[])
            axs[1].imshow(mask_view)
            axs[1].set(title=f"Predicted Lesion Mask", xticks=[], yticks=[])
            axs[2].imshow(xai_view)
            axs[2].set(title=f"CARE-XAI Heatmap Overlay", xticks=[], yticks=[])
            plt.show()

            print(f"\n------------------------------------------------------------")
            print(f"  DIAGNOSIS:             {top_class.upper()} ({top_conf*100:.2f}%)")
            print(f"  Epistemic Uncertainty: {unc:.4f} (Reliability: {max(0.0, 1.0 - unc):.4f})")
            print(f"------------------------------------------------------------")
            for c_name, p in confidences.items():
                print(f"  {c_name:<12}: {'█'*int(p*25):<25} {p*100:.1f}%")
            print(f"------------------------------------------------------------\n")

    upload_btn.observe(on_file_uploaded, names='value')
    print("Click 'Upload MRI' below to test any brain scan image:")
    display(upload_btn)
    display(out_widget)


def launch_gradio_web_app():
    """Method B: Gradio Web App with Public Shareable Link for remote testing."""
    import gradio as gr
    engine = InteractiveDiagnosticEngine()

    def predict_gradio(img, discard_ratio):
        if img is None:
            return None, "", None, None, None
        confs, top_class, top_conf, unc, orig, mask_view, xai_view = engine.analyze_image(img, discard_ratio)
        rel = max(0.0, 1.0 - unc)
        report = f"### Diagnosis: **{top_class}** ({top_conf*100:.2f}%)\n- **Reliability:** {rel:.3f}\n- **Epistemic Uncertainty:** {unc:.3f}"
        return confs, report, orig, mask_view, xai_view

    with gr.Blocks(title="X-R-MiTNet Brain Tumor Diagnosis") as demo:
        gr.Markdown("# 🧠 X-R-MiTNet Real-Time Brain Tumor Diagnostic Platform")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Brain MRI Scan")
                discard_s = gr.Slider(0.5, 0.95, value=0.85, label="XAI Noise Discard Ratio")
                btn = gr.Button("🔬 Run Real-Time Analysis", variant="primary")
            with gr.Column():
                conf_out = gr.Label(num_top_classes=4, label="Class Probabilities")
                md_out = gr.Markdown()
        with gr.Row():
            v1 = gr.Image(label="1. Standardized MRI")
            v2 = gr.Image(label="2. Auxiliary Lesion Mask")
            v3 = gr.Image(label="3. CARE-XAI Attention Overlay")

        btn.click(predict_gradio, inputs=[input_img, discard_s], outputs=[conf_out, md_out, v1, v2, v3])

    demo.launch(inline=True, share=True)


# ==============================================================================
# EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    trained_model = train_and_evaluate_pipeline(epochs=20, batch_size=32, lr=3e-4)

    print("\n[Launching In-Notebook Interactive Uploader Widget]")
    launch_in_notebook_uploader()

    print("\n[Launching Gradio Public Web Application]")
    launch_gradio_web_app()

