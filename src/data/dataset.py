"""Unified PyTorch Dataset implementations for Brain Tumor Detection and Segmentation.

Supports all 4 research cohorts:
1. "multitask" / "primary"    : BRISC 2025 Multi-Task (Detection + Auxiliary Segmentation + Small-Tumour)
2. "benchmark" / "literature" : Standard Cleaned 7,153 (Literature Comparison)
3. "robustness"               : Augmented 7,200 (Perturbation & Degradation Robustness)
4. "clinical"                 : Epic & CSCR Hospital 12,064 (Cross-Center Clinical Validation)

Fully cross-platform compatible (Windows, Linux, Google Colab, Kaggle).
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

LABEL_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3,
    "notumor": 3,
}

IDX_TO_LABEL = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
    3: "no_tumor",
}

COHORT_SPLIT_MAP = {
    "multitask": {
        "train": "data/splits/multitask_train.csv",
        "val": "data/splits/multitask_val.csv",
        "test": "data/splits/multitask_test.csv",
        "small_tumor_test": "data/splits/small_tumor_test.csv",
    },
    "benchmark": {
        "train": "data/splits/literature_benchmark_train.csv",
        "test": "data/splits/literature_benchmark_test.csv",
    },
    "robustness": {
        "train": "data/splits/robustness_train.csv",
        "test": "data/splits/robustness_test.csv",
    },
    "clinical": {
        "train": "data/splits/clinical_hospital_train.csv",
        "test": "data/splits/clinical_hospital_test.csv",
    },
}


def resolve_file_path(path_str: Optional[Union[str, Path]], data_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolves an image or mask path across Windows, Linux, and Google Colab environments.

    Handles absolute Windows paths recorded in CSVs (e.g. 'E:\\Brain tumor\\data\\...')
    when running on Linux/Colab, as well as relative paths.
    """
    if path_str is None or pd.isna(path_str):
        return None
    s = str(path_str).strip()
    if s.lower() in ["none", "nan", ""]:
        return None

    # Check direct path first
    direct_p = Path(s)
    if direct_p.is_file():
        return direct_p

    # Normalize separators
    norm_s = s.replace("\\", "/")

    # Extract relative candidate starting from 'data/' or '01_primary_multitask'
    candidates = []
    if "data/" in norm_s:
        idx = norm_s.index("data/")
        candidates.append(norm_s[idx:])             # data/...
        candidates.append(norm_s[idx + len("data/"):]) # 01_primary_multitask/...
    elif "01_primary_multitask" in norm_s:
        idx = norm_s.index("01_primary_multitask")
        candidates.append("data/" + norm_s[idx:])
        candidates.append(norm_s[idx:])

    # Search base directories
    search_bases = []
    if data_root is not None:
        p_root = Path(data_root)
        search_bases.extend([p_root, p_root / "data"])
    search_bases.append(Path.cwd())
    search_bases.append(Path.cwd() / "data")
    # Repo root relative to this file
    repo_root = Path(__file__).resolve().parent.parent.parent
    search_bases.extend([repo_root, repo_root / "data"])

    # Try candidate relative paths against bases
    for base in search_bases:
        for cand in candidates:
            target = base / cand
            if target.is_file():
                return target

    # Fallback: search by filename in standard folders
    fname = direct_p.name
    for base in search_bases:
        for sub in [
            "01_primary_multitask/segmentation/Train/images",
            "01_primary_multitask/segmentation/Test/images",
            "01_primary_multitask/segmentation/Train/masks",
            "01_primary_multitask/segmentation/Test/masks",
            "01_primary_multitask/classification/Train/glioma",
            "01_primary_multitask/classification/Train/meningioma",
            "01_primary_multitask/classification/Train/pituitary",
            "01_primary_multitask/classification/Train/no_tumor",
            "01_primary_multitask/classification/Test/glioma",
            "01_primary_multitask/classification/Test/meningioma",
            "01_primary_multitask/classification/Test/pituitary",
            "01_primary_multitask/classification/Test/no_tumor",
        ]:
            target = base / sub / fname
            if target.is_file():
                return target

    return None


class BrainTumorDataset(Dataset):
    """Unified Multi-Cohort Brain Tumor Dataset.

    Loads 2D MRI scans with corresponding ground-truth segmentation masks (if available),
    classification labels, and quantitative tumor metrics.
    """

    def __init__(
        self,
        split_csv: Optional[Union[str, Path, pd.DataFrame]] = None,
        cohort: str = "multitask",
        split: str = "train",
        data_root: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        is_training: bool = False,
        augment: bool = False,
    ):
        """Args:
            split_csv: Direct path to a split CSV or loaded DataFrame. If None, resolves from cohort & split.
            cohort: One of ['multitask', 'benchmark', 'robustness', 'clinical'].
            split: One of ['train', 'val', 'test', 'small_tumor_test'].
            data_root: Base project or data directory (useful for Google Colab).
            image_size: Target (height, width) for image and mask resizing.
            transform: Optional custom transform function.
            is_training: Whether dataset is used for training.
            augment: Whether to apply safe geometric/intensity augmentations for training.
        """
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training
        self.augment = augment and is_training
        self.data_root = Path(data_root) if data_root is not None else None

        # Resolve CSV path
        if split_csv is not None:
            if isinstance(split_csv, (str, Path)):
                csv_resolved = resolve_file_path(split_csv, data_root=self.data_root) or Path(split_csv)
                self.df = pd.read_csv(csv_resolved)
            else:
                self.df = split_csv.copy()
        else:
            cohort_key = cohort.lower().strip()
            if cohort_key in ["primary", "multitask_primary"]:
                cohort_key = "multitask"
            elif cohort_key in ["literature", "phase2"]:
                cohort_key = "benchmark"
            elif cohort_key in ["perturbation", "phase1"]:
                cohort_key = "robustness"
            elif cohort_key in ["hospital", "phase3"]:
                cohort_key = "clinical"

            rel_split_path = COHORT_SPLIT_MAP.get(cohort_key, {}).get(split)
            if rel_split_path is None:
                raise ValueError(f"Unknown split '{split}' for cohort '{cohort}'.")

            resolved_csv = resolve_file_path(rel_split_path, data_root=self.data_root)
            if resolved_csv is None or not resolved_csv.is_file():
                # Try finding under cwd or data_root
                bases = [Path.cwd()]
                if self.data_root:
                    bases.insert(0, self.data_root)
                found = False
                for b in bases:
                    candidate = b / rel_split_path
                    if candidate.is_file():
                        resolved_csv = candidate
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(
                        f"Could not find split CSV '{rel_split_path}'. Please provide data_root."
                    )
            self.df = pd.read_csv(resolved_csv)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Applies synchronized spatial and photometric augmentations to image and mask."""
        # Random horizontal flip (brain bilateral symmetry)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random gentle rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random slight brightness/contrast jitter on image only
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. Resolve and Load MRI Image
        raw_img_path = row["image_path"]
        resolved_img = resolve_file_path(raw_img_path, data_root=self.data_root)
        if resolved_img is None or not resolved_img.is_file():
            raise FileNotFoundError(
                f"Image not found at '{raw_img_path}'. Checked relative to '{self.data_root}'."
            )

        image = Image.open(resolved_img).convert("L")  # Single-channel grayscale MRI
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # 2. Resolve and Load Segmentation Mask (if available)
        has_mask = bool(row.get("has_mask", False))
        raw_mask_path = row.get("mask_path", None)
        resolved_mask = resolve_file_path(raw_mask_path, data_root=self.data_root) if has_mask else None

        if has_mask and resolved_mask is not None and resolved_mask.is_file():
            mask = Image.open(resolved_mask).convert("L")
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        else:
            # Control cases or cohorts without segmentation masks
            mask = Image.new("L", (self.image_size[1], self.image_size[0]), 0)

        # 3. Apply Training Augmentations (if enabled)
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        # 4. Convert Image to Normalized Tensor
        img_np = np.array(image, dtype=np.float32) / 255.0
        # Standardize mean and std for brain MRI
        mean = 0.28
        std = 0.22
        img_norm = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: (1, H, W)

        # 5. Convert Mask to Binary Float Tensor
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Shape: (1, H, W)

        # 6. Label & Target Attributes
        raw_label = str(row["tumor_label"]).lower().strip().replace("notumor", "no_tumor")
        label_idx = LABEL_TO_IDX.get(raw_label, 3)

        is_small = bool(row.get("is_small_tumor", False))
        tumor_area = float(row.get("tumor_pixel_area", 0.0))

        return {
            "image": img_tensor,                                  # (1, H, W)
            "mask": mask_tensor,                                  # (1, H, W)
            "label": torch.tensor(label_idx, dtype=torch.long),   # 0..3
            "is_tumor": torch.tensor(1.0 if label_idx != 3 else 0.0, dtype=torch.float32),
            "is_small_tumor": torch.tensor(1.0 if is_small else 0.0, dtype=torch.float32),
            "tumor_area": torch.tensor(tumor_area, dtype=torch.float32),
            "has_mask": torch.tensor(1.0 if (has_mask and resolved_mask is not None) else 0.0, dtype=torch.float32),
            "cohort": str(row.get("cohort", "unknown")),
            "case_id": str(row.get("case_id", row.get("filename", f"idx_{idx}"))),
        }
