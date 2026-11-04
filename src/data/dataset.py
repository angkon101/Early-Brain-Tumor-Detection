"""Unified PyTorch Dataset implementations for Brain Tumor Detection and Segmentation.

Supports all 4 research cohorts:
1. "multitask" / "primary"    : BRISC 2025 Multi-Task (Detection + Auxiliary Segmentation + Small-Tumour)
2. "benchmark" / "literature" : Standard Cleaned 7,153 (Literature Comparison)
3. "robustness"               : Augmented 7,200 (Perturbation & Degradation Robustness)
4. "clinical"                 : Epic & CSCR Hospital 12,064 (Cross-Center Clinical Validation)
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
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
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """Args:

        split_csv: Direct path to a split CSV or loaded DataFrame. If None, resolves from cohort & split.
        cohort: One of ['multitask', 'benchmark', 'robustness', 'clinical'].
        split: One of ['train', 'val', 'test', 'small_tumor_test'].
        image_size: Target (height, width) for image and mask resizing.
        transform: Optional augmentation or transform pipeline.
        is_training: Whether dataset is used for training.
        """
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training

        # Resolve CSV path
        if split_csv is not None:
            if isinstance(split_csv, (str, Path)):
                self.df = pd.read_csv(split_csv)
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

            split_path = COHORT_SPLIT_MAP.get(cohort_key, {}).get(split)
            if split_path is None or not os.path.exists(split_path):
                raise ValueError(f"Invalid cohort='{cohort}' and split='{split}'. Path: {split_path}")
            self.df = pd.read_csv(split_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. Load MRI Image
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("L")  # Single-channel grayscale MRI
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_np = np.array(image, dtype=np.float32) / 255.0

        # Standardize mean and std for brain MRI
        mean = 0.28
        std = 0.22
        img_norm = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: (1, H, W)

        # 2. Load Segmentation Mask (if available)
        has_mask = bool(row.get("has_mask", False))
        mask_path = str(row.get("mask_path", "None"))

        if has_mask and mask_path != "None" and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            mask_np = np.array(mask, dtype=np.float32)
            mask_np = (mask_np > 127).astype(np.float32)
        else:
            # Control cases or cohorts without segmentation masks
            mask_np = np.zeros(self.image_size, dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Shape: (1, H, W)

        # 3. Label & Target Attributes
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
            "has_mask": torch.tensor(1.0 if has_mask else 0.0, dtype=torch.float32),
            "cohort": str(row.get("cohort", "unknown")),
            "case_id": str(row.get("case_id", row.get("filename", f"idx_{idx}"))),
        }
