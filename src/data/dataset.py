"""PyTorch Dataset implementations for Brain Tumor Detection and Segmentation."""

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


class BrainTumorDataset(Dataset):
    """Multi-task Brain Tumor Dataset for classification and auxiliary segmentation.

    Loads 2D MRI scans with corresponding ground-truth segmentation masks,
    classification labels, and quantitative tumor size metrics.
    """

    def __init__(
        self,
        split_csv: Union[str, Path, pd.DataFrame],
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """Args:

        split_csv: Path to split CSV (e.g., train.csv) or loaded DataFrame.
        image_size: Target (height, width) for image and mask resizing.
        transform: Optional Albumentations or PyTorch transform pipeline.
        is_training: Whether dataset is used for training.
        """
        if isinstance(split_csv, (str, Path)):
            self.df = pd.read_csv(split_csv)
        else:
            self.df = split_csv.copy()

        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training

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
        img_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Standardize mean and std for grayscale MRI
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
            mask_np = (mask_np > 127).astype(np.float32)  # Binary mask [0.0, 1.0]
        else:
            # Healthy controls or external hospital cohort without masks
            mask_np = np.zeros(self.image_size, dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Shape: (1, H, W)

        # 3. Label & Target Attributes
        raw_label = str(row["tumor_label"]).lower().strip()
        label_idx = LABEL_TO_IDX.get(raw_label, 3)

        is_small = bool(row.get("is_small_tumor", False))
        tumor_area = float(row.get("tumor_pixel_area", 0.0))

        return {
            "image": img_tensor,                                  # (1, H, W)
            "mask": mask_tensor,                                  # (1, H, W)
            "label": torch.tensor(label_idx, dtype=torch.long),   # Scaled class idx
            "is_tumor": torch.tensor(1 if label_idx != 3 else 0, dtype=torch.float32),
            "is_small_tumor": torch.tensor(1.0 if is_small else 0.0, dtype=torch.float32),
            "tumor_area": torch.tensor(tumor_area, dtype=torch.float32),
            "has_mask": torch.tensor(1.0 if has_mask else 0.0, dtype=torch.float32),
            "case_id": str(row.get("case_id", row.get("filename", f"idx_{idx}"))),
        }
