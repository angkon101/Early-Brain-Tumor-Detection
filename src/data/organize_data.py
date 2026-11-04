"""Comprehensive Data Organization and Stratified Split Generation for CARE-XAI Research.

Processes all 4 research cohorts in data/:
1. 01_primary_multitask: BRISC 2025 Multi-task (Detection + Segmentation + Small Tumor)
2. 02_literature_benchmark: Standard Cleaned 7,153 (Literature Baseline Comparison)
3. 03_perturbation_robustness: Augmented 7,200 (Degradation & Explanation Stability)
4. 04_clinical_hospital: Epic & CSCR Hospital 12,064 (Cross-Center Clinical Generalization)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

WORKSPACE_DIR = Path(r"E:\Brain tumor")
DATA_DIR = WORKSPACE_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("COMPREHENSIVE DATASET ORGANIZATION & SPLIT GENERATION PIPELINE")
print("=" * 75)

# =========================================================================
# COHORT 1: 01_primary_multitask (BRISC 2025)
# =========================================================================
print("\n[Cohort 1] Processing Primary Multi-Task Dataset (BRISC 2025)...")
c1_dir = DATA_DIR / "01_primary_multitask"
seg_train_img_dir = c1_dir / "segmentation" / "Train" / "images"
seg_train_mask_dir = c1_dir / "segmentation" / "Train" / "masks"
seg_test_img_dir = c1_dir / "segmentation" / "Test" / "images"
seg_test_mask_dir = c1_dir / "segmentation" / "Test" / "masks"
cls_train_dir = c1_dir / "classification" / "Train"
cls_test_dir = c1_dir / "classification" / "Test"

# Map segmentation pairs
seg_pairs = []
for split_name, img_dir, mask_dir in [
    ("train", seg_train_img_dir, seg_train_mask_dir),
    ("test", seg_test_img_dir, seg_test_mask_dir),
]:
    for img_p in img_dir.glob("*.jpg"):
        stem = img_p.stem
        mask_p = mask_dir / f"{stem}.png"
        if not mask_p.exists():
            continue
        
        # Parse filename tokens: brisc2025_<split>_<index>_<tumor>_<plane>_<seq>.jpg
        parts = stem.split("_")
        tumor_code = parts[3] if len(parts) > 3 else "unknown"
        plane_code = parts[4] if len(parts) > 4 else "unknown"
        code_to_tumor = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor"}
        code_to_plane = {"ax": "axial", "co": "coronal", "sa": "sagittal"}

        seg_pairs.append({
            "case_id": stem,
            "filename": img_p.name,
            "image_path": str(img_p.resolve()),
            "mask_path": str(mask_p.resolve()),
            "has_mask": True,
            "tumor_label": code_to_tumor.get(tumor_code, tumor_code),
            "tumor_code": tumor_code,
            "plane_label": code_to_plane.get(plane_code, plane_code),
            "plane_code": plane_code,
            "sequence": "T1",
            "original_split": split_name,
            "cohort": "primary_multitask"
        })

print(f"  Matched {len(seg_pairs)} segmentation image-mask pairs.")

# Compute tumor pixel areas and bounding boxes
print("  Calculating mask areas and spatial coordinates...")
for rec in seg_pairs:
    with Image.open(rec["mask_path"]) as m_img:
        m_arr = np.array(m_img)
        t_pixels = int(np.count_nonzero(m_arr > 0))
        rec["tumor_pixel_area"] = t_pixels
        rec["relative_tumor_area"] = float(t_pixels / (512 * 512))
        if t_pixels > 0:
            y_idx, x_idx = np.where(m_arr > 0)
            rec["bbox_x1"] = int(x_idx.min())
            rec["bbox_y1"] = int(y_idx.min())
            rec["bbox_x2"] = int(x_idx.max())
            rec["bbox_y2"] = int(y_idx.max())
            rec["bbox_width"] = int(x_idx.max() - x_idx.min() + 1)
            rec["bbox_height"] = int(y_idx.max() - y_idx.min() + 1)
        else:
            rec["bbox_x1"] = rec["bbox_y1"] = rec["bbox_x2"] = rec["bbox_y2"] = 0
            rec["bbox_width"] = rec["bbox_height"] = 0

seg_df = pd.DataFrame(seg_pairs)

# Determine small tumor threshold (25th percentile of tumor-positive area)
pos_areas = seg_df[seg_df["tumor_pixel_area"] > 0]["tumor_pixel_area"]
q25 = float(pos_areas.quantile(0.25))
print(f"  Small-Tumour Threshold (Q25): {q25:.1f} pixels (~{q25/(512*512)*100:.2f}% of image area)")

seg_df["is_small_tumor"] = seg_df["tumor_pixel_area"] <= q25
seg_df["size_category"] = np.where(seg_df["is_small_tumor"], "small", "medium_large")

# Include healthy control slices (no_tumor) from classification
ctrl_records = []
for split_name, cls_dir in [("train", cls_train_dir / "no_tumor"), ("test", cls_test_dir / "no_tumor")]:
    if not cls_dir.exists():
        continue
    for img_p in cls_dir.glob("*.jpg"):
        stem = img_p.stem
        parts = stem.split("_")
        plane_code = parts[4] if len(parts) > 4 else "unknown"
        code_to_plane = {"ax": "axial", "co": "coronal", "sa": "sagittal"}

        ctrl_records.append({
            "case_id": stem,
            "filename": img_p.name,
            "image_path": str(img_p.resolve()),
            "mask_path": "None",
            "has_mask": False,
            "tumor_label": "no_tumor",
            "tumor_code": "no",
            "plane_label": code_to_plane.get(plane_code, plane_code),
            "plane_code": plane_code,
            "sequence": "T1",
            "original_split": split_name,
            "cohort": "primary_multitask",
            "tumor_pixel_area": 0,
            "relative_tumor_area": 0.0,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 0,
            "bbox_y2": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "is_small_tumor": False,
            "size_category": "no_tumor"
        })

ctrl_df = pd.DataFrame(ctrl_records)
multitask_df = pd.concat([seg_df, ctrl_df], ignore_index=True)
print(f"  Total Primary Multi-Task cohort size: {len(multitask_df)} cases (Tumors: {len(seg_df)}, Controls: {len(ctrl_df)})")

# Save complete cohort 1 manifest
multitask_df.to_csv(METADATA_DIR / "multitask_master_manifest.csv", index=False)

# Create stratified splits (70% train, 15% val, 15% test)
multitask_df["strat_key"] = multitask_df["tumor_label"] + "_" + multitask_df["plane_label"] + "_" + multitask_df["size_category"]
train_c1, temp_c1 = train_test_split(multitask_df, test_size=0.30, random_state=42, stratify=multitask_df["strat_key"])
val_c1, test_c1 = train_test_split(temp_c1, test_size=0.50, random_state=42, stratify=temp_c1["strat_key"])

small_test_c1 = test_c1[(test_c1["is_small_tumor"] == True) | (test_c1["tumor_label"] == "no_tumor")].copy()

train_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "multitask_train.csv", index=False)
val_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "multitask_val.csv", index=False)
test_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "multitask_test.csv", index=False)
small_test_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "small_tumor_test.csv", index=False)

# Also maintain standard train.csv, val.csv, test.csv for default loaders
train_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "train.csv", index=False)
val_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "val.csv", index=False)
test_c1.drop(columns=["strat_key"]).to_csv(SPLITS_DIR / "test.csv", index=False)

print(f"  Saved Multi-Task Splits: Train={len(train_c1)}, Val={len(val_c1)}, Test={len(test_c1)}, SmallTumorTest={len(small_test_c1)}")

# =========================================================================
# COHORT 2: 02_literature_benchmark (Clean 7,153)
# =========================================================================
print("\n[Cohort 2] Processing Standard Literature Benchmark Cohort (Phase 2 - 7,153)...")
c2_dir = DATA_DIR / "02_literature_benchmark"
c2_records = []

for split_folder in ["Train", "Test"]:
    s_dir = c2_dir / split_folder
    if not s_dir.exists():
        continue
    for class_folder in s_dir.iterdir():
        if not class_folder.is_dir():
            continue
        c_label = class_folder.name.lower().replace("notumor", "no_tumor")
        for img_p in class_folder.glob("*.jpg"):
            c2_records.append({
                "case_id": img_p.stem,
                "filename": img_p.name,
                "image_path": str(img_p.resolve()),
                "mask_path": "None",
                "has_mask": False,
                "tumor_label": c_label,
                "split": split_folder.lower(),
                "cohort": "literature_benchmark"
            })

c2_df = pd.DataFrame(c2_records)
c2_df.to_csv(METADATA_DIR / "literature_benchmark_manifest.csv", index=False)
c2_df[c2_df["split"] == "train"].to_csv(SPLITS_DIR / "literature_benchmark_train.csv", index=False)
c2_df[c2_df["split"] == "test"].to_csv(SPLITS_DIR / "literature_benchmark_test.csv", index=False)
print(f"  Indexed {len(c2_df)} images (Train: {len(c2_df[c2_df['split'] == 'train'])}, Test: {len(c2_df[c2_df['split'] == 'test'])})")

# =========================================================================
# COHORT 3: 03_perturbation_robustness (Augmented 7,200)
# =========================================================================
print("\n[Cohort 3] Processing Perturbation & Robustness Cohort (Phase 1 - 7,200)...")
c3_dir = DATA_DIR / "03_perturbation_robustness"
c3_records = []

for split_folder in ["Train", "Test"]:
    s_dir = c3_dir / split_folder
    if not s_dir.exists():
        continue
    for class_folder in s_dir.iterdir():
        if not class_folder.is_dir():
            continue
        c_label = class_folder.name.lower().replace("notumor", "no_tumor")
        for img_p in class_folder.glob("*.jpg"):
            is_augmented = "aug" in img_p.name.lower()
            c3_records.append({
                "case_id": img_p.stem,
                "filename": img_p.name,
                "image_path": str(img_p.resolve()),
                "mask_path": "None",
                "has_mask": False,
                "tumor_label": c_label,
                "is_augmented": is_augmented,
                "split": split_folder.lower(),
                "cohort": "perturbation_robustness"
            })

c3_df = pd.DataFrame(c3_records)
c3_df.to_csv(METADATA_DIR / "robustness_manifest.csv", index=False)
c3_df[c3_df["split"] == "train"].to_csv(SPLITS_DIR / "robustness_train.csv", index=False)
c3_df[c3_df["split"] == "test"].to_csv(SPLITS_DIR / "robustness_test.csv", index=False)
print(f"  Indexed {len(c3_df)} images (Train: {len(c3_df[c3_df['split'] == 'train'])}, Test: {len(c3_df[c3_df['split'] == 'test'])})")

# =========================================================================
# COHORT 4: 04_clinical_hospital (Epic & CSCR Hospital - 12,064)
# =========================================================================
print("\n[Cohort 4] Processing External Clinical Hospital Cohort (Phase 3 - 12,064)...")
c4_dir = DATA_DIR / "04_clinical_hospital"
c4_records = []

for split_folder in ["Train", "Test"]:
    s_dir = c4_dir / split_folder
    if not s_dir.exists():
        continue
    for class_folder in s_dir.iterdir():
        if not class_folder.is_dir():
            continue
        c_label = class_folder.name.lower().replace("notumor", "no_tumor")
        for img_p in class_folder.glob("*.*"):
            if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                c4_records.append({
                    "case_id": img_p.stem,
                    "filename": img_p.name,
                    "image_path": str(img_p.resolve()),
                    "mask_path": "None",
                    "has_mask": False,
                    "tumor_label": c_label,
                    "split": split_folder.lower(),
                    "source_hospital": "Epic_and_CSCR_Hospital",
                    "cohort": "clinical_hospital"
                })

c4_df = pd.DataFrame(c4_records)
c4_df.to_csv(METADATA_DIR / "clinical_hospital_manifest.csv", index=False)
c4_df[c4_df["split"] == "train"].to_csv(SPLITS_DIR / "clinical_hospital_train.csv", index=False)
c4_df[c4_df["split"] == "test"].to_csv(SPLITS_DIR / "clinical_hospital_test.csv", index=False)
# Maintain external_hospital_test.csv for backward compatibility
c4_df[c4_df["split"] == "test"].to_csv(SPLITS_DIR / "external_hospital_test.csv", index=False)
print(f"  Indexed {len(c4_df)} scans (Train: {len(c4_df[c4_df['split'] == 'train'])}, Test: {len(c4_df[c4_df['split'] == 'test'])})")

# =========================================================================
# Master Summary Table
# =========================================================================
print("\n[Step 5] Compiling Master Dataset Summary Table...")
summary_data = [
    {
        "Cohort ID": "01_primary_multitask",
        "Cohort Name": "BRISC 2025 Multi-Task Development",
        "Total Samples": len(multitask_df),
        "Train Count": len(train_c1),
        "Val Count": len(val_c1),
        "Test Count": len(test_c1),
        "Segmentation Masks": f"{len(seg_df)} radiologist masks",
        "Modality": "T1-MRI (Axial, Coronal, Sagittal)",
        "Classes": "Glioma, Meningioma, Pituitary, No Tumor",
        "Research Role": "Primary training (RO1-RO4), small-tumor detection (H1), auxiliary segmentation (H4)"
    },
    {
        "Cohort ID": "02_literature_benchmark",
        "Cohort Name": "Standard Literature Benchmark (Phase 2)",
        "Total Samples": len(c2_df),
        "Train Count": len(c2_df[c2_df["split"] == "train"]),
        "Val Count": "-",
        "Test Count": len(c2_df[c2_df["split"] == "test"]),
        "Segmentation Masks": "None",
        "Modality": "MRI 2D Slices (512x512)",
        "Classes": "Glioma, Meningioma, Pituitary, No Tumor",
        "Research Role": "Direct 1:1 baseline comparison with published SOTA literature (RO6 / H1 baseline)"
    },
    {
        "Cohort ID": "03_perturbation_robustness",
        "Cohort Name": "Perturbation & Degradation Robustness (Phase 1)",
        "Total Samples": len(c3_df),
        "Train Count": len(c3_df[c3_df["split"] == "train"]),
        "Val Count": "-",
        "Test Count": len(c3_df[c3_df["split"] == "test"]),
        "Segmentation Masks": "None",
        "Modality": "MRI 2D Slices (Augmented/Distorted)",
        "Classes": "Glioma, Meningioma, Pituitary, No Tumor",
        "Research Role": "Stress-testing model stability & CARE-XAI attribution faithfulness (RO7, RO8, RQ3)"
    },
    {
        "Cohort ID": "04_clinical_hospital",
        "Cohort Name": "Epic & CSCR Hospital Clinical Validation (Phase 3)",
        "Total Samples": len(c4_df),
        "Train Count": len(c4_df[c4_df["split"] == "train"]),
        "Val Count": "-",
        "Test Count": len(c4_df[c4_df["split"] == "test"]),
        "Segmentation Masks": "None",
        "Modality": "Authentic Clinical MRI Scans",
        "Classes": "Glioma, Meningioma, Pituitary, No Tumor",
        "Research Role": "Cross-center clinical generalization and hospital domain shift evaluation"
    }
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(METADATA_DIR / "master_dataset_summary.csv", index=False)
print(f"Master summary saved to: {METADATA_DIR / 'master_dataset_summary.csv'}")

print("\n" + "=" * 75)
print(f"ALL 4 DATASET COHORTS ORGANIZED SUCCESSFULLY!")
print(f"Total Medical Scans Across All Cohorts: {len(multitask_df) + len(c2_df) + len(c3_df) + len(c4_df):,} images")
print("=" * 75)
