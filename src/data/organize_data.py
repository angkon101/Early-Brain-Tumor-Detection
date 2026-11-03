import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

WORKSPACE_DIR = Path(r"E:\Brain tumor")
DATASETS_DIR = WORKSPACE_DIR / "Datasets"
DATA_DIR = WORKSPACE_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
EXTERNAL_HOSPITAL_DIR = DATA_DIR / "external_hospital"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ORGANIZING DATASET FOR CARE-XAI RESEARCH PIPELINE")
print("="*70)

# ---------------------------------------------------------
# Step 1: Process Primary Dataset (BRISC 2025)
# ---------------------------------------------------------
brisc_root = DATASETS_DIR / "archive (2)" / "brisc2025"
manifest_path = brisc_root / "manifest.csv"

if not manifest_path.exists():
    raise FileNotFoundError(f"BRISC manifest not found at: {manifest_path}")

print("\n[Step 1] Loading BRISC 2025 manifest...")
df = pd.read_csv(manifest_path)
print(f"Total entries in original manifest: {len(df)}")

# Separate segmentation image-mask pairs
seg_images = df[(df['task'] == 'segmentation') & (df['is_mask'] == False)].copy()
seg_masks = df[(df['task'] == 'segmentation') & (df['is_mask'] == True)].copy()

# Map image filename to mask filename and relative paths
mask_map = {}
for _, row in seg_masks.iterrows():
    # Mask filename corresponds to linked_image or matches image filename with .png extension
    base_name = Path(row['filename']).stem
    mask_map[base_name] = {
        'mask_filename': row['filename'],
        'mask_rel_path': row['relative_path'],
        'mask_abs_path': str((brisc_root / row['relative_path']).resolve())
    }

print(f"Matched {len(mask_map)} segmentation masks.")

# Calculate mask statistics: pixel area, bounding box, relative area
print("\n[Step 2] Calculating ground-truth tumor spatial metrics from masks...")
records = []
for idx, row in seg_images.iterrows():
    base_name = Path(row['filename']).stem
    img_rel_path = row['relative_path']
    img_abs_path = str((brisc_root / img_rel_path).resolve())
    
    mask_info = mask_map.get(base_name)
    if not mask_info:
        continue
    
    mask_abs_path = mask_info['mask_abs_path']
    tumor_pixels = 0
    bbox = (0, 0, 0, 0)
    w_box, h_box = 0, 0
    
    if os.path.exists(mask_abs_path):
        with Image.open(mask_abs_path) as m_img:
            m_arr = np.array(m_img)
            tumor_pixels = int(np.count_nonzero(m_arr > 0))
            if tumor_pixels > 0:
                y_idx, x_idx = np.where(m_arr > 0)
                y1, y2 = int(y_idx.min()), int(y_idx.max())
                x1, x2 = int(x_idx.min()), int(x_idx.max())
                w_box = x2 - x1 + 1
                h_box = y2 - y1 + 1
                bbox = (x1, y1, x2, y2)
                
    rel_area = tumor_pixels / (row['width'] * row['height'])
    
    records.append({
        'case_id': base_name,
        'image_filename': row['filename'],
        'mask_filename': mask_info['mask_filename'],
        'image_rel_path': img_rel_path,
        'mask_rel_path': mask_info['mask_rel_path'],
        'image_path': img_abs_path,
        'mask_path': mask_abs_path,
        'has_mask': True,
        'tumor_label': row['tumor_label'],
        'tumor_code': row['tumor_code'],
        'plane_label': row['plane_label'],
        'plane_code': row['plane_code'],
        'sequence': row['sequence'],
        'width': row['width'],
        'height': row['height'],
        'tumor_pixel_area': tumor_pixels,
        'relative_tumor_area': rel_area,
        'bbox_x1': bbox[0],
        'bbox_y1': bbox[1],
        'bbox_x2': bbox[2],
        'bbox_y2': bbox[3],
        'bbox_width': w_box,
        'bbox_height': h_box,
        'original_split': row['split']
    })

seg_df = pd.DataFrame(records)
print(f"Processed {len(seg_df)} segmentation image-mask pairs.")

# Compute size percentiles for tumor-positive cases
positive_areas = seg_df[seg_df['tumor_pixel_area'] > 0]['tumor_pixel_area']
q25 = float(positive_areas.quantile(0.25))
q50 = float(positive_areas.quantile(0.50))
q75 = float(positive_areas.quantile(0.75))

print(f"\nTumor Area Distribution (pixels):")
print(f"  Min: {positive_areas.min()} px")
print(f"  25th percentile (Small tumor threshold): {q25:.1f} px (~{q25/(512*512)*100:.2f}% of image)")
print(f"  50th percentile (Median): {q50:.1f} px")
print(f"  75th percentile: {q75:.1f} px")
print(f"  Max: {positive_areas.max()} px")

# Assign small_tumor flag: area <= 25th percentile
seg_df['is_small_tumor'] = seg_df['tumor_pixel_area'] <= q25
seg_df['size_category'] = 'medium_large'
seg_df.loc[seg_df['is_small_tumor'], 'size_category'] = 'small'

# Include Healthy Control (no_tumor) slices
no_tumor_rows = df[(df['tumor_label'] == 'no_tumor') & (df['is_mask'] == False)].copy()
print(f"\nAdding {len(no_tumor_rows)} healthy control slices from BRISC 2025...")

ctrl_records = []
for idx, row in no_tumor_rows.iterrows():
    base_name = Path(row['filename']).stem
    img_abs_path = str((brisc_root / row['relative_path']).resolve())
    ctrl_records.append({
        'case_id': base_name,
        'image_filename': row['filename'],
        'mask_filename': 'None',
        'image_rel_path': row['relative_path'],
        'mask_rel_path': 'None',
        'image_path': img_abs_path,
        'mask_path': 'None',
        'has_mask': False,
        'tumor_label': 'no_tumor',
        'tumor_code': 'no',
        'plane_label': row['plane_label'],
        'plane_code': row['plane_code'],
        'sequence': row['sequence'],
        'width': row['width'],
        'height': row['height'],
        'tumor_pixel_area': 0,
        'relative_tumor_area': 0.0,
        'bbox_x1': 0,
        'bbox_y1': 0,
        'bbox_x2': 0,
        'bbox_y2': 0,
        'bbox_width': 0,
        'bbox_height': 0,
        'original_split': row['split'],
        'is_small_tumor': False,
        'size_category': 'no_tumor'
    })

ctrl_df = pd.DataFrame(ctrl_records)
full_primary_df = pd.concat([seg_df, ctrl_df], ignore_index=True)
print(f"Combined primary dataset size: {len(full_primary_df)} images (Tumor: {len(seg_df)}, Control: {len(ctrl_df)})")

# Save complete unified manifest
unified_manifest_path = METADATA_DIR / "brisc2025_processed_manifest.csv"
full_primary_df.to_csv(unified_manifest_path, index=False)
print(f"Saved processed manifest to: {unified_manifest_path}")

# ---------------------------------------------------------
# Step 3: Stratified Data Splitting (Patient / Slice Level)
# ---------------------------------------------------------
print("\n[Step 3] Creating Stratified Splits (70% Train, 15% Val, 15% Test)...")
# Stratify based on composite key: tumor_label + plane_label + size_category
full_primary_df['strat_key'] = full_primary_df['tumor_label'] + "_" + full_primary_df['plane_label'] + "_" + full_primary_df['size_category']

train_df, temp_df = train_test_split(
    full_primary_df, 
    test_size=0.30, 
    random_state=42, 
    stratify=full_primary_df['strat_key']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    random_state=42, 
    stratify=temp_df['strat_key']
)

# Extract dedicated Small-Tumour Test Set (Hypothesis H1 & H4)
# Contains small tumors + balanced no_tumor control slices
small_test_df = test_df[(test_df['is_small_tumor'] == True) | (test_df['tumor_label'] == 'no_tumor')].copy()

# Save splits
train_path = SPLITS_DIR / "train.csv"
val_path = SPLITS_DIR / "val.csv"
test_path = SPLITS_DIR / "test.csv"
small_test_path = SPLITS_DIR / "small_tumor_test.csv"

train_df.drop(columns=['strat_key']).to_csv(train_path, index=False)
val_df.drop(columns=['strat_key']).to_csv(val_path, index=False)
test_df.drop(columns=['strat_key']).to_csv(test_path, index=False)
small_test_df.drop(columns=['strat_key']).to_csv(small_test_path, index=False)

print(f"  Train set: {len(train_df)} samples -> {train_path}")
print(f"  Validation set: {len(val_df)} samples -> {val_path}")
print(f"  Test set: {len(test_df)} samples -> {test_path}")
print(f"  Small-Tumour Evaluation set: {len(small_test_df)} samples -> {small_test_path}")

# ---------------------------------------------------------
# Step 4: Index External Clinical Cohort (Epic & CSCR Hospital)
# ---------------------------------------------------------
print("\n[Step 4] Indexing External Clinical Hospital Dataset...")
hospital_records = []
if EXTERNAL_HOSPITAL_DIR.exists():
    for img_path in EXTERNAL_HOSPITAL_DIR.glob("**/*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            parts = img_path.relative_to(EXTERNAL_HOSPITAL_DIR).parts
            # Path structure: split/class/filename
            split_name = parts[0].lower() if len(parts) > 1 else 'unknown'
            tumor_class = parts[1].lower() if len(parts) > 2 else 'unknown'
            hospital_records.append({
                'filename': img_path.name,
                'rel_path': str(img_path.relative_to(WORKSPACE_DIR)),
                'image_path': str(img_path.resolve()),
                'split': split_name,
                'tumor_label': tumor_class,
                'source': 'Epic_CSCR_Hospital'
            })

hosp_df = pd.DataFrame(hospital_records)
if len(hosp_df) > 0:
    hosp_manifest_path = METADATA_DIR / "external_hospital_manifest.csv"
    hosp_test_path = SPLITS_DIR / "external_hospital_test.csv"
    hosp_df.to_csv(hosp_manifest_path, index=False)
    # The dedicated test split of external hospital dataset
    hosp_test = hosp_df[hosp_df['split'] == 'test']
    if len(hosp_test) == 0:
        hosp_test = hosp_df
    hosp_test.to_csv(hosp_test_path, index=False)
    print(f"Indexed {len(hosp_df)} hospital clinical images.")
    print(f"Saved external test split ({len(hosp_test)} images) to: {hosp_test_path}")
else:
    print("External hospital folder not yet populated or still extracting.")

print("\n" + "="*70)
print("DATASET ORGANIZATION COMPLETED SUCCESSFULLY!")
print("="*70)
