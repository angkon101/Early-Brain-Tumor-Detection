# Dataset Organization & Processing Specification

This document details the standardized dataset organization, ground-truth metrics, and experimental partitions prepared for the **CARE-XAI / X-R-MiTNet** research framework.

---

## 1. Directory Structure

```text
data/
├── metadata/
│   ├── brisc2025_processed_manifest.csv   # Unified primary manifest with mask metrics (6,000 cases)
│   └── external_hospital_manifest.csv     # Manifest for Epic & CSCR Hospital dataset (12,064 scans)
├── splits/
│   ├── train.csv                          # Primary training cohort (4,200 samples - 70%)
│   ├── val.csv                            # Internal validation cohort (900 samples - 15%)
│   ├── test.csv                           # Internal test cohort (900 samples - 15%)
│   ├── small_tumor_test.csv               # Dedicated Small-Tumour benchmark (362 samples)
│   └── external_hospital_test.csv         # External clinical evaluation set (2,414 scans)
└── external_hospital/                     # Extracted clinical hospital scans
    ├── Test/                              # 2,414 images (glioma, meningioma, pituitary, notumor)
    └── Train/                             # 9,650 images (glioma, meningioma, pituitary, notumor)
```

---

## 2. Quantitative Ground-Truth Metrics & Small-Tumour Definition

In accordance with Section 1.1 of [`docs/RESEARCH_SCOPE.md`](file:///e:/Brain%20tumor/docs/RESEARCH_SCOPE.md), an early-detection claim requires an objective, measurable definition:

$$\text{Small-Tumour Condition} \iff \text{Tumor Area} \le Q_{0.25} = 2,129 \text{ pixels}$$

### Primary Cohort (BRISC 2025) Spatial Distribution:
- **Total Evaluated Image–Mask Pairs:** 4,793
- **Tumor Area Range:** 108 pixels to 76,039 pixels
- **25th Percentile ($Q_{0.25}$):** **2,129.0 px** ($\approx 0.81\%$ of total $512 \times 512$ image area)
- **Median ($Q_{0.50}$):** 3,637.0 px ($\approx 1.39\%$)
- **75th Percentile ($Q_{0.75}$):** 5,801.0 px ($\approx 2.21\%$)
- **Healthy Controls Included:** 1,207 true negative slices (`no_tumor`, area = 0 px)
- **Combined Primary Cohort Size:** 6,000 cases

---

## 3. Experimental Splitting Protocol

All primary cohort splits are stratified across three orthogonal factors:
1. **Tumor Class:** `glioma`, `meningioma`, `pituitary`, `no_tumor`
2. **Anatomical Plane:** `axial`, `coronal`, `sagittal`
3. **Size Category:** `small`, `medium_large`, `no_tumor`

| Partition | Sample Count | Purpose |
|---|---|---|
| `train.csv` | 4,200 (70%) | Multi-task X-R-MiTNet model optimization (Detection + Auxiliary Segmentation). |
| `val.csv` | 900 (15%) | Checkpointing, early stopping, and hyperparameter tuning. |
| `test.csv` | 900 (15%) | Unbiased internal multi-class detection and segmentation evaluation. |
| `small_tumor_test.csv` | 362 samples | Hypothesis **H1** & **H4** validation: specifically evaluating sensitivity on early/small lesions vs healthy controls. |
| `external_hospital_test.csv` | 2,414 scans | Real-world generalizability and domain transfer evaluation (Epic & CSCR Hospital cohort). |

---

## 4. PyTorch Data Pipeline

The data loader is implemented in [`src/data/dataset.py`](file:///e:/Brain%20tumor/src/data/dataset.py):

```python
from src.data.dataset import BrainTumorDataset
from torch.utils.data import DataLoader

train_dataset = BrainTumorDataset("data/splits/train.csv", image_size=(224, 224), is_training=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for batch in train_loader:
    images = batch["image"]           # (B, 1, 224, 224) normalized MRI tensor
    masks = batch["mask"]             # (B, 1, 224, 224) binary ground-truth segmentation
    labels = batch["label"]           # (B,) class indices [0: glioma, 1: meningioma, 2: pituitary, 3: no_tumor]
    is_small = batch["is_small_tumor"]# (B,) binary indicator for small/early lesion
    break
```
