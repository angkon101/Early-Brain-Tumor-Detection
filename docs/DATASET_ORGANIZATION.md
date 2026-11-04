# Comprehensive Multi-Cohort Dataset Organization Specification

This document details the standardized multi-cohort dataset architecture, quantitative ground-truth metrics, and experimental partitions prepared for the **CARE-XAI / X-R-MiTNet** research framework.

---

## 1. Directory Structure

All datasets are now unified and categorized within [`data/`](file:///e:/Brain%20tumor/data) according to their precise scientific role in the research objectives:

```text
data/
├── 01_primary_multitask/              # Primary Multi-Task Development Cohort (BRISC 2025)
│   ├── classification/
│   │   ├── Train/                     # 5,000 images (glioma, meningioma, pituitary, no_tumor)
│   │   └── Test/                      # 1,000 images (glioma, meningioma, pituitary, no_tumor)
│   └── segmentation/
│       ├── Train/                     # 3,933 paired images + 3,933 pixel-wise masks
│       └── Test/                      # 860 paired images + 860 pixel-wise masks
│
├── 02_literature_benchmark/           # Standard Cleaned 7,153 Benchmark (Phase 2)
│   ├── Train/                         # 5,723 images (glioma, meningioma, pituitary, notumor)
│   └── Test/                          # 1,430 images (glioma, meningioma, pituitary, notumor)
│
├── 03_perturbation_robustness/        # Perturbation & Degradation Cohort (Phase 1 - 7,200)
│   ├── Train/                         # 5,600 images (augmented with rotations, contrast shifts)
│   └── Test/                          # 1,600 images (augmented with rotations, contrast shifts)
│
├── 04_clinical_hospital/              # Real-World Hospital Cohort (Phase 3 - Epic & CSCR)
│   ├── Train/                         # 9,650 clinical scans from Epic Healthcare & CSCR Hospital
│   └── Test/                          # 2,414 clinical scans from Epic Healthcare & CSCR Hospital
│
├── metadata/                          # Complete manifests and audit records
│   ├── master_dataset_summary.csv     # High-level comparison across all 4 cohorts
│   ├── multitask_master_manifest.csv  # 6,000 cases with mask metrics, planes, and sizes
│   ├── literature_benchmark_manifest.csv # 7,153 indexed benchmark images
│   ├── robustness_manifest.csv        # 7,200 indexed augmented images
│   └── clinical_hospital_manifest.csv # 12,064 indexed hospital scans
│
└── splits/                            # Standardized CSV partitions for training & evaluation
    ├── multitask_train.csv            # 4,200 cases (70% primary train)
    ├── multitask_val.csv              # 900 cases (15% primary val)
    ├── multitask_test.csv             # 900 cases (15% primary test)
    ├── small_tumor_test.csv           # 362 cases (dedicated early/small lesion benchmark for H1 & H4)
    ├── literature_benchmark_train.csv # 5,723 cases (literature baseline training)
    ├── literature_benchmark_test.csv  # 1,430 cases (literature baseline testing - RO6)
    ├── robustness_train.csv           # 5,600 cases (stress-testing training)
    ├── robustness_test.csv            # 1,600 cases (explanation stability testing - RO7, RO8)
    ├── clinical_hospital_train.csv    # 9,650 cases (hospital domain transfer training)
    └── clinical_hospital_test.csv     # 2,414 cases (cross-center clinical evaluation)
```

---

## 2. Research Roles of the Four Cohorts

| Cohort | Source / Nature | Total Scans | Scientific Role in CARE-XAI Research |
|---|---|---|---|
| **01: Primary Multi-Task** | BRISC 2025 (ArXiv:2506.14318) | **6,000 cases** (4,793 masks + 1,207 controls) | **Core Architecture Development (RO1–RO4):** Trains the multi-task CNN-free ViT for simultaneous tumor detection and pixel-level segmentation. Provides ground truth for **H1** (small-tumor detection) and **H4** (auxiliary segmentation supervision). |
| **02: Literature Benchmark** | Standard Cleaned 7,153 Cohort | **7,153 scans** (5,723 train, 1,430 test) | **Literature Comparison (RO6):** The most cited 4-class brain tumor benchmark in modern literature. Enables direct apples-to-apples comparison against published CNN and ViT baselines. |
| **03: Perturbation Robustness** | Augmented 7,200 Cohort | **7,200 scans** (5,600 train, 1,600 test) | **Input Degradation & XAI Stability (RO7, RO8, RQ3):** Contains heavy geometric and intensity perturbations (`Tr-aug-me_*`). Tests model resilience under degraded imaging and evaluates CARE-XAI attribution faithfulness under input shifts. |
| **04: Clinical Hospital** | Epic Healthcare & CSCR Hospital | **12,064 scans** (9,650 train, 2,414 test) | **Clinical External Validation:** Authentic patient MRI scans from two tertiary care hospitals. Tests domain generalization and deployment readiness on noisy real-world clinical acquisitions. |

**Total Medical Scans Unified Across All Cohorts:** **32,417 unique images**

---

## 3. Quantitative Ground-Truth Metrics & Small-Tumour Definition

In accordance with Section 1.1 of [`docs/RESEARCH_SCOPE.md`](file:///e:/Brain%20tumor/docs/RESEARCH_SCOPE.md), early detection is objectively operationalized via mask pixel area:

$$\text{Small-Tumour Condition} \iff \text{Tumor Area} \le Q_{0.25} = 2,129.0 \text{ pixels}$$

### Primary Cohort (BRISC 2025) Distribution:
- **Min Tumor Area:** 108 pixels
- **25th Percentile ($Q_{0.25}$):** **2,129.0 px** ($\approx 0.81\%$ of $512 \times 512$ scan area)
- **Median ($Q_{0.50}$):** 3,637.0 px ($\approx 1.39\%$)
- **75th Percentile ($Q_{0.75}$):** 5,801.0 px ($\approx 2.21\%$)
- **Max Area:** 76,039 pixels
- **Healthy Controls:** 1,207 true negative slices (`no_tumor`, area = 0 px)

---

## 4. PyTorch Multi-Cohort Data Pipeline

Implemented in [`src/data/dataset.py`](file:///e:/Brain%20tumor/src/data/dataset.py):

```python
from src.data.dataset import BrainTumorDataset
from torch.utils.data import DataLoader

# 1. Primary Multi-Task (Detection + Segmentation + Small Tumor)
ds_multitask = BrainTumorDataset(cohort="multitask", split="train")

# 2. Dedicated Small-Tumour Test Split (Hypothesis H1 & H4)
ds_small = BrainTumorDataset(split_csv="data/splits/small_tumor_test.csv")

# 3. Literature Benchmark Comparison (RO6)
ds_benchmark = BrainTumorDataset(cohort="benchmark", split="test")

# 4. Perturbation & Robustness Evaluation (RO7 / RO8)
ds_robustness = BrainTumorDataset(cohort="robustness", split="test")

# 5. External Clinical Hospital Evaluation (Domain Generalization)
ds_clinical = BrainTumorDataset(cohort="clinical", split="test")
```
