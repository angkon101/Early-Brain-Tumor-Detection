# Master Dataset Registry & Folder Index

This registry provides a complete, authoritative mapping of all dataset folders, paths, image counts, classes, and scientific roles for the **CARE-XAI** research project.

---

## 1. Master Cohort Summary Table

| Folder ID | Cohort Name | Root Path | Total Images | Modality & Resolution | Healthy Controls | Primary Research Role |
|---|---|---|:---:|---|:---:|---|
| **Cohort 01** | Primary Multi-Task Development (BRISC 2025) | [`data/01_primary_multitask/`](file:///e:/Brain%20tumor/data/01_primary_multitask) | **6,000 cases**<br>(4,793 masks) | T1-MRI (Ax, Co, Sa)<br>$512 \times 512$ | **1,207** | Core X-R-MiTNet multi-task training (RO1–RO4), small-tumor detection (**H1**), segmentation (**H4**), CARE-XAI evaluation (**RO5, RO8**). |
| **Cohort 02** | Standard Literature Benchmark (Phase 2) | [`data/02_literature_benchmark/`](file:///e:/Brain%20tumor/data/02_literature_benchmark) | **7,153 scans** | MRI 2D Slices<br>$512 \times 512$ | **2,000** | Literature baseline comparison (**RO6**): ResNet, EfficientNet, Swin-T, DeiT comparative benchmarks. |
| **Cohort 03** | Perturbation & Robustness (Phase 1) | [`data/03_perturbation_robustness/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness) | **7,200 scans** | Augmented MRI Slices<br>Varied Resolution | **1,800** | Degradation robustness (**RO7, RQ3**) and explanation stability testing under geometric/intensity shifts. |
| **Cohort 04** | External Clinical Hospital (Phase 3) | [`data/04_clinical_hospital/`](file:///e:/Brain%20tumor/data/04_clinical_hospital) | **12,064 scans** | Authentic Hospital MRI<br>Real Clinical Noise | **2,432** | Cross-center clinical generalization and hospital domain adaptation testing (Epic Healthcare & CSCR). |
| **TOTALS** | **All 4 Cohorts** | [`data/`](file:///e:/Brain%20tumor/data) | **32,417 scans** | | **7,439** | |

---

## 2. Exhaustive Folder-Level Directory Index

### Cohort 01: `data/01_primary_multitask/`
- [`data/01_primary_multitask/classification/Train/glioma/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Train/glioma) $\rightarrow$ 1,147 images
- [`data/01_primary_multitask/classification/Train/meningioma/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Train/meningioma) $\rightarrow$ 1,329 images
- [`data/01_primary_multitask/classification/Train/pituitary/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Train/pituitary) $\rightarrow$ 1,457 images
- [`data/01_primary_multitask/classification/Train/no_tumor/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Train/no_tumor) $\rightarrow$ 1,067 images *(Healthy Controls)*
- [`data/01_primary_multitask/classification/Test/glioma/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Test/glioma) $\rightarrow$ 254 images
- [`data/01_primary_multitask/classification/Test/meningioma/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Test/meningioma) $\rightarrow$ 306 images
- [`data/01_primary_multitask/classification/Test/pituitary/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Test/pituitary) $\rightarrow$ 300 images
- [`data/01_primary_multitask/classification/Test/no_tumor/`](file:///e:/Brain%20tumor/data/01_primary_multitask/classification/Test/no_tumor) $\rightarrow$ 140 images *(Healthy Controls)*
- [`data/01_primary_multitask/segmentation/Train/images/`](file:///e:/Brain%20tumor/data/01_primary_multitask/segmentation/Train/images) $\rightarrow$ 3,933 images
- [`data/01_primary_multitask/segmentation/Train/masks/`](file:///e:/Brain%20tumor/data/01_primary_multitask/segmentation/Train/masks) $\rightarrow$ 3,933 ground-truth binary masks
- [`data/01_primary_multitask/segmentation/Test/images/`](file:///e:/Brain%20tumor/data/01_primary_multitask/segmentation/Test/images) $\rightarrow$ 860 images
- [`data/01_primary_multitask/segmentation/Test/masks/`](file:///e:/Brain%20tumor/data/01_primary_multitask/segmentation/Test/masks) $\rightarrow$ 860 ground-truth binary masks

### Cohort 02: `data/02_literature_benchmark/`
- [`data/02_literature_benchmark/Train/glioma/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Train/glioma) $\rightarrow$ 1,297 images
- [`data/02_literature_benchmark/Train/meningioma/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Train/meningioma) $\rightarrow$ 1,420 images
- [`data/02_literature_benchmark/Train/pituitary/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Train/pituitary) $\rightarrow$ 1,406 images
- [`data/02_literature_benchmark/Train/notumor/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Train/notumor) $\rightarrow$ 1,600 images *(Healthy Controls)*
- [`data/02_literature_benchmark/Test/glioma/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Test/glioma) $\rightarrow$ 324 images
- [`data/02_literature_benchmark/Test/meningioma/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Test/meningioma) $\rightarrow$ 355 images
- [`data/02_literature_benchmark/Test/pituitary/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Test/pituitary) $\rightarrow$ 351 images
- [`data/02_literature_benchmark/Test/notumor/`](file:///e:/Brain%20tumor/data/02_literature_benchmark/Test/notumor) $\rightarrow$ 400 images *(Healthy Controls)*

### Cohort 03: `data/03_perturbation_robustness/`
- [`data/03_perturbation_robustness/Train/glioma/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Train/glioma) $\rightarrow$ 1,400 images
- [`data/03_perturbation_robustness/Train/meningioma/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Train/meningioma) $\rightarrow$ 1,400 images *(Augmented)*
- [`data/03_perturbation_robustness/Train/pituitary/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Train/pituitary) $\rightarrow$ 1,400 images
- [`data/03_perturbation_robustness/Train/notumor/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Train/notumor) $\rightarrow$ 1,400 images *(Healthy Controls)*
- [`data/03_perturbation_robustness/Test/glioma/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Test/glioma) $\rightarrow$ 400 images
- [`data/03_perturbation_robustness/Test/meningioma/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Test/meningioma) $\rightarrow$ 400 images *(Augmented)*
- [`data/03_perturbation_robustness/Test/pituitary/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Test/pituitary) $\rightarrow$ 400 images
- [`data/03_perturbation_robustness/Test/notumor/`](file:///e:/Brain%20tumor/data/03_perturbation_robustness/Test/notumor) $\rightarrow$ 400 images *(Healthy Controls)*

### Cohort 04: `data/04_clinical_hospital/`
- [`data/04_clinical_hospital/Train/glioma/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Train/glioma) $\rightarrow$ 3,018 clinical scans
- [`data/04_clinical_hospital/Train/meningioma/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Train/meningioma) $\rightarrow$ 2,183 clinical scans
- [`data/04_clinical_hospital/Train/pituitary/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Train/pituitary) $\rightarrow$ 2,504 clinical scans
- [`data/04_clinical_hospital/Train/notumor/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Train/notumor) $\rightarrow$ 1,945 clinical scans *(Hospital Controls)*
- [`data/04_clinical_hospital/Test/glioma/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Test/glioma) $\rightarrow$ 755 clinical scans
- [`data/04_clinical_hospital/Test/meningioma/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Test/meningioma) $\rightarrow$ 546 clinical scans
- [`data/04_clinical_hospital/Test/pituitary/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Test/pituitary) $\rightarrow$ 626 clinical scans
- [`data/04_clinical_hospital/Test/notumor/`](file:///e:/Brain%20tumor/data/04_clinical_hospital/Test/notumor) $\rightarrow$ 487 clinical scans *(Hospital Controls)*
