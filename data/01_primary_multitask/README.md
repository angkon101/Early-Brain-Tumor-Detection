# Cohort 01: Primary Multi-Task Development (BRISC 2025)

## 1. Scientific Overview & Role
- **Citation Source:** [BRISC 2025 (Fateh et al., arXiv:2506.14318)](https://arxiv.org/abs/2506.14318)
- **Primary Research Role:** Core training, internal validation, and testing of the **X-R-MiTNet** multi-task architecture.
- **Specific Research Objectives:**
  - **RO1:** Verified multimodal multi-task MRI cohort.
  - **RO2:** Dual-task Vision Transformer training (Detection + Auxiliary Segmentation).
  - **RO3:** Reliability estimation ($\omega$) and uncertainty calibration ($u$).
  - **RO4:** Simultaneous multi-class prediction and pixel-level tumor contouring.
  - **RO5 & RO8:** CARE-XAI Attention Rollout, Pointing Game, and Deletion curve evaluation.
- **Hypotheses Tested:**
  - **H1:** Early / Small-Tumour Detection ($Q_{0.25} \le 2,129\text{ px}$).
  - **H4:** Auxiliary segmentation supervision forcing attention onto genuine pathological boundaries.

---

## 2. Directory Layout
```text
01_primary_multitask/
├── classification/
│   ├── Train/ (5,000 scans: 1,147 glioma, 1,329 meningioma, 1,457 pituitary, 1,067 no_tumor)
│   └── Test/  (1,000 scans: 254 glioma, 306 meningioma, 300 pituitary, 140 no_tumor)
└── segmentation/
    ├── Train/ (3,933 image-mask pairs: images/ and masks/)
    └── Test/  (860 image-mask pairs: images/ and masks/)
```

---

## 3. Dataset Specifications
- **Modality:** T1-Weighted Magnetic Resonance Imaging (T1-MRI).
- **Anatomical Planes:** Axial (`ax`), Coronal (`co`), Sagittal (`sa`).
- **Image Resolution:** $512 \times 512$ (standardized to $224 \times 224$ in PyTorch pipeline).
- **Total Cases:** 6,000 (4,793 tumor-affected with radiologist masks, 1,207 healthy controls).

---

## 4. PyTorch Usage
```python
from src.data.dataset import BrainTumorDataset

# Load primary multi-task training set (4,200 cases)
train_ds = BrainTumorDataset(cohort="multitask", split="train")

# Load dedicated small-tumor evaluation benchmark (362 cases)
small_ds = BrainTumorDataset(split_csv="data/splits/small_tumor_test.csv")
```

