# Cohort 03: Perturbation & Degradation Robustness (Phase 1)

## 1. Scientific Overview & Role
- **Origin:** Augmented 7,200 4-class MRI collection containing heavy synthetic geometric, contrast, and affine transformations (`Tr-aug-me_*`).
- **Primary Research Role:** 
  - **RO7 & RQ3 (Missing/Degraded Input Robustness):** Stress-testing model resilience under noisy, rotated, or degraded scans.
  - **RO8 (CARE-XAI Explanation Stability):** Measuring whether attention attribution remains stable or collapses when an image undergoes intensity and rotational shifts.
- **Hypothesis Tested:**
  - **H3:** Robustness under degraded imaging inputs.

---

## 2. Directory Layout & Class Breakdown
```text
03_perturbation_robustness/
├── Train/ (5,600 scans)
│   ├── glioma:     1,400 scans
│   ├── meningioma: 1,400 scans (contains augmented variants)
│   ├── notumor:    1,400 scans (Healthy Controls)
│   └── pituitary:  1,400 scans
└── Test/  (1,600 scans)
    ├── glioma:       400 scans
    ├── meningioma:   400 scans
    ├── notumor:      400 scans (Healthy Controls)
    └── pituitary:    400 scans
```
- **Total Images:** 7,200 scans.
- **Healthy Controls:** 1,800 scans total.

---

## 3. PyTorch Usage
```python
from src.data.dataset import BrainTumorDataset

# Load robustness stress-test cohort
robustness_test = BrainTumorDataset(cohort="robustness", split="test")
```
