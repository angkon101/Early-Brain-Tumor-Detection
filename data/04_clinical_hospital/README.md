# Cohort 04: External Clinical Hospital Validation (Phase 3)

## 1. Scientific Overview & Role
- **Origin:** Authentic patient scans acquired from **Epic Healthcare** and **CSCR Hospital (Centre for Specialized Care and Research, Chittagong)**.
- **Primary Research Role:** **External Real-World Clinical Validation & Domain Adaptation**.
- **Clinical Value:** Evaluating whether a model trained on curated benchmark data can generalize to real hospital acquisitions with different MRI scanner hardware, slice thicknesses, motion artifacts, and patient demographics.

---

## 2. Directory Layout & Class Breakdown
```text
04_clinical_hospital/
├── Train/ (9,650 scans)
│   ├── glioma:     3,018 scans
│   ├── meningioma: 2,183 scans
│   ├── notumor:    1,945 scans (Healthy Hospital Controls)
│   └── pituitary:  2,504 scans
└── Test/  (2,414 scans)
    ├── glioma:       755 scans
    ├── meningioma:   546 scans
    ├── notumor:      487 scans (Healthy Hospital Controls)
    └── pituitary:    626 scans
```
- **Total Images:** 12,064 authentic clinical MRI scans.
- **Healthy Controls:** 2,432 clinical control scans.

---

## 3. PyTorch Usage
```python
from src.data.dataset import BrainTumorDataset

# Load external hospital clinical test cohort
hospital_test = BrainTumorDataset(cohort="clinical", split="test")
```

