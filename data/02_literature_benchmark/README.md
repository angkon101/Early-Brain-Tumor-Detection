# Cohort 02: Standard Literature Benchmark (Phase 2)

## 1. Scientific Overview & Role
- **Origin:** Standard cleaned 7,153 4-class brain MRI collection widely cited in recent medical literature.
- **Primary Research Role:** Objective **RO6 (Comparative Baseline Evaluation)**.
- **Purpose in Paper:** Enables a fair, direct, apples-to-apples performance comparison against published CNNs (ResNet, EfficientNet) and Vision Transformers (DeiT, Swin-T) on the exact same benchmark split used by other authors.

---

## 2. Directory Layout & Class Breakdown
```text
02_literature_benchmark/
├── Train/ (5,723 scans)
│   ├── glioma:     1,297 scans
│   ├── meningioma: 1,420 scans
│   ├── notumor:    1,600 scans (Healthy Controls)
│   └── pituitary:  1,406 scans
└── Test/  (1,430 scans)
    ├── glioma:       324 scans
    ├── meningioma:   355 scans
    ├── notumor:      400 scans (Healthy Controls)
    └── pituitary:    351 scans
```
- **Total Images:** 7,153 scans ($512 \times 512$ resolution).
- **Healthy Controls:** 2,000 scans total.

---

## 3. PyTorch Usage
```python
from src.data.dataset import BrainTumorDataset

# Load benchmark training cohort
benchmark_train = BrainTumorDataset(cohort="benchmark", split="train")

# Load benchmark test cohort for literature baseline comparison (RO6)
benchmark_test = BrainTumorDataset(cohort="benchmark", split="test")
```
