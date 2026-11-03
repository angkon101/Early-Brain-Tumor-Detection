# Dataset Discovery and Audit Report (Step 2)

This document provides a comprehensive audit of all datasets currently available in the project workspace under [`Datasets/`](file:///e:/Brain%20tumor/Datasets). It evaluates each dataset against the mandatory selection criteria and go/no-go conditions established in [Step 1 --- Finalised Research Scope](file:///e:/Brain%20tumor/docs/RESEARCH_SCOPE.md).

---

## 1. Summary of Identified Datasets

| Dataset Identifier | Directory / Source | Total Files | Format | Modality | Segmentation Masks | Metadata / Labels |
|---|---|---|---|---|---|---|
| **D1: Kaggle Nickparvar (Augmented)** | `Datasets/archive` | 7,200 | `.jpg` | MRI (2D slices) | ❌ None | 4 Classes (glioma, meningioma, pituitary, notumor); train/test splits. Augmented slices present. |
| **D2: Kaggle Cleaned (7,153)** | `Datasets/archive (1)` | 7,153 | `.jpg` | MRI (2D slices) | ❌ None | 4 Classes; 5,723 train, 1,430 test; normalized 512x512. |
| **D3: BRISC 2025** | `Datasets/archive (2)/brisc2025` | 15,591 | `.jpg` (images) + `.png` (masks) | MRI (T1-weighted) | ✅ Yes (4,793 aligned mask-image pairs) | Manifest (`manifest.csv`, `manifest.json`), 4 classes, anatomical planes (axial, coronal, sagittal), pixel-wise radiologist masks. |
| **D4: Epic & CSCR Hospital Dataset** | `Datasets/Brain Tumor MRI Dataset.../Epic and CSCR hospital Dataset.zip` | 12,064 | `.jpg`, `.jpeg` | MRI (2D slices) | ❌ None | Real hospital cohort (Epic & CSCR); 4 classes; 9,650 train, 2,414 test. |

---

## 2. In-Depth Evaluation per Dataset

### Dataset 1: `Datasets/archive` (Kaggle Augmented 7,200)
- **Origin:** Curated Kaggle compilation (derived from Figshare, SARTAJ, and Br35H by M. Nickparvar / S. Bhuvaji).
- **Split Breakdown:**
  - `Training`: 5,600 images (1,400 glioma, 1,400 meningioma, 1,400 notumor, 1,400 pituitary).
  - `Testing`: 1,600 images (400 glioma, 400 meningioma, 400 notumor, 400 pituitary).
- **Image Properties:** Varied resolutions (e.g., 512×512, 510×602, 339×406); 3-channel RGB representation.
- **Limitations:**
  - Contains synthetic offline augmentations (e.g., `Tr-aug-me_1.jpg`), which can distort anatomical fidelity and introduce slice leakage if split across patients.
  - No patient identifiers or acquisition timestamps.
  - No CT modality, no segmentation masks, no early-stage/tumor size metadata.

### Dataset 2: `Datasets/archive (1)` (Kaggle Cleaned 7,153)
- **Origin:** Standard cleaned and de-duplicated 4-class brain MRI dataset.
- **Split Breakdown:**
  - `train`: 5,723 images (1,297 glioma, 1,420 meningioma, 1,600 notumor, 1,406 pituitary).
  - `test`: 1,430 images (324 glioma, 355 meningioma, 400 notumor, 351 pituitary).
- **Image Properties:** Standardized 512×512 resolution.
- **Limitations:**
  - MRI only; no paired CT.
  - No patient-level grouping or clinical staging metadata.

### Dataset 3: `Datasets/archive (2)/brisc2025` (BRISC 2025 Benchmark)
- **Origin:** Fateh et al., 2025 ([arXiv:2506.14318](https://arxiv.org/abs/2506.14318)): *BRISC: A Curated, Expert-Annotated T1 MRI Dataset for Multi-Class Brain Tumor Classification and Pixel-Wise Segmentation*.
- **Total Entries in Manifest:** 15,586 annotated items.
- **Tasks & Modalities:**
  - **Classification task:** 6,000 images (5,000 train, 1,000 test; balanced across 4 classes).
  - **Segmentation task:** 9,586 files (4,793 image–mask pairs; 1:1 correspondence between `.jpg` and `.png`).
- **Anatomical Planes:** Explicitly partitioned into **Axial** (5,171), **Coronal** (5,227), and **Sagittal** (5,188).
- **Modality:** Uniformly T1-weighted MRI (`sequence: T1`).
- **Key Strengths:**
  - Gold-standard, radiologist-verified pixel-level segmentation masks for 4,793 slices.
  - Aligned filenames and full manifest tracking with SHA-256 hashes.
  - Tumor area can be quantitatively computed from masks to support an objective **small-tumour detection** threshold.
- **Limitations:**
  - T1 MRI only; no CT scans included.

### Dataset 4: `Epic and CSCR hospital Dataset.zip` (Clinical Hospital Cohort)
- **Origin:** Authentic clinical scans acquired from Epic Healthcare and CSCR (Centre for Specialized Care and Research, Chittagong).
- **Total Images:** 12,064 images (241.37 MB compressed).
- **Split Breakdown:**
  - `Train`: 9,650 images (3,018 glioma, 2,183 meningioma, 1,945 notumor, 2,504 pituitary).
  - `Test`: 2,414 images (755 glioma, 546 meningioma, 487 notumor, 626 pituitary).
- **Key Strengths:** High clinical diversity, real-world hospital acquisition noise, and multiclass representation.
- **Limitations:**
  - MRI only; no CT scans.
  - Slice-level JPEG files without segmentation masks or clinical stage metadata.

---

## 3. 10-Point Dataset Comparison Matrix

| # | Evaluation Criterion | D1: Kaggle Aug (7.2k) | D2: Kaggle Clean (7.1k) | D3: BRISC 2025 (15.5k) | D4: Epic & CSCR (12k) |
|---|---|---|---|---|---|
| 1 | **Official Source** | Kaggle | Kaggle | arXiv:2506.14318 / BRISC 2025 | Hospital clinical archive |
| 2 | **MRI Sequence** | Mixed T1/T2 (2D slices) | Mixed (2D slices) | **T1-weighted** (Axial, Coronal, Sagittal) | Clinical T1/T2 |
| 3 | **CT Availability** | ❌ None | ❌ None | ❌ None | ❌ None |
| 4 | **Verified Patient Pairing** | ❌ None | ❌ None | ❌ Slice-level indexing | ❌ Slice-level indexing |
| 5 | **Unique Patients** | Unknown | Unknown | Estimated ~5,000 cases | Unknown |
| 6 | **Early-Stage / Size Labels** | ❌ None | ❌ None | ⚠️ Computable via mask area | ❌ None |
| 7 | **Negative Controls** | ✅ Yes (notumor) | ✅ Yes (notumor) | ✅ Yes (no_tumor: 1,207) | ✅ Yes (notumor: 2,432) |
| 8 | **Segmentation Masks** | ❌ None | ❌ None | ✅ **Yes (4,793 masks)** | ❌ None |
| 9 | **File Format** | `.jpg` | `.jpg` | `.jpg` + `.png` + `manifest.csv` | `.jpg`, `.jpeg` |
| 10 | **Step 1 Suitability** | Low (unpaired MRI only) | Moderate (classification only) | **High (Multi-task Classification + Segmentation)** | High for clinical external test |

---

## 4. Critical Scientific Finding & Go/No-Go Evaluation

> [!WARNING]
> **Mandatory Criterion Check: Paired MRI–CT Scans**
> None of the four datasets in `Datasets/` contain CT scans. All four datasets consist exclusively of **brain MRI scans**.
> 
> In accordance with the **Go/No-Go condition** established in Section 9 of [RESEARCH_SCOPE.md](file:///e:/Brain%20tumor/docs/RESEARCH_SCOPE.md):
> *"If no suitable public dataset contains paired MRI–CT scans and defensible early-disease labels, revise the scope before training. It would be scientifically unsound to train on unpaired MRI and CT images and claim patient-specific multimodal fusion."*

### Scientific Pathways Forward

1. **Pathway A (Recommended based on provided data):** Refocus multimodal / multi-task architecture on **BRISC 2025** and **Epic & CSCR Hospital Dataset**:
   - Utilize BRISC 2025 as the primary development cohort for joint **Brain Tumor Detection (Primary)** and **Auxiliary Segmentation (Auxiliary)**.
   - Utilize mask pixel area to define an objective, quantifiable **Small-Tumour Detection** threshold (e.g., lower quartile of tumor area).
   - Utilize multi-view (Axial + Coronal + Sagittal) or multi-sequence MRI fusion, or cross-dataset external validation (using Epic & CSCR hospital dataset).
2. **Pathway B (Retain strict MRI–CT fusion):** Acquire external paired MRI–CT datasets (e.g., RIDER Neuro MRI-CT, CQ500, TCIA Glioblastoma MRI/CT collections) before proceeding with model training.
