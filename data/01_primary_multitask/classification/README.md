# BRISC 2025: Classification Component

## Scientific Usage in Research
- **Purpose:** Diagnostic classification across 4 classes:
  - Class 0: `glioma` (intra-axial infiltrative glial tumor)
  - Class 1: `meningioma` (extra-axial dural-based benign/atypical tumor)
  - Class 2: `pituitary` (sellar/suprasellar endocrine adenoma)
  - Class 3: `no_tumor` (healthy control brain)
- **Class Breakdown:**
  - `Train/glioma`: 1,147 scans
  - `Train/meningioma`: 1,329 scans
  - `Train/no_tumor`: 1,067 scans (Negative Controls)
  - `Train/pituitary`: 1,457 scans
  - `Test/glioma`: 254 scans
  - `Test/meningioma`: 306 scans
  - `Test/no_tumor`: 140 scans (Negative Controls)
  - `Test/pituitary`: 300 scans
- **Total Images:** 6,000 scans
- **Filename Convention:** `brisc2025_<split>_<index>_<class>_<plane>_t1.jpg`

