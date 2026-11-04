# BRISC 2025: Segmentation Component

## Scientific Usage in Research
- **Purpose:** Ground-truth pixel-level supervision for the auxiliary segmentation decoder and spatial localization validation in CARE-XAI.
- **Image-Mask 1:1 Alignment:**
  - `Train/images`: 3,933 scans (`.jpg`)
  - `Train/masks`: 3,933 binary masks (`.png`)
  - `Test/images`: 860 scans (`.jpg`)
  - `Test/masks`: 860 binary masks (`.png`)
- **Total Mask Pairs:** 4,793 aligned pairs.
- **Tumor Area Statistics:**
  - Min Area: 108 px
  - 25th Percentile ($Q_{0.25}$): **2,129.0 px** (Objective Small-Tumour Threshold)
  - Median ($Q_{0.50}$): 3,637.0 px
  - Max Area: 76,039 px
- **CARE-XAI Role:** Evaluates the **Pointing Game Localization Precision** and **Intersection-over-Union (IoU)** of CARE-XAI attention maps against radiologist boundaries.
