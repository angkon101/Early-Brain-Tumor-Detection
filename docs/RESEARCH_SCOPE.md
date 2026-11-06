# Finalised Journal Research Scope

## Working Research Title

**X-R-MiTNet: An Explainable, Reliability-Aware Hierarchical Vision Transformer with Bilateral Hemispheric Symmetry Attention and Evidential Deep Learning for Multi-Cohort Brain Tumor Detection and Auxiliary Pixel-Level Segmentation**

**Target Journal Submissions:** *IEEE Transactions on Medical Imaging (TMI)* / *Nature Medicine* / *Elsevier Computers in Biology and Medicine*.

---

## 1. Executive Scientific Summary

Early and accurate diagnosis of intracranial neoplasms is crucial for surgical planning, radiotherapy guidance, and maximizing patient survival. While deep learning has demonstrated promising diagnostic accuracy on isolated benchmarks, clinical translation has been critically hindered by three persistent deficiencies:
1. **Neglect of Anatomical Priors:** Standard Vision Transformers treat brain MRI scans as generic, isotropic natural images, completely ignoring the fundamental neuroanatomical property of **bilateral mirror symmetry** across the cerebral sagittal midline.
2. **Overconfident Pathology Hallucination:** Traditional Softmax networks produce artificially peaked probabilities even on severe motion artifacts, out-of-distribution scanner noise, or subtle non-neoplastic white matter lesions, lacking calibrated **epistemic uncertainty quantification**.
3. **Black-Box Explanations Without Quantitative Faithfulness:** Post-hoc visual heatmaps (e.g., Grad-CAM) often produce diffuse, unfaithful attributions that fail rigorous spatial localization tests when benchmarked against expert radiologist masks.

To overcome these fundamental barriers, this study establishes **X-R-MiTNet (Explainable Reliability-Aware Multi-Task Imaging Transformer Network)**:
- **Imaging Modality:** Exclusively Brain Magnetic Resonance Imaging (MRI), encompassing **37,210 verified scans** across multi-center clinical cohorts.
- **Primary Task:** Multi-class brain tumor classification and subtyping (Glioma, Meningioma, Pituitary Adenoma, and Healthy Control / No Tumor).
- **Early/Small-Tumour Benchmark:** Explicit, defensible evaluation on the earliest $25\text{th}$ percentile of tumor areas ($A \le 350\text{ px}^2$).
- **Auxiliary Task:** Dense pixel-level tumor boundary segmentation via a multi-scale Feature Pyramid Network (FPN) decoder, providing spatial inductive bias and auxiliary supervision.
- **Architectural Core:**
  1. *Gated Overlapping Patch Stem* ($7\times 7$, stride 4) eliminating transformer boundary artifacts.
  2. *Bilateral Hemispheric Symmetry Multi-Head Self-Attention ($S$-MHSA)* that explicitly mirrors tokens across the sagittal axis ($c \leftrightarrow W - 1 - c$) to detect mass effect, ventricular compression, and unilateral parenchymal infiltration.
  3. *Multi-Scale Evidential Dirichlet Uncertainty Head* parameterized via Subjective Logic, trained with Focal Evidential Loss and Annealed KL divergence to decouple predictive confidence from epistemic vacuity.
  4. *CARE-XAI (Clinical Attribution & Reliability Explanations)* benchmarked quantitatively using the Pointing Game and Deletion/Insertion Area Under Curve (AUDC/AUIC).

---

## 2. Modality Audit & Elimination of CT References

> [!IMPORTANT]
> **Resolution of Dataset Modality Audit (Step 2 Verification)**
> Prior exploratory project notes hypothesized paired MRI--CT fusion. However, an exhaustive algorithmic audit of the entire raw dataset workspace (`docs/DATASET_AUDIT.md`) proved that **100% of the available data (37,210 scans) consists exclusively of Brain MRI scans**. Zero CT scans exist in the dataset.
> 
> In strict accordance with the Go/No-Go condition established in our research protocol:
> *"If no suitable public dataset contains genuinely paired MRI--CT scans and defensible early-disease labels, revise the scope before training. It is scientifically unsound to train on unpaired MRI and CT images and claim patient-specific multimodal fusion."*
> 
> Consequently, the research scope is formally solidified around **Multi-Cohort, Multi-Scale Brain MRI** with multi-center clinical cross-validation, completely eliminating unsubstantiated CT fusion claims.

---

## 3. Research Objectives and Testable Hypotheses

### 3.1 Specific Research Objectives (RO)

| ID | Research Objective | Empirical Deliverable |
|---|---|---|
| **RO1** | **Multi-Cohort MRI Dataset Harmonization** | Harmonize 37,210 MRI scans across 4 distinct cohorts (BRISC 2025 Multi-Task, Kaggle Literature Cleaned, Epic & CSCR Hospital, and Perturbation Robustness) with zero patient slice-leakage. |
| **RO2** | **Hierarchical Symmetry-Aware Architecture** | Construct X-R-MiTNet v2 combining a 4-stage hierarchical transformer ($56\times 56 \to 28\times 28 \to 14\times 14 \to 7\times 7$) with Bilateral Hemispheric Symmetry Attention ($S$-MHSA). |
| **RO3** | **Evidential Uncertainty Quantification** | Formulate Subjective Logic Dirichlet density estimation with Focal Evidential Loss, achieving low Expected Calibration Error (ECE) and statistically significant separation of errors. |
| **RO4** | **Auxiliary Multi-Scale Lesion Segmentation** | Engineer a Dense Feature Pyramid Network (FPN) auxiliary decoder directly coupled to multi-scale transformer stage representations. |
| **RO5** | **Early/Small-Tumour Detection Evaluation** | Benchmark diagnostic sensitivity and F1-score specifically on small-tumor cases ($A \le 25\text{th percentile}$ of radiologist masks) to validate early-detection efficacy. |
| **RO6** | **Quantitative XAI Faithfulness Verification** | Validate CARE-XAI attention rollout against expert radiologist masks using the Pointing Game and Deletion/Insertion curves. |
| **RO7** | **Multi-Center Clinical Generalization** | Demonstrate out-of-distribution robustness on independent clinical hospital cohorts (Epic & CSCR) and synthetic sensor perturbations. |

### 3.2 Formal Testable Hypotheses (H)

- **H1 (Symmetry Prior):** Integrating Bilateral Hemispheric Symmetry Attention ($S$-MHSA) yields a statistically significant gain in Macro-F1 compared to isotropic self-attention, specifically on asymmetric infiltrative tumors (Gliomas).
- **H2 (Evidential Calibration):** Dirichlet evidential formulation with focal modulation reduces Expected Calibration Error (ECE $\le 0.04$) and separates misclassified instances with $p < 0.001$.
- **H3 (Auxiliary Multi-Task Regularization):** Jointly supervising with auxiliary pixel-level segmentation enhances classification feature representations and improves small-tumor sensitivity without sacrificing diagnostic specificity.
- **H4 (XAI Faithfulness):** CARE-XAI Attention Rollout achieves $>85\%$ Pointing Game accuracy against gold-standard radiologist masks and outperforms conventional Grad-CAM in Deletion AUC.

---

## 4. Multi-Cohort Dataset Architecture (37,210 MRI Scans)

The empirical validation spans **4 distinct cohorts** organized to evaluate internal accuracy, external generalization, small-tumor sensitivity, and sensor degradation robustness:

```
data/
├── 01_primary_multitask/        # BRISC 2025 Benchmark (Fateh et al., 2025)
│   ├── classification/          # 6,000 scans (T1-weighted: Axial, Coronal, Sagittal)
│   └── segmentation/            # 4,793 aligned Image-Mask pairs (radiologist annotated)
├── 02_literature_benchmark/     # Cleaned 7,153 MRI cohort (Standard Literature Benchmark)
├── 03_perturbation_robustness/  # 7,200 MRI cohort (Gaussian noise, blur, intensity bias)
├── 04_clinical_hospital/        # 12,064 MRI cohort (Epic Healthcare & CSCR Hospital archive)
├── metadata/                    # Master manifests with patient IDs and SHA-256 hashes
└── splits/                      # Patient-stratified Train/Val/Test CSV manifests
```

### Zero Slice-Leakage Guarantee
All splits are generated using patient-level tokens (`case_id = patient_id + sequence`). In `01_primary_multitask`, exactly 6,000 unique patient tokens correspond to 6,000 scans (1 scan per patient). Zero patient or slice overlap exists between training, validation, and test sets.

---

## 5. Methodological & Model Innovations

```
                       X-R-MiTNet (v2) ARCHITECTURAL FLOW
+-----------------------------------------------------------------------------------+
| 1. Input MRI (224x224x1) -> Gated Overlapping Stem (7x7 Conv, Stride 4, C=64)    |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
| 2. Hierarchical Transformer Stages:                                               |
|    - Stage 1: 56x56 (C=64,  D=2)  -> Overlapping Patch Merging                     |
|    - Stage 2: 28x28 (C=128, D=2)  -> Overlapping Patch Merging                     |
|    - Stage 3: 14x14 (C=256, D=4)  -> Bilateral Hemispheric Symmetry ($S$-MHSA)    |
|    - Stage 4:  7x7  (C=512, D=2)  -> Global LayerNorm                              |
+-----------------------------------------------------------------------------------+
                       /                                     \
                      v                                       v
+-----------------------------------------+   +-------------------------------------+
| 3. Multi-Scale Evidential Dirichlet     |   | 4. Dense FPN Auxiliary Segmentation |
|    Head (768-d multi-scale pooled)      |   |    Decoder (Lateral skip fusion)    |
|    -> Dirichlet Evidence $\alpha_k$     |   |    -> 224x224 Dense Binary Mask     |
|    -> Epistemic Uncertainty $u = K/S$   |   |    -> Dice + Focal BCE Loss         |
|    -> Focal Evidential + Annealed KL    |   +-------------------------------------+
+-----------------------------------------+
```

### 5.1 Bilateral Hemispheric Symmetry Attention ($S$-MHSA)
The human brain possesses approximate bilateral symmetry across the sagittal midline. Neoplastic lesions break this symmetry. In Stage 3, each patch embedding $x_{(r, c)}$ queries both its spatial neighborhood and its sagittal counterpart $x_{(r, W - 1 - c)}$:
$$A_{sym}(Q, K) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + \lambda_{sym} \mathcal{M}_{sagittal}\right) V$$
This enables the network to explicitly capture mass effect, midline shift, and unilateral tissue hyperintensity.

### 5.2 Evidential Uncertainty Quantification
Rather than a point Softmax estimate, X-R-MiTNet parameterizes a Dirichlet distribution over class probabilities:
$$p(p_1, \dots, p_K | \alpha) = \frac{\Gamma(S)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}, \quad S = \sum_{k=1}^K \alpha_k$$
The expected probability is $\hat{p}_k = \alpha_k / S$, and the total epistemic uncertainty (vacuity of evidence) is:
$$u = \frac{K}{S}$$
High uncertainty ($u \to 1$) directly signals ambiguous, low-quality, or atypical scans that require mandatory expert radiologist review.

### 5.3 CARE-XAI Quantitative Faithfulness
Attention rollout across transformer layers is evaluated using:
1. **Pointing Game Hit Rate:**
   $$\text{Accuracy}_{PG} = \frac{1}{N_{tumor}} \sum_{i=1}^{N_{tumor}} \mathbb{I}\left(\arg\max_{(x,y)} H_i(x,y) \in \mathcal{M}_{gt, i}\right)$$
2. **Deletion & Insertion AUC (AUDC / AUIC):**
   Pixel-perturbation curves tracking the monotonic degradation of the true class probability as top-attributed features are masked.

---

## 6. Journal Reporting Standards & Statistical Integrity

To satisfy the highest peer-reviewed journal criteria (IEEE TMI / Nature Medicine / Elsevier CBM):
1. **95% Bootstrap Confidence Intervals:** All reported metrics (Accuracy, Balanced Accuracy, Precision, Recall, Specificity, Macro-F1, MCC, Cohen's Kappa, and AUC-ROC) must be accompanied by non-parametric percentile bootstrap confidence intervals ($B = 1,000$ iterations).
2. **Calibration Rigor:** Model reliability must be verified via Expected Calibration Error (ECE) with 10 bins and multi-class Brier score.
3. **Reproducibility:** Self-contained execution pipeline (`colab_all_in_one.py`) executable end-to-end on Google Colab (Tesla T4 GPU) with automated inline 300 DPI figures and Overleaf-ready LaTeX tables.
