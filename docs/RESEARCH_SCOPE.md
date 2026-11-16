# Step 1 --- Finalised Research Scope

## Working research title

**CARE-XAI: An Explainable Reliability-Aware Multimodal Vision Transformer for Early Brain Tumour Detection Using MRI--CT Fusion Under Missing-Modality Conditions**

This is a proposed research title. The early-detection claim must be supported by suitable clinical or imaging metadata in the selected dataset.

------------------------------------------------------------------------

## 1. Final research definition

The research focuses on **early brain tumour detection using paired MRI and CT scans from the same patients**. The proposed model is a CNN-free, reliability-aware multimodal Vision Transformer, with explainable AI integrated into its workflow.

- **Primary task:** early brain tumour detection.
- **Auxiliary task:** tumour segmentation to support spatial localisation and additional supervision.
- **Imaging modalities:** MRI and CT.
- **Data structure:** paired scans from the same patients.
- **Main architecture:** X-R-MiTNet --- Explainable Reliability-Aware Multimodal Imaging Transformer Network.
- **Fusion:** reliability-conditioned, bidirectional cross-modal attention.
- **Explainability:** CARE-XAI --- Cross-Modal Attribution and Reliability Explanation.
- **Robustness conditions:** MRI-only, CT-only, MRI+CT, and degraded-modality inputs.
- **Implementation:** PyTorch from scratch, without CNN layers or pretrained backbones.
- **Intended compute:** Google Colab or Kaggle GPU.
- **Publication target:** a suitable Scopus-indexed journal; acceptance is not guaranteed.

### Operational definition of early detection

Early detection means identifying a tumour from imaging evidence at an early disease stage, ideally before advanced progression. Three concepts must not be conflated:

1. **Early-stage detection:** cases have clinically or radiologically documented early-stage disease.
2. **Small-tumour detection:** cases have tumours meeting a predefined size or volume criterion.
3. **Early identification:** the model identifies a tumour before a diagnosis was established at the time of imaging.

A dataset containing ordinary tumour-positive images does not automatically support an early-detection claim. If clinical stage or diagnostic timeline is unavailable, the study should describe the task as *small-tumour detection* only if tumour size or volume can be measured reliably; otherwise, use *brain tumour detection*.

------------------------------------------------------------------------

## 2. Research problem

### 2.1 Background

MRI and CT provide different and potentially complementary information for brain imaging. Single-modality systems cannot exploit information from both modalities, while multimodal systems may be affected by differences in modality quality, availability, acquisition protocols and spatial alignment.

Deep-learning predictions can also be difficult to interpret. A detection score alone does not reveal which image regions influenced the prediction, how much each modality contributed, or whether the model relied on relevant image evidence.

This research proposes a CNN-free multimodal Vision Transformer that learns from paired MRI and CT scans, estimates modality reliability, and combines information through cross-modal attention. CARE-XAI will analyse spatial attribution, modality contribution, cross-modal interaction and counterfactual prediction changes.

### 2.2 Research problem statement

Existing brain tumour detection approaches face challenges including limited use of complementary information from multiple imaging modalities, sensitivity to missing or degraded imaging inputs, and insufficient interpretability of deep-learning predictions. Single-modality systems cannot fully exploit potentially complementary information in paired MRI and CT scans, while simple multimodal fusion approaches may not adequately account for differences in modality quality or availability.

There is also a need to investigate whether multimodal models can identify early-stage or small brain tumours reliably, rather than merely classify established tumour cases. In addition, automated predictions do not inherently reveal which anatomical regions or imaging modalities influenced a decision.

To address these challenges, this research proposes X-R-MiTNet, a CNN-free, reliability-aware multimodal Vision Transformer that integrates paired MRI and CT images through bidirectional cross-modal attention and reliability-conditioned fusion. The model will perform early brain tumour detection as its primary task, with tumour segmentation as an auxiliary task. CARE-XAI will investigate spatial feature attribution, modality contributions, cross-modal interactions and counterfactual prediction changes. The research will evaluate detection performance, segmentation quality, missing-modality robustness and explanation faithfulness using patient-level experimental protocols.

------------------------------------------------------------------------

## 3. Research aim and objectives

### 3.1 Overall aim

To develop and evaluate a CNN-free, explainable, reliability-aware multimodal Vision Transformer for early brain tumour detection using paired MRI and CT scans, with auxiliary tumour segmentation and robustness to missing imaging modalities.

### 3.2 Specific objectives

| ID | Objective | Expected outcome |
|---|---|---|
| RO1 | Identify and prepare public datasets containing paired MRI--CT scans with reliable patient-level diagnoses and early-tumour evidence. | Verified multimodal dataset |
| RO2 | Design a dual-branch CNN-free Vision Transformer with modality-specific encoders and bidirectional cross-modal attention. | X-R-MiTNet architecture |
| RO3 | Develop reliability-conditioned fusion to account for modality quality and availability. | Reliability-aware fusion module |
| RO4 | Develop an early tumour detection head and an auxiliary segmentation head. | Multi-task prediction model |
| RO5 | Integrate Attention Rollout, Integrated Gradients, grouped SHAP, counterfactual explanations and modality analysis. | CARE-XAI framework |
| RO6 | Compare the model with single-modality and conventional multimodal baselines. | Comparative results |
| RO7 | Assess robustness under missing and degraded MRI or CT inputs. | Missing-modality analysis |
| RO8 | Quantitatively evaluate explanation faithfulness, localisation and stability. | XAI evaluation results |

**Priority:** RO1 is the first critical objective. The architecture and claims depend on genuinely paired data and valid early-tumour labels.

### 3.3 Primary and secondary objectives

- **Primary objective --- Early brain tumour detection:** Determine whether paired MRI--CT fusion improves identification of early-stage or small tumours compared with MRI-only and CT-only models.
- **Secondary objective --- Auxiliary tumour segmentation:** Localise tumour regions to support spatial analysis and investigate whether segmentation supervision improves detection or explanation localisation.
- **Supporting objectives:**
  - Missing-modality robustness.
  - Reliability estimation.
  - Cross-modal interaction analysis.
  - Quantitative explainability.

------------------------------------------------------------------------

## 4. Research questions

- **RQ1 --- Multimodal detection performance:** Does reliability-aware fusion of paired MRI and CT images improve early brain tumour detection compared with single-modality Vision Transformer models?
- **RQ2 --- Cross-modal fusion:** Does bidirectional cross-modal attention provide better detection performance than simple feature concatenation or conventional multimodal fusion?
- **RQ3 --- Missing-modality robustness:** How does X-R-MiTNet perform when MRI or CT is unavailable or degraded, and can reliability-conditioned fusion reduce the associated performance loss?
- **RQ4 --- Auxiliary segmentation:** Does joint training with tumour segmentation improve early tumour detection or spatial localisation compared with detection-only training?
- **RQ5 --- Explainability:** Can CARE-XAI identify relevant tumour-associated image regions and quantify the individual and interacting contributions of MRI and CT to model predictions?
- **RQ6 --- Explanation reliability:** How faithful, stable and spatially meaningful are CARE-XAI explanations under different input-quality and modality-availability conditions?

### 4.1 Testable hypotheses

| Hypothesis | Expected result |
|---|---|
| H1 | Paired MRI--CT fusion improves early tumour detection compared with the best single-modality baseline. |
| H2 | Reliability-aware cross-modal fusion outperforms simple feature concatenation. |
| H3 | Missing-modality training improves performance when one modality is unavailable at inference. |
| H4 | Auxiliary segmentation improves detection or explanation localisation; improvement in detection is not assumed. |
| H5 | CARE-XAI provides quantitatively useful explanations according to predefined faithfulness, localisation and stability tests. |

------------------------------------------------------------------------

## 5. Dataset-selection criteria

A dataset may contain MRI images, CT images and tumour masks yet still be unsuitable if MRI and CT do not belong to the same patients, or if early-disease status is unknown.

### 5.1 Mandatory requirements

| Criterion | Requirement | Importance |
|---|---|---|
| MRI availability | Brain MRI images | Mandatory |
| CT availability | Brain CT images | Mandatory for paired multimodal experiments |
| Patient pairing | MRI and CT linked to the same patient | Mandatory |
| Diagnosis | Reliable tumour-positive and tumour-negative labels | Mandatory |
| Early-disease evidence | Early-stage label, tumour size/volume or documented disease status | Mandatory for a defensible early-detection claim |
| Patient identifiers | Consistent anonymised patient IDs | Mandatory |
| Image quality | Sufficient spatial resolution and diagnostic information | Mandatory |
| Data provenance | Public source and clear usage conditions | Mandatory |
| Segmentation masks | Tumour masks for a useful subset | Strongly preferred |
| Clinical metadata | Tumour type, stage, size, diagnosis date and scan date | Strongly preferred |
| Acquisition metadata | Scanner, sequence, contrast and acquisition details | Preferred |

### 5.2 Early-detection-specific requirements

- **Group A --- Early tumour cases:** confirmed tumours with independently supported early-stage or predefined small-tumour characteristics.
- **Group B --- Later-stage tumour cases:** larger or more advanced tumours, allowing analysis across disease extent.
- **Group C --- Tumour-negative controls:** patients without brain tumours according to a reliable reference diagnosis.

### 5.3 MRI--CT pairing requirements

1. MRI and CT belong to the same anonymised patient.
2. Scan dates and their temporal relationship are known.
3. Both scans represent a clinically comparable disease state.
4. Both images are suitable for the intended task.
5. The scans can be registered when spatially corresponding fusion or cross-modal spatial analysis is required.

### 5.4 Registration and alignment

1. DICOM orientation and coordinate consistency.
2. Image resampling to documented voxel spacing.
3. Rigid or affine registration where appropriate.
4. Registration quality assessment.
5. Consistent brain-region cropping.
6. Alignment of segmentation masks with reference images.

### 5.5 Segmentation-label requirements

- Tumour masks with clear mask-to-image correspondence.
- Consistent label definitions.
- Sufficient early-tumour cases with masks.
- Do not use an MRI mask as a CT ground-truth mask unless scans are registered and the mask has been transformed and verified.

### 5.6 Exclusion criteria

- Missing or inconsistent patient identifiers.
- Unverifiable MRI--CT pairing.
- Uncertain tumour diagnosis.
- Unclear early-stage or tumour-size labels.
- Severe image corruption or incomplete brain coverage.
- Unreliable segmentation masks.
- Duplicate scans or patients across data partitions.
- MRI and CT acquired so far apart that disease progression makes them unsuitable as a paired observation.

------------------------------------------------------------------------

## 6. Dataset quality assessment

| Criterion | Weight |
|---|---|
| Paired MRI--CT from the same patients | 25% |
| Reliable diagnosis and negative controls | 20% |
| Early-stage or tumour-size information | 20% |
| Patient identifiers and scan dates | 10% |
| Tumour segmentation masks | 10% |
| Image quality and acquisition metadata | 10% |
| Public access and clear usage terms | 5% |

`Weighted score = sum(weight × criterion score / 2)`

### Recommended dataset strategy

- **Primary development dataset:** Paired MRI--CT, diagnostic labels, early-case evidence, preferably masks (training, validation, internal testing).
- **External validation dataset:** Independent patients and comparable labels; paired modalities preferred.
- **Auxiliary segmentation dataset:** Reliable tumour masks, preferably from the primary cohort.

------------------------------------------------------------------------

## 7. Experimental population and data organisation

Unit of data separation: Patient-level partition (no slice leakage).

### Target directory structure

```text
Brain_Tumour_Research/
├── dataset/
│   ├── patient_001/
│   │   ├── MRI/
│   │   │   ├── image.nii.gz
│   │   │   └── metadata.json
│   │   ├── CT/
│   │   │   ├── image.nii.gz
│   │   │   └── metadata.json
│   │   ├── masks/
│   │   │   └── tumour_mask.nii.gz
│   │   └── labels.json
│   └── patient_002/
├── splits/
│   ├── train_patients.csv
│   ├── validation_patients.csv
│   └── test_patients.csv
├── preprocessing/
├── models/
├── explainability/
│   ├── attention_rollout/
│   ├── integrated_gradients/
│   ├── shap/
│   └── counterfactuals/
├── evaluation/
└── reports/
```

### Data splitting

- Patient-level split: 70% train, 15% validation, 15% test (or cross-validation if cohort is small).
- Stratify by diagnosis, early-case status, and tumour type.
- Never use the test set for thresholding, XAI baselines, or hyperparameter tuning.

------------------------------------------------------------------------

## 8. Step 1 completion and go/no-go conditions

### Go/No-Go Condition

> **CRITICAL:** If no suitable public dataset contains genuinely paired MRI--CT scans and defensible early-disease labels, revise the scope before training. It is scientifically unsound to train on unpaired MRI and CT images and claim patient-specific multimodal fusion.

------------------------------------------------------------------------

## 9. Next step: dataset discovery

Systematic search and verification of public datasets using the 10-point dataset comparison matrix.

