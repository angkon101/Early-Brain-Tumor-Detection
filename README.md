# Early Brain Tumor Detection

An automated computer vision and deep learning framework for the early detection and classification of brain tumors from magnetic resonance imaging (MRI) scans.

## Overview

Early detection of brain tumors is vital for effective clinical diagnosis and treatment planning. This repository hosts deep learning models and data preprocessing pipelines designed to detect, classify (e.g., Glioma, Meningioma, Pituitary, or No Tumor), and segment brain tumor regions from MRI modalities (T1, T1-CE, T2, FLAIR).

## Project Structure

```text
├── data/                  # Dataset storage and documentation
│   └── raw/               # Raw MRI images (git-ignored)
├── notebooks/             # Exploratory data analysis and experimental notebooks
├── src/                   # Source code modules
│   ├── data/              # Data loaders and preprocessing pipelines
│   ├── models/            # Neural network architectures
│   └── utils/             # Evaluation metrics and visualization helpers
├── .gitignore             # Standard ML and Python exclusions
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

## Getting Started

### 1. Clone & Setup Environment

```bash
git clone https://github.com/angkon101/Early-Brain-Tumor-Detection.git
cd Early-Brain-Tumor-Detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place MRI datasets (such as the Brain Tumor MRI Dataset or BraTS) into the `data/` directory according to the instructions in `data/README.md`.

## License

This project is licensed under the MIT License.
