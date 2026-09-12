# IMUSA: Indic Meme Understanding & Sentiment Analysis

> **Official Repository for the FIRE 2026 Shared Task on Multimodal Punjabi Meme Sentiment Classification**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d6ttoeYVnHMZ48-XVz6C-tagsXT47qUf?usp=sharing)
[![Code Quality](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Abstract

**IMUSA** (Indic Meme Understanding & Sentiment Analysis) addresses the FIRE 2026 challenge of classifying highly imbalanced, multimodal Punjabi internet memes into four sentiment categories: **Sarcasm**, **Neutral**, **Offensive**, and **Motivational**. 

This repository implements our **V2 Architecture**, a dual-stream multimodal pipeline that leverages state-of-the-art vision and language models fused via a Gated Multimodal Unit (GMU), trained with a Label-Smoothed $\alpha$-Balanced Focal Loss to mitigate severe class imbalance. We achieve robust generalization through Stratified 5-Fold Cross-Validation, Two-Stage Linear Probing + Fine-Tuning (LP-FT), and post-hoc threshold calibration via Nelder-Mead optimization.

For a rigorous theoretical deep dive into our methodology, see the complete [Academic Research Paper (`docs/paper.md`)](./docs/paper.md).

---

## 🚀 Key Architectural Innovations

### 1. Dual-Stream Feature Extraction
- **Vision Backbone**: `google/vit-base-patch16-224` (Vision Transformer).
- **Text Backbone**: `google/muril-base-cased` (Multilingual Representations for Indian Languages, explicitly robust on Gurmukhi script).

### 2. Gated Multimodal Fusion (GMU)
Instead of naive concatenation, we employ a **Gated Multimodal Unit**. The gate vector $\mathbf{z}$ modulates the visual representation $\mathbf{h}_v$ and the linguistic representation $\mathbf{h}_t$, allowing the model to dynamically prioritize the most informative modality for a given meme.

### 3. $\alpha$-Balanced Focal Loss with Label Smoothing
To counteract the dataset's extreme class imbalance, we optimize a modified Focal Loss ($\gamma = 2.0$) combined with class-specific $\alpha$-weights and label smoothing ($\epsilon = 0.1$). This prevents the dominant 'Sarcasm' class from overwhelming the gradients.

### 4. Post-Hoc Threshold Calibration (Nelder-Mead)
We utilize derivative-free **Nelder-Mead optimization** on out-of-fold (OOF) validation logits to discover an optimal decision threshold vector $\boldsymbol{\tau}^*$. This threshold scaling directly maximizes the discrete Macro F1 score, providing a significant performance boost over standard argmax inference.

---

## 📊 V2 Empirical Results

Our ensemble model was evaluated via rigorous Stratified 5-Fold Cross-Validation on the provided Punjabi meme dataset. 

| Metric | Cross-Validation Score |
| :--- | :--- |
| **Mean Accuracy** | $60.53\% \pm 1.79\%$ |
| **Uncalibrated Macro F1** | $0.4548$ |
| **Calibrated Macro F1** | $\mathbf{0.4630}$ |

*Post-hoc Nelder-Mead calibration yielded an absolute Macro F1 improvement of **+0.83%**.*

### Optimal Decision Thresholds ($\boldsymbol{\tau}^*$)
The Nelder-Mead optimization discovered the following class-specific logit scaling thresholds:
- `[1.0319889, 0.8517324, 1.0506044, 1.1507181]`

### Test Set Prediction Distribution (N=500)
| Sarcasm | Neutral | Motivational | Offensive |
| :---: | :---: | :---: | :---: |
| 374 | 87 | 37 | 2 |

---

## 💻 Reproducibility & Code Run-books

Due to the heavy compute requirements of ViT + MuRIL fine-tuning, the official training pipeline is executed via Google Colab Pro instances.

### Training Colab Notebooks
We provide three reproducible notebooks that document the full 5-Fold Stratified CV process, training, OOF generation, and final ensemble inference:

1. **[Folds 0 & 1 Training Pipeline](https://colab.research.google.com/drive/12Y7JOXljUUjnqbpdD2dLOkmB91gjSCc_?usp=sharing)**
2. **[Folds 2 & 3 Training Pipeline](https://colab.research.google.com/drive/1C1YY5cxF_s1G60ddmG5OdyzUjpR-o2YI?usp=sharing)**
3. **[Fold 4, Nelder-Mead Calibration, & Ensemble Inference](https://colab.research.google.com/drive/1d6ttoeYVnHMZ48-XVz6C-tagsXT47qUf?usp=sharing)**

### Local Workspace Setup (For Development)

If you wish to run the data pipelines, tests, or develop locally:

**Prerequisites:**
- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

```bash
# Clone the repository
git clone https://github.com/BlackKnight05/imusa-multimodal-sentiment.git
cd imusa-multimodal-sentiment

# Synchronize all workspace dependencies & create virtual environment
make install

# Install pre-commit quality hooks
make setup-hooks
```

---

## 🛠 Monorepo Architecture

This repository operates as a production-grade ML ecosystem.

```text
multimodal-ai-project/
├── apps/                 # Application Entrypoints (API, Web)
├── libs/imusa/           # Core Python Package (PyTorch Models, Datasets, Config)
│   ├── src/imusa/
│   │   ├── models/       # ViT, MuRIL, GMU Fusion Definitions
│   │   ├── training/     # PyTorch DDP Trainers & Loss Functions
│   │   ├── data/         # Stratified K-Fold & Torch Datasets
│   │   └── evaluation/   # Nelder-Mead Calibration & Metrics
│   └── tests/            # Comprehensive Pytest Suite
├── notebooks/            # Executable Colab Notebooks for V2 Pipeline
├── docs/                 # Academic Research Paper (paper.md)
└── scripts/              # CLI Entrypoints (Data Cleaning, EDA)
```

### Data Pipeline Commands
```bash
# 1. Clean the raw CSV dataset (Handles multi-line Gurmukhi text)
make clean-data

# 2. Generate Exploratory Data Analysis (EDA) reports and distributions
make explore
```

### Quality Assurance
The codebase strictly adheres to Staff-Engineer quality standards.
```bash
# Code Linting & Strict Type Checking (Ruff + Mypy)
make lint

# Auto-format codebase
make format

# Run test suite with coverage
make test
```

---

## 📜 Citation

If you use this codebase or our methodology in your research, please refer to the detailed academic formulation in `docs/paper.md` and cite the repository.

---
*Developed for the IMUSA Shared Task at FIRE 2026.*
