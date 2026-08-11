# Indic Meme Understanding & Sentiment Analysis (IMUSA): Multimodal Punjabi Sentiment Classification

## Abstract
This document synthesizes the experimental methodologies, architectural decisions, data engineering pipelines, empirical findings, and literature context for the **Indic Meme Understanding & Sentiment Analysis (IMUSA)** shared task at FIRE 2026. The primary objective is four-class multimodal sentiment classification (`Sarcasm`, `Motivational`, `Neutral`, `Offensive`) of social media memes in Punjabi (Gurmukhi script).

---

## 1. Task Definition & Problem Formulation

### 1.1 Modalities & Objective
- **Input Modality 1 (Text)**: Extracted Gurmukhi script text string $T$.
- **Input Modality 2 (Vision)**: RGB meme image $V \in \mathbb{R}^{H \times W \times 3}$.
- **Output Target**: Discrete class label $y \in \{\text{Sarcasm}, \text{Motivational}, \text{Neutral}, \text{Offensive}\}$.

### 1.2 Mathematical Formulation
Let $\mathcal{D} = \{(V_i, T_i, y_i)\}_{i=1}^N$ denote the dataset. The multimodal model estimates the class probability distribution:

$$
P(y \mid V, T; \Theta) = \text{softmax}(W \cdot f_{\text{fusion}}(e_v(V), e_t(T)) + b)
$$

where $e_v(V)$ is the visual representation, $e_t(T)$ is the textual embedding, and $f_{\text{fusion}}$ integrates both representations.

---

## 2. Dataset Engineering & Exploratory Data Analysis (EDA)

### 2.1 Cleaning Pipeline & Audit
The raw dataset contains 3,002 entries. The cleaning pipeline (`imusa.data.cleaning`) performs three deterministic sanitization steps:
1. **Raw CSV Parsing**: Handles embedded multiline strings in Gurmukhi text fields using quoting rules.
2. **File Resolution & Normalization**: Maps raw IDs (including 10 samples missing `.jpg` extensions like `image_punjabi_1171`) to actual on-disk files. All 3,002 image files were verified to exist.
3. **Deduplication**: Identifies and eliminates 111 duplicate `(Category, Text)` combinations.

| Pipeline Stage | Sample Count | Percentage of Raw Data |
|---|---|---|
| **Raw CSV Entries** | 3,002 | 100.0% |
| **Missing Image Resolution** | 0 dropped (10 normalized) | 0.0% |
| **Duplicates Removed** | 111 dropped | 3.7% |
| **Final Clean Dataset** | **2,891** | **96.3%** |

### 2.2 Class Distribution & Severe Imbalance

![IMUSA Dataset Sentiment Class Distribution](assets/class_distribution.png)

```
Category Breakdown:
  Sarcasm:      1,274 samples (44.07%)
  Motivational:   836 samples (28.92%)
  Neutral:        730 samples (25.25%)
  Offensive:       51 samples ( 1.76%)
  Total:        2,891 samples (100.0%)
```

#### Analytical Impact of the 25:1 Class Imbalance
The imbalance ratio between the majority class (`Sarcasm`) and minority class (`Offensive`) is **24.98 : 1**.
- **Theoretical Failure Mode**: Under standard cross-entropy loss $\mathcal{L} = -\sum y_i \log \hat{y}_i$, gradients dominated by `Sarcasm` will collapse minority class predictions to zero. A naive model predicting `Sarcasm` for all samples achieves **44.07% accuracy** while yielding **0.0 recall on `Offensive`**.
- **Mitigation Strategy (Phase 2)**:
  - Inverse Class-Frequency Weighting:
    $$
    w_c = \frac{N}{K \cdot N_c}
    $$
  - Focal Loss $\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with dynamic focal scaling parameter $\gamma = 2.0$.
  - Oversampling / Cross-modal Data Augmentation for `Offensive` samples.

### 2.3 Text Length Dynamics

![Punjabi Text Word Count per Sentiment Category](assets/text_length_distribution.png)

- **Median Word Count**: ~15 words across all categories.
- **Outliers**: Sequences extending up to 72 words (long poetry/quotes).
- **Finding**: Text length distributions are nearly identical across categories. Consequently, visual features and deep semantic embeddings are strictly required to differentiate sentiment.

### 2.4 Visual Aspect Ratio & Dimensions

![Meme Image Dimensions Scatter Plot](assets/image_resolution_distribution.png)

- **Resolution Range**: Width: 200px to 750px; Height: 200px to 1,500px.
- **Preprocessing Requirement**: Standardized bilinear resizing to $224 \times 224$ pixels with ImageNet normalization parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 2.5 Qualitative Multimodal Samples

![Sample Memes Grid per Category](assets/sample_meme_grid.png)

---

## 3. System Architecture & Methodology

### 3.1 Dual-Encoder Feature Extraction
The multimodal architecture leverages pre-trained deep transformers for both modalities:
1. **Vision Encoder $e_v$**: Given meme image $V \in \mathbb{R}^{3 \times 224 \times 224}$, `google/vit-base-patch16-224` extracts 196 patch tokens of dimension 768. The pooled `[CLS]` visual embedding is $\mathbf{h}_v \in \mathbb{R}^{768}$.
2. **Text Encoder $e_t$**: Given Gurmukhi token sequence $T$, `xlm-roberta-base` processes sequence embeddings to yield textual pooled representation $\mathbf{h}_t \in \mathbb{R}^{768}$.

### 3.2 Gated Multimodal Fusion Module
To dynamically balance textual and visual evidence, representations are fused via a learned gating mechanism:

$$
\mathbf{x}_{\text{concat}} = [\mathbf{h}_v \;;\; \mathbf{h}_t] \in \mathbb{R}^{1536}
$$

$$
\mathbf{g} = \sigma(W_g \mathbf{x}_{\text{concat}} + \mathbf{b}_g) \in (0, 1)^{1536}
$$

$$
\mathbf{h}_{\text{fused}} = \text{GELU}\left(\text{LayerNorm}(W_f (\mathbf{g} \odot \mathbf{x}_{\text{concat}}) + \mathbf{b}_f)\right) \in \mathbb{R}^{1536}
$$

### 3.3 Classification Head & Objective Function
Logits are computed via a multi-layer perceptron:

$$
\mathbf{z} = W_2 \left(\text{Dropout}\left(\text{GELU}\left(\text{LayerNorm}(W_1 \mathbf{h}_{\text{fused}} + \mathbf{b}_1)\right)\right)\right) + \mathbf{b}_2 \in \mathbb{R}^4
$$

To resolve the 25:1 imbalance, optimization uses Focal Loss with balanced class weights $\alpha_c = \frac{N}{K \cdot N_c}$:

$$
\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^K \alpha_c (1 - p_c)^\gamma y_c \log(p_c)
$$

where $p_c = \text{softmax}(\mathbf{z})_c$ and $\gamma = 2.0$.

---

## 4. Experimental Setup & Reproducibility

### 4.1 Environment & Tooling
- **Python Version**: 3.12 (pinned via `.python-version`)
- **Package Manager**: `uv` workspaces
- **Code Quality**: `ruff` (linting/formatting), `mypy` (strict static typing)
- **Test Framework**: `pytest` (unit testing with coverage tracking)

---

## 5. References & Literature Review

1. **IMUSA Shared Task @ FIRE 2026**: Indic Meme Understanding & Sentiment Analysis Guidelines.
2. **XLM-RoBERTa**: Conneau et al. (2020), *"Unsupervised Cross-lingual Representation Learning at Scale"*.
3. **Vision Transformer (ViT)**: Dosovitskiy et al. (2021), *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*.
4. **Focal Loss**: Lin et al. (2017), *"Focal Loss for Dense Object Detection"*.
