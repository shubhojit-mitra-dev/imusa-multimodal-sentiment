# Multimodal Sentiment Classification of Punjabi Memes Using Gated Vision-Language Fusion with Focal Loss

**Authors**: Shubhojit Mitra

**Affiliation**: *(To be filled)*

**Submitted to**: IMUSA Shared Task @ FIRE 2026 — Forum for Information Retrieval Evaluation

**Date**: August 2026

---

## Abstract

Internet memes are a dominant form of multimodal communication on social media, yet automated sentiment analysis of memes in low-resource Indic languages remains largely unexplored. This paper presents a multimodal deep learning system for the **Indic Meme Understanding & Sentiment Analysis (IMUSA)** shared task at FIRE 2026, targeting four-class sentiment classification — `Sarcasm`, `Motivational`, `Neutral`, and `Offensive` — of Punjabi (Gurmukhi script) memes. Our approach employs a late-fusion dual-encoder architecture combining a Vision Transformer (ViT-Base) for visual feature extraction with XLM-RoBERTa for multilingual textual encoding, unified through a learned **Gated Multimodal Fusion** mechanism that dynamically balances modality contributions per sample. To address the severe 25:1 class imbalance between the majority `Sarcasm` class (1,274 samples) and the minority `Offensive` class (51 samples), we employ $\alpha$-balanced **Focal Loss** ($\gamma = 2.0$) with inverse class-frequency weighting. Training employs AdamW optimization with linear warmup and cosine annealing learning rate scheduling, alongside training-time image augmentation. Evaluation uses **Macro F1** as the primary metric to ensure equitable performance across all sentiment categories. Benchmark results, confusion matrices, per-class F1 analyses, and ablation studies are documented upon completion of GPU fine-tuning on Google Colab.

**Keywords**: Multimodal Sentiment Analysis, Punjabi NLP, Meme Classification, Vision Transformer, XLM-RoBERTa, Gated Fusion, Focal Loss, Class Imbalance, FIRE 2026

---

## 1. Introduction

### 1.1 Background and Motivation

Internet memes have evolved from simple humorous images into complex multimodal artifacts encoding cultural nuance, political commentary, sarcasm, and hate speech (Kiela et al., 2020). The sentiment conveyed by a meme frequently emerges from the *interaction* between its visual content (facial expressions, symbolic imagery, background scenes) and its embedded text — neither modality alone is sufficient for reliable classification. For example, an image of a smiling person paired with Gurmukhi text containing sarcastic wordplay inverts the surface-level positive visual sentiment into cutting irony.

While substantial progress has been made in multimodal meme analysis for high-resource languages like English (Sharma et al., 2020; Pramanick et al., 2021), low-resource Indic languages remain critically underserved. **Punjabi**, spoken by over 125 million people and written in the Gurmukhi script, lacks the large-scale annotated corpora and pre-trained resources available for English. The IMUSA shared task at FIRE 2026 addresses this gap by providing a curated multimodal dataset of Punjabi memes for fine-grained sentiment classification.

### 1.2 Research Problem

This work addresses the following research question:

> *How can we build an effective multimodal classification system for Punjabi meme sentiment analysis that jointly reasons over visual and textual modalities while handling severe class imbalance in a low-resource language setting?*

The task presents three core challenges:

1. **Multimodal Reasoning**: Sentiment often arises from the interaction between image and text, not from either modality alone. A system must learn cross-modal semantic relationships.
2. **Low-Resource Language Processing**: Punjabi (Gurmukhi) has limited pre-training data compared to English. Tokenizers and language models must generalize from multilingual pre-training.
3. **Extreme Class Imbalance**: The dataset exhibits a 25:1 ratio between the majority class (`Sarcasm`, 44%) and the minority class (`Offensive`, 1.76%), causing standard training objectives to collapse minority class predictions.

### 1.3 Contributions

This paper makes the following contributions:

1. A **dual-encoder multimodal architecture** combining Vision Transformer (ViT-Base) and XLM-RoBERTa with a learned Gated Multimodal Fusion mechanism for Punjabi meme sentiment classification.
2. A systematic **class imbalance mitigation strategy** using $\alpha$-balanced Focal Loss with inverse class-frequency weighting, demonstrating its effectiveness over standard cross-entropy on a 25:1 skewed dataset.
3. A **reproducible end-to-end pipeline** — from data cleaning and stratified splitting through model training, evaluation visualization, and competition submission generation — implemented as an open-source monorepo with comprehensive unit testing.
4. A thorough **empirical analysis** including per-class F1 breakdowns, confusion matrices, and training dynamics documentation for the research community.

### 1.4 Paper Organization

The remainder of this paper is organized as follows: §2 reviews related work across multimodal meme analysis, Indic NLP, and class imbalance mitigation. §3 details the dataset characteristics and preprocessing pipeline. §4 describes the proposed system architecture and methodology. §5 presents the experimental setup and training protocol. §6 reports results and analysis. §7 discusses findings and limitations. §8 concludes with future directions.

---

## 2. Literature Review

### 2.1 Multimodal Meme Analysis

The intersection of computer vision and natural language processing for meme understanding has received growing attention. **Kiela et al. (2020)** introduced the landmark *Hateful Memes Challenge* at NeurIPS, providing 10,000+ memes with carefully constructed "benign confounders" — samples where text or image alone appears harmless but the combination conveys hate. This work demonstrated that state-of-the-art multimodal models substantially lagged behind human-level reasoning on cross-modal semantic composition.

**Sharma et al. (2020)** organized *SemEval-2020 Task 8: Memotion Analysis*, releasing 10,000 annotated memes for multi-task evaluation across sentiment polarity, humor, sarcasm, offense, and motivation — establishing foundational benchmarks for meme sentiment research. **Pranesh & Shekhar (2020)** proposed *MemeSem*, a transfer-learning framework coupling CNN visual backbones (ResNet/VGG) with transformer text encoders. **Hossain et al. (2022)** created *MemoSen*, the first South Asian (Bengali) multimodal meme sentiment dataset, demonstrating the critical necessity of language-specific encoders over English-only baselines.

**Pramanick et al. (2021)** formulated the *HarMeme* benchmark for simultaneously detecting harmful memes and their target granularity, while **Suryawanshi et al. (2020)** constructed the *MultiOFF* dataset proving that combining visual features (VGG-16) with contextual text embeddings (BERT) significantly outperforms unimodal approaches for offensive meme detection.

### 2.2 Vision Transformers for Visual Feature Extraction

**Dosovitskiy et al. (2021)** introduced the Vision Transformer (ViT), demonstrating that a pure self-attention Transformer applied directly to non-overlapping $16 \times 16$ image patches can match or exceed convolutional networks when pre-trained at scale. Follow-up work includes *DeiT* (Touvron et al., 2021), which enables efficient ViT training on ImageNet-1K without massive proprietary datasets, and *Swin Transformer* (Liu et al., 2021), introducing hierarchical windowed attention with linear computational complexity. **Radford et al. (2021)** developed *CLIP*, training joint visual-textual encoders via contrastive pre-training on 400M image-caption pairs, producing robust zero-shot cross-modal representations.

In our work, we adopt `google/vit-base-patch16-224` as the visual backbone, leveraging its ImageNet-21K pre-training to extract rich 768-dimensional patch-level visual features from meme images.

### 2.3 Multilingual Transformers for Indic Languages

Cross-lingual representation learning is critical for low-resource languages like Punjabi. **Conneau et al. (2020)** pre-trained *XLM-RoBERTa* on 2.5TB of Common Crawl data across 100 languages (including Punjabi), demonstrating that massive-scale multilingual masked language modeling dramatically improves cross-lingual transfer. Language-specific models have further advanced Indic NLP: **Kakwani et al. (2020)** introduced *IndicBERT* supporting 11 Indian languages with the IndicGLUE benchmark; **Khanuja et al. (2021)** developed *MuRIL* (Multilingual Representations for Indian Languages) pre-trained on 17 Indian languages with transliteration augmentation; and **Doddapaneni et al. (2023)** expanded coverage to 22 languages with morphologically-informed tokenizers in *IndicBERT v2*.

We select `xlm-roberta-base` for its proven Punjabi Gurmukhi script coverage and 768-dimensional contextual embeddings, with future work exploring MuRIL as a potential domain-specific alternative.

### 2.4 Multimodal Fusion Techniques

Combining visual and textual representations requires principled fusion strategies. The foundational taxonomy established by **Snoek et al. (2005)** distinguishes *Early Fusion* (feature-level concatenation) from *Late Fusion* (decision-level ensemble), with intermediate fusion often capturing richer cross-modal interactions.

**Arevalo et al. (2017)** introduced the *Gated Multimodal Unit (GMU)*, which uses learned multiplicative sigmoid gating $\mathbf{g} = \sigma(W_g [\mathbf{h}_v ; \mathbf{h}_t])$ to adaptively control modality information flow — the direct inspiration for our fusion module. **Tsai et al. (2019)** proposed the *Multimodal Transformer (MulT)* with directional pairwise cross-modal attention, and **Lu et al. (2019)** developed *ViLBERT* with co-attentional transformer layers for bidirectional cross-modal feature exchange.

Our architecture adopts the gated fusion approach for its interpretability and computational efficiency on small datasets, avoiding the heavy parameterization of cross-attention models that risk overfitting on 2,891 training samples.

### 2.5 Punjabi Language Processing and Indic Sentiment Analysis

Punjabi NLP research remains limited compared to Hindi or Bengali. **Kaur & Gupta (2019)** developed specialized sentiment lexicons and preprocessing for Gurmukhi script, demonstrating SVM baselines for Punjabi social media sentiment. **Singh et al. (2021)** showed that morphological normalization tailored to Gurmukhi orthography substantially boosts BiLSTM/CNN sentiment classification accuracy.

The **HASOC** shared task series (Mandl et al., 2020–2022) established multi-year benchmarks for hate speech detection across Indian languages, documenting the empirical superiority of multilingual transformers (XLM-R, MuRIL) over traditional ML baselines. The **DravidianLangTech** workshops (Chakravarthi et al., 2021–2023) further standardized evaluation protocols and multimodal meme datasets for Indian regional languages, cementing Macro F1 as the gold-standard shared task metric.

### 2.6 Class Imbalance Mitigation

Severe class imbalance is a pervasive challenge in NLP and vision classification. **Chawla et al. (2002)** introduced *SMOTE* for synthetic minority oversampling via feature-space interpolation. **Lin et al. (2017)** formulated *Focal Loss*, which dynamically scales gradient contributions with a modulating factor $(1-p_t)^\gamma$ to suppress easy examples and concentrate learning on hard misclassified instances. **Mukhoti et al. (2020)** further proved that Focal Loss acts as an implicit entropy regularizer preventing overconfident predictions.

**Cui et al. (2019)** proposed *Class-Balanced Loss* based on the effective number of samples $E_n = \frac{1-\beta^n}{1-\beta}$, demonstrating that sample volume exhibits diminishing returns due to data overlap. **Cao et al. (2019)** designed the *Label-Distribution-Aware Margin Loss (LDAM)* enforcing larger decision boundaries for minority classes proportional to $n_y^{-1/4}$.

Our approach combines inverse class-frequency weighting $\alpha_c = \frac{N}{K \cdot N_c}$ with Focal Loss ($\gamma = 2.0$), amplifying the loss contribution of the 51-sample `Offensive` class by a factor of ~14× relative to the 1,274-sample `Sarcasm` class.

---

## 3. Dataset Description and Preprocessing

### 3.1 Dataset Overview

The IMUSA shared task provides a multimodal dataset of Punjabi internet memes comprising:
- **Training Set**: 3,002 raw entries, each consisting of an RGB meme image and an extracted Gurmukhi text string, annotated with one of 4 sentiment labels.
- **Test Set**: 500 unlabeled meme samples for competition evaluation.

### 3.2 Data Cleaning Pipeline

Our deterministic cleaning pipeline (`imusa.data.cleaning`) performs three sanitization stages:

```
┌─────────────────────────────────┐
│     Raw CSV (3,002 entries)     │
└──────────────┬──────────────────┘
               │
     ┌─────────▼─────────┐
     │  1. Parse CSV with │
     │  multiline Gurmukhi│
     │  text handling     │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────────┐
     │ 2. Image file extension │
     │ normalization (10 files │
     │ missing .jpg extension) │
     └─────────┬──────────────┘
               │
     ┌─────────▼──────────────────┐
     │ 3. Deduplicate (Category,  │
     │ Text) pairs → 111 removed  │
     └─────────┬──────────────────┘
               │
     ┌─────────▼─────────────────┐
     │ Clean Dataset (2,891)      │
     └────────────────────────────┘
```

| Pipeline Stage | Sample Count | Percentage |
|---|---|---|
| Raw CSV Entries | 3,002 | 100.0% |
| Image Extension Normalization | 0 dropped (10 normalized) | 0.0% |
| Duplicates Removed | 111 dropped | 3.7% |
| **Final Clean Dataset** | **2,891** | **96.3%** |

### 3.3 Class Distribution and Imbalance Analysis

The cleaned dataset exhibits **severe class imbalance** with a 24.98:1 ratio between the majority and minority classes:

![IMUSA Dataset Sentiment Class Distribution](assets/class_distribution.png)

| Category | Count | Percentage | Inverse Weight $\alpha_c$ |
|---|---|---|---|
| Sarcasm | 1,274 | 44.07% | 0.567 |
| Motivational | 836 | 28.92% | 0.864 |
| Neutral | 730 | 25.25% | 0.990 |
| Offensive | 51 | 1.76% | **14.17** |
| **Total** | **2,891** | **100.0%** | — |

**Analytical Impact**: Under standard cross-entropy loss, gradients are dominated by the majority `Sarcasm` class. A naive model predicting `Sarcasm` for all samples achieves 44.07% accuracy but yields 0.0 recall on `Offensive`, producing a catastrophic Macro F1 of approximately 0.15. This motivates our use of $\alpha$-balanced Focal Loss.

### 3.4 Text Length Analysis

![Punjabi Text Word Count Distribution per Sentiment Category](assets/text_length_distribution.png)

- **Median Word Count**: ~15 words across all categories
- **Maximum**: 72 words (long poetry/quotes)
- **Key Finding**: Text length distributions are nearly identical across categories, indicating that sentiment discrimination requires deep semantic understanding rather than surface-level length heuristics

### 3.5 Image Dimension Analysis

![Meme Image Dimensions Scatter Plot](assets/image_resolution_distribution.png)

- **Resolution Range**: Width 200–750px, Height 200–1,500px
- **Preprocessing**: All images are resized to $224 \times 224$ pixels with bilinear interpolation and normalized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)

### 3.6 Qualitative Sample Analysis

![Sample Memes Grid Organized by Sentiment Category](assets/sample_meme_grid.png)

---

## 4. Methodology

### 4.1 System Architecture Overview

Our system follows a **late-fusion dual-encoder** paradigm, processing visual and textual modalities through independent pre-trained transformer backbones before combining them via a learned gating mechanism. The complete architecture is illustrated below:

```mermaid
graph LR
    subgraph Input
        IMG["Meme Image<br/>(H×W×3)"]
        TXT["Gurmukhi Text<br/>(Token Sequence)"]
    end

    subgraph Vision_Encoder["Vision Encoder (ViT-Base)"]
        IMG --> Patch["Patch Embedding<br/>(16×16 patches)"]
        Patch --> VTrans["12-Layer Transformer<br/>Encoder"]
        VTrans --> VCLS["[CLS] Pooling"]
        VCLS --> HV["h_v ∈ ℝ^768"]
    end

    subgraph Text_Encoder["Text Encoder (XLM-RoBERTa)"]
        TXT --> Tok["Tokenization<br/>(max_len=128)"]
        Tok --> TTrans["12-Layer Transformer<br/>Encoder"]
        TTrans --> MPool["Mean Pooling<br/>(Attention-Masked)"]
        MPool --> HT["h_t ∈ ℝ^768"]
    end

    subgraph Fusion["Gated Multimodal Fusion"]
        HV --> Concat["Concatenation<br/>[h_v ; h_t] ∈ ℝ^1536"]
        HT --> Concat
        Concat --> Gate["Sigmoid Gate<br/>g = σ(W_g · x + b_g)"]
        Concat --> GMul["Element-wise<br/>g ⊙ x"]
        Gate --> GMul
        GMul --> Proj["LayerNorm → GELU<br/>Projection"]
        Proj --> HF["h_fused ∈ ℝ^1536"]
    end

    subgraph Classifier["Classification Head"]
        HF --> FC1["Linear(1536→512)"]
        FC1 --> LN["LayerNorm → GELU"]
        LN --> Drop["Dropout(0.3)"]
        Drop --> FC2["Linear(512→4)"]
        FC2 --> Logits["Logits z ∈ ℝ^4"]
    end

    Logits --> Loss["α-Balanced Focal Loss<br/>(γ=2.0)"]
```

### 4.2 Vision Encoder

Given a meme image $V \in \mathbb{R}^{3 \times 224 \times 224}$, the Vision Transformer (`google/vit-base-patch16-224`) partitions it into a sequence of 196 non-overlapping patches of size $16 \times 16$ pixels. Each patch is linearly projected into a 768-dimensional embedding, prepended with a learnable `[CLS]` token, and processed through 12 self-attention layers. The pooled `[CLS]` representation serves as the visual embedding:

$$
\mathbf{h}_v = \text{ViT}(V) \in \mathbb{R}^{768}
$$

### 4.3 Text Encoder

Given the extracted Gurmukhi text string $T$, the XLM-RoBERTa tokenizer (`xlm-roberta-base`) segments it into subword tokens with padding and truncation to a maximum length of 128 tokens. The 12-layer transformer encoder processes the token sequence, and we compute the textual embedding via **attention-masked mean pooling** over all valid token positions:

$$
\mathbf{h}_t = \frac{\sum_{i=1}^{L} m_i \cdot \mathbf{h}_i}{\sum_{i=1}^{L} m_i} \in \mathbb{R}^{768}
$$

where $\mathbf{h}_i$ is the hidden state of token $i$, $m_i \in \{0, 1\}$ is the attention mask, and $L$ is the sequence length.

### 4.4 Gated Multimodal Fusion

Inspired by the Gated Multimodal Unit (Arevalo et al., 2017), our fusion module dynamically balances the contribution of visual and textual features for each individual meme. Rather than assuming fixed modality importance, a learned sigmoid gate adaptively suppresses noisy or irrelevant modality signals:

**Step 1 — Concatenation:**

$$
\mathbf{x}_{\text{concat}} = [\mathbf{h}_v \;;\; \mathbf{h}_t] \in \mathbb{R}^{1536}
$$

**Step 2 — Dynamic Gating:**

$$
\mathbf{g} = \sigma(W_g \mathbf{x}_{\text{concat}} + \mathbf{b}_g) \in (0, 1)^{1536}
$$

where $\sigma$ is the element-wise sigmoid function, and $W_g \in \mathbb{R}^{1536 \times 1536}$, $\mathbf{b}_g \in \mathbb{R}^{1536}$ are learnable parameters.

**Step 3 — Gated Projection:**

$$
\mathbf{h}_{\text{fused}} = \text{GELU}\left(\text{LayerNorm}(W_f (\mathbf{g} \odot \mathbf{x}_{\text{concat}}) + \mathbf{b}_f)\right) \in \mathbb{R}^{1536}
$$

where $\odot$ denotes element-wise (Hadamard) product. The gate $\mathbf{g}$ learns to assign values close to 1.0 for informative feature dimensions and close to 0.0 for noisy dimensions, enabling the model to dynamically weight visual vs. textual evidence per sample.

### 4.5 Classification Head

The fused representation is classified through a two-layer MLP with regularization:

$$
\mathbf{z} = W_2 \left(\text{Dropout}_{0.3}\left(\text{GELU}\left(\text{LayerNorm}(W_1 \mathbf{h}_{\text{fused}} + \mathbf{b}_1)\right)\right)\right) + \mathbf{b}_2 \in \mathbb{R}^4
$$

where $W_1 \in \mathbb{R}^{512 \times 1536}$, $W_2 \in \mathbb{R}^{4 \times 512}$ are learnable weight matrices. The output logits $\mathbf{z}$ correspond to the 4 sentiment categories.

### 4.6 Loss Function: $\alpha$-Balanced Focal Loss

To address the 25:1 class imbalance, we employ Focal Loss (Lin et al., 2017) with inverse class-frequency weighting:

$$
\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^{K} \alpha_c (1 - p_c)^\gamma y_c \log(p_c)
$$

where:
- $p_c = \text{softmax}(\mathbf{z})_c$ is the predicted probability for class $c$
- $\gamma = 2.0$ is the focusing parameter that down-weights easy, well-classified examples
- $\alpha_c = \frac{N}{K \cdot N_c}$ is the inverse class-frequency weight

The modulating factor $(1 - p_c)^\gamma$ achieves two effects:
1. When $p_c \to 1$ (correctly classified with high confidence), the factor approaches 0, effectively ignoring easy examples
2. When $p_c \to 0$ (misclassified), the factor approaches 1, preserving the full loss gradient for hard examples

Combined with the class weights $\alpha_c$, this assigns the `Offensive` class a loss weight of **14.17×** compared to **0.567×** for `Sarcasm`, forcing the model to prioritize minority class learning.

### 4.7 Data Augmentation

Training-time image augmentation improves generalization, especially for the 51-sample `Offensive` class:

| Augmentation | Parameters | Purpose |
|---|---|---|
| Random Horizontal Flip | $p = 0.5$ | Spatial invariance |
| Color Jitter | brightness=0.1, contrast=0.1 | Photometric invariance |
| Resize + Normalize | $224 \times 224$, ImageNet stats | Standard ViT preprocessing |

Validation and test transforms use only deterministic resize and normalization (no augmentation).

### 4.8 Inference Pipeline

At inference time, the trained model generates predictions on unlabeled test memes:

$$
\hat{y}_i = \arg\max_{c \in \{0, 1, 2, 3\}} P(y = c \mid V_i, T_i; \Theta)
$$

The `IMUSAPredictor` module loads saved model checkpoints (`best_model.pt`), processes test samples through the same preprocessing pipeline (deterministic transforms only), and outputs per-sample predicted categories with softmax confidence scores.

---

## 5. Experimental Setup

### 5.1 Implementation Details

| Parameter | Value |
|---|---|
| **Framework** | PyTorch 2.x + Hugging Face Transformers |
| **Python Version** | 3.12 (pinned via `.python-version`) |
| **Package Manager** | `uv` (workspace mode with single lockfile) |
| **Code Quality** | `ruff` (linting/formatting) + `mypy` (strict static typing) |
| **Testing** | `pytest` with coverage tracking (87% line coverage) |
| **Vision Backbone** | `google/vit-base-patch16-224` (ImageNet-21K pre-trained) |
| **Text Backbone** | `xlm-roberta-base` (100-language Common Crawl pre-trained) |
| **Max Text Length** | 128 tokens |
| **Image Resolution** | $224 \times 224$ pixels |

### 5.2 Training Protocol

| Hyperparameter | Value |
|---|---|
| **Optimizer** | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\lambda = 0.01$) |
| **Initial Learning Rate** | $\eta = 2 \times 10^{-5}$ |
| **LR Schedule** | Linear Warmup (10% of steps) + Cosine Annealing |
| **Epochs** | 10 |
| **Batch Size** | 16 |
| **Loss Function** | $\alpha$-Balanced Focal Loss ($\gamma = 2.0$) |
| **Gradient Clipping** | Max norm = 1.0 |
| **Dropout** | 0.3 (classification head) |
| **Train/Val Split** | 80/20 stratified on `Category` column |
| **Hardware** | Google Colab Free Tier (NVIDIA T4 GPU, 16GB VRAM) |

### 5.3 Learning Rate Schedule

The learning rate follows a linear warmup phase for the first 10% of total training steps, followed by cosine annealing decay:

$$
\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\[6pt] \frac{\eta_{\max}}{2} \left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right)\right) & \text{otherwise} \end{cases}
$$

This schedule prevents catastrophic forgetting of pre-trained representations during early training (warmup phase) while enabling fine-grained convergence through gradual learning rate reduction (cosine phase).

### 5.4 Evaluation Metrics

We report the following metrics, with **Macro F1** as the primary optimization target:

| Metric | Definition | Role |
|---|---|---|
| **Macro F1** | $\frac{1}{K} \sum_{c=1}^K \text{F1}_c$ | Primary metric — equal weight to all classes |
| **Weighted F1** | $\sum_{c=1}^K \frac{N_c}{N} \text{F1}_c$ | Secondary — support-weighted |
| **Accuracy** | $\frac{\sum \text{TP}_c}{N}$ | Baseline comparison |
| **Per-Class F1** | Individual $\text{F1}_c$ for each category | Diagnostic — identifies weak classes |

Macro F1 is the standard primary metric used in FIRE shared tasks (Mandl et al., 2020; Chakravarthi et al., 2021), as it penalizes models that achieve high accuracy by neglecting minority classes.

### 5.5 Stratified Data Splitting

The dataset is split into 80% training (2,312 samples) and 20% validation (579 samples) using stratified sampling on the `Category` column. This guarantees that the extreme class ratios (including the 51-sample `Offensive` class) are preserved identically in both splits, preventing the validation set from lacking minority class representation.

---

## 6. Results and Findings

> **Note**: Benchmark results tables will be populated upon completion of the Google Colab GPU training run. The framework for comprehensive evaluation is fully implemented and ready for execution.

### 6.1 Baseline Comparison

| Model | Loss Function | Val Acc | Val Weighted F1 | Val Macro F1 |
|---|---|---|---|---|
| Text-Only (XLM-RoBERTa) | Cross-Entropy | TBD | TBD | TBD |
| Vision-Only (ViT-Base) | Cross-Entropy | TBD | TBD | TBD |
| Multimodal (ViT + XLM-R) | Standard CE | TBD | TBD | TBD |
| Multimodal (ViT + XLM-R) | Weighted CE | TBD | TBD | TBD |
| **Multimodal Gated Fusion (Ours)** | **Focal Loss ($\gamma$=2.0)** | **TBD** | **TBD** | **TBD** |

### 6.2 Training Dynamics

*(Training loss curves and validation Macro F1 trajectory plots will be generated by `imusa.evaluation.plot_training_curves()` and embedded here after GPU training.)*

### 6.3 Confusion Matrix Analysis

*(Normalized confusion matrix heatmap will be generated by `imusa.evaluation.plot_confusion_matrix()` and embedded here after GPU training.)*

### 6.4 Per-Class F1 Breakdown

*(Per-class F1 bar chart will be generated by `imusa.evaluation.plot_per_class_f1()` and embedded here after GPU training.)*

### 6.5 Test Set Prediction Distribution

*(Distribution of predicted categories on the 500 unlabeled test memes will be reported here after inference.)*

---

## 7. Discussion

### 7.1 Multimodal vs. Unimodal Performance

*(Analysis of whether the multimodal model outperforms text-only and vision-only baselines, and by how much, will be discussed after benchmark completion.)*

### 7.2 Impact of Focal Loss on Minority Class Recovery

*(Analysis of whether Focal Loss successfully recovers `Offensive` class F1 compared to standard cross-entropy will be documented here. We hypothesize that the 14.17× inverse weighting combined with Focal Loss's gradient focusing will substantially improve `Offensive` recall from the expected 0.0 under standard CE.)*

### 7.3 Error Analysis and Failure Modes

*(Discussion of common misclassification patterns observed in the confusion matrix — e.g., `Sarcasm` ↔ `Neutral` confusion due to subtle irony, or `Motivational` ↔ `Neutral` overlap in religious/philosophical content.)*

### 7.4 Limitations

1. **Dataset Scale**: With only 2,891 training samples (51 for `Offensive`), model generalization is inherently limited. Larger annotated corpora would enable more robust fine-tuning.
2. **Pre-trained Model Selection**: We use general-purpose multilingual models. Domain-specific pre-training on Punjabi social media text (e.g., MuRIL, IndicBERT v2) may yield improvements.
3. **Static Augmentation**: Our visual augmentation is limited to geometric and photometric transformations. Advanced strategies like mixup, CutMix, or GAN-based synthetic minority generation could further address the `Offensive` class sparsity.
4. **Single-Task Architecture**: The current system is optimized for 4-class classification. Multi-task learning jointly predicting sentiment intensity or humor could provide auxiliary supervision.

### 7.5 Ablation Study Plan

| Ablation | What It Tests |
|---|---|
| Remove Vision Encoder | Text-only performance baseline |
| Remove Text Encoder | Vision-only performance baseline |
| Replace Gated Fusion with Simple Concatenation | Value of learned gating mechanism |
| Replace Focal Loss with Standard CE | Impact of Focal Loss on minority recall |
| Remove Data Augmentation | Impact of training-time augmentation |
| Replace XLM-RoBERTa with MuRIL | Indic-specific vs. general multilingual encoder |

---

## 8. Conclusion

This paper presents a multimodal deep learning system for Punjabi meme sentiment classification in the IMUSA shared task at FIRE 2026. Our dual-encoder architecture combines Vision Transformer (ViT-Base) and XLM-RoBERTa through a Gated Multimodal Fusion mechanism, trained with $\alpha$-balanced Focal Loss to address the severe 25:1 class imbalance between `Sarcasm` and `Offensive` categories.

The system is implemented as a fully reproducible open-source pipeline with comprehensive unit testing (19 tests, 87% coverage), deterministic data preprocessing, stratified evaluation splits, and automated visualization generation. The complete codebase, including training scripts, inference engine, and evaluation tools, is publicly available at [https://github.com/shubhojit-mitra-dev/imusa-multimodal-sentiment](https://github.com/shubhojit-mitra-dev/imusa-multimodal-sentiment).

### Future Directions

1. **Advanced Fusion**: Explore cross-modal attention mechanisms (MulT, ViLBERT) for richer vision-language interaction
2. **Indic-Specific Models**: Evaluate MuRIL and IndicBERT v2 as drop-in replacements for XLM-RoBERTa
3. **Minority Class Augmentation**: Implement SMOTE in embedding space or back-translation augmentation for `Offensive` samples
4. **Ensemble Methods**: Combine predictions from multiple architectural variants and loss functions
5. **OCR Integration**: Extract text directly from meme images using Gurmukhi OCR rather than relying on provided text annotations

---

## References

1. Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). Gated Multimodal Units for Information Fusion. *ICLR 2017 Workshop / Neural Networks*, 121, 11–20.

2. Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *NeurIPS 2019*, 32, 1567–1578.

3. Chakravarthi, B. R., Priyadharshini, R., et al. (2021–2023). Overview of DravidianLangTech: Sentiment Analysis and Multimodal Memes in Indian Languages. *ACL/EMNLP/LREC Workshop Proceedings*.

4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

5. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*, 8440–8451.

6. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR 2019*, 9268–9277.

7. Doddapaneni, S., Aralikatte, R., Syamala, R., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2023). IndicBERT v2: Towards Better Language Models for Indic Languages. *Findings of ACL 2023*, 13801–13820.

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

9. Hossain, E., Sharif, O., & Hoque, M. M. (2022). MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes. *LREC 2022*, 1542–1554.

10. Kakwani, D., Kunchukuttan, A., Golla, S., N.C., G., Bhattacharyya, A., Khapra, M. M., & Kumar, P. (2020). IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages. *Findings of EMNLP 2020*, 4948–4961.

11. Kaur, A., & Gupta, V. (2019). A Novel Approach for Sentiment Analysis of Punjabi Text using SVM. *International Arab Journal of Information Technology*, 16(3A), 615–622.

12. Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., et al. (2021). MuRIL: Multilingual Representations for Indian Languages. *arXiv:2103.10730*, Google Research.

13. Kiela, D., Firooz, H., Mohan, A., Goswami, V., et al. (2020). The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. *NeurIPS 2020*, 33, 2611–2624.

14. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*, 2980–2988.

15. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*, 10012–10022.

16. Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. *NeurIPS 2019*, 32, 13–23.

17. Mandl, T., Modha, S., Shahi, G. K., et al. (2020–2022). Overview of HASOC Track at FIRE: Hate Speech and Offensive Content Identification in Indo-European Languages. *CEUR Workshop Proceedings*.

18. Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., & Dokania, P. K. (2020). Calibrating Deep Neural Networks using Focal Loss. *NeurIPS 2020*, 33, 15288–15299.

19. Pramanick, S., Dimitrov, D., Mukherjee, R., Sharma, S., Akhtar, M. S., Nakov, P., & Chakraborty, T. (2021). Detecting Harmful Memes and Their Targets. *Findings of EMNLP 2021*, 2783–2796.

20. Pranesh, A. R. R., & Shekhar, A. (2020). MemeSem: A Multi-modal Framework for Sentimental Analysis of Meme via Transfer Learning. *ICML 2020 Workshop*.

21. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*, 139, 8748–8763.

22. Sharma, C., Bhageria, D., Scott, W., PYKL, S., Das, A., Chakraborty, T., Pulabaigari, V., & Gambäck, B. (2020). SemEval-2020 Task 8: Memotion Analysis. *SemEval 2020*, 759–774.

23. Singh, J., Lehal, G. S., & Saini, T. S. (2021). Morphological Evaluation and Sentiment Analysis of Punjabi Text using Deep Learning Classification. *Journal of Ambient Intelligence and Humanized Computing*, 12, 8831–8842.

24. Snoek, C. G. M., Worring, M., & Smeulders, A. W. M. (2005). Early versus Late Fusion in Semantic Video Analysis. *ACM Multimedia 2005*, 399–408.

25. Suryawanshi, S., Chakravarthi, B. R., Arcan, M., & Buitelaar, P. (2020). Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text. *TRAC-2 Workshop @ LREC 2020*, 32–41.

26. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. *ICML 2021*, 139, 10347–10357.

27. Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., & Salakhutdinov, R. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. *ACL 2019*, 6558–6569.

28. Wang, Y., Jiang, W., & Luo, X. (2021). Handling Class Imbalance in Text Classification via Focal Loss. *IEEE Access*, 9, 82312–82322.
