# Indic Meme Understanding & Sentiment Analysis (IMUSA)

> Multimodal AI System for Punjabi Meme Sentiment Analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i3uWNATbQFnO9fIJS-JiX-1qcjWIdxOr)
[![Code Quality](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

---

## Project Overview

**IMUSA** is a multimodal AI system designed to analyze and classify Punjabi memes (visual images + embedded Gurmukhi script text) into 4 distinct sentiment categories:

1. **Sarcasm** — Irony, satire, or humor where visual context modifies textual meaning.
2. **Neutral** — Objective observations or everyday statements without strong emotional polarity.
3. **Offensive** — Harmful, toxic, or abusive content targeting individuals or groups.
4. **Motivational** — Inspiring messages, quotes, or positive life advice.

The project is architected as an **end-to-end production ML ecosystem**, combining:
- **Multimodal Deep Learning**: Vision Transformer (ViT) / CLIP + Multilingual Transformer (XLM-RoBERTa / MuRIL) with cross-attention fusion.
- **Distributed Training**: PyTorch Distributed Data Parallel (DDP) orchestrated via Kubeflow on Kubernetes.
- **MLOps & Governance**: Experiment tracking and model registry via MLflow.
- **High-Performance Serving**: Asynchronous FastAPI inference backend with Redis queueing.
- **Web Dashboard**: Interactive Next.js frontend with visual sentiment breakdown.

---

## Monorepo Architecture

```
multimodal-ai-project/
├── apps/                 # Product Entrypoints
│   ├── api/              # FastAPI Inference Service (Async prediction engine)
│   └── frontend/         # Next.js Web Dashboard
├── libs/                 # Reusable Core Infrastructure
│   └── imusa/            # Core ML Package (Dataset, Models, Training, Inference)
├── infra/                # Infrastructure as Code
│   ├── docker/           # Production Dockerfiles (Training & Serving)
│   └── k8s/              # Kubernetes Job & Deployment Manifests
├── scripts/              # Command Line Interface Scripts
│   ├── clean_data.py     # Raw dataset parser and cleaner
│   └── explore_data.py   # Statistical explorer & report generator
├── data/                 # Raw & Processed Datasets (Gitignored)
└── outputs/              # Artifacts, Plots & Checkpoints (Gitignored)
```

---

## Quickstart Guide

### Prerequisites
- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Environment Setup

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

## Data Pipeline

### 1. Data Cleaning
The raw CSV dataset contains multiline strings and unparsed character sequences. Run the cleaning pipeline to produce sanitized datasets:

```bash
make clean-data
```
*Output: `data/processed/train_clean.csv`*

### 2. Exploratory Data Analysis (EDA)
Generate statistical reports, class distribution metrics, image resolution profiling, and sample grids:

```bash
make explore
```
*Output: Visual plots saved to `outputs/exploration/`*

---

## Development & Quality Assurance

```bash
# Code Linting & Type Checking (Ruff + Mypy)
make lint

# Code Formatting (Ruff)
make format

# Run Automated Test Suite
make test
```

---

## License & Attribution

Developed for the IMUSA Shared Task FIRE 2026. Built with PyTorch, HuggingFace, FastAPI, and Next.js.
