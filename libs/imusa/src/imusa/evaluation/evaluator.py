"""Evaluation and plotting module for model metrics, confusion matrices, and training curves."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from imusa.config import settings

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save normalized confusion matrix heatmap plot.

    Args:
        y_true: Array of true integer category labels.
        y_pred: Array of predicted integer category labels.
        output_path: Optional custom output filepath. Defaults to outputs/confusion_matrix.png.

    Returns:
        Path object of saved plot file.
    """
    if output_path is None:
        output_path = settings.output_dir / "confusion_matrix.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(settings.categories))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=settings.categories,
        yticklabels=settings.categories,
        ax=ax,
        cbar_kws={"label": "Normalized Ratio"},
    )

    ax.set_title("IMUSA Multimodal Model — Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Category", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth Category", fontsize=11, labelpad=8)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved confusion matrix visualization to %s", output_path)
    return output_path


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save dual-axis training loss and validation Macro F1 history plot.

    Args:
        history: List of epoch metric dictionaries containing 'train_loss' and 'macro_f1'.
        output_path: Optional custom output filepath. Defaults to outputs/training_curves.png.

    Returns:
        Path object of saved plot file.
    """
    if output_path is None:
        output_path = settings.output_dir / "training_curves.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history) + 1))
    train_losses = [epoch_dict.get("train_loss", 0.0) for epoch_dict in history]
    val_losses = [epoch_dict.get("val_loss", 0.0) for epoch_dict in history]
    macro_f1s = [epoch_dict.get("macro_f1", 0.0) for epoch_dict in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_train = "#2b5c8f"
    color_val = "#d95f02"
    ax1.set_xlabel("Epoch", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss", color=color_train, fontsize=11, labelpad=8)
    ax1.plot(epochs, train_losses, color=color_train, marker="o", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses, color=color_val, marker="s", linewidth=2, linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color_train)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_f1 = "#2ca02c"
    ax2.set_ylabel("Validation Macro F1", color=color_f1, fontsize=11, labelpad=8)
    ax2.plot(epochs, macro_f1s, color=color_f1, marker="^", linewidth=2.5, label="Val Macro F1")
    ax2.tick_params(axis="y", labelcolor=color_f1)

    plt.title("IMUSA Fine-Tuning Training Dynamics & Macro F1 Trajectory", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved training curves visualization to %s", output_path)
    return output_path


def plot_per_class_f1(
    class_f1_scores: dict[str, float],
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save per-class F1 score bar chart.

    Args:
        class_f1_scores: Dictionary mapping category names to F1 scores (0.0 to 1.0).
        output_path: Optional custom output filepath. Defaults to outputs/per_class_f1.png.

    Returns:
        Path object of saved plot file.
    """
    if output_path is None:
        output_path = settings.output_dir / "per_class_f1.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = list(class_f1_scores.keys())
    scores = [class_f1_scores[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    bars = ax.bar(categories, scores, color=palette[: len(categories)], width=0.55, edgecolor="black", alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=11, labelpad=8)
    ax.set_xlabel("Sentiment Category", fontsize=11, labelpad=8)
    ax.set_title("IMUSA Multimodal Model — Per-Class F1 Performance", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved per-class F1 score bar chart to %s", output_path)
    return output_path
