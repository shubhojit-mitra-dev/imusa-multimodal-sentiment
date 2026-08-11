"""Exploratory Data Analysis (EDA) and Visualization Engine for IMUSA Dataset."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from rich.console import Console
from rich.table import Table

from imusa.config import settings
from imusa.data.cleaning import clean_dataset

logger = logging.getLogger(__name__)
console = Console()

# Set clean aesthetic style for plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("muted")


def explore_dataset(
    processed_csv: Path | None = None,
    images_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Perform comprehensive Exploratory Data Analysis (EDA) on cleaned dataset.

    Args:
        processed_csv: Path to cleaned dataset CSV.
        images_dir: Path to image directory.
        output_dir: Directory to save generated plot artifacts.

    Returns:
        Dictionary containing EDA summary statistics.
    """
    settings.ensure_directories()
    csv_path = processed_csv or settings.processed_train_csv
    img_dir = images_dir or settings.train_images_dir
    save_dir = output_dir or settings.exploration_output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.info("Processed CSV not found at %s. Running cleaning first...", csv_path)
        clean_dataset(output_csv=csv_path)

    df = pd.read_csv(csv_path)
    logger.info("Loaded cleaned dataset with %d samples for EDA", len(df))

    # 1. Text Metrics Analysis
    df["char_count"] = df["Text"].astype(str).str.len()
    df["word_count"] = df["Text"].astype(str).str.split().str.len()

    # 2. Image Resolution Analysis
    img_widths, img_heights, aspect_ratios = [], [], []
    for _, row in df.iterrows():
        img_path = img_dir / row["Id"]
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    img_widths.append(w)
                    img_heights.append(h)
                    aspect_ratios.append(w / h)
            except Exception as err:
                logger.warning("Failed to open image %s: %s", img_path, err)
                img_widths.append(None)
                img_heights.append(None)
                aspect_ratios.append(None)
        else:
            img_widths.append(None)
            img_heights.append(None)
            aspect_ratios.append(None)

    df["img_width"] = img_widths
    df["img_height"] = img_heights
    df["aspect_ratio"] = aspect_ratios

    # 3. Generate Visual Plots
    _plot_class_distribution(df, save_dir / "class_distribution.png")
    _plot_text_length(df, save_dir / "text_length_distribution.png")
    _plot_image_stats(df, save_dir / "image_resolution_distribution.png")
    _create_sample_grid(df, img_dir, save_dir / "sample_meme_grid.png")

    # 4. Print Summary Tables
    _print_eda_tables(df)

    summary_stats = {
        "total_samples": len(df),
        "class_counts": df["Category"].value_counts().to_dict(),
        "avg_word_count": float(df["word_count"].mean()),
        "max_word_count": int(df["word_count"].max()),
        "avg_image_width": float(df["img_width"].dropna().mean()),
        "avg_image_height": float(df["img_height"].dropna().mean()),
    }

    return summary_stats


def _plot_class_distribution(df: pd.DataFrame, output_file: Path) -> None:
    """Plot and save class distribution bar chart with percentage labels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Category"].value_counts()
    total = len(df)

    palette = sns.color_palette("Set2", len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="black", linewidth=1)

    ax.set_title("IMUSA Dataset Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Memes", fontsize=12, fontweight="bold")

    # Add count and percentage labels above bars
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info("Saved class distribution plot to %s", output_file)


def _plot_text_length(df: pd.DataFrame, output_file: Path) -> None:
    """Plot word count distributions per category."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Category", y="word_count", palette="Set2", ax=ax, width=0.5)

    ax.set_title("Punjabi Text Word Count per Sentiment Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Word Count", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info("Saved text length plot to %s", output_file)


def _plot_image_stats(df: pd.DataFrame, output_file: Path) -> None:
    """Plot image dimensions scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df.dropna(subset=["img_width", "img_height"]), x="img_width", y="img_height", hue="Category", alpha=0.7, ax=ax)

    ax.set_title("Meme Image Dimensions (Width vs Height)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Width (pixels)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Height (pixels)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info("Saved image dimensions plot to %s", output_file)


def _create_sample_grid(df: pd.DataFrame, images_dir: Path, output_file: Path) -> None:
    """Create a 4x4 sample image grid showing representative memes for each class."""
    categories = settings.categories
    fig, axes = plt.subplots(len(categories), 3, figsize=(12, 14))

    for cat_idx, cat in enumerate(categories):
        cat_df = df[df["Category"] == cat]
        samples = cat_df.sample(min(3, len(cat_df)), random_state=42)

        for col_idx, (_, row) in enumerate(samples.iterrows()):
            ax = axes[cat_idx, col_idx]
            img_path = images_dir / row["Id"]
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "Missing File", ha="center", va="center")

            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(cat, fontsize=12, fontweight="bold")

    plt.suptitle("Sample Memes Grid per Category", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    logger.info("Saved sample meme grid to %s", output_file)


def _print_eda_tables(df: pd.DataFrame) -> None:
    """Print terminal summary tables of class counts and text lengths."""
    table = Table(title="IMUSA Dataset Sentiment Class Breakdown", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Sample Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="magenta")

    total = len(df)
    for cat, count in df["Category"].value_counts().items():
        pct = (count / total) * 100
        table.add_row(str(cat), str(count), f"{pct:.2f}%")

    console.print(table)
