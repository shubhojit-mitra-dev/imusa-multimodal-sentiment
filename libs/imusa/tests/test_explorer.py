"""Unit tests for exploratory data analysis (EDA) pipeline."""

from pathlib import Path

import pandas as pd
from PIL import Image

from imusa.data.explorer import explore_dataset


def test_explore_dataset(tmp_path: Path) -> None:
    """Test dataset exploration function generating plot artifacts and metrics."""
    processed_csv = tmp_path / "train_clean.csv"
    img_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs" / "exploration"

    img_dir.mkdir(parents=True)

    # Create dummy images and clean CSV
    Image.new("RGB", (200, 300)).save(img_dir / "img1.jpg")
    Image.new("RGB", (400, 400)).save(img_dir / "img2.jpg")

    df = pd.DataFrame(
        [
            {"Id": "img1.jpg", "Category": "Sarcasm", "Text": "Sample text one"},
            {"Id": "img2.jpg", "Category": "Neutral", "Text": "Sample text two"},
        ]
    )
    df.to_csv(processed_csv, index=False)

    stats = explore_dataset(processed_csv=processed_csv, images_dir=img_dir, output_dir=output_dir)

    assert stats["total_samples"] == 2
    assert stats["class_counts"]["Sarcasm"] == 1
    assert stats["class_counts"]["Neutral"] == 1
    assert (output_dir / "class_distribution.png").exists()
    assert (output_dir / "text_length_distribution.png").exists()
    assert (output_dir / "image_resolution_distribution.png").exists()
    assert (output_dir / "sample_meme_grid.png").exists()
