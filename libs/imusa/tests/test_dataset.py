"""Unit tests for IMUSADataset and DataLoader factory functions."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from imusa.data.dataset import IMUSADataset, create_stratified_dataloaders


class DummyTokenizer:
    """Mock Hugging Face Tokenizer for testing dataset logic without heavy downloads."""

    def __call__(
        self,
        text: str,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Return dummy input_ids and attention_mask tensors of max_length."""
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_imusa_dataset_getitem(tmp_path: Path) -> None:
    """Verify item extraction from IMUSADataset produces correct tensor shapes and types."""
    # Create dummy image on disk
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    dummy_img_path = img_dir / "sample_1.jpg"
    Image.new("RGB", (100, 100), color="blue").save(dummy_img_path)

    # Create dummy DataFrame
    df = pd.DataFrame(
        [
            {"Id": "sample_1.jpg", "Category": "Sarcasm", "Text": "ਤਾਂ ਰਹੀ ਹੈ ਪਰ ਸਮਝ ਨਹੀਂ ਆਉਂਦੀ"},
        ]
    )

    tokenizer = DummyTokenizer()
    dataset = IMUSADataset(df=df, images_dir=img_dir, tokenizer=tokenizer, max_length=64)

    assert len(dataset) == 1
    sample: dict[str, Any] = dataset[0]

    # Assert Vision Modality
    assert "image" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)

    # Assert Text Modality
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert sample["input_ids"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)

    # Assert Label (Sarcasm is first index 0)
    assert "label" in sample
    assert sample["label"].item() == 0


def test_create_stratified_dataloaders(tmp_path: Path) -> None:
    """Verify stratified splitting creates proportional train and validation DataLoaders."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 10 dummy images
    records = []
    categories = ["Sarcasm", "Motivational", "Neutral", "Offensive"]
    for i in range(20):
        img_name = f"img_{i}.jpg"
        Image.new("RGB", (50, 50)).save(img_dir / img_name)
        cat = categories[i % 4]
        records.append({"Id": img_name, "Category": cat, "Text": f"Sample text {i}"})

    df = pd.DataFrame(records)
    tokenizer = DummyTokenizer()

    train_loader, val_loader = create_stratified_dataloaders(
        df=df,
        images_dir=img_dir,
        tokenizer=tokenizer,
        val_ratio=0.2,
        batch_size=4,
        num_workers=0,
    )

    # 20 samples total: 16 train, 4 val
    assert len(train_loader.dataset) == 16  # type: ignore[arg-type]
    assert len(val_loader.dataset) == 4  # type: ignore[arg-type]

    batch = next(iter(train_loader))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["input_ids"].shape == (4, 128)
    assert batch["label"].shape == (4,)
