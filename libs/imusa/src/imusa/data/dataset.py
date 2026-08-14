"""PyTorch Dataset and Stratified DataLoader Module for IMUSA.

This module handles loading multimodal (image + text) data from the IMUSA dataset,
applying visual pre-processing transforms, tokenizing Punjabi (Gurmukhi) text using
Hugging Face transformers, and constructing stratified train/validation DataLoaders.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from imusa.config import settings

logger = logging.getLogger(__name__)


def get_default_image_transform() -> transforms.Compose:
    """Return standard ImageNet preprocessing transforms for Vision Transformers.

    Resizes input images to 224x224, converts PIL Image to PyTorch Tensor,
    and normalizes RGB channels using standard ImageNet mean and standard deviation.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_train_image_transform() -> transforms.Compose:
    """Return training image transformation pipeline with subtle data augmentations.

    Includes RandomHorizontalFlip, RandomRotation, and ColorJitter to improve model
    generalization on imbalanced classes.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class IMUSADataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for Multimodal Punjabi Meme Sentiment Analysis.

    Concurrently loads image files and tokenizes Gurmukhi text strings.

    Attributes:
        df: Cleaned pandas DataFrame containing columns ['Id', 'Category', 'Text'].
        images_dir: Path to directory containing image files.
        tokenizer: Hugging Face tokenizer instance for text encoding.
        max_length: Maximum token sequence length for text padding/truncation.
        is_test: True if loading test set without sentiment labels.
        img_transform: torchvision transform pipeline for image preprocessing.
        category_to_idx: Mapping dictionary from sentiment label strings to integer IDs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        tokenizer: Any,
        max_length: int = 128,
        is_test: bool = False,
        img_transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize the IMUSA Dataset.

        Args:
            df: Cleaned pandas DataFrame.
            images_dir: Path to directory containing image files.
            tokenizer: Pre-initialized Hugging Face AutoTokenizer.
            max_length: Max sequence length for tokenization (default: 128).
            is_test: Set True for un-labeled test dataset.
            img_transform: Custom torchvision transforms (default: ImageNet standard).
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.img_transform = img_transform or get_default_image_transform()

        # Map sentiment categories to zero-indexed integer targets
        self.category_to_idx = {cat: idx for idx, cat in enumerate(settings.categories)}

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch a single multimodal sample at index `idx`.

        Args:
            idx: Integer index.

        Returns:
            Dictionary containing:
                - 'image': PyTorch float Tensor of shape (3, 224, 224)
                - 'input_ids': LongTensor of token IDs of shape (max_length,)
                - 'attention_mask': LongTensor of attention masks of shape (max_length,)
                - 'label': LongTensor target class index (0-3), omitted if is_test=True
                - 'image_id': String filename of sample image
        """
        row = self.df.iloc[idx]
        image_id = str(row["Id"])
        text_content = str(row["Text"])

        # 1. Load and transform vision modality
        img_path = self.images_dir / image_id
        if not img_path.exists():
            stem = Path(image_id).stem
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = self.images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as err:
            logger.warning("Failed to open image %s: %s. Using blank image.", img_path, err)
            image = Image.new("RGB", (224, 224), color="black")

        image_tensor = self.img_transform(image)

        # 2. Tokenize text modality using Hugging Face tokenizer
        encoded = self.tokenizer(
            text_content,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        sample: dict[str, Any] = {
            "image": image_tensor,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image_id": image_id,
        }

        # 3. Attach sentiment label if not test set
        if not self.is_test and "Category" in row and pd.notna(row["Category"]):
            cat_str = str(row["Category"])
            sample["label"] = torch.tensor(self.category_to_idx.get(cat_str, 0), dtype=torch.long)

        return sample


def create_stratified_dataloaders(
    df: pd.DataFrame,
    images_dir: Path,
    tokenizer: Any,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Split dataset into stratified train/validation sets and create PyTorch DataLoaders.

    Ensures that extreme class imbalances (e.g. Offensive class with ~1.8% of samples)
    are preserved identically across training and validation splits.

    Args:
        df: Cleaned pandas DataFrame containing dataset rows.
        images_dir: Path to directory containing meme images.
        tokenizer: Hugging Face tokenizer instance.
        val_ratio: Fraction of dataset reserved for validation (default: 0.2).
        batch_size: Mini-batch size for DataLoader (default: 16).
        num_workers: Data loading parallel worker threads (default: 2).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    from sklearn.model_selection import train_test_split

    logger.info(
        "Creating stratified train/val DataLoaders (val_ratio=%.2f, batch_size=%d)",
        val_ratio,
        batch_size,
    )

    # Perform stratified split on the Category column
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["Category"],
        random_state=seed,
    )

    train_dataset = IMUSADataset(
        train_df, images_dir, tokenizer, img_transform=get_train_image_transform()
    )
    val_dataset = IMUSADataset(
        val_df, images_dir, tokenizer, img_transform=get_default_image_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders successfully initialized: Train samples=%d, Val samples=%d",
        len(train_dataset),
        len(val_dataset),
    )

    return train_loader, val_loader
