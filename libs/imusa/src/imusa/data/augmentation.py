"""Multimodal Data Augmentation Module for IMUSA.

Provides vision (spatial/color/erasing transforms) and text (random subword swap/deletion)
augmentation functions to increase sample diversity for minority sentiment classes.
"""

import logging
import random

from torchvision import transforms

logger = logging.getLogger(__name__)


def get_advanced_train_image_transform() -> transforms.Compose:
    """Return advanced training image augmentation pipeline.

    Includes RandomHorizontalFlip, RandomRotation, ColorJitter, and RandomErasing (Cutout)
    to mitigate vision backbone overfitting on small meme image collections.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0),
        ]
    )


def augment_text(text: str, p_swap: float = 0.1, p_delete: float = 0.1) -> str:
    """Apply Easy Data Augmentation (EDA) word-level swap and deletion on text.

    Following Wei & Zou (2019), randomly swaps adjacent words or deletes words
    with probabilities p_swap and p_delete.

    Args:
        text: Input Gurmukhi text string.
        p_swap: Probability of swapping a word with its adjacent neighbor.
        p_delete: Probability of deleting a word.

    Returns:
        Augmented text string.
    """
    words = text.strip().split()
    if len(words) <= 2:
        return text

    # 1. Random word deletion
    new_words = [w for w in words if random.random() > p_delete]
    if not new_words:
        new_words = [random.choice(words)]

    # 2. Random adjacent word swap
    if len(new_words) > 1 and random.random() < p_swap:
        idx = random.randint(0, len(new_words) - 2)
        new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]

    return " ".join(new_words)
