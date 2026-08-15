"""Unit tests for Phase F multimodal data augmentation module."""

import torch
from PIL import Image

from imusa.data.augmentation import augment_text, get_advanced_train_image_transform


def test_advanced_image_transform() -> None:
    """Verify advanced vision transformation pipeline converts PIL Image to tensor with erasing."""
    img = Image.new("RGB", (300, 300), color="blue")
    transform = get_advanced_train_image_transform()
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_augment_text() -> None:
    """Verify text augmentation modifies or preserves Gurmukhi word strings cleanly."""
    raw_text = "ਇਹ ਇੱਕ ਬਹੁਤ ਹੀ ਵਧੀਆ ਮੀਮ ਹੈ"
    aug1 = augment_text(raw_text, p_swap=0.0, p_delete=0.0)
    assert aug1 == raw_text

    aug2 = augment_text(raw_text, p_swap=1.0, p_delete=0.5)
    assert isinstance(aug2, str)
    assert len(aug2) > 0
