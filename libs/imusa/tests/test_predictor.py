"""Unit tests for IMUSAPredictor inference module."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from imusa.config import settings
from imusa.inference.predictor import IMUSAPredictor
from imusa.models.multimodal import IMUSAMultimodalClassifier


class TinyTestDataset(Dataset):  # type: ignore[type-arg]
    """Dummy dataset for testing predictor batch inference."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(128, dtype=torch.long),
            "attention_mask": torch.ones(128, dtype=torch.long),
            "image_id": f"test_image_{idx}.jpg",
        }


def test_predictor_initialization() -> None:
    """Verify IMUSAPredictor initializes properly with default device."""
    dummy_model = IMUSAMultimodalClassifier(
        num_classes=4, vit_model_name="dummy/vit", text_model_name="dummy/xlm-r"
    )
    predictor = IMUSAPredictor(model=dummy_model, device="cpu")
    assert predictor.device == "cpu"
    assert isinstance(predictor.model, nn.Module)


def test_predictor_predict_batch() -> None:
    """Verify predict_batch produces expected keys and valid category assignments."""
    dataset = TinyTestDataset(size=4)
    loader = DataLoader(dataset, batch_size=2)

    dummy_model = IMUSAMultimodalClassifier(
        num_classes=4, vit_model_name="dummy/vit", text_model_name="dummy/xlm-r"
    )
    predictor = IMUSAPredictor(model=dummy_model, device="cpu")
    results = predictor.predict_batch(loader)

    assert len(results) == 4
    for record in results:
        assert "image_id" in record
        assert "predicted_category" in record
        assert "confidence" in record
        assert "probabilities" in record

        assert record["predicted_category"] in settings.categories
        assert 0.0 <= record["confidence"] <= 1.0
        assert len(record["probabilities"]) == 4
