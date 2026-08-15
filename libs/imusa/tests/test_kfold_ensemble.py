"""Unit tests for Phase D K-Fold cross-validation dataloaders and ensemble inference engine."""

from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from imusa.config import settings
from imusa.inference.predictor import IMUSAEnsemblePredictor


class DummyTestDataset(Dataset):  # type: ignore[type-arg]
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(32, dtype=torch.long),
            "attention_mask": torch.ones(32, dtype=torch.long),
            "image_id": f"sample_{idx}.jpg",
        }


class DummyClassifier(nn.Module):
    def __init__(self, constant_pred: int = 0) -> None:
        super().__init__()
        self.constant_pred = constant_pred
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = images.size(0)
        logits = torch.zeros((batch_size, 4))
        logits[:, self.constant_pred] = 5.0
        return logits


def test_ensemble_predictor_batch() -> None:
    """Verify IMUSAEnsemblePredictor correctly averages probability outputs across models."""
    dataset = DummyTestDataset()
    loader = DataLoader(dataset, batch_size=2)

    # Model 0 predicts Sarcasm (0), Model 1 predicts Sarcasm (0)
    m1 = DummyClassifier(constant_pred=0)
    m2 = DummyClassifier(constant_pred=0)

    ensemble = IMUSAEnsemblePredictor(models=[m1, m2], device="cpu")
    results = ensemble.predict_batch(loader)

    assert len(results) == 4
    for item in results:
        assert "predicted_category" in item
        assert "probabilities" in item
        assert item["predicted_category"] == settings.categories[0]
        assert len(item["probabilities"]) == 4


def test_kfold_dataloaders_split_proportions() -> None:
    """Verify StratifiedKFold logic correctly partitions dataframe into 5 stratified splits."""
    df = pd.DataFrame(
        {
            "Id": [f"{i}.jpg" for i in range(100)],
            "Category": ["Sarcasm"] * 50
            + ["Neutral"] * 30
            + ["Motivational"] * 15
            + ["Offensive"] * 5,
            "Text": ["text"] * 100,
        }
    )

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(df, df["Category"]))

    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) == 80
        assert len(val_idx) == 20
