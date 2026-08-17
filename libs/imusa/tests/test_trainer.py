"""Unit and Integration tests for Trainer module."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from imusa.training.trainer import Trainer


class TinyDummyDataset(Dataset):  # type: ignore[type-arg]
    """Tiny dataset for fast unit testing of trainer execution loop."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(64, dtype=torch.long),
            "attention_mask": torch.ones(64, dtype=torch.long),
            "label": torch.tensor(idx % 4, dtype=torch.long),
        }


class DummyClassifier(nn.Module):
    """Dummy classifier returning constant logits for fast testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 4)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fc(images.view(images.size(0), -1))


def test_trainer_fit_loop(tmp_path: Path) -> None:
    """Verify Trainer executes 1 epoch training and validation loop without error."""
    dataset = TinyDummyDataset()
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    model = DummyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        output_dir=tmp_path / "checkpoints",
    )

    results: dict[str, Any] = trainer.fit(epochs=1)

    assert "best_macro_f1" in results
    assert "history" in results
    assert len(results["history"]) == 1
    assert "train_loss" in results["history"][0]
    assert "macro_f1" in results["history"][0]
    assert hasattr(trainer, "best_checkpoint_path")
    assert trainer.best_checkpoint_path.exists()
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
