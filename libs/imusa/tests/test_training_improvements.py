"""Unit tests for Phase C training strategy improvements (Label-Smoothed Focal Loss, Manifold Mixup, LP-FT)."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from imusa.models.loss import FocalLoss
from imusa.models.multimodal import IMUSAMultimodalClassifier
from imusa.training.trainer import Trainer


class DummyDataset(Dataset):  # type: ignore[type-arg]
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(32, dtype=torch.long),
            "attention_mask": torch.ones(32, dtype=torch.long),
            "label": torch.tensor(idx % 4, dtype=torch.long),
        }


def test_focal_loss_label_smoothing() -> None:
    """Verify FocalLoss with label_smoothing computes valid loss tensor."""
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2, 3])

    loss = criterion(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar mean
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_focal_loss_soft_targets() -> None:
    """Verify FocalLoss with 2D soft targets (for mixup compatibility)."""
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 4)
    soft_targets = torch.softmax(torch.randn(4, 4), dim=-1)

    loss = criterion(logits, soft_targets)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_multimodal_forward_with_mixup() -> None:
    """Verify forward_with_mixup produces valid logits, permutation, and lambda value."""
    model = IMUSAMultimodalClassifier(
        num_classes=4, vit_model_name="dummy/vit", text_model_name="dummy/xlm-r"
    )
    model.train()
    images = torch.randn(4, 3, 224, 224)
    input_ids = torch.ones(4, 32, dtype=torch.long)
    attention_mask = torch.ones(4, 32, dtype=torch.long)

    logits, perm, lam = model.forward_with_mixup(
        images, input_ids, attention_mask=attention_mask, alpha=0.2
    )

    assert logits.shape == (4, 4)
    assert perm.shape == (4,)
    assert 0.0 <= lam <= 1.0


class DummyLPFTClassifier(torch.nn.Module):
    """Lightweight model implementing freeze/unfreeze and mixup for trainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3 * 224 * 224, 64)
        self.classifier = torch.nn.Linear(64, 4)
        self.is_frozen = False

    def freeze_backbones(self) -> None:
        self.is_frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        self.is_frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.backbone(images.view(images.size(0), -1))
        return self.classifier(feat)

    def forward_with_mixup(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        alpha: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        feat = self.backbone(images.view(images.size(0), -1))
        perm = torch.randperm(images.size(0))
        lam = 0.5
        mixed_feat = lam * feat + (1.0 - lam) * feat[perm]
        logits = self.classifier(mixed_feat)
        return logits, perm, lam


def test_trainer_fit_lpft_loop(tmp_path: Path) -> None:
    """Verify Trainer.fit_lpft executes linear probing and fine-tuning stages."""
    dataset = DummyDataset()
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    model = DummyLPFTClassifier()
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        output_dir=tmp_path / "checkpoints",
    )

    results = trainer.fit_lpft(
        lp_epochs=1, ft_epochs=1, lp_lr=1e-3, ft_lr=1e-4, use_mixup=True, mixup_alpha=0.2
    )

    assert "best_macro_f1" in results
    assert len(results["history"]) == 2
    assert results["history"][0]["phase"] == "lp"
    assert results["history"][1]["phase"] == "ft"
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
