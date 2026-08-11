"""Unit tests for imbalance-aware loss functions."""

import pandas as pd
import torch

from imusa.models.loss import FocalLoss, compute_inverse_class_weights


def test_compute_inverse_class_weights() -> None:
    """Verify minority class gets significantly higher loss weight than majority class."""
    # Dummy DataFrame with 10 Sarcasm, 5 Motivational, 4 Neutral, 1 Offensive
    data = (
        [{"Category": "Sarcasm"}] * 10
        + [{"Category": "Motivational"}] * 5
        + [{"Category": "Neutral"}] * 4
        + [{"Category": "Offensive"}] * 1
    )
    df = pd.DataFrame(data)

    from imusa.config import settings

    weights = compute_inverse_class_weights(df, num_classes=4)

    assert weights.shape == (4,)
    sarcasm_idx = settings.categories.index("Sarcasm")
    offensive_idx = settings.categories.index("Offensive")

    # Offensive weight should be 10x higher than Sarcasm weight
    assert weights[offensive_idx].item() > weights[sarcasm_idx].item()
    assert torch.isclose(
        weights[offensive_idx] / weights[sarcasm_idx], torch.tensor(10.0), atol=1e-1
    )


def test_focal_loss_forward() -> None:
    """Verify FocalLoss forward pass produces valid scalar loss."""
    loss_fn = FocalLoss(gamma=2.0)

    # Dummy logits for batch_size=4, num_classes=4
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = loss_fn(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)
