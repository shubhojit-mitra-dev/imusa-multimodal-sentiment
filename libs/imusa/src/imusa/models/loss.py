"""Imbalance-Aware Loss Functions Module for IMUSA Sentiment Classification.

This module provides class weight computation and custom focal loss implementations
to tackle extreme class imbalances in the IMUSA dataset (e.g. Offensive class representing
only ~1.8% of training samples vs Sarcasm representing ~44%).
"""

import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from imusa.config import settings

logger = logging.getLogger(__name__)


def compute_inverse_class_weights(
    df: pd.DataFrame,
    num_classes: int = 4,
) -> torch.Tensor:
    """Calculate inverse class weights to penalize minority class classification errors.

    The formula used is standard balanced weighting:
        w_c = N / (K * N_c)
    where N is total samples, K is number of classes, and N_c is count of samples in class c.

    For IMUSA dataset:
        - Sarcasm (1274 samples): weight ~ 0.567
        - Motivational (836 samples): weight ~ 0.864
        - Neutral (730 samples): weight ~ 0.990
        - Offensive (51 samples): weight ~ 14.17 (25x higher penalty!)

    Args:
        df: Cleaned training DataFrame containing a 'Category' column.
        num_classes: Total number of sentiment classes (default: 4).

    Returns:
        PyTorch float Tensor of shape (num_classes,) containing class weights.
    """
    counts = df["Category"].value_counts().to_dict()
    total_samples = len(df)

    weights = []
    for cat in settings.categories:
        count = counts.get(cat, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(
        "Computed balanced class weights: %s", dict(zip(settings.categories, weights, strict=False))
    )
    return weight_tensor


class FocalLoss(nn.Module):
    """Focal Loss for Dense/Imbalanced Multi-Class Sentiment Classification.

    Focal loss addresses class imbalance by down-weighting well-classified (easy)
    examples and focusing model training on hard, misclassified minority samples.

    Formula:
        L_focal = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Attributes:
        gamma: Focusing parameter scaling down easy samples (default: 2.0).
        alpha: Optional class weight tensor of shape (num_classes,).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            gamma: Focusing exponent (higher values focus more on hard samples).
            alpha: Class weights tensor of shape (num_classes,).
            reduction: 'mean', 'sum', or 'none' loss reduction.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between raw logits and target class indices.

        Args:
            inputs: Logits tensor of shape (batch_size, num_classes).
            targets: Integer target tensor of shape (batch_size,).

        Returns:
            Computed scalar loss tensor (if reduction='mean').
        """
        # Calculate standard cross entropy probabilities p_t
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Gather log_p_t and p_t corresponding to true class targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1]).to(inputs.dtype)
        log_p_t = (log_probs * targets_one_hot).sum(dim=-1)
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Calculate focal factor (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        # Apply class weights alpha if provided
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_weight = (self.alpha * targets_one_hot).sum(dim=-1)
            loss = alpha_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
