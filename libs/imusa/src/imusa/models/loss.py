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
    Supports label smoothing and soft target distributions (e.g. from mixup).

    Formula:
        L_focal = - sum_c [ alpha_c * y_c * (1 - p_c)^gamma * log(p_c) ]

    Attributes:
        gamma: Focusing parameter scaling down easy samples (default: 2.0).
        alpha: Optional class weight tensor of shape (num_classes,).
        label_smoothing: Float label smoothing factor (default: 0.0).
        reduction: 'mean', 'sum', or 'none' loss reduction.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            gamma: Focusing exponent (higher values focus more on hard samples).
            alpha: Class weights tensor of shape (num_classes,).
            label_smoothing: Smoothing ratio in [0, 1) applied to 1-hot targets.
            reduction: 'mean', 'sum', or 'none' loss reduction.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between raw logits and target class indices or soft targets.

        Args:
            inputs: Logits tensor of shape (batch_size, num_classes).
            targets: Integer target tensor of shape (batch_size,) or soft targets of shape (batch_size, num_classes).

        Returns:
            Computed scalar loss tensor (if reduction='mean').
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        num_classes = inputs.shape[-1]

        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).to(inputs.dtype)
            if self.label_smoothing > 0.0:
                smooth_val = self.label_smoothing / num_classes
                targets_one_hot = targets_one_hot * (1.0 - self.label_smoothing) + smooth_val
        else:
            targets_one_hot = targets.to(inputs.dtype)

        # Calculate focal factor (1 - p_c)^gamma for each class
        focal_weight = (1.0 - probs) ** self.gamma
        loss_per_class = -focal_weight * log_probs * targets_one_hot

        # Apply class weights alpha if provided
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(inputs.device)
            loss_per_class = loss_per_class * alpha_tensor.unsqueeze(0)

        loss = loss_per_class.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
