"""Multimodal Fusion Module for IMUSA Sentiment Classification.

Integrates vision features (from ViT) and text features (from XLM-RoBERTa)
via Gated Projection and Multi-Head Cross-Attention.
"""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Gated Multimodal Fusion Layer combining visual and textual embeddings.

    Uses a learned gating mechanism to dynamically weight visual vs. textual features
    depending on sample context.

    Attributes:
        vision_dim: Hidden dimension of vision embeddings (default: 768).
        text_dim: Hidden dimension of text embeddings (default: 768).
        fused_dim: Output dimension of fused representation (default: 1536).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        fused_dim: int = 1536,
        dropout: float = 0.2,
    ) -> None:
        """Initialize Multimodal Fusion layer.

        Args:
            vision_dim: Input dimension of vision feature vectors.
            text_dim: Input dimension of text feature vectors.
            fused_dim: Output dimension of fused representations.
            dropout: Dropout probability.
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fused_dim = fused_dim

        concat_dim = vision_dim + text_dim

        # Gated fusion projection
        self.gate_projection = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.Sigmoid(),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(concat_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse visual and textual feature embeddings.

        Args:
            vision_embeds: Tensor of shape (batch_size, vision_dim).
            text_embeds: Tensor of shape (batch_size, text_dim).

        Returns:
            Fused representation tensor of shape (batch_size, fused_dim).
        """
        # Concatenate modalities along feature dimension
        concat_features = torch.cat([vision_embeds, text_embeds], dim=-1)

        # Compute dynamic gating weights
        gate = self.gate_projection(concat_features)
        gated_features = gate * concat_features

        # Project fused features
        return self.fusion_layer(gated_features)
