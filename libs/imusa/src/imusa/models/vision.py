"""Vision Encoder Module for IMUSA Multimodal Classification.

Extracts visual representation vectors from meme images using a Vision Transformer
(ViT) or Convolutional Backbone.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """Vision Encoder extracting feature embeddings from meme images.

    Supports pre-trained Hugging Face Vision Transformers (e.g. google/vit-base-patch16-224).

    Attributes:
        model_name: Hugging Face model hub identifier string.
        hidden_dim: Output dimensionality of feature embedding vector (default: 768).
        freeze_backbone: If True, freezes pre-trained backbone parameters during training.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize Vision Encoder.

        Args:
            model_name: Hugging Face model hub path.
            hidden_dim: Hidden dimension size of visual embeddings.
            freeze_backbone: Whether to freeze backbone weights for feature extraction.
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        try:
            from transformers import AutoModel

            logger.info("Loading Vision Transformer backbone from %s", model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        except Exception as err:
            logger.warning(
                "Could not load Hugging Face model %s: %s. Using Linear fallback.", model_name, err
            )
            # Fallback for offline unit testing without network downloads
            self.backbone = None
            self.dummy_projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 224 * 224, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual embeddings from batch of meme images.

        Args:
            images: Tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Embedding tensor of shape (batch_size, hidden_dim).
        """
        if self.backbone is not None:
            outputs = self.backbone(pixel_values=images)
            # Use pooler_output if available, else CLS token at index 0
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return cast(torch.Tensor, outputs.pooler_output)
            return cast(torch.Tensor, outputs.last_hidden_state[:, 0, :])
        return cast(torch.Tensor, self.dummy_projection(images))
