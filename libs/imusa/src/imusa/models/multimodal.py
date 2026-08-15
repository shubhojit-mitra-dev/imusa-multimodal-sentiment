"""End-to-End Multimodal Classifier Model for IMUSA.

Combines VisionEncoder, TextEncoder, and MultimodalFusion into a unified
deep neural network architecture for 4-class Punjabi meme sentiment classification.
"""

import logging
from typing import cast

import torch
import torch.nn as nn

from imusa.models.fusion import MultimodalFusion
from imusa.models.text import TextEncoder
from imusa.models.vision import VisionEncoder

logger = logging.getLogger(__name__)


class IMUSAMultimodalClassifier(nn.Module):
    """IMUSA Multimodal Sentiment Classifier Model.

    Architecture:
        1. VisionEncoder (ViT): Images -> (B, 768)
        2. TextEncoder (XLM-RoBERTa): Gurmukhi Text -> (B, 768)
        3. MultimodalFusion Layer: (B, 768) + (B, 768) -> (B, 1536)
        4. Classification Head: Linear(1536, 512) -> LayerNorm -> GELU -> Dropout -> Linear(512, 4)

    Attributes:
        num_classes: Number of sentiment categories (default: 4).
        vision_encoder: VisionEncoder instance.
        text_encoder: TextEncoder instance.
        fusion: MultimodalFusion instance.
        classifier: Feed-forward neural network output head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        vit_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "xlm-roberta-base",
        freeze_backbones: bool = False,
        dropout: float = 0.3,
        vision_model_name: str | None = None,
    ) -> None:
        """Initialize Multimodal Sentiment Classifier.

        Args:
            num_classes: Output class count (4 for Sarcasm, Motivational, Neutral, Offensive).
            vit_model_name: Hugging Face model identifier for Vision Transformer.
            text_model_name: Hugging Face model identifier for Text Transformer.
            freeze_backbones: If True, freezes pre-trained transformer backbones.
            dropout: Dropout probability in classification head.
            vision_model_name: Alias for vit_model_name.
        """
        super().__init__()
        if vision_model_name is not None:
            vit_model_name = vision_model_name

        self.num_classes = num_classes

        self.vision_encoder = VisionEncoder(
            model_name=vit_model_name,
            hidden_dim=768,
            freeze_backbone=freeze_backbones,
        )

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            hidden_dim=768,
            freeze_backbone=freeze_backbones,
        )

        self.fusion = MultimodalFusion(
            vision_dim=768,
            text_dim=768,
            fused_dim=1536,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        logger.info(
            "Initialized IMUSAMultimodalClassifier (num_classes=%d, freeze_backbones=%s)",
            num_classes,
            freeze_backbones,
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through multimodal neural network.

        Args:
            images: Tensor of shape (batch_size, 3, 224, 224).
            input_ids: Tensor of shape (batch_size, max_length).
            attention_mask: Tensor of shape (batch_size, max_length).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # 1. Extract visual representations
        vision_embeds = self.vision_encoder(images)

        # 2. Extract textual representations
        text_embeds = self.text_encoder(input_ids, attention_mask=attention_mask)

        # 3. Fuse modalities
        fused_embeds = self.fusion(vision_embeds, text_embeds)

        # 4. Predict sentiment logits
        return cast(torch.Tensor, self.classifier(fused_embeds))

    def freeze_backbones(self) -> None:
        """Freeze both vision and text encoder backbone weights for Linear Probing phase."""
        self.vision_encoder.freeze()
        self.text_encoder.freeze()
        logger.info("Froze pre-trained backbones for Linear Probing phase.")

    def unfreeze_backbones(self) -> None:
        """Unfreeze both vision and text encoder backbone weights for Fine-Tuning phase."""
        self.vision_encoder.unfreeze()
        self.text_encoder.unfreeze()
        logger.info("Unfroze pre-trained backbones for end-to-end Fine-Tuning phase.")

    def forward_with_mixup(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        alpha: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass applying Manifold Mixup in the fused embedding space.

        Args:
            images: Tensor of shape (batch_size, 3, 224, 224).
            input_ids: Tensor of shape (batch_size, max_length).
            attention_mask: Tensor of shape (batch_size, max_length).
            alpha: Beta distribution parameter for mixup ratio sampling.

        Returns:
            Tuple of (mixed_logits, permutation_indices, lambda_value).
        """
        vision_embeds = self.vision_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask=attention_mask)
        fused_embeds = self.fusion(vision_embeds, text_embeds)

        batch_size = fused_embeds.size(0)
        perm = torch.randperm(batch_size, device=fused_embeds.device)

        if alpha > 0.0 and self.training:
            import numpy as np

            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0

        mixed_fused = lam * fused_embeds + (1.0 - lam) * fused_embeds[perm]
        logits = self.classifier(mixed_fused)

        return cast(torch.Tensor, logits), perm, lam
