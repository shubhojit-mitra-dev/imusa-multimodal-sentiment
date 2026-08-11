"""Text Encoder Module for IMUSA Multimodal Classification.

Extracts text embeddings from Punjabi (Gurmukhi) sequences using multilingual
Transformers (e.g. XLM-RoBERTa).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Text Encoder extracting semantic representation vectors from Gurmukhi text.

    Supports pre-trained Hugging Face Transformers (e.g. xlm-roberta-base).

    Attributes:
        model_name: Hugging Face model hub identifier string.
        hidden_dim: Output dimensionality of feature embedding vector (default: 768).
        freeze_backbone: If True, freezes pre-trained backbone parameters.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        hidden_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize Text Encoder.

        Args:
            model_name: Hugging Face model hub path.
            hidden_dim: Hidden dimension size of text embeddings.
            freeze_backbone: Whether to freeze backbone weights during fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        try:
            from transformers import AutoModel

            logger.info("Loading Text Transformer backbone from %s", model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        except Exception as err:
            logger.warning("Could not load Hugging Face model %s: %s. Using Embedding fallback.", model_name, err)
            self.backbone = None
            self.dummy_embedding = nn.Embedding(250000, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract textual embeddings from batch of token sequences.

        Args:
            input_ids: Tensor of token indices of shape (batch_size, max_length).
            attention_mask: Tensor of attention masks of shape (batch_size, max_length).

        Returns:
            Embedding tensor of shape (batch_size, hidden_dim).
        """
        if self.backbone is not None:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            # Mean pooling over tokens using attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                return sum_embeddings / sum_mask
            return outputs.last_hidden_state[:, 0, :]

        # Fallback embedding
        embedded = self.dummy_embedding(input_ids)
        return embedded.mean(dim=1)
