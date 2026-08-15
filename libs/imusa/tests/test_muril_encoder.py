"""Unit tests for MuRIL text encoder integration and backbone freeze/unfreeze functionality."""

import torch

from imusa.models.text import TextEncoder
from imusa.models.vision import VisionEncoder


def test_text_encoder_muril_initialization() -> None:
    """Verify TextEncoder initializes cleanly with MuRIL configuration."""
    encoder = TextEncoder(model_name="dummy/muril-base-cased", hidden_dim=768)
    assert encoder.hidden_dim == 768
    assert encoder.model_name == "dummy/muril-base-cased"

    input_ids = torch.ones((2, 16), dtype=torch.long)
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    output = encoder(input_ids, attention_mask=attention_mask)

    assert output.shape == (2, 768)
    assert not torch.isnan(output).any()


def test_text_encoder_freeze_unfreeze() -> None:
    """Verify freeze() and unfreeze() toggle requires_grad on text encoder parameters."""
    encoder = TextEncoder(model_name="dummy/xlm-roberta-base", hidden_dim=768)
    encoder.freeze()
    if encoder.backbone is not None:
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    encoder.unfreeze()
    if encoder.backbone is not None:
        for param in encoder.backbone.parameters():
            assert param.requires_grad


def test_vision_encoder_freeze_unfreeze() -> None:
    """Verify freeze() and unfreeze() toggle requires_grad on vision encoder parameters."""
    encoder = VisionEncoder(model_name="dummy/vit-base-patch16-224", hidden_dim=768)
    encoder.freeze()
    if encoder.backbone is not None:
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    encoder.unfreeze()
    if encoder.backbone is not None:
        for param in encoder.backbone.parameters():
            assert param.requires_grad
