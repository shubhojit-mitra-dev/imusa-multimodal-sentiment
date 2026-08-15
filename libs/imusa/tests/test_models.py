"""Unit tests for Multimodal Model Architecture components."""

import torch

from imusa.models.fusion import MultimodalFusion
from imusa.models.multimodal import IMUSAMultimodalClassifier
from imusa.models.text import TextEncoder
from imusa.models.vision import VisionEncoder


def test_vision_encoder_forward() -> None:
    """Verify VisionEncoder output tensor shape (batch_size, 768)."""
    encoder = VisionEncoder(model_name="dummy/vit-base", hidden_dim=768)
    dummy_images = torch.randn(2, 3, 224, 224)

    output = encoder(dummy_images)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 768)


def test_text_encoder_forward() -> None:
    """Verify TextEncoder output tensor shape (batch_size, 768)."""
    encoder = TextEncoder(model_name="dummy/xlm-roberta", hidden_dim=768)
    dummy_input_ids = torch.ones(2, 64, dtype=torch.long)
    dummy_mask = torch.ones(2, 64, dtype=torch.long)

    output = encoder(dummy_input_ids, attention_mask=dummy_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 768)


def test_multimodal_fusion_forward() -> None:
    """Verify MultimodalFusion output tensor shape (batch_size, 1536)."""
    fusion = MultimodalFusion(vision_dim=768, text_dim=768, fused_dim=1536)
    v_embeds = torch.randn(2, 768)
    t_embeds = torch.randn(2, 768)

    fused = fusion(v_embeds, t_embeds)

    assert isinstance(fused, torch.Tensor)
    assert fused.shape == (2, 1536)


def test_multimodal_classifier_end_to_end() -> None:
    """Verify end-to-end forward pass through IMUSAMultimodalClassifier produces (batch_size, 4)."""
    model = IMUSAMultimodalClassifier(
        num_classes=4, vit_model_name="dummy/vit", text_model_name="dummy/xlm-r"
    )
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_input_ids = torch.ones(2, 64, dtype=torch.long)
    dummy_mask = torch.ones(2, 64, dtype=torch.long)

    logits = model(dummy_images, dummy_input_ids, attention_mask=dummy_mask)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
