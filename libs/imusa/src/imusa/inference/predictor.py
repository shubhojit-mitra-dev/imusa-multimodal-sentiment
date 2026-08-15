"""Inference module for loading trained multimodal sentiment models and running batch predictions."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from imusa.config import settings
from imusa.models.multimodal import IMUSAMultimodalClassifier

logger = logging.getLogger(__name__)


class IMUSAPredictor:
    """Predictor engine for loading model checkpoints and generating sentiment predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: IMUSAMultimodalClassifier | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize predictor from checkpoint path or pre-instantiated model instance.

        Args:
            checkpoint_path: Path to best_model.pt saved checkpoint file.
            model: Optional pre-loaded IMUSAMultimodalClassifier instance.
            device: Computing device ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(Path(checkpoint_path))
        else:
            logger.info("No checkpoint provided; instantiating default initialized model.")
            self.model = IMUSAMultimodalClassifier()

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: Path) -> IMUSAMultimodalClassifier:
        """Load trained state dict from PyTorch checkpoint.

        Args:
            path: Path to checkpoint file (.pt).

        Returns:
            Instantiated and state-loaded IMUSAMultimodalClassifier model.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at path: {path}")

        logger.info("Loading model checkpoint from %s...", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = IMUSAMultimodalClassifier()
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                "Successfully loaded checkpoint (Epoch %d, Macro F1: %.4f)",
                checkpoint.get("epoch", -1),
                checkpoint.get("macro_f1", 0.0),
            )
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Invalid checkpoint format loaded from {path}")

        return model

    def predict_batch(
        self,
        dataloader: DataLoader[Any],
        thresholds: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate category predictions and confidence scores for a dataset DataLoader.

        Args:
            dataloader: PyTorch DataLoader containing batch items.
            thresholds: Optional calibrated threshold vector for threshold-adjusted classification.

        Returns:
            List of prediction dictionary records containing image_id, predicted_category,
            confidence score, and full class probability breakdown.
        """
        self.model.eval()
        results: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                logits = self.model(images, input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=-1)

                if thresholds is not None:
                    from imusa.evaluation.calibration import apply_calibrated_thresholds

                    probs_np = probabilities.cpu().numpy()
                    pred_indices_np = apply_calibrated_thresholds(probs_np, thresholds)
                else:
                    pred_indices_np = torch.max(probabilities, dim=-1)[1].cpu().numpy()

                confidences, _ = torch.max(probabilities, dim=-1)
                batch_ids = batch.get("image_id", [f"sample_{i}" for i in range(len(images))])

                for i in range(len(images)):
                    pred_idx = int(pred_indices_np[i])
                    conf_val = float(confidences[i].item())
                    prob_dict = {
                        category: float(probabilities[i, c_idx].item())
                        for c_idx, category in enumerate(settings.categories)
                    }

                    results.append(
                        {
                            "image_id": batch_ids[i],
                            "predicted_category": settings.categories[pred_idx],
                            "confidence": conf_val,
                            "probabilities": prob_dict,
                        }
                    )

        logger.info("Generated predictions for %d test samples.", len(results))
        return results


class IMUSAEnsemblePredictor:
    """Ensemble predictor combining predictions across K-fold checkpoints."""

    def __init__(
        self,
        checkpoint_paths: list[str | Path] | None = None,
        models: list[IMUSAMultimodalClassifier] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize Ensemble Predictor.

        Args:
            checkpoint_paths: List of paths to fold checkpoints.
            models: Optional pre-loaded list of model instances.
            device: Computing device ('cuda' or 'cpu').
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.models: list[IMUSAMultimodalClassifier] = []
        if models is not None:
            self.models = models
        elif checkpoint_paths is not None:
            for path in checkpoint_paths:
                p = Path(path)
                if p.exists():
                    predictor = IMUSAPredictor(checkpoint_path=p, device=self.device)
                    self.models.append(predictor.model)
                else:
                    logger.warning("Checkpoint path %s not found; skipping fold.", p)

        for m in self.models:
            m.to(self.device)
            m.eval()

        logger.info("IMUSAEnsemblePredictor initialized with %d fold models.", len(self.models))

    def predict_batch(
        self,
        dataloader: DataLoader[Any],
        thresholds: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate ensemble predictions by averaging class probabilities across all folds."""
        if not self.models:
            raise ValueError("No fold models loaded in IMUSAEnsemblePredictor.")

        results: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                prob_sum = torch.zeros((images.size(0), settings.num_classes), device=self.device)
                for model in self.models:
                    model.eval()
                    logits = model(images, input_ids, attention_mask)
                    prob_sum += F.softmax(logits, dim=-1)

                ensemble_probs = prob_sum / float(len(self.models))

                if thresholds is not None:
                    from imusa.evaluation.calibration import apply_calibrated_thresholds

                    probs_np = ensemble_probs.cpu().numpy()
                    pred_indices_np = apply_calibrated_thresholds(probs_np, thresholds)
                else:
                    pred_indices_np = torch.max(ensemble_probs, dim=-1)[1].cpu().numpy()

                confidences, _ = torch.max(ensemble_probs, dim=-1)
                batch_ids = batch.get("image_id", [f"sample_{i}" for i in range(len(images))])

                for i in range(len(images)):
                    pred_idx = int(pred_indices_np[i])
                    conf_val = float(confidences[i].item())
                    prob_dict = {
                        category: float(ensemble_probs[i, c_idx].item())
                        for c_idx, category in enumerate(settings.categories)
                    }

                    results.append(
                        {
                            "image_id": batch_ids[i],
                            "predicted_category": settings.categories[pred_idx],
                            "confidence": conf_val,
                            "probabilities": prob_dict,
                        }
                    )

        logger.info("Ensemble predictor generated predictions for %d samples.", len(results))
        return results

