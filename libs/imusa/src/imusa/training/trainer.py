"""Training Orchestration and Evaluation Engine for IMUSA.

Implements the Trainer class managing PyTorch training loops, mixed precision,
Macro F1 evaluation, per-class metric reporting, and checkpoint management.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader

from imusa.config import settings

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer instance.
        num_warmup_steps: Number of steps for linear warmup phase.
        num_training_steps: Total number of training optimization steps.
        min_lr_ratio: Minimum LR ratio relative to initial LR.

    Returns:
        torch.optim.lr_scheduler.LambdaLR instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """IMUSA Model Trainer managing training, evaluation, and checkpointing.

    Attributes:
        model: PyTorch neural network model.
        train_loader: Training DataLoader instance.
        val_loader: Validation DataLoader instance.
        criterion: PyTorch loss function (e.g. FocalLoss or WeightedCrossEntropy).
        optimizer: PyTorch optimizer (e.g. AdamW).
        scheduler: Optional learning rate scheduler.
        device: Target execution device ('cuda', 'mps', or 'cpu').
        output_dir: Destination directory for saving model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        device: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize Trainer instance.

        Args:
            model: Neural network model.
            train_loader: DataLoader containing training samples.
            val_loader: DataLoader containing validation samples.
            criterion: Loss function module.
            optimizer: Optimizer instance.
            scheduler: Optional learning rate scheduler.
            device: Computing device string (default: CUDA if available, else CPU).
            output_dir: Directory for storing output checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.output_dir = output_dir or (settings.output_dir / "checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        logger.info("Trainer initialized on device: %s", self.device)

    def train_epoch(self) -> float:
        """Train model for one full epoch.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    def evaluate(self) -> dict[str, Any]:
        """Evaluate model performance on validation set.

        Returns:
            Dictionary containing 'val_loss', 'accuracy', 'macro_f1', 'weighted_f1',
            and per-class precision, recall, and f1-scores.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images, input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        acc = float(accuracy_score(labels_arr, preds_arr))
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_arr, preds_arr, average="macro", zero_division=0
        )
        weighted_f1 = float(
            precision_recall_fscore_support(
                labels_arr, preds_arr, average="weighted", zero_division=0
            )[2]
        )

        metrics: dict[str, Any] = {
            "val_loss": avg_loss,
            "accuracy": acc,
            "macro_f1": float(f1),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "weighted_f1": weighted_f1,
            "classification_report": classification_report(
                labels_arr,
                preds_arr,
                target_names=settings.categories[: len(np.unique(labels_arr))],
                zero_division=0,
            ),
        }

        return metrics

    def fit(self, epochs: int = 5) -> dict[str, Any]:
        """Execute full model training over specified number of epochs.

        Args:
            epochs: Number of training epochs (default: 5).

        Returns:
            Dictionary containing best validation Macro F1 score and epoch logs.
        """
        best_macro_f1 = -1.0
        best_checkpoint_path = self.output_dir / "best_model.pt"

        logger.info("Starting training loop for %d epochs...", epochs)

        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()

            macro_f1 = val_metrics["macro_f1"]
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro F1: %.4f",
                epoch,
                epochs,
                train_loss,
                val_metrics["val_loss"],
                val_metrics["accuracy"],
                macro_f1,
            )

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
            history.append(epoch_log)

            # Checkpoint best model based on Macro F1 score
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "macro_f1": best_macro_f1,
                    },
                    best_checkpoint_path,
                )
                logger.info(
                    "Saved new best model checkpoint to %s (Macro F1=%.4f)",
                    best_checkpoint_path,
                    best_macro_f1,
                )

        return {"best_macro_f1": best_macro_f1, "history": history}
