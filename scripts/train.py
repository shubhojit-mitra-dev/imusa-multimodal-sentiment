#!/usr/bin/env python3
"""CLI Entrypoint Script for Multimodal Baseline Model Training."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add libs/imusa/src to sys.path so imusa is importable in any execution environment
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "libs" / "imusa" / "src"))

from imusa.config import settings  # noqa: E402
from imusa.data.cleaning import clean_dataset  # noqa: E402
from imusa.data.dataset import create_stratified_dataloaders  # noqa: E402
from imusa.models.loss import FocalLoss, compute_inverse_class_weights  # noqa: E402
from imusa.models.multimodal import IMUSAMultimodalClassifier  # noqa: E402
from imusa.training.trainer import Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("imusa.scripts.train")


def parse_args() -> argparse.Namespace:
    """Parse command line options for model training."""
    parser = argparse.ArgumentParser(description="Train IMUSA Multimodal Sentiment Model")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["weighted_ce", "focal"],
        default="focal",
        help="Imbalance-aware loss function type (default: focal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute fast 1-epoch dry-run on 20 samples for verification",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup step ratio for cosine learning rate scheduler (default: 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute model training command."""
    args = parse_args()

    logger.info("=== IMUSA Multimodal Model Training Initialization ===")
    df = clean_dataset()

    if args.dry_run:
        logger.info("Executing dry-run mode on top 20 dataset samples...")
        df = df.head(20).copy()
        epochs = 1
    else:
        epochs = args.epochs

    # Create dummy tokenizer or HF tokenizer wrapper
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    except Exception as err:
        logger.warning("Could not download HF tokenizer: %s. Using dummy tokenizer fallback.", err)

        class DummyTokenizerWrapper:
            def __call__(self, text: str, **kwargs: str | int | bool) -> dict[str, torch.Tensor]:
                max_len = int(kwargs.get("max_length", 128))
                return {
                    "input_ids": torch.ones((1, max_len), dtype=torch.long),
                    "attention_mask": torch.ones((1, max_len), dtype=torch.long),
                }

        tokenizer = DummyTokenizerWrapper()

    train_loader, val_loader = create_stratified_dataloaders(
        df=df,
        images_dir=settings.train_images_dir,
        tokenizer=tokenizer,
        val_ratio=0.2,
        batch_size=args.batch_size,
        num_workers=0 if args.dry_run else 2,
    )

    # Compute inverse class weights for loss function
    class_weights = compute_inverse_class_weights(df, num_classes=4)

    criterion: nn.Module
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = IMUSAMultimodalClassifier(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    from imusa.training.trainer import get_cosine_schedule_with_warmup

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=settings.output_dir / "checkpoints",
    )

    results = trainer.fit(epochs=epochs)
    logger.info("Training finished. Best Validation Macro F1 Score: %.4f", results["best_macro_f1"])


if __name__ == "__main__":
    main()
