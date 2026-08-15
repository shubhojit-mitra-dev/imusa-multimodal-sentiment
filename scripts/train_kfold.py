"""CLI script for Stratified K-Fold Training & OOF Prediction Collection for IMUSA V2.

Allows running specific folds (0..4) independently across multiple Google Colab accounts,
saving fold model checkpoints and out-of-fold (OOF) validation probability matrices.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Add libs/imusa/src to sys.path so imusa is importable in any execution environment
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "libs" / "imusa" / "src"))

from imusa.config import settings  # noqa: E402
from imusa.data.cleaning import clean_dataset  # noqa: E402
from imusa.data.dataset import create_kfold_dataloaders  # noqa: E402
from imusa.evaluation.calibration import optimize_thresholds, save_thresholds  # noqa: E402
from imusa.models.multimodal import IMUSAMultimodalClassifier  # noqa: E402
from imusa.training.trainer import Trainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for K-Fold training."""
    parser = argparse.ArgumentParser(description="IMUSA V2 Stratified K-Fold Training")
    parser.add_argument(
        "--fold", type=int, default=0, help="Target fold index to train (0 to num_folds - 1)"
    )
    parser.add_argument(
        "--num-folds", type=int, default=5, help="Total number of stratified folds (default: 5)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs per fold")
    parser.add_argument(
        "--lp-epochs", type=int, default=3, help="Linear Probing initial epochs (default: 3)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Fine-tuning learning rate")
    parser.add_argument(
        "--text-model", type=str, default="google/muril-base-cased", help="Text encoder model"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="google/vit-base-patch16-224",
        help="Vision encoder model",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run threshold calibration across saved OOF files"
    )
    return parser.parse_args()


def train_single_fold(args: argparse.Namespace, fold_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Train a single fold model and save OOF validation predictions.

    Args:
        args: Command line argument namespace.
        fold_idx: Integer fold index (0 to args.num_folds - 1).

    Returns:
        Tuple of (oof_probs_array, oof_targets_array).
    """
    logger.info(
        "--- Starting Stratified K-Fold Training: Fold %d/%d ---", fold_idx + 1, args.num_folds
    )

    # 1. Clean dataset
    df = clean_dataset()

    # 2. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    train_loader, val_loader = create_kfold_dataloaders(
        df=df,
        images_dir=settings.train_images_dir,
        tokenizer=tokenizer,
        num_folds=args.num_folds,
        fold=fold_idx,
        batch_size=args.batch_size,
    )

    # 3. Model & Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IMUSAMultimodalClassifier(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
    ).to(device)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=device,
        output_dir=settings.versioned_output_dir,
    )

    # Override checkpoint path for specific fold
    fold_ckpt = settings.checkpoint_dir / f"best_model_fold_{fold_idx}.pt"

    # 4. Run LP-FT multi-stage training
    trainer.fit_lpft(lp_epochs=args.lp_epochs, ft_epochs=args.epochs - args.lp_epochs)

    # Copy best checkpoint to fold checkpoint name
    if trainer.best_checkpoint_path and trainer.best_checkpoint_path.exists():
        import shutil

        shutil.copy(trainer.best_checkpoint_path, fold_ckpt)
        logger.info("Saved Fold %d best checkpoint to %s", fold_idx, fold_ckpt)

    # 5. Extract OOF probabilities on validation set
    oof_probs, oof_targets = trainer.evaluate_probabilities(val_loader)
    np.save(settings.versioned_output_dir / f"oof_probs_fold_{fold_idx}.npy", oof_probs)
    np.save(settings.versioned_output_dir / f"oof_targets_fold_{fold_idx}.npy", oof_targets)

    return oof_probs, oof_targets


def main() -> None:
    """Main execution entrypoint for K-Fold training."""
    args = parse_args()

    # Create output directories
    settings.versioned_output_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        logger.info("--- Performing Post-Hoc Threshold Calibration Across Available Folds ---")
        all_probs = []
        all_targets = []

        for k in range(args.num_folds):
            p_file = settings.versioned_output_dir / f"oof_probs_fold_{k}.npy"
            t_file = settings.versioned_output_dir / f"oof_targets_fold_{k}.npy"
            if p_file.exists() and t_file.exists():
                all_probs.append(np.load(p_file))
                all_targets.append(np.load(t_file))
            else:
                logger.warning("Fold %d OOF file missing; skipping.", k)

        if all_probs:
            concat_probs = np.concatenate(all_probs, axis=0)
            concat_targets = np.concatenate(all_targets, axis=0)
            opt_tau, best_f1 = optimize_thresholds(concat_targets, concat_probs)
            save_thresholds(opt_tau)
            logger.info("Calibration successful: Macro F1 = %.4f", best_f1)
        else:
            logger.error("No OOF files found to perform threshold calibration.")
    else:
        train_single_fold(args, args.fold)


if __name__ == "__main__":
    main()
