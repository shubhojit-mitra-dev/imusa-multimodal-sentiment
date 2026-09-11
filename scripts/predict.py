"""CLI entrypoint script for running inference on unlabeled test dataset and generating submission CSV."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

# Add libs/imusa/src to sys.path so imusa is importable in any execution environment
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "libs" / "imusa" / "src"))

from imusa.config import settings  # noqa: E402
from imusa.data.dataset import IMUSADataset  # noqa: E402
from imusa.inference.predictor import IMUSAPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_prediction_pipeline(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    model_version: str = "v2",
    use_ensemble: bool = False,
    use_calibration: bool = False,
) -> Path:
    """Execute end-to-end test set inference and generate submission CSV.

    Args:
        checkpoint_path: Optional path to trained model checkpoint file (.pt).
        output_path: Destination path for formatted submission CSV.
        batch_size: Mini-batch size for prediction dataloader.
        device: Computing device ('cuda' or 'cpu').
        model_version: Model version identifier (v1 or v2).
        use_ensemble: Whether to average predictions across K-fold checkpoints.
        use_calibration: Whether to apply post-hoc calibrated decision thresholds.

    Returns:
        Path object of saved submission CSV file.
    """
    settings.model_version = model_version
    settings.ensure_directories()

    if output_path is None:
        output_path = settings.submission_path
    else:
        output_path = Path(output_path)

    test_csv_path = settings.raw_test_csv
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Raw test CSV file not found at: {test_csv_path}")

    logger.info("Reading test dataset from %s...", test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    logger.info("Loaded %d test samples.", len(test_df))

    from transformers import AutoTokenizer

    # Choose text model based on version
    model_name = "google/muril-base-cased" if model_version == "v2" else settings.text_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = IMUSADataset(
        df=test_df,
        images_dir=settings.test_images_dir,
        tokenizer=tokenizer,
        is_test=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    thresholds: list[float] | None = None
    if use_calibration:
        from imusa.evaluation.calibration import load_thresholds

        thresholds = load_thresholds()
        logger.info("Loaded calibrated thresholds: %s", thresholds)

    if use_ensemble:
        import glob

        from imusa.inference.predictor import IMUSAEnsemblePredictor

        # Collect fold checkpoints from v2 (or fallback to v1)
        ckpts = sorted(glob.glob(str(settings.checkpoint_dir / "best_model_fold_*.pt")))
        if not ckpts:
            ckpts = sorted(
                glob.glob(str(settings.output_dir / "v1" / "checkpoints" / "best_model_fold_*.pt"))
            )
        logger.info("Initializing EnsemblePredictor with %d checkpoints: %s", len(ckpts), ckpts)
        predictor = IMUSAEnsemblePredictor(checkpoint_paths=ckpts, device=device)  # type: ignore[assignment]
        predictions = predictor.predict_batch(test_loader, thresholds=thresholds)  # type: ignore[attr-defined]
    else:
        if checkpoint_path is None:
            candidate_checkpoint = settings.checkpoint_dir / "best_model.pt"
            if not candidate_checkpoint.exists():
                candidate_checkpoint = settings.output_dir / "checkpoints" / "best_model.pt"
            checkpoint_path = candidate_checkpoint
        else:
            checkpoint_path = Path(checkpoint_path)

        logger.info("Initializing single predictor from %s...", checkpoint_path)
        single_predictor = IMUSAPredictor(checkpoint_path=checkpoint_path, device=device)
        predictions = single_predictor.predict_batch(test_loader)

    # Format submission DataFrame matching competition format (Id, Category, Text)
    submission_df = test_df.copy()
    pred_map = {item["image_id"]: item["predicted_category"] for item in predictions}

    submission_df["Category"] = submission_df["Id"].map(pred_map).fillna("Neutral")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logger.info("Saved submission CSV to %s (%d rows)", output_path, len(submission_df))

    # Log category distribution of predictions
    dist = submission_df["Category"].value_counts().to_dict()
    logger.info("Test set prediction category distribution: %s", dist)

    return output_path


def main() -> None:
    """Parse CLI arguments and run prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="IMUSA Multimodal Sentiment Test Set Prediction Script"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint file (.pt). Defaults to outputs/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output submission CSV. Defaults to outputs/submission.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for test set dataloader (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        help="Model version (v1 or v2, default: v2)",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use 5-fold ensemble predictor across all trained folds",
    )
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Apply post-hoc calibrated decision thresholds",
    )

    args = parser.parse_args()
    run_prediction_pipeline(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        model_version=args.model_version,
        use_ensemble=args.use_ensemble,
        use_calibration=args.use_calibration,
    )


if __name__ == "__main__":
    main()
