"""CLI entrypoint script for running inference on unlabeled test dataset and generating submission CSV."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from imusa.config import settings
from imusa.data.dataset import IMUSADataset
from imusa.inference.predictor import IMUSAPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_prediction_pipeline(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> Path:
    """Execute end-to-end test set inference and generate submission CSV.

    Args:
        checkpoint_path: Optional path to trained model checkpoint file (.pt).
        output_path: Destination path for formatted submission CSV.
        batch_size: Mini-batch size for prediction dataloader.
        device: Computing device ('cuda' or 'cpu').

    Returns:
        Path object of saved submission CSV file.
    """
    settings.ensure_directories()

    if checkpoint_path is None:
        candidate_checkpoint = settings.checkpoint_dir / "best_model.pt"
        if not candidate_checkpoint.exists():
            candidate_checkpoint = settings.output_dir / "checkpoints" / "best_model.pt"
        checkpoint_path = candidate_checkpoint
    else:
        checkpoint_path = Path(checkpoint_path)

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

    model_name = getattr(settings, "text_model_name", "xlm-roberta-base")
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

    logger.info("Initializing predictor from %s...", checkpoint_path)
    predictor = IMUSAPredictor(checkpoint_path=checkpoint_path, device=device)

    logger.info("Generating predictions...")
    predictions = predictor.predict_batch(test_loader)

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

    args = parser.parse_args()
    run_prediction_pipeline(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
