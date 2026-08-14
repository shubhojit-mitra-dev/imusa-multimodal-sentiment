"""Unit tests for evaluation and plotting functions."""

from pathlib import Path

from imusa.evaluation.evaluator import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
)


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Verify plot_confusion_matrix generates PNG file without error."""
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 2, 0, 1, 3, 3]
    out_file = tmp_path / "confusion_matrix.png"

    result_path = plot_confusion_matrix(y_true, y_pred, output_path=out_file)

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_plot_training_curves(tmp_path: Path) -> None:
    """Verify plot_training_curves generates plot file from history dicts."""
    history = [
        {"epoch": 1, "train_loss": 0.8, "val_loss": 0.6, "macro_f1": 0.25},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4, "macro_f1": 0.45},
    ]
    out_file = tmp_path / "training_curves.png"

    result_path = plot_training_curves(history, output_path=out_file)

    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_plot_per_class_f1(tmp_path: Path) -> None:
    """Verify plot_per_class_f1 generates bar chart PNG file."""
    scores = {
        "Sarcasm": 0.75,
        "Neutral": 0.60,
        "Offensive": 0.40,
        "Motivational": 0.65,
    }
    out_file = tmp_path / "per_class_f1.png"

    result_path = plot_per_class_f1(scores, output_path=out_file)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
