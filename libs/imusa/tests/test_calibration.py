"""Unit tests for Phase E post-hoc threshold calibration module."""

from pathlib import Path

import numpy as np

from imusa.evaluation.calibration import (
    apply_calibrated_thresholds,
    load_thresholds,
    optimize_thresholds,
    save_thresholds,
)


def test_apply_calibrated_thresholds() -> None:
    """Verify apply_calibrated_thresholds correctly shifts decision boundaries."""
    # Probabilities: Class 0=0.45, Class 1=0.55 (default argmax would pick Class 1)
    probs = np.array([[0.45, 0.55, 0.0, 0.0]], dtype=np.float32)

    # Thresholds: tau_0 = 0.5, tau_1 = 2.0 -> scaled: Class 0 = 0.45/0.5 = 0.9, Class 1 = 0.55/2.0 = 0.275
    thresholds = [0.5, 2.0, 1.0, 1.0]
    preds = apply_calibrated_thresholds(probs, thresholds)

    assert preds.shape == (1,)
    assert preds[0] == 0  # Shifted to Class 0 due to threshold adjustment


def test_optimize_thresholds() -> None:
    """Verify optimize_thresholds runs Nelder-Mead search and returns valid threshold vector."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    # Construct noisy probabilities
    probs = np.random.dirichlet(np.ones(4), size=8)

    optimal_tau, best_f1 = optimize_thresholds(y_true, probs, num_classes=4)

    assert len(optimal_tau) == 4
    assert all(t > 0.0 for t in optimal_tau)
    assert 0.0 <= best_f1 <= 1.0


def test_save_load_thresholds(tmp_path: Path) -> None:
    """Verify save_thresholds and load_thresholds JSON persistence."""
    thresholds = [0.8, 1.2, 0.5, 1.5]
    file_path = tmp_path / "thresholds.json"

    saved_path = save_thresholds(thresholds, path=file_path)
    assert saved_path.exists()

    loaded = load_thresholds(path=file_path)
    assert len(loaded) == 4
    assert np.allclose(loaded, thresholds)
