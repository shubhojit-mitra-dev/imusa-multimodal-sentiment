"""Post-Hoc Threshold Calibration Engine for IMUSA Macro F1 Maximization.

Optimizes class-specific decision thresholds using Nelder-Mead simplex search on validation
probabilities to counteract majority-class prior bias on extreme class imbalances.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from imusa.config import settings

logger = logging.getLogger(__name__)


def apply_calibrated_thresholds(
    probs: np.ndarray, thresholds: list[float] | np.ndarray
) -> np.ndarray:
    """Apply calibrated decision threshold scaling factors to probability matrix.

    Formula:
        y_hat = argmax_c ( P_{i, c} / tau_c )

    Args:
        probs: Probability matrix of shape (N, K).
        thresholds: Threshold vector of shape (K,) with positive non-zero entries.

    Returns:
        1D numpy array of predicted class indices of shape (N,).
    """
    tau = np.array(thresholds, dtype=np.float64)
    tau = np.maximum(tau, 1e-5)  # Avoid division by zero
    scaled_probs = probs / tau
    return np.argmax(scaled_probs, axis=-1)


def optimize_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 4,
) -> tuple[list[float], float]:
    """Optimize decision thresholds using Nelder-Mead search to maximize validation Macro F1 score.

    Args:
        y_true: Ground truth target array of shape (N,).
        probs: Model predicted probability matrix of shape (N, K).
        num_classes: Total number of target categories (default: 4).

    Returns:
        Tuple of (optimal_threshold_list, best_macro_f1_score).
    """
    initial_tau = np.ones(num_classes, dtype=np.float64)

    def objective(tau: np.ndarray) -> float:
        preds = apply_calibrated_thresholds(probs, tau)
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        return -macro_f1  # Minimize negative Macro F1

    result = minimize(
        objective,
        initial_tau,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3},
    )

    optimal_tau = list(np.maximum(result.x, 1e-5))
    best_macro_f1 = -float(result.fun)

    uncalibrated_preds = np.argmax(probs, axis=-1)
    uncalibrated_f1 = float(f1_score(y_true, uncalibrated_preds, average="macro", zero_division=0))

    logger.info(
        "Threshold calibration complete: Uncalibrated Macro F1=%.4f -> Calibrated Macro F1=%.4f (+%.2f%%)",
        uncalibrated_f1,
        best_macro_f1,
        (best_macro_f1 - uncalibrated_f1) * 100.0,
    )
    logger.info("Optimal threshold vector: %s", optimal_tau)

    return optimal_tau, best_macro_f1


def save_thresholds(thresholds: list[float], path: Path | None = None) -> Path:
    """Save calibrated threshold vector to JSON configuration file.

    Args:
        thresholds: List of float threshold entries.
        path: Target file path (defaults to settings.calibration_dir / 'thresholds.json').

    Returns:
        Path object of saved threshold JSON file.
    """
    dest = path or (settings.calibration_dir / "thresholds.json")
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "categories": settings.categories,
        "thresholds": [float(t) for t in thresholds],
    }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved calibrated thresholds to %s", dest)
    return dest


def load_thresholds(path: Path | None = None) -> list[float]:
    """Load calibrated threshold vector from JSON configuration file.

    Args:
        path: Source file path (defaults to settings.calibration_dir / 'thresholds.json').

    Returns:
        List of float threshold entries.
    """
    src = path or (settings.calibration_dir / "thresholds.json")
    if not src.exists():
        logger.warning("Threshold configuration file not found at %s. Returning default ones.", src)
        return [1.0] * settings.num_classes

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    return [float(t) for t in data["thresholds"]]
