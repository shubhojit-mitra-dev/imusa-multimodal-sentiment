#!/usr/bin/env python3
"""Migration script to organize output artifacts into versioned structures.

Moves legacy outputs/checkpoints/best_model.pt to outputs/v1/checkpoints/best_model.pt
and legacy outputs/submission.csv to outputs/v1/submission.csv while creating
necessary directory structures for V1 and V2 outputs.
"""

import logging
import shutil

from imusa.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("imusa.scripts.migrate_v1_outputs")


def migrate_outputs() -> None:
    """Migrate legacy outputs to outputs/v1 and prepare outputs/v2 directory structure."""
    settings.ensure_directories()

    output_dir = settings.output_dir
    v1_dir = output_dir / "v1"
    v1_checkpoints = v1_dir / "checkpoints"
    v2_dir = output_dir / "v2"
    v2_checkpoints = v2_dir / "checkpoints"
    v2_calibration = v2_dir / "calibration"

    for d in [v1_dir, v1_checkpoints, v2_dir, v2_checkpoints, v2_calibration]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Migrate legacy checkpoint
    legacy_checkpoint = output_dir / "checkpoints" / "best_model.pt"
    v1_checkpoint = v1_checkpoints / "best_model.pt"

    if legacy_checkpoint.exists() and not v1_checkpoint.exists():
        logger.info(
            "Migrating legacy checkpoint from %s to %s...", legacy_checkpoint, v1_checkpoint
        )
        shutil.move(str(legacy_checkpoint), str(v1_checkpoint))
    elif v1_checkpoint.exists():
        logger.info("V1 checkpoint already present at %s.", v1_checkpoint)
    else:
        logger.info("No legacy checkpoint found at %s.", legacy_checkpoint)

    # 2. Migrate legacy submission CSV
    legacy_submission = output_dir / "submission.csv"
    v1_submission = v1_dir / "submission.csv"

    if legacy_submission.exists() and not v1_submission.exists():
        logger.info(
            "Migrating legacy submission CSV from %s to %s...", legacy_submission, v1_submission
        )
        shutil.move(str(legacy_submission), str(v1_submission))
    elif v1_submission.exists():
        logger.info("V1 submission CSV already present at %s.", v1_submission)
    else:
        logger.info("No legacy submission CSV found at %s.", legacy_submission)

    logger.info("Output directory migration complete. Structure:")
    logger.info("  V1 Dir: %s", v1_dir)
    logger.info("  V2 Dir: %s", v2_dir)


if __name__ == "__main__":
    migrate_outputs()
