#!/usr/bin/env python3
"""CLI Script to execute Exploratory Data Analysis (EDA) on dataset."""

import argparse
import logging
from pathlib import Path

from imusa.data.explorer import explore_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    """Parse CLI args and trigger EDA visualization pipeline."""
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on IMUSA dataset.")
    parser.add_argument("--processed-csv", type=Path, default=None, help="Path to cleaned dataset CSV")
    parser.add_argument("--images-dir", type=Path, default=None, help="Path to training images directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for EDA plot artifacts")

    args = parser.parse_args()

    explore_dataset(
        processed_csv=args.processed_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
