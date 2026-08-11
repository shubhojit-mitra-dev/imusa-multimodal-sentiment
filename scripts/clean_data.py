#!/usr/bin/env python3
"""CLI Script to execute dataset cleaning pipeline."""

import argparse
import logging
from pathlib import Path

from imusa.data.cleaning import clean_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    """Parse CLI args and trigger dataset cleaning."""
    parser = argparse.ArgumentParser(description="Clean raw IMUSA Punjabi dataset.")
    parser.add_argument("--raw-csv", type=Path, default=None, help="Path to raw train CSV file")
    parser.add_argument(
        "--images-dir", type=Path, default=None, help="Path to training images directory"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None, help="Destination path for cleaned CSV"
    )

    args = parser.parse_args()

    clean_dataset(
        raw_csv=args.raw_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
