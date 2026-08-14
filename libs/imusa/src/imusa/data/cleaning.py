"""Dataset Cleaning and Validation Pipeline for IMUSA."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from imusa.config import settings

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CleaningSummary:
    """Dataclass holding summary metrics of data cleaning process."""

    total_raw_rows: int
    parsed_rows: int
    missing_category_rows: int
    invalid_category_rows: int
    missing_image_rows: int
    duplicate_rows: int
    final_clean_rows: int


def parse_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Parse raw CSV file handling multiline strings and quotes cleanly.

    Args:
        csv_path: Path to the raw CSV file.

    Returns:
        DataFrame containing parsed raw rows.
    """
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        # Use python csv reader with universal newline handling
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV file at {csv_path} is empty.")

        for line_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            # Handle rows with split or extra fields
            if len(row) >= 3:
                image_id = row[0].strip()
                category = row[1].strip()
                text = ",".join(row[2:]).strip()  # In case comma was inside text
                rows.append({"Id": image_id, "Category": category, "Text": text})
            elif len(row) == 2:
                rows.append({"Id": row[0].strip(), "Category": row[1].strip(), "Text": ""})
            else:
                logger.warning("Skipping unparseable line %d: %s", line_idx, row)

    return pd.DataFrame(rows)


def clean_dataset(
    raw_csv: Path | None = None,
    images_dir: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Clean, validate, and sanitize the IMUSA training dataset.

    Args:
        raw_csv: Custom path to raw training CSV.
        images_dir: Custom path to training images directory.
        output_csv: Custom destination path for processed CSV.

    Returns:
        Cleaned pandas DataFrame.
    """
    settings.ensure_directories()
    raw_path = raw_csv or settings.raw_train_csv
    img_dir = images_dir or settings.train_images_dir
    dest_path = output_csv or settings.processed_train_csv

    logger.info("Starting dataset cleaning pipeline on %s", raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"\n\n[ERROR] Raw dataset CSV file not found at: '{raw_path}'\n"
            f"If running on Google Colab, ensure you upload your 'data.zip' or mount Google Drive containing the 'data/' directory!\n"
        )

    # 1. Read Raw Data
    df_raw = parse_raw_csv(raw_path)
    total_raw = len(df_raw)

    # 2. Filter Missing or Invalid Category
    valid_categories = set(settings.categories)
    missing_cat_mask = df_raw["Category"].isna() | (df_raw["Category"] == "")
    missing_cat_count = int(missing_cat_mask.sum())

    df_valid_cat = df_raw[~missing_cat_mask].copy()

    invalid_cat_mask = ~df_valid_cat["Category"].isin(valid_categories)
    invalid_cat_count = int(invalid_cat_mask.sum())

    df_filtered_cat = df_valid_cat[~invalid_cat_mask].copy()

    # 3. Verify Image File Existence
    # Some CSV entries may be missing the file extension (e.g., "image_punjabi_1171"
    # instead of "image_punjabi_1171.jpg"). We check for exact match first, then try
    # common image extensions as fallback. If a match is found, we normalize the Id
    # to include the extension so downstream code can always use Id directly as a filename.
    existing_images = set(p.name for p in img_dir.glob("*") if p.is_file())
    image_extensions = [".jpg", ".jpeg", ".png"]

    def _resolve_image_id(img_id: str) -> str | None:
        """Resolve an image ID to its actual filename on disk, or None if missing."""
        if img_id in existing_images:
            return img_id
        for ext in image_extensions:
            candidate = img_id + ext
            if candidate in existing_images:
                return candidate
        return None

    df_filtered_cat["Id"] = df_filtered_cat["Id"].apply(
        lambda img_id: _resolve_image_id(img_id) or img_id
    )
    image_exists_mask = df_filtered_cat["Id"].apply(lambda img_id: img_id in existing_images)
    missing_img_count = int((~image_exists_mask).sum())

    df_existing_img = df_filtered_cat[image_exists_mask].copy()

    # 4. Remove Duplicates
    initial_len = len(df_existing_img)
    df_clean = df_existing_img.drop_duplicates(subset=["Id"]).copy()
    df_clean = df_clean.drop_duplicates(subset=["Category", "Text"]).copy()
    duplicate_count = initial_len - len(df_clean)

    # 5. Clean Text String formatting (strip quotes and double spaces)
    df_clean["Text"] = df_clean["Text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # 6. Save Clean Dataset
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset (%d rows) to %s", len(df_clean), dest_path)

    # 7. Print Rich Summary Table
    summary = CleaningSummary(
        total_raw_rows=total_raw,
        parsed_rows=len(df_raw),
        missing_category_rows=missing_cat_count,
        invalid_category_rows=invalid_cat_count,
        missing_image_rows=missing_img_count,
        duplicate_rows=duplicate_count,
        final_clean_rows=len(df_clean),
    )
    _print_summary(summary)

    return df_clean


def _print_summary(summary: CleaningSummary) -> None:
    """Print clean terminal summary table of dataset pipeline status."""
    table = Table(
        title="IMUSA Dataset Cleaning Report", show_header=True, header_style="bold magenta"
    )
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Total Raw Rows Parsed", str(summary.total_raw_rows))
    table.add_row("Dropped (Missing Category)", str(summary.missing_category_rows))
    table.add_row("Dropped (Invalid Category)", str(summary.invalid_category_rows))
    table.add_row("Dropped (Missing Image File)", str(summary.missing_image_rows))
    table.add_row("Dropped (Duplicates)", str(summary.duplicate_rows))
    table.add_row("Final Clean Dataset Size", str(summary.final_clean_rows), style="bold gold1")

    console.print(table)
