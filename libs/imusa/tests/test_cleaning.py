"""Unit tests for dataset cleaning logic."""

from pathlib import Path

from PIL import Image

from imusa.data.cleaning import clean_dataset, parse_raw_csv


def test_parse_raw_csv(tmp_path: Path) -> None:
    """Test parsing raw CSV with multiline and quoted strings."""
    csv_file = tmp_path / "raw.csv"
    csv_file.write_text(
        'Id,Category,Text\nimg1.jpg,Sarcasm,"Line 1\nLine 2"\nimg2.jpg,Neutral,Simple string\n',
        encoding="utf-8",
    )

    df = parse_raw_csv(csv_file)
    assert len(df) == 2
    assert df.iloc[0]["Id"] == "img1.jpg"
    assert df.iloc[0]["Category"] == "Sarcasm"
    assert "Line 1" in df.iloc[0]["Text"]


def test_clean_dataset_pipeline(tmp_path: Path) -> None:
    """Test full cleaning pipeline filtering missing files and invalid categories."""
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text(
        "Id,Category,Text\n"
        "img1.jpg,Sarcasm,Valid text 1\n"
        "img2.jpg,InvalidCat,Text 2\n"
        "img3.jpg,,Missing cat text\n"
        "img4.jpg,Motivational,Valid text 4\n",
        encoding="utf-8",
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy images for img1 and img4
    Image.new("RGB", (100, 100)).save(img_dir / "img1.jpg")
    Image.new("RGB", (100, 100)).save(img_dir / "img4.jpg")

    out_csv = tmp_path / "clean.csv"

    df_clean = clean_dataset(raw_csv=raw_csv, images_dir=img_dir, output_csv=out_csv)

    assert len(df_clean) == 2
    assert set(df_clean["Id"]) == {"img1.jpg", "img4.jpg"}
    assert out_csv.exists()
