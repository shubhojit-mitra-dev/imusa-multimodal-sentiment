"""Central Configuration Module for IMUSA Project using Pydantic BaseSettings."""

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and Machine Learning Settings."""

    model_config = SettingsConfigDict(
        env_prefix="IMUSA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Directory (Project Root)
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent.parent

    # Data Paths
    data_dir: Path = base_dir / "data"
    raw_train_csv: Path = data_dir / "train" / "train_punjabi_dataset.csv"
    train_images_dir: Path = data_dir / "train" / "Training_images"
    raw_test_csv: Path = data_dir / "test" / "Test.csv"
    test_images_dir: Path = data_dir / "test" / "Testing_images"

    # Processed Data Outputs
    processed_dir: Path = data_dir / "processed"
    processed_train_csv: Path = processed_dir / "train_clean.csv"

    # Output Artifacts & Reports
    output_dir: Path = base_dir / "outputs"
    exploration_output_dir: Path = output_dir / "exploration"

    # Sentiment Class Definitions
    categories: ClassVar[list[str]] = ["Sarcasm", "Neutral", "Offensive", "Motivational"]
    num_classes: ClassVar[int] = 4

    def ensure_directories(self) -> None:
        """Create output and processed directories if they do not exist."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exploration_output_dir.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
