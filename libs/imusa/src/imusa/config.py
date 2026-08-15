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

    # Model Version & Architecture Configuration
    model_version: str = "v1"
    freeze_strategy: str = "none"  # Options: "none", "freeze", "lpft"
    num_folds: int = 1  # 1 = single train/val split (V1), 5 = 5-fold CV (V2)
    label_smoothing: float = 0.0  # Default 0.0 for V1; 0.05 for V2
    use_mixup: bool = False  # Enable manifold mixup in fusion space
    mixup_alpha: float = 0.2  # Beta distribution alpha for mixup ratio sampling

    # Pre-trained Model Hub Names
    vision_model_name: str = "google/vit-base-patch16-224"
    text_model_name: str = "xlm-roberta-base"

    # Sentiment Class Definitions
    categories: ClassVar[list[str]] = ["Sarcasm", "Neutral", "Offensive", "Motivational"]
    num_classes: ClassVar[int] = 4

    @property
    def versioned_output_dir(self) -> Path:
        """Return version-specific output directory (e.g. outputs/v1 or outputs/v2)."""
        return self.output_dir / self.model_version

    @property
    def checkpoint_dir(self) -> Path:
        """Return version-specific checkpoint directory (e.g. outputs/v1/checkpoints)."""
        return self.versioned_output_dir / "checkpoints"

    @property
    def submission_path(self) -> Path:
        """Return version-specific submission CSV path (e.g. outputs/v1/submission.csv)."""
        return self.versioned_output_dir / "submission.csv"

    @property
    def calibration_dir(self) -> Path:
        """Return version-specific calibration directory (e.g. outputs/v2/calibration)."""
        return self.versioned_output_dir / "calibration"

    def ensure_directories(self) -> None:
        """Create output, processed, versioned, checkpoint, and calibration directories if needed."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exploration_output_dir.mkdir(parents=True, exist_ok=True)
        self.versioned_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
