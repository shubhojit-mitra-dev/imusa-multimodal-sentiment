"""Unit tests for Settings configuration."""

from imusa.config import settings


def test_settings_categories() -> None:
    """Verify sentiment categories defined in configuration."""
    assert len(settings.categories) == 4
    assert "Sarcasm" in settings.categories
    assert "Motivational" in settings.categories
    assert settings.num_classes == 4


def test_ensure_directories(tmp_path) -> None:
    """Verify directory creation logic."""
    settings.output_dir = tmp_path / "outputs"
    settings.processed_dir = tmp_path / "data" / "processed"
    settings.exploration_output_dir = tmp_path / "outputs" / "exploration"

    settings.ensure_directories()

    assert settings.output_dir.exists()
    assert settings.processed_dir.exists()
    assert settings.exploration_output_dir.exists()
