.PHONY: help install sync lint format test clean-data explore setup-hooks all

PYTHON := uv run python

help:
	@echo "IMUSA Monorepo Developer Commands:"
	@echo "  make install     - Sync all virtual environments and dependencies"
	@echo "  make lint        - Run ruff linter and type checker"
	@echo "  make format      - Run ruff formatter"
	@echo "  make test        - Run unit tests with pytest and coverage report"
	@echo "  make clean-data  - Clean raw train CSV and generate processed data"
	@echo "  make explore     - Generate dataset statistics and visualization report"
	@echo "  make pack-data   - Package data/ directory into data.zip for Google Colab upload"
	@echo "  make setup-hooks - Install git pre-commit hooks"
	@echo "  make all         - Run lint, test, and data pipeline verification"

install:
	uv sync --all-packages

sync: install

lint:
	uv run ruff check .
	uv run mypy libs/imusa/src

format:
	uv run ruff format .

test:
	uv run pytest

clean-data:
	$(PYTHON) scripts/clean_data.py

explore:
	$(PYTHON) scripts/explore_data.py

pack-data:
	zip -r data.zip data/ -x "*.DS_Store*" "*__pycache__*"

setup-hooks:
	uv run pre-commit install

all: lint test clean-data explore
