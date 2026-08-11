# AGENTS.md — Repository Operational Instructions for AI Agents

> This document defines operational standards, architecture rules, build/test procedures, and behavioral boundaries for AI coding agents operating in this repository.

---

## 1. Executive Summary & Domain Context

- **Task**: Indic Meme Understanding & Sentiment Analysis (IMUSA) shared task.
- **Goal**: Classify multimodal Punjabi memes into **4 sentiment categories**: `Sarcasm`, `Neutral`, `Offensive`, `Motivational`.
- **Primary Input Modalities**: 
  1. Image (`.jpg`/`.png`) containing visual content and embedded Gurmukhi text.
  2. Text string (extracted Gurmukhi script).
- **Core Engineering Philosophy**: Staff-engineer level monorepo quality. Clean boundaries between data pipeline (`libs/imusa/data`), model definitions (`libs/imusa/models`), training orchestration (`libs/imusa/training`), serving (`apps/api`), and frontend (`apps/frontend`).

---

## 2. Technical Stack & Standards

| Layer | Standard / Tooling |
|---|---|
| **Python Runtime** | Python 3.12 (pinned via `.python-version`) |
| **Package Manager** | `uv` (Workspace mode, single `uv.lock`) |
| **ML Framework** | PyTorch + Hugging Face Transformers |
| **Code Style & Formatting** | `ruff` (line-length: 100, quote-style: double) |
| **Static Type Checking** | `mypy` (strict type annotations required on public interfaces) |
| **Testing** | `pytest` with coverage requirements |
| **Serving Layer** | FastAPI (ASGI) |
| **Frontend** | Next.js + React + Tailwind CSS |
| **Distributed Systems** | PyTorch DDP on Kubernetes / Kubeflow Trainer |

---

## 3. Repository Topology

```
multimodal-ai-project/
├── apps/                 # Application entrypoints (API server, web dashboard)
├── libs/                 # Core shared libraries (imusa package)
│   └── imusa/            # Primary Python package
│       ├── src/imusa/    # Source code (src-layout)
│       └── tests/        # Pytest test suite
├── experiments/          # Non-production exploration scripts and notebooks
├── infra/                # Dockerfiles and Kubernetes manifests
├── scripts/              # Command-line entry points
├── data/                 # Raw and processed datasets (gitignored)
└── outputs/              # Logs, plots, model checkpoints (gitignored)
```

---

## 4. Key Developer Commands

All commands MUST be executed via `uv` or `make`:

```bash
# Environment sync
make install              # Runs `uv sync --all-packages`

# Code Quality & Testing
make lint                 # Runs `uv run ruff check .` and `uv run mypy`
make format               # Runs `uv run ruff format .`
make test                 # Runs `uv run pytest`

# Data Pipeline
make clean-data           # Executes dataset cleaning script
make explore              # Executes dataset statistical analysis script
```

---

## 5. Strict Constraints for AI Agents

1. **No Code Without Verification**: Never claim a feature or fix works without executing `make lint` and `make test`.
2. **Never Edit Git-Ignored Outputs**: Do not commit binaries, weights (`.pt`, `.pth`), or processed data files.
3. **Atomic Commits**: Make small, incremental, logically distinct commits with descriptive commit messages following Conventional Commits format (`feat:`, `fix:`, `chore:`, `docs:`).
4. **No Hidden Logic**: Avoid magic numbers and hardcoded paths. All configuration options MUST derive from `imusa.config.IMUSAConfig` (Pydantic BaseSettings).
5. **Preserve Type Annotations**: Every new function must include complete type hints.

---

## 6. Definition of Done (DoD)

A task is considered **DONE** only when:
- [ ] Code is formatted with `ruff format`.
- [ ] Code passes `ruff check` with zero warnings.
- [ ] Code passes `mypy` type checking.
- [ ] Relevant unit tests exist in `libs/imusa/tests/` and pass via `pytest`.
- [ ] Clear documentation and docstrings accompany new functions.
