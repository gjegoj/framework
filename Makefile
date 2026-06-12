.PHONY: help install pre-commit clean test test-unit typecheck check generate-smoke smoke smoke-embeddings smoke-ranking smoke-contrastive smoke-arcface


help: ## Show help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

clean: ## Clean cache and temporary files
	@echo "Cleaning cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "mypy-report" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".gitlab-ci-local" -exec rm -rf {} + 2>/dev/null || true

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests (exclude slow)
	uv run pytest tests/ -v -m "not slow"

typecheck: ## Run mypy static analysis
	uv run mypy src tests

check: typecheck test ## Run type checks and tests

generate-smoke: ## Generate synthetic smoke dataset in data/smoke/
	uv run python scripts/generate_smoke_data.py

smoke: generate-smoke ## Run 2-epoch offline smoke training (CPU, no pretrained weights)
	uv run python main.py

smoke-embeddings: generate-smoke ## Run 2-epoch offline embeddings smoke (CPU, M6 modality)
	uv run python main.py experiment=embeddings_smoke

smoke-ranking: generate-smoke ## Run 2-epoch offline ranking smoke (CPU, M7a triplet)
	uv run python main.py experiment=ranking_smoke

smoke-contrastive: generate-smoke ## Run 2-epoch offline contrastive smoke (CPU, M7b dual-encoder)
	uv run python main.py experiment=contrastive_smoke

smoke-arcface: generate-smoke ## Run 2-epoch offline ArcFace smoke (CPU, cosine head + angular margin)
	uv run python main.py experiment=arcface_smoke
