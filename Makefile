.PHONY: help install pre-commit clean test test-unit typecheck check test-run

PET_TABLE := data/pet/data.csv


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

test-unit: ## Run unit tests only (tests/unit — the pre-commit gate)
	uv run pytest tests/unit -v -m "not slow"

typecheck: ## Run mypy static analysis
	uv run mypy src tests

check: typecheck test ## Run type checks and tests

# A file target, not a phony one: the fetch happens once and every later run reuses it.
# torchvision downloads ~800 MB of Oxford-IIIT Pet, then the script remaps 7349 trimaps
# into 0-based masks and writes the table the examples read.
$(PET_TABLE):
	uv run python scripts/prepare_pet.py

test-run: $(PET_TABLE) ## Fetch the pet dataset (once) and train the multitask example on a slice
	uv run main.py experiment=examples/multitask epochs=2 +data.max_samples=256
