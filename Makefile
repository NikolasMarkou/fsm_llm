.PHONY: help test lint format type-check lint-imports build clean install-dev coverage audit counts api-snapshot

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run test suite
	python -m pytest tests/ -v

lint: ## Run linter (ruff)
	python -m ruff check src/ tests/

format: ## Format code with ruff
	python -m ruff format src/ tests/

type-check: ## Run type checker (mypy)
	python -m mypy -p fsm_llm

lint-imports: ## Check the import contracts in .importlinter (import-linter)
	lint-imports

coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=fsm_llm --cov-branch --cov-report=term-missing --cov-report=html

build: ## Build wheel and sdist
	@echo "Building package..."
	python -m build

clean: ## Remove build artifacts and caches
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ logs/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf tests/__pycache__
	rm -rf tests/test_fsm_llm/__pycache__
	rm -rf src/logs* src/fsm_llm/logs*
	rm -rf src/fsm_llm/__pycache__
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install-dev: ## Install package in development mode with all extras
	pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,eval,synthesis,testing]"
	pre-commit install

audit: ## Audit site-packages for suspicious .pth files
	python scripts/audit_pth.py

counts: ## Fail when a documented test count differs from a collection
	python scripts/doc_counts.py

api-snapshot: ## Regenerate docs/api.txt, the public API snapshot
	python scripts/api_snapshot.py --write docs/api.txt
