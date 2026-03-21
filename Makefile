.PHONY: help test lint format type-check build clean install-dev coverage

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
	python -m mypy src/fsm_llm/ src/fsm_llm_reasoning/ src/fsm_llm_workflows/ src/fsm_llm_classification/ src/fsm_llm_agents/ --ignore-missing-imports

coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=fsm_llm --cov=fsm_llm_classification --cov=fsm_llm_reasoning --cov=fsm_llm_workflows --cov=fsm_llm_agents --cov-report=term-missing --cov-report=html

build: ## Build wheel and sdist
	@echo "Building package..."
	python -m build

clean: ## Remove build artifacts and caches
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ logs/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf tests/__pycache__
	rm -rf tests/test_fsm_llm/__pycache__
	rm -rf src/logs*
	rm -rf src/fsm_llm*egg-info
	rm -rf src/fsm_llm/__pycache__
	rm -rf src/fsm_llm_workflows/__pycache__
	rm -rf src/fsm_llm_reasoning/__pycache__
	rm -rf src/fsm_llm_classification/__pycache__
	rm -rf src/fsm_llm_agents/__pycache__
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install-dev: ## Install package in development mode with all extras
	pip install -e ".[dev,workflows,classification,reasoning,agents]"
	pre-commit install
