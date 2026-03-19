.PHONY: help test lint format type-check build clean install-dev

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run test suite
	python -m pytest tests/ -vvv

lint: ## Run linter (flake8)
	python -m flake8 src/ tests/ --max-line-length=127 --select=E,W,F --exclude=.tox,*.egg,build,data,__pycache__

format: ## Format code with black
	python -m black src/ tests/

type-check: ## Run type checker (mypy)
	python -m mypy src/fsm_llm/ --ignore-missing-imports

build: ## Build wheel and sdist
	@echo "Building package..."
	python -m build

clean: ## Remove build artifacts and caches
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ logs/
	rm -rf tests/__pycache__
	rm -rf tests/test_fsm_llm/__pycache__
	rm -rf tests/test_fsm_llm_2/__pycache__
	rm -rf src/logs*
	rm -rf src/fsm_llm*egg-info
	rm -rf src/fsm_llm/__pycache__
	rm -rf src/fsm_llm_2/__pycache__
	rm -rf src/fsm_llm_workflows/__pycache__
	rm -rf src/fsm_llm_reasoning/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install-dev: ## Install package in development mode with all extras
	pip install -e ".[dev,workflows]"
	pre-commit install
